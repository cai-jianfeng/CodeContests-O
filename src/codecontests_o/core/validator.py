"""
Solution validator

Validates solution performance on generated test cases
"""

import queue
import threading
import json
import os
from typing import List, Dict, Any, Optional

from ..data.models import Sample, TestCase, Language
from ..clients.sandbox_client import SandboxClient
from ..utils.helpers import check_output_equal, extract_code
from ..utils.logger import log_sample
from ..parallel.api_pool import acquire_api, release_api
from ..prompts.templates import SOLUTION_CODE_TEMPLATE


class SolutionValidator:
    """
    Solution Validator
    
    Validates the performance of correct and incorrect solutions on generated test cases
    """
    
    def __init__(
        self,
        sandbox_client: SandboxClient,
        testlib_files: Dict[str, str]
    ):
        """
        Initialize validator
        
        Args:
            sandbox_client: Sandbox client
            testlib_files: testlib files (for checker)
        """
        self.sandbox_client = sandbox_client
        self.testlib_files = testlib_files
    
    def validate_solutions(
        self,
        sample: Sample,
        corner_cases: List[TestCase],
        max_workers: int = 4,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Validate all solutions
        
        Args:
            sample: Sample data
            corner_cases: List of test cases
            max_workers: Max parallel worker threads
            save_dir: Result saving directory
            
        Returns:
            Dict: {
                'solution_result': Failed results among correct solutions,
                'incorrect_solution_result': Passed results among incorrect solutions
            }
        """
        sample_id = sample.id
        
        # Convert test case format
        test_cases = [tc.to_dict() for tc in corner_cases]
        
        # Validate correct solutions
        log_sample(sample_id, "Validating correct solutions...")
        correct_results = self._validate_solution_list(
            solutions=sample.correct_solutions,
            test_cases=test_cases,
            checker=sample.checker,
            sample_id=sample_id,
            max_workers=max_workers,
            save_dir=save_dir,
            is_correct=True
        )
        
        # Validate incorrect solutions
        log_sample(sample_id, "Validating incorrect solutions...")
        incorrect_results = self._validate_solution_list(
            solutions=sample.incorrect_solutions,
            test_cases=test_cases,
            checker=sample.checker,
            sample_id=sample_id,
            max_workers=max_workers,
            save_dir=save_dir,
            is_correct=False
        )
        
        # Filter results
        # Correct solutions should all pass, return failed ones (false negatives)
        solution_result = [r for r in correct_results if not r['result'].get('accepted', False)]
        # Incorrect solutions should fail, return passed ones (false positives)
        incorrect_solution_result = [r for r in incorrect_results if r['result'].get('accepted', False)]
        
        log_sample(sample_id, f"Found {len(solution_result)} false negatives, {len(incorrect_solution_result)} false positives")
        
        return {
            'solution_result': solution_result,
            'incorrect_solution_result': incorrect_solution_result
        }
    
    def _validate_solution_list(
        self,
        solutions: List,
        test_cases: List[Dict],
        checker: Optional[str],
        sample_id: str,
        max_workers: int,
        save_dir: Optional[str],
        is_correct: bool
    ) -> List[Dict]:
        """Validate solution list"""
        if not solutions:
            return []
        
        results = []
        results_lock = threading.Lock()
        
        # Create task queue
        task_queue = queue.Queue()
        for idx, solution in enumerate(solutions):
            if solution.language == Language.UNKNOWN:
                continue
            task_queue.put((idx, solution))
        
        if task_queue.empty():
            return []
        
        def worker(worker_id: str):
            api_path = None
            session = None
            
            while True:
                try:
                    idx, solution = task_queue.get(timeout=1)
                except queue.Empty:
                    if api_path:
                        release_api(api_path, session)
                    break
                
                # Check if there are cached results
                if save_dir:
                    result_file = os.path.join(
                        save_dir,
                        f"{'solution' if is_correct else 'incorrect_solution'}_{idx}.json"
                    )
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r') as f:
                                cached = json.load(f)
                            if len(cached.get('tests', [])) == len(test_cases):
                                with results_lock:
                                    results.append({
                                        'language': solution.language.value,
                                        'solution': solution.code,
                                        'result': cached
                                    })
                                task_queue.task_done()
                                continue
                        except Exception:
                            pass
                
                # Get API endpoint
                if api_path is None:
                    api_pair = acquire_api(timeout=0.1)
                    if api_pair is None:
                        task_queue.put((idx, solution))
                        import time
                        time.sleep(0.01)
                        continue
                    api_path, session = api_pair
                
                try:
                    language = solution.language.value
                    
                    # Validate solution
                    result = self._validate_single_solution(
                        api_path=api_path,
                        code=solution.code,
                        language=language,
                        test_cases=test_cases,
                        checker=checker,
                        sample_id=sample_id,
                        session=session
                    )
                    
                    # Save results
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        result_file = os.path.join(
                            save_dir,
                            f"{'solution' if is_correct else 'incorrect_solution'}_{idx}.json"
                        )
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)
                    
                    with results_lock:
                        results.append({
                            'language': language,
                            'solution': solution.code,
                            'result': result
                        })
                
                except Exception as e:
                    log_sample(sample_id, f"Error validating solution {idx}: {e}")
                
                finally:
                    task_queue.task_done()
        
        # Start worker threads
        threads = []
        num_workers = min(max_workers, task_queue.qsize())
        for i in range(num_workers):
            t = threading.Thread(target=worker, args=(f"validator_{i}",))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        return results
    
    def _validate_single_solution(
        self,
        api_path: str,
        code: str,
        language: str,
        test_cases: List[Dict],
        checker: Optional[str],
        sample_id: str,
        session
    ) -> Dict[str, Any]:
        """Validate a single solution"""
        # Extract code
        extracted_code = extract_code(code, language)
        
        tests_result = []
        accepted = True
        
        for idx, test_case in enumerate(test_cases):
            stdin = test_case.get('input', {}).get('stdin', '')
            expected_stdout = test_case.get('output', {}).get('stdout', '')
            
            # Run solution
            result = self.sandbox_client.run_solution(
                api_path=api_path,
                code=extracted_code,
                language=language,
                stdin=stdin,
                session=session,
                sample_id=sample_id,
                logger=log_sample
            )
            
            if not result['success']:
                passed = False
            else:
                passed = check_output_equal(expected_stdout, result['output'])
            
            # If failed and checker exists, try to validate using checker
            if not passed and checker and result['success']:
                checker_result = self.sandbox_client.run_checker(
                    api_path=api_path,
                    checker_code=checker,
                    stdin=stdin,
                    actual_output=result['output'],
                    expected_output=expected_stdout,
                    files=self.testlib_files,
                    session=session,
                    sample_id=sample_id,
                    logger=log_sample
                )
                if checker_result['success']:
                    passed = True
            
            if not passed:
                accepted = False
            
            tests_result.append({
                'passed': passed,
                'exec_info': result.get('response'),
                'test_index': idx,
            })
        
        return {
            'id': sample_id,
            'accepted_nochecker': all(t['passed'] for t in tests_result),
            'accepted': accepted,
            'extracted_code': extracted_code,
            'tests': tests_result,
        }
