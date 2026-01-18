"""
Sandbox API Client Wrapper

Used for code execution and solution validation
"""

import requests
from typing import Dict, Optional, List, Any
from ..utils.helpers import check_output_equal, extract_code


class SandboxClient:
    """
    Sandbox API Client
    
    Provides code execution and solution validation functionality
    """
    
    def __init__(
        self,
        compile_timeout: int = 20,
        run_timeout: int = 20,
        request_timeout: int = 50
    ):
        """
        Initialize Sandbox Client
        
        Args:
            compile_timeout: Compile timeout
            run_timeout: Run timeout
            request_timeout: HTTP request timeout
        """
        self.compile_timeout = compile_timeout
        self.run_timeout = run_timeout
        self.request_timeout = request_timeout
        self.session = requests.Session()
    
    def call_api(
        self,
        api_path: str,
        payload: Dict,
        session: Optional[requests.Session] = None,
        sample_id: str = "",
        logger=None
    ) -> Optional[Dict]:
        """
        Call Sandbox API
        
        Args:
            api_path: API path
            payload: Request data
            session: HTTP session (optional)
            sample_id: Sample ID (for logging)
            logger: Logger function
            
        Returns:
            Dict: API response, None if failed
        """
        try:
            sess = session or self.session
            response = sess.post(api_path, json=payload, timeout=self.request_timeout)
            return response.json()
        except Exception as e:
            error_msg = f"Request failed ({api_path}): {e}"
            if logger:
                logger(sample_id, error_msg)
            return None
    
    def run_code(
        self,
        api_path: str,
        code: str,
        language: str,
        stdin: str = "",
        extra_args: str = "",
        files: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
        sample_id: str = "",
        logger=None
    ) -> Optional[Dict]:
        """
        Execute code
        
        Args:
            api_path: API path
            code: Code content
            language: Programming language
            stdin: Standard input
            extra_args: Extra command line arguments
            files: Additional files (base64 encoded)
            session: HTTP session
            sample_id: Sample ID
            logger: Logger function
            
        Returns:
            Dict: Execution result
        """
        payload = {
            "code": code,
            "language": language,
            "stdin": stdin,
            "compile_timeout": self.compile_timeout,
            "run_timeout": self.run_timeout,
        }
        
        if extra_args:
            payload["extra_args"] = extra_args
        if files:
            payload["files"] = files
        
        return self.call_api(
            api_path + "run_code",
            payload,
            session=session,
            sample_id=sample_id,
            logger=logger
        )
    
    def run_generator(
        self,
        api_path: str,
        generator_code: str,
        command: str,
        files: Dict[str, str],
        session: Optional[requests.Session] = None,
        sample_id: str = "",
        logger=None
    ) -> Dict[str, Any]:
        """
        Run test case generator
        
        Args:
            api_path: API path
            generator_code: Generator code
            command: Command (e.g. "./gen --n 100")
            files: Additional files (including testlib.h)
            session: HTTP session
            sample_id: Sample ID
            logger: Logger function
            
        Returns:
            Dict: Result containing success, output, error
        """
        # Extract command arguments
        extra_args = command.replace("./gen ", "").strip()
        
        response = self.run_code(
            api_path=api_path,
            code=generator_code,
            language="cpp",
            extra_args=extra_args,
            files=files,
            session=session,
            sample_id=sample_id,
            logger=logger
        )
        
        if response is None:
            return {
                "success": False,
                "output": "",
                "error": "No response from sandbox API"
            }
        
        # Check execution status
        if response.get('status') == "Success":
            return {
                "success": True,
                "output": response.get('run_result', {}).get('stdout', ''),
                "error": ""
            }
        
        # Handle special case: compilation success but with testlib warnings
        compile_result = response.get('compile_result', {})
        run_result = response.get('run_result', {})
        
        if (compile_result and compile_result.get('status') == "Finished" and
            run_result and run_result.get('status') == 'Finished' and
            'FAIL Opts: unused key' in run_result.get('stderr', '') and
            run_result.get('stdout', '').strip()):
            return {
                "success": True,
                "output": run_result.get('stdout', ''),
                "error": ""
            }
        
        # Construct error message
        compile_error = ""
        if compile_result:
            compile_error = f"Compile error (code {compile_result.get('return_code', '')}): {compile_result.get('stderr', '')}"
        
        run_error = ""
        if run_result:
            run_error = f"Runtime error (code {run_result.get('return_code', '')}): {run_result.get('stderr', '')}"
        
        return {
            "success": False,
            "output": "",
            "error": f"{compile_error}; {run_error}".strip("; ")
        }
    
    def run_solution(
        self,
        api_path: str,
        code: str,
        language: str,
        stdin: str,
        session: Optional[requests.Session] = None,
        sample_id: str = "",
        logger=None
    ) -> Dict[str, Any]:
        """
        Run solution code
        
        Args:
            api_path: API path
            code: Solution code
            language: Programming language
            stdin: Test input
            session: HTTP session
            sample_id: Sample ID
            logger: Logger function
            
        Returns:
            Dict: Result containing success, output, error
        """
        response = self.run_code(
            api_path=api_path,
            code=code,
            language=language,
            stdin=stdin,
            session=session,
            sample_id=sample_id,
            logger=logger
        )
        
        if response is None:
            return {
                "success": False,
                "output": "",
                "error": "No response from sandbox API",
                "response": None
            }
        
        if response.get('status') == "Success":
            return {
                "success": True,
                "output": response.get('run_result', {}).get('stdout', ''),
                "error": "",
                "response": response
            }
        
        return {
            "success": False,
            "output": "",
            "error": str(response),
            "response": response
        }
    
    def validate_solution(
        self,
        api_path: str,
        code: str,
        language: str,
        test_cases: List[Dict],
        session: Optional[requests.Session] = None,
        sample_id: str = "",
        logger=None
    ) -> Dict[str, Any]:
        """
        Validate solution
        
        Args:
            api_path: API path
            code: Solution code
            language: Programming language
            test_cases: List of test cases [{"input": {"stdin": ...}, "output": {"stdout": ...}}]
            session: HTTP session
            sample_id: Sample ID
            logger: Logger function
            
        Returns:
            Dict: Validation result
        """
        # Extract code
        extracted_code = extract_code(code, language)
        
        tests_result = []
        accepted = True
        
        for idx, test_case in enumerate(test_cases):
            stdin = test_case.get('input', {}).get('stdin', '')
            expected_stdout = test_case.get('output', {}).get('stdout', '')
            
            result = self.run_solution(
                api_path=api_path,
                code=extracted_code,
                language=language,
                stdin=stdin,
                session=session,
                sample_id=sample_id,
                logger=logger
            )
            
            if not result['success']:
                passed = False
            else:
                passed = check_output_equal(expected_stdout, result['output'])
            
            if not passed:
                accepted = False
            
            tests_result.append({
                "passed": passed,
                "exec_info": result.get('response'),
                "test_info": test_case,
                "test_index": idx,
            })
        
        return {
            'id': sample_id,
            'accepted_nochecker': accepted,
            'extracted_code': extracted_code,
            'tests': tests_result,
        }
    
    def run_checker(
        self,
        api_path: str,
        checker_code: str,
        stdin: str,
        actual_output: str,
        expected_output: str,
        files: Dict[str, str],
        session: Optional[requests.Session] = None,
        sample_id: str = "",
        logger=None
    ) -> Dict[str, Any]:
        """
        Run checker to verify output
        
        Args:
            api_path: API path
            checker_code: Checker code
            stdin: Original input
            actual_output: Actual output
            expected_output: Expected output
            files: Additional files (including testlib.h and input/output files)
            session: HTTP session
            sample_id: Sample ID
            logger: Logging function
            
        Returns:
            Dict: Checker result
        """
        import base64
        
        # Encode files
        checker_files = files.copy()
        checker_files["input.txt"] = base64.b64encode(stdin.encode('utf-8')).decode('utf-8')
        checker_files["output.txt"] = base64.b64encode(actual_output.encode('utf-8')).decode('utf-8')
        checker_files["answer.txt"] = base64.b64encode(expected_output.encode('utf-8')).decode('utf-8')
        
        response = self.run_code(
            api_path=api_path,
            code=checker_code,
            language="cpp",
            extra_args="input.txt output.txt answer.txt",
            files=checker_files,
            session=session,
            sample_id=sample_id,
            logger=logger
        )
        
        if response is None:
            return {"success": False, "error": "No response from checker"}
        
        return {
            "success": response.get('status') == "Success",
            "response": response
        }
    
    def setup_session_pool(self, pool_size: int):
        """Set connection pool size"""
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            pool_maxsize=pool_size
        ))
