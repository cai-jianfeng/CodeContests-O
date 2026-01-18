"""
Corner Case Generator

Core test case generation logic
"""

import json
import os
import queue
import threading
import time
import random
import base64
from typing import List, Dict, Tuple, Any
from collections import Counter

from ..data.models import Sample, TestCase, IterationResult
from ..clients.openai_client import OpenAIClient, InitCommandModel
from ..clients.sandbox_client import SandboxClient
from ..config.base import Config
from ..utils.helpers import apply_code_patches
from ..utils.logger import log_sample
from ..parallel.api_pool import acquire_api, release_api
from ..prompts.templates import (
    SYSTEM_PROMPT,
    TESTLIB_HEADER_COMMENT,
    INIT_PROMPT_TEMPLATE_WITH_GENERATOR,
    INIT_PROMPT_TEMPLATE_WITHOUT_GENERATOR,
    REFINE_PROMPT_TEMPLATE,
    INIT_RESPONSE_TEMPLATE_WGEN,
    INIT_RESPONSE_TEMPLATE_WOGEN,
    REFINE_RESPONSE_TEMPLATE,
    SOLUTION_RESULT_TEMPLATE,
    TEST_CASE_RESULT_TEMPLATE,
    CANONICAL_SOLUTION_TEMPLATE,
)


class CornerCaseGenerator:
    """
    Corner Case Generator
    
    Implements feedback-driven iterative test case generation
    """
    
    def __init__(
        self,
        openai_client: OpenAIClient,
        sandbox_client: SandboxClient,
        config: Config,
        testlib_files: Dict[str, str]
    ):
        """
        Initialize the generator
        
        Args:
            openai_client: OpenAI client
            sandbox_client: Sandbox client
            config: Configuration
            testlib_files: testlib files (base64 encoded)
        """
        self.openai_client = openai_client
        self.sandbox_client = sandbox_client
        self.config = config
        self.testlib_files = testlib_files
        
        self.max_iterations = config.processing.max_iterations
        self.max_sample_solutions = config.processing.max_sample_solutions
        self.truncated_length = config.processing.truncated_length
        self.compress_messages = config.processing.compress_messages
    
    def generate_for_sample(
        self,
        sample: Sample,
        output_workers: int = 4,
        validation_workers: int = 4,
        only_generate: bool = False,
        validator = None,
        results_dir: str = ""
    ) -> Tuple[List[TestCase], List[str], List[IterationResult]]:
        """
        Generate test cases for a single sample

        Args:
            sample: Sample data
            output_workers: Output generation concurrency
            validation_workers: Validation concurrency
            only_generate: Generate only without validation
            validator: Solution validator
            results_dir: Result saving directory (for resume)

        Returns:
            Tuple: (List of test cases, List of commands, List of iteration results)
        """
        sample_id = sample.id

        # State variables
        commands: List[str] = []
        corner_cases: List[TestCase] = []
        all_results: List[IterationResult] = []
        current_generator = sample.generator
        input_constraints_summary = ""
        all_search_replace_blocks: List[str] = []
        begin = 0
        last_only_generate = False

        # Check if there are previous results to resume from
        if results_dir and os.path.exists(os.path.join(results_dir, f"{sample_id}.json.bak")) and not only_generate:
            backup_path = os.path.join(results_dir, f"{sample_id}.json.bak")
            try:
                with open(backup_path, "r") as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get('result', [])

                    if existing_results:
                        begin = len(existing_results)
                        last_result = existing_results[-1]

                        # Resume condition: all_results is not empty and the last result is not in only_generate mode
                        if not last_result.get('only_generate', False):
                            log_sample(sample_id, f"Resuming from iteration {begin} with {len(existing_results)} previous results")

                            # Restore state
                            corner_cases = [TestCase.from_dict(tc) for tc in last_result.get('corner_cases', [])]
                            input_constraints_summary = existing_results[0].get('input_constraints_summary', input_constraints_summary)
                            commands = last_result.get('generate_commands', last_result.get('commands', []))
                            if isinstance(commands, str):
                                commands = eval(commands)

                            # Restore generator: find the last iteration with improved_generator among all results
                            for result in existing_results:
                                if result.get('improved_generator', ''):
                                    current_generator = result.get('improved_generator')
                                all_search_replace_blocks.extend(result.get('search_replace_generator_blocks', []))

                            # Restore existing results
                            all_results = [IterationResult.from_dict(r) for r in existing_results]

                            log_sample(sample_id, f"Resumed generator from previous iteration")

                        # If the restored result is in only_generate mode, skip the first generation and proceed only with validation
                        elif begin == 1 and last_result.get('only_generate', False):
                            last_only_generate = True
                            begin = 0
                            log_sample(sample_id, f"Resuming from iteration {begin} with {len(existing_results)} previous results in only_generate mode")

                            # Restore state
                            corner_cases = [TestCase.from_dict(tc) for tc in last_result.get('corner_cases', [])]
                            input_constraints_summary = existing_results[0].get('input_constraints_summary', input_constraints_summary)
                            commands = last_result.get('generate_commands', last_result.get('commands', []))
                            if isinstance(commands, str):
                                commands = eval(commands)

                            current_generator = last_result.get('improved_generator', '') or current_generator
                            all_search_replace_blocks.extend(last_result.get('search_replace_generator_blocks', []))

                            # Restore existing results
                            all_results = [IterationResult.from_dict(r) for r in existing_results]

                            log_sample(sample_id, f"Resumed generator from previous iteration in only_generate mode")

            except Exception as e:
                log_sample(sample_id, f"Failed to restore from backup: {e}, starting fresh")
                begin = 0
                last_only_generate = False


        # Build initial prompt and messages
        # If in resume mode and messages exist, use restored messages; otherwise build initial messages
        if all_results and all_results[-1].messages:
            messages = all_results[-1].messages
            log_sample(sample_id, f"Restored messages from previous iteration")
        else:
            if sample.generator and sample.generator.strip():
                prompt = INIT_PROMPT_TEMPLATE_WITH_GENERATOR.format(
                    testlib_header=TESTLIB_HEADER_COMMENT,
                    problem_statement=sample.problem_statement,
                    generator=sample.generator
                )
            else:
                testlib_content = TESTLIB_HEADER_COMMENT
                if "testlib.h" in self.testlib_files:
                    try:
                        testlib_content = base64.b64decode(self.testlib_files["testlib.h"]).decode('utf-8')
                    except Exception as e:
                        log_sample(sample_id, f"Error decoding testlib.h: {e}")

                prompt = INIT_PROMPT_TEMPLATE_WITHOUT_GENERATOR.format(
                    problem_statement=sample.problem_statement,
                    testlib_content=testlib_content
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

        # If in only_generate mode, limit the number of iterations
        if only_generate:
            max_iterations = 1
        else:
            max_iterations = self.max_iterations

        for iteration in range(begin, max_iterations):
            log_sample(sample_id, f"Starting iteration {iteration}")

            # If in last_only_generate mode, skip command generation and test case generation, proceed directly to validation
            if last_only_generate and iteration == 0:
                log_sample(sample_id, "Skipping command generation in last_only_generate mode, using restored data")
                # Already restored corner_cases and commands from backup
                # Reset last_only_generate, used only for the first resume
                last_only_generate = False
                is_first = True

                # Use restored data to build dummy variables for subsequent processes
                search_replace_blocks = all_results[-1].search_replace_blocks if all_results else []
                unmatched_blocks = []
                case_inputs = all_results[-1].case_inputs if all_results else []
                input_errors = []
                command_to_input = {cmd: inp for cmd, inp in zip(commands, case_inputs)} if commands and case_inputs else {}
                output_errors = []

                # Build dummy command_response for subsequent processes
                command_response = InitCommandModel(
                    input_constraints_summary=input_constraints_summary,
                    command_list=commands,
                    search_replace_generator_blocks=search_replace_blocks,
                    generator=current_generator if not search_replace_blocks else None
                )

            else:
                # Generate commands
                is_first = (iteration == 0)
                command_response = self.openai_client.generate_command(
                    messages=messages,
                    is_first=is_first,
                    sample_id=sample_id,
                    logger=log_sample
                )

                # Process response
                if command_response == "Exceeded":
                    log_sample(sample_id, "Context length exceeded, stopping iteration")
                    if all_results:
                        all_results[-1].exceed_context = True
                    break

                if command_response is None:
                    log_sample(sample_id, "Failed to generate commands, stopping iteration")
                    break

                # Parse commands
                search_replace_blocks = command_response.search_replace_generator_blocks or []
                unmatched_blocks = []
                is_generator_updated = False

                if hasattr(command_response, 'generator') and command_response.generator:
                    current_generator = command_response.generator
                    is_generator_updated = True
                    log_sample(sample_id, "Received generated generator")
                elif search_replace_blocks:
                    current_generator, unmatched_blocks = apply_code_patches(
                        current_generator, search_replace_blocks
                    )
                    is_generator_updated = True
                    log_sample(sample_id, f"Applied {len(search_replace_blocks)} patches")
                    if unmatched_blocks:
                        log_sample(sample_id, f"Warning: {len(unmatched_blocks)} patches unmatched")

                all_search_replace_blocks.extend(search_replace_blocks)

                # Update command list
                if is_first:
                    input_constraints_summary = command_response.input_constraints_summary
                    commands.extend(command_response.command_list)
                else:
                    # Check if commands to be replaced are in the original list
                    missing_commands = [c for c in command_response.replace_command_list if c not in commands]
                    if missing_commands:
                        log_sample(sample_id, f"Warning: The following commands to replace are not in the original list: {missing_commands}")

                    # Remove commands to be replaced
                    commands = [c for c in commands if c not in command_response.replace_command_list]
                    # Add new commands
                    commands.extend(command_response.add_command_list)

                log_sample(sample_id, f"Total commands: {len(commands)}")

                if not commands:
                    log_sample(sample_id, "No commands to execute")
                    break

                # Generate test inputs
                log_sample(sample_id, "Generating test inputs...")
                inputs_result = self._generate_test_inputs(
                    commands=commands,
                    generator=current_generator,
                    sample_id=sample_id,
                    max_workers=output_workers
                )

                case_inputs = inputs_result['inputs']
                input_errors = inputs_result['errors']
                command_to_input = inputs_result['command_to_input']

                if not case_inputs:
                    log_sample(sample_id, "No test inputs generated")
                    break

                log_sample(sample_id, f"Generated {len(case_inputs)} test inputs")

                # Generate test outputs

                log_sample(sample_id, "Generating test outputs...")
                outputs_result = self._generate_test_outputs(
                    case_inputs=case_inputs,
                    sample=sample,
                    sample_id=sample_id,
                    max_workers=output_workers
                )

                corner_cases = outputs_result['corner_cases']
                output_errors = outputs_result['errors']

                if not corner_cases:
                    log_sample(sample_id, "No test outputs generated")
                    break

                log_sample(sample_id, f"Generated {len(corner_cases)} complete test cases")
            
            # Validation
            validation_result = {'solution_result': [], 'incorrect_solution_result': []}
            
            if not only_generate and validator:
                log_sample(sample_id, "Validating solutions...")
                validation_result = validator.validate_solutions(
                    sample=sample,
                    corner_cases=corner_cases,
                    max_workers=validation_workers
                )
                log_sample(sample_id, "Validation completed")
            
            # Save iteration results
            # In last_only_generate mode, command_response might not exist
            commands_add = []
            commands_replace = []
            if 'command_response' in locals() and command_response and not is_first:
                commands_add = command_response.add_command_list
                commands_replace = command_response.replace_command_list

            iteration_result = IterationResult(
                iteration=iteration,
                corner_cases=corner_cases,
                commands=commands.copy(),
                commands_add=commands_add,
                commands_replace=commands_replace,
                case_inputs=case_inputs,
                improved_generator=current_generator if is_generator_updated else "",
                search_replace_blocks=search_replace_blocks,
                unmatched_blocks=[str(b) for b in unmatched_blocks],
                input_constraints_summary=input_constraints_summary if is_first else "",
                validation_result=validation_result,
                errors={'input_errors': input_errors, 'output_errors': output_errors},
                only_generate=only_generate,
                messages=messages.copy()  # Save current messages for resume
            )
            all_results.append(iteration_result)
            
            # Check if iteration needs to continue
            if only_generate:
                break
            
            if not validation_result['solution_result'] and not validation_result['incorrect_solution_result']:
                log_sample(sample_id, "All solutions validated correctly, stopping iteration")
                break
            
            # Prepare next round of feedback
            messages = self._prepare_feedback_messages(
                messages=messages,
                command_response=command_response,
                commands=commands,
                command_to_input=command_to_input,
                input_errors=input_errors,
                corner_cases=corner_cases,
                output_errors=output_errors,
                validation_result=validation_result,
                input_constraints_summary=input_constraints_summary,
                improved_generator=current_generator if is_generator_updated else "",
                all_search_replace_blocks=all_search_replace_blocks,
                is_first=is_first,
                is_generator_updated=is_generator_updated,
                current_generator=current_generator
            )
        
        return corner_cases, commands, all_results
    
    def _generate_test_inputs(
        self,
        commands: List[str],
        generator: str,
        sample_id: str,
        max_workers: int
    ) -> Dict[str, Any]:
        """Generate test inputs in parallel"""
        results = {
            'inputs': [],
            'errors': [],
            'command_to_input': {}
        }
        
        task_queue = queue.Queue()
        for i, command in enumerate(commands):
            task_queue.put((i, command))
        
        results_lock = threading.Lock()
        temp_results = {}
        
        def worker(worker_id: str):
            api_path = None
            session = None
            
            while True:
                try:
                    i, command = task_queue.get(timeout=1)
                except queue.Empty:
                    if api_path:
                        release_api(api_path, session)
                    break
                
                if api_path is None:
                    api_pair = acquire_api(timeout=0.1)
                    if api_pair is None:
                        task_queue.put((i, command))
                        time.sleep(0.01)
                        continue
                    api_path, session = api_pair
                
                try:
                    result = self.sandbox_client.run_generator(
                        api_path=api_path,
                        generator_code=generator,
                        command=command,
                        files=self.testlib_files,
                        session=session,
                        sample_id=sample_id,
                        logger=log_sample
                    )
                    
                    with results_lock:
                        if result['success']:
                            temp_results[i] = ('success', command, result['output'])
                        else:
                            temp_results[i] = ('error', command, result['error'])
                
                except Exception as e:
                    with results_lock:
                        temp_results[i] = ('error', command, str(e))
                
                finally:
                    task_queue.task_done()
        
        # Start worker threads
        threads = []
        for i in range(min(max_workers, len(commands))):
            t = threading.Thread(target=worker, args=(f"input_gen_{i}",))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        # Organize results
        for i in sorted(temp_results.keys()):
            status, command, data = temp_results[i]
            if status == 'success':
                results['inputs'].append(data)
                results['command_to_input'][command] = data
            else:
                results['errors'].append(f"command: {command}; error: {data}")
        
        return results
    
    def _generate_test_outputs(
        self,
        case_inputs: List[str],
        sample: Sample,
        sample_id: str,
        max_workers: int
    ) -> Dict[str, Any]:
        """Generate test outputs in parallel"""
        results = {
            'corner_cases': [],
            'errors': []  # format: [(language, code, case_input, error_msg), ...]
        }

        # Get canonical solutions
        canonical_solutions = sample.get_canonical_solutions_by_language()
        code_lang_pairs = []
        for lang in ['python', 'cpp', 'java']:
            for code in canonical_solutions.get(lang, []):
                code_lang_pairs.append((lang, code))

        if not code_lang_pairs:
            results['errors'].append(('unknown', '', '', "No canonical solution available"))
            return results

        task_queue = queue.Queue()
        for case_input in case_inputs:
            for lang, code in code_lang_pairs:
                task_queue.put((case_input, lang, code))

        results_lock = threading.Lock()
        output_map = {inp: [] for inp in case_inputs}
        error_map = {inp: [] for inp in case_inputs}  # Store detailed error messages

        def worker(worker_id: str):
            api_path = None
            session = None

            while True:
                try:
                    case_input, lang, code = task_queue.get(timeout=1)
                except queue.Empty:
                    if api_path:
                        release_api(api_path, session)
                    break

                if api_path is None:
                    api_pair = acquire_api(timeout=0.1)
                    if api_pair is None:
                        task_queue.put((case_input, lang, code))
                        time.sleep(0.01)
                        continue
                    api_path, session = api_pair

                try:
                    result = self.sandbox_client.run_solution(
                        api_path=api_path,
                        code=code,
                        language=lang,
                        stdin=case_input,
                        session=session,
                        sample_id=sample_id,
                        logger=log_sample
                    )

                    with results_lock:
                        if result['success'] and result['output']:
                            output_map[case_input].append(result['output'])
                        else:
                            # Save detailed error info: (language, code, error_detail, response)
                            error_detail = {
                                'error_msg': result.get('error', 'Unknown error'),
                                'response': result.get('response')
                            }
                            error_map[case_input].append({
                                'type': 'error_code',
                                'data': (lang, code, error_detail)
                            })

                except Exception as e:
                    with results_lock:
                        error_map[case_input].append({
                            'type': 'error_specific',
                            'data': (lang, code, str(e))
                        })

                finally:
                    task_queue.task_done()

        # Start worker threads
        threads = []
        for i in range(min(max_workers, len(case_inputs) * len(code_lang_pairs))):
            t = threading.Thread(target=worker, args=(f"output_gen_{i}",))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Organize results (choose the most frequent output)
        for case_input in case_inputs:
            outputs = [o for o in output_map[case_input] if o]
            if outputs:
                output, _ = Counter(outputs).most_common(1)[0]
                results['corner_cases'].append(TestCase(input=case_input, output=output))
            else:
                # Construct detailed error information
                error_list = error_map.get(case_input, [])

                # Prioritize compilation/runtime errors
                error_code_found = False
                for error_item in error_list:
                    if error_item['type'] == 'error_code':
                        lang, code, error_detail = error_item['data']
                        response = error_detail.get('response')

                        # Construct detailed error message
                        error_msg = f"Stdin: {case_input}; "
                        if response:
                            compile_result = response.get('compile_result', {}) or {}
                            run_result = response.get('run_result', {}) or {}
                            error_msg += (
                                f"Compile error (code {compile_result.get('return_code', '')}; "
                                f"status: {compile_result.get('status', '')}): {compile_result.get('stderr', '')}; "
                                f"Runtime error (code {run_result.get('return_code', '')}; "
                                f"status: {run_result.get('status', '')}): {run_result.get('stderr', '')}"
                            )
                        else:
                            error_msg += error_detail.get('error_msg', 'Unknown error')

                        results['errors'].append((lang, code, case_input, error_msg))
                        error_code_found = True
                        break

                # If no compilation/runtime errors, handle API errors
                if not error_code_found:
                    for error_item in error_list:
                        if error_item['type'] == 'error_specific':
                            lang, code, error_msg = error_item['data']
                            full_error_msg = f"Stdin: {case_input}; API Error: {error_msg}"
                            results['errors'].append((lang, code, case_input, full_error_msg))
                            break

        return results
    
    def _prepare_feedback_messages(
        self,
        messages: List[Dict],
        command_response,
        commands: List[str],
        command_to_input: Dict[str, str],
        input_errors: List[str],
        corner_cases: List[TestCase],
        output_errors: List[Tuple[str, str, str, str]],  # (language, code, stdin, error_msg)
        validation_result: Dict,
        input_constraints_summary: str,
        improved_generator: str,
        all_search_replace_blocks: List[str],
        is_first: bool,
        is_generator_updated: bool,
        current_generator: str = ""
    ) -> List[Dict]:
        """Prepare feedback messages for the next iteration"""
        # Sample validation results
        solution_results = random.sample(
            validation_result.get('solution_result', []),
            min(self.max_sample_solutions, len(validation_result.get('solution_result', [])))
        )
        incorrect_results = random.sample(
            validation_result.get('incorrect_solution_result', []),
            min(self.max_sample_solutions, len(validation_result.get('incorrect_solution_result', [])))
        )

        # Format results
        formatted_solution_results = self._format_validation_results(
            solution_results, corner_cases, command_to_input
        )
        formatted_incorrect_results = self._format_validation_results(
            incorrect_results, corner_cases, command_to_input
        )

        # Format command input mapping (truncate long inputs)
        command_to_input_display = {}
        for cmd, inp in command_to_input.items():
            if len(inp) >= self.truncated_length:
                command_to_input_display[cmd] = "[input]"
            else:
                command_to_input_display[cmd] = inp

        # Build input-to-command reverse mapping (for long input replacement)
        input_to_command = {
            inp: cmd for cmd, inp in command_to_input.items()
            if len(inp) >= self.truncated_length
        }

        # Format output_errors using CANONICAL_SOLUTION_TEMPLATE
        if output_errors:
            canonical_solution_results = []
            for error in output_errors:
                language, code, stdin, error_msg = error
                # If input is too long, replace with command
                if stdin in input_to_command:
                    stdin_display = input_to_command[stdin] + " [command]"
                else:
                    stdin_display = stdin

                canonical_solution_result = CANONICAL_SOLUTION_TEMPLATE.format(
                    language=language,
                    solution=code,
                    stdin=stdin_display,
                    output=error_msg
                )
                canonical_solution_results.append(canonical_solution_result)
            formatted_output_errors = canonical_solution_results
        else:
            formatted_output_errors = "No errors in generating outputs for corner cases."

        # Add assistant response and new user message
        # Message compression logic
        if is_first:
            # First iteration: use initial response template
            if hasattr(command_response, 'generator') and command_response.generator:
                assistant_response = INIT_RESPONSE_TEMPLATE_WOGEN.format(
                    input_constraints_summary=json.dumps(command_response.input_constraints_summary),
                    generator=json.dumps(command_response.generator),
                    command_list=json.dumps(command_response.command_list)
                )
            else:
                assistant_response = INIT_RESPONSE_TEMPLATE_WGEN.format(
                    input_constraints_summary=json.dumps(command_response.input_constraints_summary),
                    search_replace_generator_blocks=json.dumps(command_response.search_replace_generator_blocks),
                    command_list=json.dumps(command_response.command_list)
                )
            # Build feedback prompt
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                improved_generator=improved_generator,
                current_command_list=commands,
                command_to_input_map=command_to_input_display,
                command_run_errors=input_errors,
                input_constraints_summary=input_constraints_summary,
                correct_results=formatted_solution_results,
                incorrect_results=formatted_incorrect_results,
                outputs=formatted_output_errors
            )
        elif self.compress_messages and len(messages) >= 4:
            # Message compression mode: remove the last two messages
            messages = messages[:-2]

            # In compression mode, use accumulated final generator
            # current_generator parameter is cumulatively updated, displayed as long as is_generator_updated is True
            final_improved_generator = current_generator if is_generator_updated else ""

            # Check if this is "without generator" mode by checking if the first user message contains distinctive text
            # INIT_PROMPT_TEMPLATE_WITHOUT_GENERATOR contains "Your task is to create a high-quality C++ test case generator"
            is_wogen_mode = "Your task is to create a high-quality C++ test case generator" in messages[1]['content']

            if is_wogen_mode:
                # Use WOGEN template with current generator
                assistant_response = INIT_RESPONSE_TEMPLATE_WOGEN.format(
                    input_constraints_summary=json.dumps(input_constraints_summary),
                    generator=json.dumps(current_generator),
                    command_list=json.dumps(commands)
                )
            else:
                # Use WGEN template with accumulated blocks
                assistant_response = INIT_RESPONSE_TEMPLATE_WGEN.format(
                    input_constraints_summary=json.dumps(input_constraints_summary),
                    search_replace_generator_blocks=json.dumps(all_search_replace_blocks),
                    command_list=json.dumps(commands)
                )

            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                improved_generator=final_improved_generator,
                current_command_list=commands,
                command_to_input_map=command_to_input_display,
                command_run_errors=input_errors,
                input_constraints_summary=input_constraints_summary,
                correct_results=formatted_solution_results,
                incorrect_results=formatted_incorrect_results,
                outputs=formatted_output_errors
            )
        else:
            # Normal mode: use refine response template
            assistant_response = REFINE_RESPONSE_TEMPLATE.format(
                search_replace_generator_blocks=json.dumps(command_response.search_replace_generator_blocks),
                replace_command_list=json.dumps(command_response.replace_command_list),
                add_command_list=json.dumps(command_response.add_command_list)
            )
            # Build feedback prompt
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                improved_generator=improved_generator,
                current_command_list=commands,
                command_to_input_map=command_to_input_display,
                command_run_errors=input_errors,
                input_constraints_summary=input_constraints_summary,
                correct_results=formatted_solution_results,
                incorrect_results=formatted_incorrect_results,
                outputs=formatted_output_errors
            )
        
        messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": refine_prompt})
        
        return messages
    
    def _format_validation_results(
        self,
        results: List[Dict],
        corner_cases: List[TestCase],
        command_to_input: Dict[str, str]
    ) -> List[str]:
        """Format validation results"""
        formatted = []
        
        # Reverse mapping (long input -> command)
        input_to_command = {
            inp: cmd for cmd, inp in command_to_input.items()
            if len(inp) >= self.truncated_length
        }
        
        for res in results:
            if not isinstance(res.get('result'), dict):
                continue
            
            test_results = []
            for test in res['result'].get('tests', []):
                test_idx = test.get('test_index', 0)
                if test_idx >= len(corner_cases):
                    continue
                
                test_case = corner_cases[test_idx]
                stdin = test_case.input
                
                # Replace long input with command
                if stdin in input_to_command:
                    stdin = input_to_command[stdin] + " [command]"
                
                stdout = test_case.output
                if len(stdout) >= self.truncated_length:
                    stdout = "[output]"
                
                exec_info = test.get('exec_info', {}) or {}
                run_result = exec_info.get('run_result', {}) or {}
                expected_output = run_result.get('stdout', '')
                if len(expected_output) >= self.truncated_length:
                    expected_output = "[expected output]"
                
                error_info = ""
                if exec_info.get('status') != "Success":
                    compile_result = exec_info.get('compile_result', {}) or {}
                    error_info = (
                        f"Compile error (code {compile_result.get('return_code', '')}; "
                        f"status: {compile_result.get('status', '')}): {compile_result.get('stderr', '')}; "
                        f"Runtime error (code {run_result.get('return_code', '')}; "
                        f"status: {run_result.get('status', '')}): {run_result.get('stderr', '')}"
                    )
                
                test_results.append(TEST_CASE_RESULT_TEMPLATE.format(
                    passed=test.get('passed', False),
                    stdin=stdin,
                    stdout=stdout,
                    expected_output=expected_output,
                    error_info=error_info
                ))
            
            formatted.append(SOLUTION_RESULT_TEMPLATE.format(
                language=res.get('language', 'unknown'),
                solution=res.get('solution', '')[:500] + "..." if len(res.get('solution', '')) > 500 else res.get('solution', ''),
                output=test_results
            ))
        
        return formatted
