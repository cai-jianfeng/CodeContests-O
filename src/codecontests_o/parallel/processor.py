"""
Parallel Processor

Handles dataset-level parallel task scheduling
"""

import os
import json
import time
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from ..data.base import DatasetReader
from ..data.models import Sample, TestCase, GenerationResult
from ..clients.openai_client import OpenAIClient
from ..clients.sandbox_client import SandboxClient
from ..config.base import Config
from ..core.validator import SolutionValidator
from ..utils.logger import log_global, log_sample, initialize_logger_manager, cleanup_logger
from .api_pool import initialize_api_pool, reset_api_pool


class ParallelProcessor:
    """
    Parallel Processor
    
    Manages dataset-level parallel task scheduling and result aggregation
    """
    
    def __init__(
        self,
        config: Config,
        testlib_files: Dict[str, str]
    ):
        """
        Initialize parallel processor
        
        Args:
            config: Configuration
            testlib_files: testlib files (base64 encoded)
        """
        self.config = config
        self.testlib_files = testlib_files
        
        # Initialize API pool
        api_paths = config.sandbox.get_api_paths()
        initialize_api_pool(api_paths)
        log_global(f"Initialized API pool with {len(api_paths)} endpoints")
        
        # Initialize clients
        self.openai_client = OpenAIClient(
            api_base=config.openai.api_base,
            api_key=config.openai.api_key,
            model=config.openai.model,
            max_tokens=config.openai.max_tokens,
            no_reasoning=config.openai.no_reasoning,
            max_attempts=config.openai.max_attempts,
            timeout=config.openai.timeout
        )
        
        self.sandbox_client = SandboxClient(
            compile_timeout=config.sandbox.compile_timeout,
            run_timeout=config.sandbox.run_timeout
        )
        self.sandbox_client.setup_session_pool(len(api_paths))
        
        # Initialize generator and validator
        from ..core.generator import CornerCaseGenerator
        self.generator = CornerCaseGenerator(
            openai_client=self.openai_client,
            sandbox_client=self.sandbox_client,
            config=config,
            testlib_files=testlib_files
        )
        
        self.validator = SolutionValidator(
            sandbox_client=self.sandbox_client,
            testlib_files=testlib_files
        )
    
    def process_dataset(
        self,
        dataset: DatasetReader,
        results_dir: str,
        only_generate: bool = False
    ) -> Dict[str, Any]:
        """
        Process entire dataset
        
        Args:
            dataset: Dataset reader
            results_dir: Results saving directory
            only_generate: Generate only, no validation
            
        Returns:
            Dict: Processing statistics
        """
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logger
        initialize_logger_manager(
            results_dir,
            self.config.processing.start,
            self.config.processing.end
        )
        
        # Get samples to process
        print("Getting samples to process...")
        samples_to_process = self._get_samples_to_process(dataset, results_dir, only_generate)
        
        if not samples_to_process:
            log_global("No samples to process")
            return {"processed": 0, "total": len(dataset)}
        
        log_global(f"Processing {len(samples_to_process)} samples out of {len(dataset)} total")
        
        # Statistics
        stats = {
            "total": len(dataset),
            "to_process": len(samples_to_process),
            "completed": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        # Parallel processing
        sample_workers = min(
            self.config.processing.sample_level_workers,
            len(samples_to_process)
        )
        
        completed_lock = threading.Lock()
        
        def process_single(sample: Sample) -> Dict:
            """Process single sample"""
            sample_id = sample.id
            result_data = {
                'id': sample_id,
                'status': 'processing',
                'error': None,
                'corner_cases': [],
                'commands': [],
                'result': []
            }
            
            # Backup existing results
            result_path = os.path.join(results_dir, f"{sample_id}.json")
            backup_path = os.path.join(results_dir, f"{sample_id}.json.bak")
            
            if os.path.exists(result_path) and not only_generate:
                shutil.move(result_path, backup_path)
            
            try:
                log_sample(sample_id, "Starting generation...")
                
                corner_cases, commands, iterations = self.generator.generate_for_sample(
                    sample=sample,
                    output_workers=self.config.processing.output_generation_workers,
                    validation_workers=self.config.processing.solution_validation_workers,
                    only_generate=only_generate,
                    validator=self.validator,
                    results_dir=results_dir
                )
                
                result_data['corner_cases'] = [tc.to_dict() for tc in corner_cases]
                result_data['commands'] = commands
                result_data['result'] = [it.to_dict() for it in iterations]
                result_data['status'] = 'completed'
                
                # Add original sample data
                from ..data.codecontests import CodeContestsReader
                if hasattr(dataset, 'to_dict_format'):
                    result_data.update(dataset.to_dict_format(sample))
                else:
                    result_data.update(sample.to_dict())
                
                log_sample(sample_id, f"Completed with {len(corner_cases)} test cases")
                
                with completed_lock:
                    stats['completed'] += 1
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                log_sample(sample_id, error_msg)
                result_data['status'] = 'error'
                result_data['error'] = error_msg
                
                # Restore backup
                if os.path.exists(backup_path):
                    shutil.move(backup_path, result_path)
                    log_sample(sample_id, "Restored backup")
                
                with completed_lock:
                    stats['errors'] += 1
            
            finally:
                # Save results
                try:
                    save_path = result_path
                    if result_data['status'] == 'error':
                        error_dir = os.path.join(results_dir, "error")
                        os.makedirs(error_dir, exist_ok=True)
                        save_path = os.path.join(error_dir, f"{sample_id}.json")
                    elif not result_data['corner_cases']:
                        empty_dir = os.path.join(results_dir, "empty")
                        os.makedirs(empty_dir, exist_ok=True)
                        save_path = os.path.join(empty_dir, f"{sample_id}.json")
                    
                    with open(save_path, 'w') as f:
                        json.dump(result_data, f, indent=2)
                    
                    log_sample(sample_id, f"Saved to {save_path}")
                    
                except Exception as save_e:
                    log_sample(sample_id, f"Failed to save result: {save_e}")
            
            return result_data
        
        # Process using thread pool
        with ThreadPoolExecutor(max_workers=sample_workers) as executor:
            futures = {
                executor.submit(process_single, sample): sample
                for sample in samples_to_process
            }
            
            with tqdm(total=len(samples_to_process), desc="Processing samples") as pbar:
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        result = future.result()
                        if self.config.processing.debug:
                            log_global(f"Sample {sample.id}: {result.get('status')}")
                    except Exception as e:
                        log_global(f"Future error for {sample.id}: {e}")
                    finally:
                        pbar.update(1)
        
        # Calculate statistics
        stats['end_time'] = time.time()
        stats['duration'] = stats['end_time'] - stats['start_time']
        stats['avg_time_per_sample'] = stats['duration'] / len(samples_to_process) if samples_to_process else 0
        
        # Save statistics
        self._save_stats(stats, results_dir)
        
        # Cleanup
        cleanup_logger()
        reset_api_pool()
        
        return stats
    
    def _get_samples_to_process(
        self,
        dataset: DatasetReader,
        results_dir: str,
        only_generate: bool
    ) -> List[Sample]:
        """Get list of samples to process"""
        existing_ids = set()
        
        # Check completed results
        for fname in os.listdir(results_dir):
            if not fname.endswith('.json'):
                continue
            
            sample_id = os.path.splitext(fname)[0]
            
            if only_generate:
                existing_ids.add(sample_id)
                continue
            
            try:
                with open(os.path.join(results_dir, fname), 'r') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                if not results:
                    continue
                
                last_result = results[-1]
                
                # Skip conditions
                # 1. Max iterations reached
                if (not last_result.get('only_generate', False) and
                    len(results) >= self.config.processing.max_iterations):
                    existing_ids.add(sample_id)
                    log_global(f"Skip {sample_id}: max iterations reached")
                # 2. All solutions covered
                elif (not last_result.get('only_generate', False) and
                      not last_result.get('result', {}).get('solution_result', []) and
                      not last_result.get('result', {}).get('incorrect_solution_result', [])):
                    existing_ids.add(sample_id)
                    log_global(f"Skip {sample_id}: all solutions covered")
                # 3. Context limit exceeded
                elif last_result.get('exceed', False):
                    existing_ids.add(sample_id)
                    log_global(f"Skip {sample_id}: context exceeded")
                    
            except Exception as e:
                log_global(f"Error reading {fname}: {e}")
        
        # Check error directory, skip error samples
        error_dir = os.path.join(results_dir, "error")
        if os.path.exists(error_dir):
            for fname in os.listdir(error_dir):
                if fname.endswith('.json'):
                    existing_ids.add(os.path.splitext(fname)[0])
        
        log_global(f"Found {len(existing_ids)} already processed samples")
        
        # Filter samples to process
        samples = []
        for sample in dataset:
            if sample.id not in existing_ids:
                samples.append(sample)
        
        return samples
    
    def _save_stats(self, stats: Dict, results_dir: str):
        """Save processing statistics"""
        stats_file = os.path.join(results_dir, "processing_stats.json")
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Also save as readable text
        stats_txt = os.path.join(results_dir, "processing_stats.txt")
        with open(stats_txt, 'w') as f:
            f.write("=== Processing Statistics ===\n")
            f.write(f"Total samples: {stats['total']}\n")
            f.write(f"Samples to process: {stats['to_process']}\n")
            f.write(f"Completed: {stats['completed']}\n")
            f.write(f"Errors: {stats['errors']}\n")
            f.write(f"Duration: {stats['duration']:.2f} seconds\n")
            f.write(f"Avg time per sample: {stats['avg_time_per_sample']:.2f} seconds\n")
        
        log_global(f"Stats saved to {stats_file}")
