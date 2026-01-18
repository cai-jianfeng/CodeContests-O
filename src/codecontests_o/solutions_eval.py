#!/usr/bin/env python3
"""
CodeContests-O Solutions Evaluator

Evaluates solutions in the dataset against existing test cases to find false negatives/positives.
Analogous to solutions_eval_plus_test_cases.py but integrated into the package structure.
"""

import os
import sys
import argparse
import time
import threading
import json
import base64
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .config import Config, get_preset_config
from .data import CodeContestsReader
from .data.models import Sample
from .clients.sandbox_client import SandboxClient
from .core.validator import SolutionValidator
from .parallel.api_pool import initialize_api_pool
from .utils.logger import log_global, log_sample, initialize_logger_manager


def load_testlib_files(testlib_path: str) -> dict:
    """Load testlib.h file"""
    if not os.path.exists(testlib_path):
        # Allow missing testlib if checks don't strictly require it (though validator might)
        print(f"Warning: testlib.h not found at: {testlib_path}")
        return {}
    
    with open(testlib_path, 'rb') as f:
        testlib_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return {"testlib.h": testlib_b64}


class Evaluator:
    def __init__(self, config: Config, testlib_files: Dict[str, str]):
        self.config = config
        self.testlib_files = testlib_files
        
        # Initialize API pool
        api_paths = config.sandbox.get_api_paths()
        initialize_api_pool(api_paths)
        log_global(f"Initialized API pool with {len(api_paths)} endpoints")
        
        # Initialize client
        self.sandbox_client = SandboxClient(
            compile_timeout=config.sandbox.compile_timeout,
            run_timeout=config.sandbox.run_timeout
        )
        self.sandbox_client.setup_session_pool(len(api_paths))
        
        self.validator = SolutionValidator(
            sandbox_client=self.sandbox_client,
            testlib_files=testlib_files
        )

    def evaluate_sample(self, sample: Sample, results_dir: str) -> Dict[str, Any]:
        """Evaluate a single sample"""
        sample_id = sample.id
        sample_dir = os.path.join(results_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        stats = {'completed': 0, 'errors': 0}
        
        try:
            log_sample(sample_id, f"Validating {len(sample.correct_solutions)} correct and {len(sample.incorrect_solutions)} incorrect solutions")
            
            # Check if we have test cases
            if not sample.test_cases:
                log_sample(sample_id, "No test cases found, skipping validation")
                return stats
            
            # Validator handles parallelism for solutions internally if we passed max_workers
            # But the 'validate_solutions' method runs synchronously waiting for all threads
            self.validator.validate_solutions(
                sample=sample,
                corner_cases=sample.test_cases,
                max_workers=self.config.processing.solution_validation_workers,
                save_dir=sample_dir
            )
            
            # Log summary
            log_sample(sample_id, "Validation completed")
            stats['completed'] = 1
            
        except Exception as e:
            log_sample(sample_id, f"Error evaluating sample: {e}")
            stats['errors'] = 1
            
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="CodeContests-O Solution Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--results_dir", type=str, default="./results_eval",
                       help="Directory to save evaluation results")
    parser.add_argument("--preset", type=str, default="production",
                       help="Configuration preset")
    
    # Dataset filtering
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index")
    parser.add_argument("--subset", type=str, help="Dataset subset (e.g. 1x for CodeContestsPlus)")
    
    # Worker configuration
    parser.add_argument("--sample_workers", type=int, default=4,
                       help="Number of parallel sample workers")
    parser.add_argument("--validation_workers", type=int, default=16,
                       help="Number of parallel validation workers per sample")
    
    # Sandbox configuration
    parser.add_argument("--sandbox_hosts", type=str, nargs="+", default=["localhost"],
                       help="Sandbox hosts")
    parser.add_argument("--sandbox_port", type=int, default=8080,
                       help="Sandbox base port")
    parser.add_argument("--port_range", type=int, default=1,
                       help="Number of sandbox ports per host")
    
    parser.add_argument("--testlib_path", type=str, default="testlib.h",
                       help="Path to testlib.h")

    args = parser.parse_args()
    
    # Setup config
    config_dict = get_preset_config(args.preset)
    config = Config.from_dict(config_dict)
    
    # Overrides
    config.dataset.data_path = args.data_path
    config.dataset.results_dir = args.results_dir
    config.processing.start = args.start
    config.processing.end = args.end
    config.processing.sample_level_workers = args.sample_workers
    config.processing.solution_validation_workers = args.validation_workers
    config.sandbox.hosts = args.sandbox_hosts
    config.sandbox.base_port = args.sandbox_port
    config.sandbox.port_range = args.port_range
    config.resource.testlib_path = args.testlib_path
    
    # Load resources
    testlib_files = load_testlib_files(config.resource.testlib_path)
    
    # Initialize logger
    os.makedirs(config.dataset.results_dir, exist_ok=True)
    initialize_logger_manager(
        config.dataset.results_dir,
        config.processing.start,
        config.processing.end
    )
    
    # Load dataset
    print(f"Loading dataset from {config.dataset.data_path}...")
    dataset = CodeContestsReader(
        data_path=config.dataset.data_path,
        start=config.processing.start,
        end=config.processing.end,
        subset=args.subset
    )
    
    print(f"Found {len(dataset)} samples to process")
    
    # Initialize evaluator
    evaluator = Evaluator(config, testlib_files)
    
    # Run evaluation
    log_global("Starting evaluation...")
    start_time = time.time()
    
    total_stats = {'completed': 0, 'errors': 0}
    
    # Since validator uses threads, we should be careful with max_workers for samples
    # If sample_workers is high and validation_workers is high, we might exhaust threads/connections
    # The api_pool limits concurrency to available sandbox endpoints, so it should be safe.
    
    with ThreadPoolExecutor(max_workers=config.processing.sample_level_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(evaluator.evaluate_sample, sample, config.dataset.results_dir): sample 
            for sample in dataset
        }
        
        for future in tqdm(as_completed(future_to_sample), total=len(dataset), desc="Evaluating"):
            try:
                stats = future.result()
                total_stats['completed'] += stats['completed']
                total_stats['errors'] += stats['errors']
            except Exception as e:
                log_global(f"Sample processing failed: {e}")
                total_stats['errors'] += 1

    duration = time.time() - start_time
    print("\n" + "="*50)
    print(f"Evaluation Complete in {duration:.2f}s")
    print(f"Completed: {total_stats['completed']}")
    print(f"Errors: {total_stats['errors']}")
    print("="*50)
    log_global(f"Evaluation Complete. Duration: {duration:.2f}s. Completed: {total_stats['completed']}, Errors: {total_stats['errors']}")


if __name__ == "__main__":
    main()
