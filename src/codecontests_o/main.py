#!/usr/bin/env python3
"""
CodeContests-O Main Entry Point

Usage:
    # Use local JSON files
    python -m codecontests_o.main --data_path /path/to/json/files --results_dir ./results

    # Use HuggingFace dataset
    python -m codecontests_o.main --data_path ByteDance-Seed/Code-Contests-Plus --results_dir ./results

    # Use preset configuration
    python -m codecontests_o.main --preset production --data_path /path/to/data

    # Custom dataset
    python -m codecontests_o.main --custom_reader my_reader.py --data_path /path/to/data
"""

import os
import sys
import argparse
import base64
import importlib.util

from .config import Config, get_preset_config
from .data import CodeContestsReader, DatasetReader
from .parallel import ParallelProcessor


def load_testlib_files(testlib_path: str) -> dict:
    """Load testlib.h file"""
    if not os.path.exists(testlib_path):
        raise FileNotFoundError(f"testlib.h not found at: {testlib_path}")
    
    with open(testlib_path, 'rb') as f:
        testlib_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return {"testlib.h": testlib_b64}


def load_custom_reader(reader_path: str) -> type:
    """
    Dynamically load custom dataset reader
    
    Args:
        reader_path: Path to Python file containing a class inheriting from DatasetReader
        
    Returns:
        type: Dataset reader class
    """
    spec = importlib.util.spec_from_file_location("custom_reader", reader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find DatasetReader subclass
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, DatasetReader) and 
            obj is not DatasetReader):
            return obj
    
    raise ValueError(f"No DatasetReader subclass found in {reader_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CodeContests-O: Feedback-Driven Test Case Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with local JSON files
    python -m codecontests_o.main --data_path ./data/json --results_dir ./results

    # Use HuggingFace dataset
    python -m codecontests_o.main --data_path ByteDance-Seed/Code-Contests-Plus --results_dir ./results

    # Use production preset
    python -m codecontests_o.main --preset production --data_path ./data

    # Custom dataset reader
    python -m codecontests_o.main --custom_reader my_reader.py --data_path ./my_data

    # Only generate test cases (no validation)
    python -m codecontests_o.main --data_path ./data --only_generate

    # Process specific range
    python -m codecontests_o.main --data_path ./data --start 0 --end 100
        """
    )
    
    # Data configuration
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset (local directory with JSON files or HuggingFace dataset name like 'ByteDance-Seed/Code-Contests-Plus')")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Directory to save results")
    
    # Preset configuration
    parser.add_argument("--preset", type=str, 
                       choices=["development", "production", "test", "quick"], default="production",
                       help="Use preset configuration")
    
    # Custom reader
    parser.add_argument("--custom_reader", type=str,
                       help="Path to custom DatasetReader implementation")
    
    # Processing configuration
    parser.add_argument("--start", type=int, default=0,
                       help="Start index for processing")
    parser.add_argument("--end", type=int, default=-1,
                       help="End index for processing (-1 for all)")
    parser.add_argument("--max_iterations", type=int, default=2,
                       help="Maximum iterations per sample")
    parser.add_argument("--only_generate", action="store_true",
                       help="Only generate test cases, skip validation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # API configuration
    parser.add_argument("--api_base", type=str,
                       help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str,
                       help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="Model name")
    
    # Sandbox configuration
    parser.add_argument("--sandbox_hosts", type=str, nargs="+",
                       default=["localhost"],
                       help="Sandbox API hosts")
    parser.add_argument("--sandbox_port", type=int, default=8080,
                       help="Sandbox base port")
    parser.add_argument("--port_range", type=int, default=1,
                       help="Number of ports per host")
    
    # Parallel configuration
    parser.add_argument("--sample_workers", type=int, default=1,
                       help="Number of parallel sample workers")
    parser.add_argument("--output_workers", type=int, default=32,
                       help="Number of parallel output generation workers")
    parser.add_argument("--validation_workers", type=int, default=32,
                       help="Number of parallel validation workers")
    
    # Resource configuration
    parser.add_argument("--testlib_path", type=str,  default="testlib.h",
                       help="Path to testlib.h")
    
    args = parser.parse_args()
    
    # Build configuration
    if args.preset:
        config_dict = get_preset_config(args.preset)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Override command line arguments
    config.dataset.data_path = args.data_path
    config.dataset.results_dir = args.results_dir
    
    config.processing.start = args.start
    config.processing.end = args.end
    config.processing.max_iterations = args.max_iterations
    config.processing.only_generate = args.only_generate
    config.processing.debug = args.debug
    config.processing.sample_level_workers = args.sample_workers
    config.processing.output_generation_workers = args.output_workers
    config.processing.solution_validation_workers = args.validation_workers
    
    if args.api_base:
        config.openai.api_base = args.api_base
    else:
        # Read from environment variables
        env_api_base = os.getenv("OPENAI_BASE_URL")
        if env_api_base:
            config.openai.api_base = env_api_base
    if args.api_key:
        config.openai.api_key = args.api_key
    else:
        # Read from environment variables
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_api_key:
            config.openai.api_key = env_api_key
        else:
            raise ValueError("OpenAI API key must be specified via --api_key or OPENAI_API_KEY environment variable")
        
    config.openai.model = args.model
    
    config.sandbox.hosts = args.sandbox_hosts
    config.sandbox.base_port = args.sandbox_port
    config.sandbox.port_range = args.port_range
    
    config.resource.testlib_path = args.testlib_path
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Load testlib files
    print("Loading testlib files...")
    testlib_files = load_testlib_files(config.resource.testlib_path)
    
    # Create dataset reader
    if args.custom_reader:
        ReaderClass = load_custom_reader(args.custom_reader)
        dataset = ReaderClass(
            data_path=config.dataset.data_path,
            start=config.processing.start,
            end=config.processing.end
        )
    else:
        print("Using default CodeContestsReader...")
        dataset = CodeContestsReader(
            data_path=config.dataset.data_path,
            start=config.processing.start,
            end=config.processing.end
        )
    
    # Print configuration info
    print("=" * 60)
    print("CodeContests-O Configuration")
    print("=" * 60)
    runtime_info = config.get_runtime_info()
    for key, value in runtime_info.items():
        print(f"  {key}: {value}")
    print(f"  dataset: {dataset.name}")
    print(f"  samples: {len(dataset)}")
    print("=" * 60)
    
    # Create processor and run
    processor = ParallelProcessor(config=config, testlib_files=testlib_files)
    
    stats = processor.process_dataset(
        dataset=dataset,
        results_dir=config.dataset.results_dir,
        only_generate=config.processing.only_generate
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"  Total samples: {stats['total']}")
    print(f"  Processed: {stats['to_process']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Duration: {stats['duration']:.2f}s")
    print(f"  Avg time/sample: {stats['avg_time_per_sample']:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
