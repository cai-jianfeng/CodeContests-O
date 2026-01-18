# CodeContests-O

Feedback-Driven Iterative Test Case Generation Framework

## Overview

CodeContests-O is a framework for generating high-quality competitive programming test cases. It uses a feedback-driven iterative optimization method, working synergistically with Large Language Models and a code execution sandbox to generate test cases that effectively distinguish between correct and incorrect solutions.

## Directory Structure

```
codecontests_o/
├── pyproject.toml           # Build configuration
├── setup.py                 # Installation script
├── src/
│   └── codecontests_o/      # Source code package
│       ├── __init__.py      # Package entry point
│       ├── main.py          # Main generation entry point
│       ├── solutions_eval.py # Solution evaluation entry point
│       ├── analyze_results.py # Result analysis entry point
│       ├── config/          # Configuration management
│       ├── data/            # Data processing
│       ├── clients/         # API clients
│       ├── core/            # Core logic
│       ├── parallel/        # Parallel processing
│       ├── prompts/         # Prompt templates
│       ├── utils/           # Utility functions
│       └── examples/        # Examples
└── README.md                # Documentation
```

## Installation

Ensure your environment is using Python 3.8 or higher.

```bash
# Clone the repository
git clone <repository_url>
cd codecontests_o

# Install (development mode)
pip install -e .

# Or standard install
pip install .
```

## Quick Start

### Using Local JSON Files

```bash
python -m codecontests_o.main \
    --data_path /path/to/codecontests/json \
    --results_dir ./results \
    --api_key YOUR_OPENAI_API_KEY \
    --sandbox_hosts localhost \
    --testlib_path ./testlib.h
```

### Using HuggingFace Datasets

CodeContests-O supports loading datasets directly from HuggingFace, such as `ByteDance-Seed/Code-Contests-Plus`:

```bash
# Load from HuggingFace Hub
python -m codecontests_o.main \
    --data_path ByteDance-Seed/Code-Contests-Plus \
    --results_dir ./results \
    --api_key YOUR_OPENAI_API_KEY \
    --sandbox_hosts localhost \
    --testlib_path ./testlib.h
```

Python API Usage:

```python
from codecontests_o import CodeContestsReader, ParallelProcessor

# Automatically detect data source type
dataset = CodeContestsReader(
    data_path="ByteDance-Seed/Code-Contests-Plus",
    split="test",  # train, test, or validation
    start=0,
    end=10
)

# Or explicitly specify to use HuggingFace
dataset = CodeContestsReader(
    data_path="ByteDance-Seed/Code-Contests-Plus",
    use_hf=True,
    split="test"
)
```

### Using Preset Configurations

```bash
# Development environment (low parallelism, debug mode)
python -m codecontests_o.main --preset development --data_path ./data

# Production environment (high parallelism)
python -m codecontests_o.main --preset production --data_path ./data

# Quick test (generate only, no validation)
python -m codecontests_o.main --preset quick --data_path ./data
```

### Python API Usage

```python
from codecontests_o import (
    Config,
    CodeContestsReader,
    ParallelProcessor,
    get_preset_config,
)
import base64

# 1. Create configuration
config = Config.from_dict(get_preset_config("development"))
config.openai.api_key = "your-api-key"
config.dataset.data_path = "/path/to/data"
config.dataset.results_dir = "./results"

# 2. Load testlib.h
with open("testlib.h", "rb") as f:
    testlib_files = {"testlib.h": base64.b64encode(f.read()).decode()}

# 3. Create dataset reader
# Method 1: From local JSON files
dataset = CodeContestsReader(
    data_path=config.dataset.data_path,
    start=0,
    end=10  # Process only the first 10 samples
)

# Method 2: From HuggingFace dataset
dataset = CodeContestsReader(
    data_path="ByteDance-Seed/Code-Contests-Plus",
    split="test",
    start=0,
    end=10
)

# 4. Create processor and run
processor = ParallelProcessor(config=config, testlib_files=testlib_files)
stats = processor.process_dataset(dataset, config.dataset.results_dir)

print(f"Completed: {stats['completed']}/{stats['total']}")
```

## Solution Evaluation

If you only want to evaluate the performance of solutions in the dataset on existing test cases (detecting false negatives and false positives), you can use the `solutions_eval` module. This is useful for analyzing dataset quality.

### Basic Usage

```bash
python -m codecontests_o.solutions_eval \
    --data_path ByteDance-Seed/Code-Contests-Plus \
    --subset 1x \
    --results_dir ./results_eval \
    --start 0 --end 10 \
    --sandbox_hosts localhost
```

### Parameters

*   `--data_path`: Dataset path (local directory or HuggingFace dataset name)
*   `--subset`: Dataset subset (e.g., `1x`, `2x`, only for Code-Contests-Plus)
*   `--results_dir`: Directory to save results
*   `--start`/`--end`: Range of samples to process
*   `--sample_workers`: Sample-level parallelism
*   `--validation_workers`: Validation parallelism within a single sample

### Result Analysis

After running the evaluation, use `analyze_results` to calculate overall metrics (TPR/TNR):

```bash
python -m codecontests_o.analyze_results --results_dir ./results_eval
```

This will output detailed statistics, including:
*   **TPR (True Positive Rate)**: The proportion of correct solutions identified as correct (higher is better).
*   **TNR (True Negative Rate)**: The proportion of incorrect solutions identified as incorrect (higher is better).
*   **Intersection**: Average value for samples that have both valid TPR and TNR.

## Custom Dataset Integration

The framework is designed to be extensible, allowing you to integrate your own datasets by implementing the `DatasetReader` interface.

### Step 1: Create Custom Reader

```python
# my_reader.py
from codecontests_o.data import DatasetReader, Sample, Solution, TestCase, Language

class MyDatasetReader(DatasetReader):
    def __init__(self, data_path: str, start: int = 0, end: int = -1):
        self.data_path = data_path
        self._samples = self._load_data()
    
    def _load_data(self):
        samples = []
        # Load your data...
        for item in your_data:
            sample = Sample(
                id=item['id'],
                name=item['name'],
                description=item['description'],
                generator=item.get('generator_code'),  # Optional. If None, will specify generated.
                canonical_solutions=[
                    Solution(code=item['solution'], language=Language.PYTHON)
                ],
                correct_solutions=[
                    Solution(code=item['solution'], language=Language.PYTHON)
                ],
                incorrect_solutions=[
                    Solution(code=item['wrong_solution'], language=Language.PYTHON)
                ],
                test_cases=[
                    # Required if using solutions_eval to evaluate existing test cases
                    TestCase(input="1\n", output="2\n")
                ]
            )
            samples.append(sample)
        return samples
    
    def __iter__(self):
        for sample in self._samples:
            yield sample
    
    def __len__(self):
        return len(self._samples)
    
    @property
    def name(self):
        return "MyDataset"
```

### Step 2: Use Custom Reader

```bash
python -m codecontests_o.main \
    --custom_reader my_reader.py \
    --data_path /path/to/your/data \
    --results_dir ./results
```

Or via Python API:

```python
from my_reader import MyDatasetReader

dataset = MyDatasetReader(data_path="/path/to/data")
processor = ParallelProcessor(config=config, testlib_files=testlib_files)
processor.process_dataset(dataset, results_dir)
```

## Sample Data Model

`Sample` is the core data structure of the framework, containing the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | ✓ | Unique identifier |
| `name` | str | ✓ | Problem name |
| `description` | str | ✓ | Full problem description |
| `generator` | str | | C++ tests generator code (optional, generated from scratch if absent) |
| `checker` | str | | C++ checker code (optional) |
| `canonical_solutions` | List[Solution] | ✓ | Canonical solutions (used for generating test outputs) |
| `correct_solutions` | List[Solution] | ✓ | List of correct solutions |
| `incorrect_solutions` | List[Solution] | ✓ | List of incorrect solutions |
| `test_cases` | List[TestCase] | | Test cases (required for solutions_eval) |
| `metadata` | Dict | | Other metadata |

## Configuration Options

### OpenAI Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `api_base` | `https://api.openai.com/v1` | API base URL |
| `api_key` | - | API key |
| `model` | `gpt-4o` | Model name |
| `max_tokens` | `8000` | Maximum tokens |
| `no_reasoning` | `True` | Whether to disable reasoning mode |

### Sandbox Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `hosts` | `["localhost"]` | Sandbox hosts list |
| `base_port` | `8080` | Base port |
| `port_range` | `4` | Number of ports per host |
| `compile_timeout` | `20` | Compilation timeout (seconds) |
| `run_timeout` | `20` | Execution timeout (seconds) |

### Processing Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `max_iterations` | `3` | Maximum iterations per sample |
| `sample_level_workers` | `4` | Sample-level parallelism |
| `output_generation_workers` | `4` | Output generation parallelism |
| `solution_validation_workers` | `4` | Validation parallelism |
| `only_generate` | `False` | Generate only, skip validation |

## Output Format

Results for each sample are saved as JSON files:

```json
{
    "id": "Codeforces_1234A",
    "status": "completed",
    "corner_cases": [
        {"input": {"stdin": "..."}, "output": {"stdout": "..."}}
    ],
    "commands": ["./gen --n 1000", "./gen --n 1 --edge"],
    "result": [
        {
            "iteration": 0,
            "corner_cases": [...],
            "generate_commands": [...],
            "improved_generator": "...",
            ...
        }
    ]
}
```

## Workflow

1.  **Initial Generation**: LLM analyzes the problem and generator to generate an initial list of commands.
2.  **Command Execution**: Execute commands in the sandbox to generate test inputs.
3.  **Output Generation**: Run test inputs using canonical solutions to generate expected outputs.
4.  **Validation**: Run test cases on correct and incorrect solutions.
5.  **Feedback Optimization**: Based on validation results, LLM optimizes commands and the generator.
6.  **Iteration**: Repeat steps 2-5 until the maximum number of iterations is reached or all solutions are correctly classified.

## License

MIT License
