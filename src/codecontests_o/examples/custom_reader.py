"""
Custom Dataset Reader Example

This file demonstrates how to implement a custom dataset reader
that integrates seamlessly with the CodeContests-O framework.

Usage:
    python -m codecontests_o.main --custom_reader examples/custom_reader.py --data_path /your/data
"""

import os
import json
from typing import Iterator, Optional, List

# Import base class and data models
from codecontests_o.data import DatasetReader, Sample, Solution, TestCase, Language


class MyCustomReader(DatasetReader):
    """
    Custom Dataset Reader Example
    
    Your dataset can be in any format, just implement the following methods:
    - __iter__: Iterate and yield Sample objects
    - __len__: Return dataset size
    - name: Return dataset name
    
    Sample object needs to include:
    - id: Unique identifier
    - name: Problem name
    - description: Problem description
    - generator: C++ test case generator code
    - checker: (Optional) C++ checker code
    - canonical_solutions: List of canonical solutions (for generating test outputs)
    - correct_solutions: List of correct solutions
    - incorrect_solutions: List of incorrect solutions
    """
    
    def __init__(
        self,
        data_path: str,
        start: int = 0,
        end: int = -1
    ):
        """
        Initialize reader
        
        Args:
            data_path: Data directory path
            start: Start index
            end: End index (-1 means to end)
        """
        self.data_path = data_path
        self.start = start
        self.end = end
        
        # Load data
        self._samples = self._load_data()
    
    def _load_data(self) -> List[Sample]:
        """
        Load data and convert to Sample format
        
        This demonstrates a hypothetical JSON data format:
        {
            "problem_id": "problem_001",
            "title": "Two Sum",
            "description": "Given an array...",
            "generator_code": "#include <testlib.h>\\n...",
            "checker_code": "#include <testlib.h>\\n...",  // optional
            "solutions": [
                {"code": "...", "language": "python", "correct": true},
                {"code": "...", "language": "cpp", "correct": false},
            ],
            "test_cases": [  // optional, public test cases
                {"input": "...", "output": "..."}
            ]
        }
        """
        samples = []
        
        # Get all JSON files
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Data path is not a directory: {self.data_path}")
        
        files = sorted([f for f in os.listdir(self.data_path) if f.endswith('.json')])
        
        # Apply slice
        end = self.end if self.end != -1 else len(files)
        files = files[self.start:end]
        
        for filename in files:
            filepath = os.path.join(self.data_path, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sample = self._transform_to_sample(data)
                if sample:
                    samples.append(sample)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return samples
    
    def _transform_to_sample(self, data: dict) -> Optional[Sample]:
        """
        Convert raw data to Sample object
        
        Modify this method according to your data format
        """
        # Extract basic info
        problem_id = data.get('problem_id', '')
        title = data.get('title', '')
        description = data.get('description', '')
        generator = data.get('generator_code', '')
        checker = data.get('checker_code')
        
        # Validate required fields
        if not all([problem_id, description, generator]):
            return None
        
        # Process solutions
        canonical_solutions = []
        correct_solutions = []
        incorrect_solutions = []
        
        for sol_data in data.get('solutions', []):
            code = sol_data.get('code', '')
            lang_str = sol_data.get('language', 'unknown')
            is_correct = sol_data.get('correct', True)
            
            # Convert language
            language = Language.from_string(lang_str)
            if language == Language.UNKNOWN:
                continue
            
            solution = Solution(code=code, language=language)
            
            if is_correct:
                correct_solutions.append(solution)
                canonical_solutions.append(solution)
            else:
                incorrect_solutions.append(solution)
        
        # At least one correct and one incorrect solution required
        if not correct_solutions or not incorrect_solutions:
            return None
        
        # Process test cases (if any)
        public_tests = []
        for tc_data in data.get('test_cases', []):
            test = TestCase(
                input=tc_data.get('input', ''),
                output=tc_data.get('output', '')
            )
            public_tests.append(test)
        
        return Sample(
            id=problem_id,
            name=title,
            description=description,
            generator=generator,
            checker=checker,
            canonical_solutions=canonical_solutions,
            correct_solutions=correct_solutions,
            incorrect_solutions=incorrect_solutions,
            public_tests=public_tests,
            metadata={
                'source': 'custom',
                'tags': data.get('tags', [])
            }
        )
    
    def __iter__(self) -> Iterator[Sample]:
        """Iterate samples"""
        for sample in self._samples:
            yield sample
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self._samples)
    
    @property
    def name(self) -> str:
        """Dataset name"""
        return "MyCustomDataset"


# ============================================================================
# More examples: Read from CSV
# ============================================================================

class CSVDatasetReader(DatasetReader):
    """
    Example of reading dataset from CSV files
    
    Assumed CSV format:
    problem_id,title,description,generator_path,correct_solution_path,incorrect_solution_path
    """
    
    def __init__(self, csv_path: str, base_dir: str = ""):
        import csv
        
        self.csv_path = csv_path
        self.base_dir = base_dir or os.path.dirname(csv_path)
        self._samples = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = self._row_to_sample(row)
                if sample:
                    self._samples.append(sample)
    
    def _row_to_sample(self, row: dict) -> Optional[Sample]:
        """Convert CSV row to Sample"""
        try:
            # Read generator code
            gen_path = os.path.join(self.base_dir, row['generator_path'])
            with open(gen_path, 'r') as f:
                generator = f.read()
            
            # Read correct solution
            correct_path = os.path.join(self.base_dir, row['correct_solution_path'])
            with open(correct_path, 'r') as f:
                correct_code = f.read()
            
            # Read incorrect solution
            incorrect_path = os.path.join(self.base_dir, row['incorrect_solution_path'])
            with open(incorrect_path, 'r') as f:
                incorrect_code = f.read()
            
            # Infer language (based on file extension)
            correct_lang = self._infer_language(correct_path)
            incorrect_lang = self._infer_language(incorrect_path)
            
            return Sample(
                id=row['problem_id'],
                name=row.get('title', row['problem_id']),
                description=row['description'],
                generator=generator,
                canonical_solutions=[Solution(code=correct_code, language=correct_lang)],
                correct_solutions=[Solution(code=correct_code, language=correct_lang)],
                incorrect_solutions=[Solution(code=incorrect_code, language=incorrect_lang)]
            )
            
        except Exception as e:
            print(f"Error processing row {row.get('problem_id')}: {e}")
            return None
    
    def _infer_language(self, filepath: str) -> Language:
        """Infer language from file extension"""
        ext = os.path.splitext(filepath)[1].lower()
        mapping = {
            '.py': Language.PYTHON,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.java': Language.JAVA,
        }
        return mapping.get(ext, Language.UNKNOWN)
    
    def __iter__(self) -> Iterator[Sample]:
        for sample in self._samples:
            yield sample
    
    def __len__(self) -> int:
        return len(self._samples)
    
    @property
    def name(self) -> str:
        return "CSVDataset"


# ============================================================================
# More examples: Read from HuggingFace Hub
# ============================================================================

class HuggingFaceDatasetReader(DatasetReader):
    """
    Example of reading dataset from HuggingFace Hub
    
    Assumed dataset structure similar to deepmind/code_contests
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        generator_field: str = "generator",
        start: int = 0,
        end: int = -1
    ):
        from datasets import load_dataset
        
        self.dataset_name = dataset_name
        self.split = split
        self.generator_field = generator_field
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Apply slice
        end = end if end != -1 else len(dataset)
        dataset = dataset.select(range(start, min(end, len(dataset))))
        
        self._samples = [self._transform(item) for item in dataset]
        self._samples = [s for s in self._samples if s is not None]
    
    def _transform(self, item: dict) -> Optional[Sample]:
        """Convert HuggingFace item to Sample"""
        # Modify this method according to your HuggingFace dataset structure
        try:
            return Sample(
                id=item.get('name', '').split('.')[0],
                name=item.get('name', ''),
                description=item.get('description', ''),
                generator=item.get(self.generator_field, ''),
                # ... Other fields
            )
        except Exception:
            return None
    
    def __iter__(self) -> Iterator[Sample]:
        for sample in self._samples:
            yield sample
    
    def __len__(self) -> int:
        return len(self._samples)
    
    @property
    def name(self) -> str:
        return f"HuggingFace:{self.dataset_name}"
