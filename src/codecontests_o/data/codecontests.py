"""
CodeContests Dataset Reader

Supports reading CodeContests format data from JSON file directory
"""

import os
import json
from typing import Iterator, Optional, List, Set, Dict, Any
from tqdm import tqdm

from .base import DatasetReader
from .models import Sample, Solution, TestCase, Language


# CodeContests language index map
CODECONTESTS_LANGUAGE_MAP = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]


class CodeContestsReader(DatasetReader):
    """
    CodeContests Dataset Reader

    Supports two data sources:
    1. Local JSON file directory
    2. HuggingFace dataset (e.g., ByteDance-Seed/Code-Contests-Plus)

    Args:
        data_path: Data path, can be:
            - Local JSON file directory path
            - HuggingFace dataset name (e.g., "ByteDance-Seed/Code-Contests-Plus")
            - Local HuggingFace dataset path
        start: Start index
        end: End index, -1 means to end
        require_both_solutions: Whether to require both correct and incorrect solutions
        split: HuggingFace dataset split (train/test/validation), only valid when using HF dataset
        use_hf: Force use HuggingFace mode, auto-detect if None

    Example:
        ```python
        # Read from local JSON files
        reader = CodeContestsReader(
            data_path="/path/to/json/files",
            start=0,
            end=100
        )

        # Read from HuggingFace dataset
        reader = CodeContestsReader(
            data_path="ByteDance-Seed/Code-Contests-Plus",
            split="test",
            start=0,
            end=100
        )

        for sample in reader:
            print(f"Processing {sample.id}")
        ```
    """

    def __init__(
        self,
        data_path: str,
        start: int = 0,
        end: int = -1,
        require_both_solutions: bool = True,
        split: str = "train",
        use_hf: Optional[bool] = None
    ):
        self.data_path = data_path
        self.start = start
        self.end = end
        self.require_both_solutions = require_both_solutions
        self.split = split

        # Auto-detect data source type
        if use_hf is None:
            # If directory and contains .json files, use file mode
            if os.path.isdir(data_path):
                json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
                self.use_hf = len(json_files) == 0  # If no JSON files, likely HF cache directory
            else:
                # Not a directory, try as HF dataset name
                self.use_hf = True
        else:
            self.use_hf = use_hf

        # Initialize based on mode
        if self.use_hf:
            self._hf_dataset = None
        else:
            # Load data file list
            self._file_list = self._get_file_list()

        self._samples_cache: Optional[List[Sample]] = None
    
    def _get_file_list(self) -> List[str]:
        """Get JSON file list"""
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Data path is not a directory: {self.data_path}")

        files = [f for f in os.listdir(self.data_path) if f.endswith('.json')]
        files.sort()  # Ensure consistent order

        end = self.end if self.end != -1 else len(files)
        return files[self.start:end]

    def _load_hf_dataset(self):
        """Lazily load HuggingFace dataset"""
        if self._hf_dataset is not None:
            return self._hf_dataset

        try:
            from datasets import load_dataset, load_from_disk
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for loading HuggingFace datasets. "
                "Install it with: pip install datasets"
            )

        # Check if local cache directory
        if os.path.isdir(self.data_path):
            # Try loading from local
            try:
                self._hf_dataset = load_from_disk(self.data_path)
                print(f"Loaded dataset from local cache: {self.data_path}")
            except Exception as e:
                # Try loading using load_dataset
                try:
                    self._hf_dataset = load_dataset(self.data_path, split=self.split)
                    print(f"Loaded dataset from HuggingFace: {self.data_path}, split: {self.split}")
                except Exception as e2:
                    raise ValueError(
                        f"Failed to load dataset from local path '{self.data_path}': {e}. "
                        f"Also failed to load as HuggingFace dataset: {e2}"
                    )
        else:
            # Load from HuggingFace Hub
            try:
                self._hf_dataset = load_dataset(self.data_path, split=self.split)
                print(f"Loaded dataset from HuggingFace: {self.data_path}, split: {self.split}")
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset '{self.data_path}' from HuggingFace: {e}"
                )

        return self._hf_dataset
    
    def _transform_raw_data(self, raw_data: Dict[str, Any], file_name: str) -> Optional[Sample]:
        """
        Transform raw JSON data to unified Sample format
        
        Args:
            raw_data: Raw JSON data
            file_name: File name (used for generating ID)
            
        Returns:
            Sample: Transformed sample, None if condition not met
        """
        # Generate sample ID
        if 'id' in raw_data:
            data_id = "Codeforces_" + str(raw_data['id']).split('.')[0].split(' ')[0]
        else:
            data_id = file_name
        
        # Process solutions
        correct_solutions = []
        incorrect_solutions = []
        canonical_solutions = []
        
        # Helper function: process Code-Contests-Plus format submission list
        def process_submissions_list(submissions):
            processed_solutions = []
            for idx, submission in enumerate(submissions):
                # Compatible with different field names: code or solution
                solution_code = submission.get('code', submission.get('solution', ''))
                # Compatible with different language field values
                lang_str = submission.get('language', '')
                
                language = Language.from_string(lang_str)
                # Try looser language matching, see solutions_eval_plus_test_cases.py
                if language == Language.UNKNOWN:
                    if lang_str in ['py2', 'py3', 'python', 'Python', 'PYTHON']:
                        language = Language.PYTHON
                    elif lang_str in ['cpp', 'c++', 'CPP']:
                        language = Language.CPP
                    elif lang_str in ['java', 'JAVA']:
                        language = Language.JAVA
                    else:
                        continue
                
                solution = Solution(code=solution_code, language=language, index=idx)
                processed_solutions.append(solution)
            return processed_solutions

        # 1. Try CodeContests original format (Duck Typing: dict containing language and solution lists)
        if 'solutions' in raw_data and isinstance(raw_data['solutions'], dict) and 'language' in raw_data['solutions']:
            solutions_data = raw_data['solutions']
            for idx, (lang_idx, solution_code) in enumerate(
                zip(solutions_data.get('language', []), solutions_data.get('solution', []))
            ):
                language = Language.from_index(lang_idx)
                if language == Language.UNKNOWN:
                    continue
                
                solution = Solution(code=solution_code, language=language, index=idx)
                correct_solutions.append(solution)
                canonical_solutions.append(solution)
        
        # 2. Try Code-Contests-Plus format (List of dicts)
        # Supported field names: 'correct_submissions' or 'solutions' (if list)
        elif 'correct_submissions' in raw_data:
            correct_solutions = process_submissions_list(raw_data['correct_submissions'])
            canonical_solutions.extend(correct_solutions)
        elif 'solutions' in raw_data and isinstance(raw_data['solutions'], list):
            correct_solutions = process_submissions_list(raw_data['solutions'])
            canonical_solutions.extend(correct_solutions)

        # Process incorrect solutions
        # 1. Original format
        if 'incorrect_solutions' in raw_data and isinstance(raw_data['incorrect_solutions'], dict) and 'language' in raw_data['incorrect_solutions']:
            incorrect_data = raw_data['incorrect_solutions']
            for idx, (lang_idx, solution_code) in enumerate(
                zip(incorrect_data.get('language', []), incorrect_data.get('solution', []))
            ):
                language = Language.from_index(lang_idx)
                if language == Language.UNKNOWN:
                    continue
                
                solution = Solution(code=solution_code, language=language, index=idx)
                incorrect_solutions.append(solution)
        
        # 2. Plus format
        elif 'incorrect_submissions' in raw_data:
            incorrect_solutions = process_submissions_list(raw_data['incorrect_submissions'])
        elif 'incorrect_solutions' in raw_data and isinstance(raw_data['incorrect_solutions'], list):
            incorrect_solutions = process_submissions_list(raw_data['incorrect_solutions'])
        
        # Check if conditions are met
        if self.require_both_solutions:
            if not correct_solutions or not incorrect_solutions:
                return None
        
        # Get generator and checker
        generator = raw_data.get('generator_refined', raw_data.get('generator', ''))
        checker = raw_data.get('checker', None)
        
        return Sample(
            id=data_id,
            name=raw_data.get('title', ''),
            description=raw_data.get('description', ''),
            generator=generator,
            checker=checker,
            canonical_solutions=canonical_solutions,
            correct_solutions=correct_solutions,
            incorrect_solutions=incorrect_solutions,
            metadata={
                'cf_tags': raw_data.get('cf_tags', []),
                'difficulty': raw_data.get('difficulty', ''),
                'time_limit': raw_data.get('time_limit', {}),
                'memory_limit': raw_data.get('memory_limit', {}),
                'source_file': file_name
            }
        )
    
    def _extract_test_cases(self, test_data: Dict) -> List[TestCase]:
        """Extract test cases"""
        tests = []
        inputs = test_data.get('input', [])
        outputs = test_data.get('output', [])
        
        for inp, out in zip(inputs, outputs):
            tests.append(TestCase(input=inp, output=out))
        
        return tests
    
    def _load_all_samples(self) -> List[Sample]:
        """Load all samples to cache"""
        if self._samples_cache is not None:
            return self._samples_cache

        samples = []

        if self.use_hf:
            # Load from HuggingFace dataset
            dataset = self._load_hf_dataset()

            # Apply start/end slice
            end = self.end if self.end != -1 else len(dataset)
            dataset_slice = dataset.select(range(self.start, min(end, len(dataset))))

            for idx, hf_sample in enumerate(tqdm(dataset_slice, desc=f"Loading {self.name}")):
                try:
                    sample = self._transform_raw_data(hf_sample, f"hf_sample_{idx}")
                    if sample is not None:
                        samples.append(sample)
                except Exception as e:
                    print(f"Error processing HF sample {idx}: {e}")

        else:
            # Load from local JSON files
            for file_name in tqdm(self._file_list, desc=f"Loading {self.name}"):
                file_path = os.path.join(self.data_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)

                    sample = self._transform_raw_data(raw_data, file_name)
                    if sample is not None:
                        samples.append(sample)

                except json.JSONDecodeError as e:
                    print(f"JSON decode error in {file_name}: {e}")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        self._samples_cache = samples
        return samples
    
    def __iter__(self) -> Iterator[Sample]:
        """Iterate samples"""
        samples = self._load_all_samples()
        for sample in samples:
            yield sample
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self._load_all_samples())
    
    def __getitem__(self, index: int) -> Sample:
        """Get sample by index"""
        samples = self._load_all_samples()
        return samples[index]
    
    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """Get single sample by ID"""
        for sample in self._load_all_samples():
            if sample.id == sample_id:
                return sample
        return None
    
    @property
    def name(self) -> str:
        """Dataset name"""
        if self.use_hf:
            dataset_name = self.data_path.split('/')[-1] if '/' in self.data_path else self.data_path
            return f"CodeContests-HF-{dataset_name}-{self.split}"
        return "CodeContests"
    
    def to_dict_format(self, sample: Sample) -> Dict[str, Any]:
        """
        Convert Sample to original CodeContests compatible dict format
        
        Used for compatibility with existing code
        """
        # Build solutions dict
        solutions = {
            'language': [
                CODECONTESTS_LANGUAGE_MAP.index(s.language.value.upper()) 
                if s.language.value.upper() in CODECONTESTS_LANGUAGE_MAP 
                else 0
                for s in sample.correct_solutions
            ],
            'solution': [s.code for s in sample.correct_solutions]
        }
        
        incorrect_solutions = {
            'language': [
                CODECONTESTS_LANGUAGE_MAP.index(s.language.value.upper())
                if s.language.value.upper() in CODECONTESTS_LANGUAGE_MAP
                else 0
                for s in sample.incorrect_solutions
            ],
            'solution': [s.code for s in sample.incorrect_solutions]
        }
        
        return {
            'id': sample.id,
            'name': sample.name,
            'description': sample.description,
            'generator_refined': sample.generator,
            'checker': sample.checker,
            'solutions': solutions,
            'incorrect_solutions': incorrect_solutions,
            'canonical_solution': sample.get_canonical_solutions_by_language(),
            'content': sample.problem_statement,
            'labels': {
                'tag': sample.metadata.get('cf_tags', []),
                'title': sample.name.split('.')[-1].strip() if '.' in sample.name else sample.name
            },
            **sample.metadata
        }


class CodeContestsHFReader(DatasetReader):
    """
    Read CodeContests data from HuggingFace Datasets format
    
    Args:
        data_path: HuggingFace dataset path or local path
        split: Dataset split (train/test/valid)
        
    Example:
        ```python
        reader = CodeContestsHFReader(
            data_path="deepmind/code_contests",
            split="test"
        )
        ```
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "test",
        require_both_solutions: bool = True
    ):
        self.data_path = data_path
        self.split = split
        self.require_both_solutions = require_both_solutions
        
        self._dataset = None
        self._samples_cache: Optional[List[Sample]] = None
    
    def _load_dataset(self):
        """Lazily load dataset"""
        if self._dataset is not None:
            return self._dataset
        
        from datasets import load_dataset, Dataset
        
        if os.path.isdir(self.data_path):
            self._dataset = Dataset.load_from_disk(self.data_path)
        else:
            self._dataset = load_dataset(self.data_path, split=self.split)
        
        return self._dataset
    
    def _transform_hf_data(self, hf_sample: Dict) -> Optional[Sample]:
        """Convert HuggingFace format data to Sample"""
        # Similar logic to CodeContestsReader
        data_id = "Codeforces_" + hf_sample['name'].split('.')[0].split(' ')[0]
        
        correct_solutions = []
        incorrect_solutions = []
        canonical_solutions = []
        
        if 'solutions' in hf_sample:
            for idx, (lang_idx, code) in enumerate(
                zip(hf_sample['solutions']['language'], hf_sample['solutions']['solution'])
            ):
                language = Language.from_index(lang_idx)
                if language == Language.UNKNOWN:
                    continue
                
                solution = Solution(code=code, language=language, index=idx)
                correct_solutions.append(solution)
                canonical_solutions.append(solution)
        
        if 'incorrect_solutions' in hf_sample:
            for idx, (lang_idx, code) in enumerate(
                zip(hf_sample['incorrect_solutions']['language'], hf_sample['incorrect_solutions']['solution'])
            ):
                language = Language.from_index(lang_idx)
                if language == Language.UNKNOWN:
                    continue
                
                solution = Solution(code=code, language=language, index=idx)
                incorrect_solutions.append(solution)
        
        if self.require_both_solutions:
            if not correct_solutions or not incorrect_solutions:
                return None
        
        public_tests = []
        if 'public_tests' in hf_sample:
            for inp, out in zip(hf_sample['public_tests']['input'], hf_sample['public_tests']['output']):
                public_tests.append(TestCase(input=inp, output=out))
        
        private_tests = []
        if 'private_tests' in hf_sample:
            for inp, out in zip(hf_sample['private_tests']['input'], hf_sample['private_tests']['output']):
                private_tests.append(TestCase(input=inp, output=out))
        
        generator = hf_sample.get('generator_refined', hf_sample.get('generator', ''))
        checker = hf_sample.get('checker', None)
        
        return Sample(
            id=data_id,
            name=hf_sample.get('name', ''),
            description=hf_sample.get('description', ''),
            generator=generator,
            checker=checker,
            canonical_solutions=canonical_solutions,
            correct_solutions=correct_solutions,
            incorrect_solutions=incorrect_solutions,
            public_tests=public_tests,
            private_tests=private_tests,
            metadata={
                'cf_tags': hf_sample.get('cf_tags', []),
                'difficulty': hf_sample.get('difficulty', ''),
            }
        )
    
    def _load_all_samples(self) -> List[Sample]:
        """Load all samples"""
        if self._samples_cache is not None:
            return self._samples_cache
        
        dataset = self._load_dataset()
        samples = []
        
        for hf_sample in tqdm(dataset, desc=f"Loading {self.name}"):
            sample = self._transform_hf_data(hf_sample)
            if sample is not None:
                samples.append(sample)
        
        self._samples_cache = samples
        return samples
    
    def __iter__(self) -> Iterator[Sample]:
        samples = self._load_all_samples()
        for sample in samples:
            yield sample
    
    def __len__(self) -> int:
        return len(self._load_all_samples())
    
    @property
    def name(self) -> str:
        return f"CodeContests-HF-{self.split}"
