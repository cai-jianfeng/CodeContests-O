"""
Dataset Reader Abstract Base Class

Users can implement this interface to access custom datasets
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Dict, Any
from .models import Sample


class DatasetReader(ABC):
    """
    Dataset Reader Abstract Base Class
    
    Users implementing this interface can integrate custom datasets into the CodeContests-O framework.
    
    Example:
        ```python
        class MyDatasetReader(DatasetReader):
            def __init__(self, data_path: str):
                self.data = self._load_data(data_path)
            
            def __iter__(self):
                for item in self.data:
                    yield Sample(
                        id=item['id'],
                        name=item['name'],
                        description=item['description'],
                        generator=item['generator'],
                        # ...
                    )
            
            def __len__(self):
                return len(self.data)
            
            @property
            def name(self):
                return "MyDataset"
        ```
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """
        Iterate samples
        
        Yields:
            Sample: Sample data in unified format
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return dataset size
        
        Returns:
            int: Number of samples in the dataset
        """
        pass
    
    def __getitem__(self, index: int) -> Sample:
        """
        Get sample by index (optional implementation)
        
        Args:
            index: Sample index
            
        Returns:
            Sample: Sample data
        """
        for i, sample in enumerate(self):
            if i == index:
                return sample
        raise IndexError(f"Index {index} out of range")
    
    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """
        Get single sample by ID
        
        Args:
            sample_id: Unique identifier for the sample
            
        Returns:
            Sample: Sample data, None if not exists
        """
        for sample in self:
            if sample.id == sample_id:
                return sample
        return None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Dataset name
        
        Returns:
            str: Name identifier of the dataset
        """
        pass
    
    def filter(self, predicate) -> List[Sample]:
        """
        Filter samples by condition
        
        Args:
            predicate: Filter function, receives Sample returns bool
            
        Returns:
            List[Sample]: List of samples satisfying condition
        """
        return [sample for sample in self if predicate(sample)]
    
    def slice(self, start: int = 0, end: int = -1) -> List[Sample]:
        """
        Get dataset slice
        
        Args:
            start: Start index
            end: End index, -1 means to end
            
        Returns:
            List[Sample]: List of sliced samples
        """
        samples = list(self)
        if end == -1:
            end = len(samples)
        return samples[start:end]
    
    def validate_sample(self, sample: Sample) -> List[str]:
        """
        Validate sample data integrity
        
        Args:
            sample: Sample to validate
            
        Returns:
            List[str]: Validation error list, empty means valid
        """
        errors = []
        
        if not sample.id:
            errors.append("Sample ID is required")
        if not sample.name:
            errors.append("Sample name is required")
        if not sample.description:
            errors.append("Sample description is required")
        if not sample.generator:
            errors.append("Sample generator is required")
        if not sample.canonical_solutions:
            errors.append("At least one canonical solution is required")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dict: Dictionary containing various statistical metrics
        """
        total = len(self)
        total_correct = 0
        total_incorrect = 0
        total_public_tests = 0
        languages = {"python": 0, "cpp": 0, "java": 0}
        
        for sample in self:
            total_correct += len(sample.correct_solutions)
            total_incorrect += len(sample.incorrect_solutions)
            total_public_tests += len(sample.public_tests)
            for sol in sample.canonical_solutions:
                if sol.language.value in languages:
                    languages[sol.language.value] += 1
        
        return {
            "name": self.name,
            "total_samples": total,
            "total_correct_solutions": total_correct,
            "total_incorrect_solutions": total_incorrect,
            "avg_correct_per_sample": total_correct / total if total > 0 else 0,
            "avg_incorrect_per_sample": total_incorrect / total if total > 0 else 0,
            "total_public_tests": total_public_tests,
            "languages_distribution": languages
        }
