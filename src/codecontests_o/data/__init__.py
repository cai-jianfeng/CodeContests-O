"""
Data module

Contains data model definitions and dataset readers
"""

from .models import (
    Language,
    TestCase,
    Solution,
    Sample,
    IterationResult,
    GenerationResult,
)

from .base import DatasetReader

from .codecontests import (
    CodeContestsReader,
    CodeContestsHFReader,
)

__all__ = [
    # Models
    "Language",
    "TestCase",
    "Solution",
    "Sample",
    "IterationResult",
    "GenerationResult",
    # Readers
    "DatasetReader",
    "CodeContestsReader",
    "CodeContestsHFReader",
]
