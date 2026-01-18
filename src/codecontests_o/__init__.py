"""
CodeContests-O: Feedback-Driven Iterative Test Case Generation

A framework for generating high-quality test cases for competitive programming problems
using feedback-driven iterative refinement.
"""

__version__ = "0.1.0"

from .data import (
    DatasetReader,
    CodeContestsReader,
    CodeContestsHFReader,
    Sample,
    TestCase,
    Solution,
    Language,
)

from .config import (
    Config,
    get_preset_config,
)

from .core import (
    CornerCaseGenerator,
    SolutionValidator,
)

from .parallel import (
    ParallelProcessor,
)

__all__ = [
    # Version
    "__version__",
    # Data
    "DatasetReader",
    "CodeContestsReader",
    "CodeContestsHFReader",
    "Sample",
    "TestCase",
    "Solution",
    "Language",
    # Config
    "Config",
    "get_preset_config",
    # Core
    "CornerCaseGenerator",
    "SolutionValidator",
    # Parallel
    "ParallelProcessor",
]
