"""
Core module

Includes generator and validator
"""

from .generator import CornerCaseGenerator
from .validator import SolutionValidator

__all__ = [
    "CornerCaseGenerator",
    "SolutionValidator",
]
