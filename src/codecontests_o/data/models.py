"""
Data Model Definition - Unified Data Structure
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Language(Enum):
    """Supported Programming Languages"""
    UNKNOWN = "unknown"
    PYTHON = "python"
    PYTHON3 = "python"  # mapped to python
    CPP = "cpp"
    JAVA = "java"
    
    @classmethod
    def from_string(cls, lang_str: str) -> "Language":
        """Convert from string to Language Enum"""
        lang_upper = lang_str.upper()
        if "PYTHON" in lang_upper:
            return cls.PYTHON
        elif lang_upper == "CPP" or lang_upper == "C++":
            return cls.CPP
        elif lang_upper == "JAVA":
            return cls.JAVA
        return cls.UNKNOWN
    
    @classmethod
    def from_index(cls, index: int) -> "Language":
        """Convert from CodeContests language index"""
        mapping = {
            0: cls.UNKNOWN,
            1: cls.PYTHON,
            2: cls.CPP,
            3: cls.PYTHON,  # PYTHON3
            4: cls.JAVA
        }
        return mapping.get(index, cls.UNKNOWN)


@dataclass
class TestCase:
    """Test Case"""
    input: str
    output: str
    
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Convert to dictionary format"""
        return {
            "input": {"stdin": self.input},
            "output": {"stdout": self.output}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TestCase":
        """Create from dictionary"""
        input_data = data.get("input", {})
        output_data = data.get("output", {})
        return cls(
            input=input_data.get("stdin", ""),
            output=output_data.get("stdout", "")
        )


@dataclass
class Solution:
    """Solution"""
    code: str
    language: Language
    index: int = -1  # Original index, used for tracking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "code": self.code,
            "language": self.language.value,
            "index": self.index
        }


@dataclass
class Sample:
    """
    Unified Sample Data Model
    
    All dataset readers should convert data to this format
    """
    id: str
    name: str
    description: str
    generator: str  # Initial generator code
    checker: Optional[str] = None
    canonical_solutions: List[Solution] = field(default_factory=list)
    correct_solutions: List[Solution] = field(default_factory=list)
    incorrect_solutions: List[Solution] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def problem_statement(self) -> str:
        """Get full problem description"""
        title = self.name.split('. ')[-1].strip() if '. ' in self.name else self.name
        return f"{title}\n\n{self.description}"
    
    def get_canonical_solutions_by_language(self) -> Dict[str, List[str]]:
        """Get canonical solutions grouped by language"""
        result = {"python": [], "cpp": [], "java": []}
        for sol in self.canonical_solutions:
            if sol.language != Language.UNKNOWN:
                result[sol.language.value].append(sol.code)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for saving)"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "generator": self.generator,
            "checker": self.checker,
            "canonical_solution": self.get_canonical_solutions_by_language(),
            "metadata": self.metadata
        }


@dataclass
class IterationResult:
    """Result of a single iteration"""
    iteration: int
    corner_cases: List[TestCase]
    commands: List[str]
    commands_add: List[str] = field(default_factory=list)
    commands_replace: List[str] = field(default_factory=list)
    case_inputs: List[str] = field(default_factory=list)
    improved_generator: str = ""
    search_replace_blocks: List[str] = field(default_factory=list)
    unmatched_blocks: List[str] = field(default_factory=list)
    input_constraints_summary: str = ""
    validation_result: Optional[Dict] = None
    errors: Dict[str, List] = field(default_factory=dict)
    only_generate: bool = False
    exceed_context: bool = False
    messages: List[Dict] = field(default_factory=list)  # Used to restore conversation context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "iteration": self.iteration,
            "corner_cases": [tc.to_dict() for tc in self.corner_cases],
            "generate_commands": self.commands,
            "commands_add": self.commands_add,
            "commands_replace": self.commands_replace,
            "generate_case_inputs": self.case_inputs,
            "improved_generator": self.improved_generator,
            "search_replace_generator_blocks": self.search_replace_blocks,
            "unmatched_blocks": self.unmatched_blocks,
            "input_constraints_summary": self.input_constraints_summary,
            "result": self.validation_result,
            "errors": self.errors,
            "only_generate": self.only_generate,
            "exceed": self.exceed_context,
            "messages": self.messages
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationResult":
        """Create from dictionary"""
        corner_cases = [TestCase.from_dict(tc) for tc in data.get('corner_cases', [])]
        return cls(
            iteration=data.get('iteration', 0),
            corner_cases=corner_cases,
            commands=data.get('generate_commands', data.get('commands', [])),
            commands_add=data.get('commands_add', []),
            commands_replace=data.get('commands_replace', []),
            case_inputs=data.get('generate_case_inputs', []),
            improved_generator=data.get('improved_generator', ''),
            search_replace_blocks=data.get('search_replace_generator_blocks', []),
            unmatched_blocks=data.get('unmatched_blocks', []),
            input_constraints_summary=data.get('input_constraints_summary', ''),
            validation_result=data.get('result'),
            errors=data.get('errors', {}),
            only_generate=data.get('only_generate', False),
            exceed_context=data.get('exceed', False),
            messages=data.get('messages', [])
        )


@dataclass
class GenerationResult:
    """Complete generation result"""
    sample_id: str
    status: str  # "completed", "error", "processing"
    corner_cases: List[TestCase]
    commands: List[str]
    iterations: List[IterationResult]
    final_generator: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "id": self.sample_id,
            "status": self.status,
            "corner_cases": [tc.to_dict() for tc in self.corner_cases],
            "commands": self.commands,
            "result": [it.to_dict() for it in self.iterations],
            "final_generator": self.final_generator,
            "error": self.error
        }
