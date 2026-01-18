"""
Common utility functions
"""

import re
from typing import Tuple, List, Optional, Dict, Literal
from pydantic import BaseModel


Language = Literal['python', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'php', 'csharp', 'bash', 'typescript', 'sql',
                   'rust', 'cuda', 'lua', 'R', 'perl', 'D_ut', 'ruby', 'scala', 'julia', 'pytest', 'junit',
                   'kotlin_script', 'jest', 'verilog', 'python_gpu', 'lean', 'swift', 'racket']
NullableLang = Language | Literal['']


language_to_aliases = {
    'python': ['python', 'Python', 'py', 'Python3', 'python3', 'PY'],
    'cpp': ['cpp', 'c++', 'C++', 'Cpp', 'CPP'],
    'nodejs': ['javascript', 'Javascript', 'JavaScript', 'JS', 'js'],
    'go': ['go', 'Go'],
    'java': ['java', 'Java'],
    'php': ['php', 'PHP'],
    'csharp': ['csharp', 'c#', 'C#'],
    'bash': ['bash', 'Bash', 'BASH', 'sh', 'shell'],
    'typescript': ['typescript'],
    'rust': ['rust', 'Rust', 'rs'],
    'sql': ['sql', 'SQL', 'Sql'],
    'D': ['D', 'd'],
    'julia': ['julia', 'Julia', 'jl'],
    'lua': ['lua', 'Lua'],
    'perl': ['perl', 'Perl', 'PERL'],
    'R': ['R', 'r'],
    'ruby': ['ruby', 'Ruby'],
    'scala': ['scala', 'Scala'],
    'kotlin': ['kotlin', 'Kotlin'],
    'c': ['c', 'C'],
    'html': ['html', 'Html', 'HTML'],
    'javascript': ['javascript', 'Javascript', 'JavaScript'],
    'verilog': ['verilog', 'Verilog', 'VERILOG'],
    'racket': ['racket'],
    'swift': ['swift'],
}

TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""
LANGUAGE = ["UNKNOWN_LANGUAGE", "PYTHON", "CPP", "PYTHON3", "JAVA"]

fenced_code_block_pattern = re.compile(
    r'```([^\n]*)\n'
    r'(.*?)'
    r'\n\s*```',
    re.DOTALL | re.MULTILINE
)

aliases_to_language_tiled = {v: k for k, vs in language_to_aliases.items() for v in vs}

incomplete_fenced_code_block_pattern = re.compile(
    r'```([^\n]*)\n'
    r'(.*)',
    re.DOTALL | re.MULTILINE
)


class CodeBlock(BaseModel):
    priority: int
    language: str
    code: str


def extract_fenced_code(completion: str) -> List[CodeBlock]:
    code_matches = re.findall(fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), '')
        results.append(CodeBlock(priority=30, language=lang, code=m[1]))
    return results


def extract_heuristic_code(completion: str, language: NullableLang = '') -> List[CodeBlock]:
    def extract_py(text):
        code = "\n".join([line for line in text.split("\n") if line.strip() != ""]) + "\n"

        pattern_py = "(?:^(?:import|from|#)[^\n]+\n)*" \
            "^(?:def|class) [^\n]+\n" \
            r"(?:\s+[^\n]+\n)+"
        matches = re.findall(pattern_py, code, re.M)
        return matches

    def extract_sql(text):
        code = "\n".join([line for line in text.split("\n") if line.strip() != ""]) + "\n"

        pattern_sql = r"^\s*(?:select|with\s[^\n]+as)[^;]*"
        matches = re.findall(pattern_sql, code, re.M | re.IGNORECASE)
        return matches

    def extract_bash(text):
        code = "\n".join([line for line in text.split("\n") if line.strip() != ""]) + "\n"
        return code

    if language == 'python':
        return [CodeBlock(priority=10, language='python', code=m) for m in extract_py(completion)]
    elif language == 'sql':
        return [CodeBlock(priority=10, language='sql', code=m) for m in extract_sql(completion)]
    elif language == 'bash':
        return [CodeBlock(priority=10, language='bash', code=extract_bash(completion))]
    else:
        return []


def extract_incomplete_fenced_code(completion: str) -> List[CodeBlock]:
    code_matches = re.findall(incomplete_fenced_code_block_pattern, completion)
    results = []
    for m in code_matches:
        lang = aliases_to_language_tiled.get(m[0].strip(), '')
        results.append(CodeBlock(priority=20, language=lang, code=m[1]))
    return results


def extract_custom_code(completion: str, custom_logic: str) -> List[CodeBlock]:
    blocks = []

    def submit(cbs):
        for cb in cbs:
            assert isinstance(cb, CodeBlock), 'extract code type must be class CodeBlock'
            blocks.append(cb)

    context = {
        'CodeBlock': CodeBlock,
        'completion': completion,
        'submit_code_blocks': submit,
        'extract_fenced_code': extract_fenced_code,
        'extract_heuristic_code': extract_heuristic_code,
    }
    exec(custom_logic, context)
    return blocks


def filter_language(blocks: List[CodeBlock], language: NullableLang) -> List[CodeBlock]:
    return [b for b in blocks if b.language == language]


def default_extract_helper(completion: str, language: NullableLang = '', custom_extract_logic: Optional[str] = None) -> str:
    """
    Default code extraction logic
    
    By default, find all the fenced code blocks and add heuristic blocks if first one fails
    Use the first block with target language, and fallback to the first any language block
    """
    code_blocks = extract_fenced_code(completion)
    code_blocks += extract_heuristic_code(completion, language)
    code_blocks += extract_incomplete_fenced_code(completion)
    if custom_extract_logic is not None:
        code_blocks += extract_custom_code(completion, custom_extract_logic)
    
    if len(code_blocks) == 0:
        return ''

    max_priority = max([cb.priority for cb in code_blocks])
    code_blocks = [cb for cb in code_blocks if cb.priority == max_priority]

    target_blocks = filter_language(code_blocks, language)
    if len(target_blocks) > 0:
        return target_blocks[0].code
    return code_blocks[0].code


def is_float(s: str) -> bool:
    """Check if string is float"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_equal(a: float, b: float, rel_tol: float = 1e-5) -> bool:
    """Compare two floats with relative tolerance"""
    return abs(a - b) / max(abs(b), 1e-10) < rel_tol


def check_output_equal(expected_output: str, result_output: str, lower_cmp: bool = True) -> bool:
    """
    Compare if outputs are equal, supporting float comparison
    
    Args:
        expected_output: Expected output
        result_output: Actual output
        lower_cmp: Whether to ignore case
        
    Returns:
        bool: Whether equal
    """
    if expected_output == result_output:
        return True

    expected_lines = expected_output.strip().split('\n')
    result_lines = result_output.strip().split('\n')
    
    # Handle trailing empty lines difference
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == '':
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == '':
        expected_lines = expected_lines[:-1]
        
    if len(result_lines) != len(expected_lines):
        return False
        
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
            
        if rl.strip() != el.strip():
            # Try float comparison
            if is_float(el) and is_float(rl):
                if float_equal(float(rl), float(el)):
                    continue
            return False
            
    return True


def extract_code(completion: str, language: str) -> str:
    """
    Extract code from completion text
    
    Args:
        completion: Text containing code
        language: Programming language
        
    Returns:
        str: Extracted code
    """
    return default_extract_helper(completion, language)



def apply_code_patches(
    original_code: str,
    patch_blocks: List[str]
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Apply code patches
    
    Replace code based on search-replace blocks
    
    Args:
        original_code: Original code
        patch_blocks: List of patch blocks, each format:
            <<<<<<< SEARCH
            <original code>
            =======
            <replacement code>
            >>>>>>> REPLACE
            
    Returns:
        Tuple[str, List[Tuple[str, str]]]: (Modified code, List of unmatched blocks)
    """
    # Regex match patch blocks
    block_pattern = re.compile(
        r"<<<<<<<\s*SEARCH\r?\n"
        r"(.*?)\r?\n"
        r"=======\r?\n"
        r"(.*?)\r?\n"
        r">>>>>>>\s*REPLACE",
        re.DOTALL
    )
    
    modified_code = original_code
    unmatched_blocks: List[Tuple[str, str]] = []
    
    for container in patch_blocks or []:
        for search_block, replace_block in block_pattern.findall(container):
            if search_block in modified_code:
                # Replace only the first match
                modified_code = modified_code.replace(search_block, replace_block, 1)
            else:
                unmatched_blocks.append((search_block, replace_block))
    
    return modified_code, unmatched_blocks


def truncate_string(s: str, max_length: int = 1000, placeholder: str = "[truncated]") -> str:
    """
    Truncate string
    
    Args:
        s: Original string
        max_length: Maximum length
        placeholder: Truncation placeholder
        
    Returns:
        str: Truncated string
    """
    if len(s) <= max_length:
        return s
    
    half_length = (max_length - len(placeholder)) // 2
    return s[:half_length] + placeholder + s[-half_length:]


def safe_parse_list(value: str) -> Optional[List]:
    """
    Safely parse list string
    
    Args:
        value: List string
        
    Returns:
        List: Parsed result, None if failed
    """
    import ast
    
    try:
        result = ast.literal_eval(value)
        if isinstance(result, list):
            return result
    except (SyntaxError, ValueError):
        pass
    
    # Try restricted eval
    try:
        restricted_globals = {
            "__builtins__": {},
            "range": range,
            "len": len,
            "str": str,
            "int": int,
        }
        result = eval(value, restricted_globals, {})
        if isinstance(result, list):
            return result
    except Exception:
        pass
    
    return None


def inject_improved_generator(user_content: str, improved_generator: str) -> str:
    """
    Inject improved generator into user prompt
    
    Args:
        user_content: Original user prompt content
        improved_generator: Improved generator code
        
    Returns:
        str: Content with injected generator
    """
    pattern = r"(?s)(will appear as an empty string\):\r?\n).*?(\r?\nCurrent command list:)"
    
    user_content, n = re.subn(
        pattern,
        r"\1" + improved_generator + r"\2",
        user_content,
        count=1
    )
    
    if n == 0:
        raise ValueError("Insertion point not found (anchors do not match)")
    
    return user_content


def format_error_message(
    compile_result: Optional[dict],
    run_result: Optional[dict]
) -> str:
    """
    Format error message
    
    Args:
        compile_result: Compilation result
        run_result: Execution result
        
    Returns:
        str: Formatted error message
    """
    parts = []
    
    if compile_result:
        compile_error = compile_result.get('stderr', '')
        if compile_error:
            parts.append(f"Compile error (code {compile_result.get('return_code', '')}): {compile_error}")
    
    if run_result:
        run_error = run_result.get('stderr', '')
        if run_error:
            parts.append(f"Runtime error (code {run_result.get('return_code', '')}): {run_error}")
    
    return "; ".join(parts) if parts else "Unknown error"


def get_language_extension(language: str) -> str:
    """
    Get file extension for language
    
    Args:
        language: Programming language
        
    Returns:
        str: File extension
    """
    extensions = {
        "python": ".py",
        "cpp": ".cpp",
        "java": ".java",
        "c": ".c",
        "javascript": ".js",
        "typescript": ".ts",
        "go": ".go",
        "rust": ".rs",
    }
    return extensions.get(language.lower(), ".txt")
