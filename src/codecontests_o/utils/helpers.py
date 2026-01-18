"""
Common utility functions
"""

import re
from typing import Tuple, List, Optional


def check_output_equal(expected: str, actual: str, lower_cmp: bool = True) -> bool:
    """
    Compare if outputs are equal
    
    Args:
        expected: Expected output
        actual: Actual output
        lower_cmp: Whether to ignore case
        
    Returns:
        bool: Whether equal
    """
    # Normalize whitespace
    expected_normalized = expected.strip()
    actual_normalized = actual.strip()
    
    if lower_cmp:
        expected_normalized = expected_normalized.lower()
        actual_normalized = actual_normalized.lower()
    
    # Direct comparison
    if expected_normalized == actual_normalized:
        return True
    
    # Compare by line (ignore trailing whitespace)
    expected_lines = [line.rstrip() for line in expected_normalized.split('\n')]
    actual_lines = [line.rstrip() for line in actual_normalized.split('\n')]
    
    if expected_lines == actual_lines:
        return True
    
    # Remove empty lines and compare
    expected_lines = [line for line in expected_lines if line]
    actual_lines = [line for line in actual_lines if line]
    
    return expected_lines == actual_lines


def extract_code(completion: str, language: str) -> str:
    """
    Extract code from completion text
    
    Args:
        completion: Text containing code
        language: Programming language
        
    Returns:
        str: Extracted code
    """
    # Try extracting code block
    code_block_pattern = r'```(?:' + language + r')?\s*(.*?)```'
    matches = re.findall(code_block_pattern, completion, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # If no code block, return original text
    return completion.strip()


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
