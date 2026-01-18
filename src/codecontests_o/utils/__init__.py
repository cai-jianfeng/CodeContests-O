"""
Utility module
"""

from .helpers import (
    check_output_equal,
    extract_code,
    apply_code_patches,
    truncate_string,
    safe_parse_list,
    inject_improved_generator,
    format_error_message,
    get_language_extension,
)

from .logger import (
    LoggerManager,
    get_logger_manager,
    initialize_logger_manager,
    log_global,
    log_sample,
    cleanup_logger,
)

__all__ = [
    # Helpers
    "check_output_equal",
    "extract_code",
    "apply_code_patches",
    "truncate_string",
    "safe_parse_list",
    "inject_improved_generator",
    "format_error_message",
    "get_language_extension",
    # Logger
    "LoggerManager",
    "get_logger_manager",
    "initialize_logger_manager",
    "log_global",
    "log_sample",
    "cleanup_logger",
]
