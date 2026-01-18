"""
Log Management Module

Provides global log and sample-level log management
"""

import os
import logging
import threading
from typing import Dict, Optional


class LoggerManager:
    """
    Logger Manager
    
    Manages global logs and independent logs for samples
    """
    
    def __init__(
        self,
        results_dir: Optional[str] = None,
        start: int = 0,
        end: int = -1
    ):
        """
        Initialize Logger Manager
        
        Args:
            results_dir: Results directory (logs will be saved in its log subdirectory)
            start: Processing start index (used for log filename)
            end: Processing end index
        """
        self.start = start
        self.end = end
        self.results_dir = results_dir
        self.log_dir: Optional[str] = None
        
        self.global_logger: Optional[logging.Logger] = None
        self.sample_loggers: Dict[str, logging.Logger] = {}
        self._lock = threading.Lock()
        
        if self.results_dir:
            self._setup_log_directory()
            self._setup_global_logger()
    
    def _setup_log_directory(self):
        """Set up log directory"""
        self.log_dir = os.path.join(self.results_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _setup_global_logger(self):
        """Set up global logger"""
        if not self.log_dir:
            return
        
        log_filename = f"global_{self.start}_{self.end}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        # Clear existing log file
        if os.path.exists(log_path):
            open(log_path, 'w').close()
        
        # Create logger
        self.global_logger = logging.getLogger('codecontests_o_global')
        self.global_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in self.global_logger.handlers[:]:
            self.global_logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.global_logger.addHandler(file_handler)
        self.global_logger.propagate = False
    
    def _get_or_create_sample_logger(self, sample_id: str) -> Optional[logging.Logger]:
        """Get or create sample logger"""
        if not self.log_dir:
            return self.global_logger
        
        with self._lock:
            if sample_id not in self.sample_loggers:
                # Create sample log file
                log_path = os.path.join(self.log_dir, f"{sample_id}.log")
                
                if os.path.exists(log_path):
                    open(log_path, 'w').close()
                
                # Create logger
                logger = logging.getLogger(f'codecontests_o_sample_{sample_id}')
                logger.setLevel(logging.INFO)
                
                # Clear existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                
                # Add file handler
                file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                
                formatter = logging.Formatter(
                    '[%(asctime)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(formatter)
                
                logger.addHandler(file_handler)
                logger.propagate = False
                
                self.sample_loggers[sample_id] = logger
            
            return self.sample_loggers[sample_id]
    
    def log_global(self, message: str):
        """
        Record global log
        
        Args:
            message: Log message
        """
        if self.global_logger:
            self.global_logger.info(message)
        else:
            print(f"[GLOBAL] {message}")
    
    def log_sample(self, sample_id: str, message: str):
        """
        Record sample log
        
        Args:
            sample_id: Sample ID
            message: Log message
        """
        logger = self._get_or_create_sample_logger(sample_id)
        if logger:
            logger.info(message)
        else:
            print(f"[{sample_id}] {message}")
    
    def cleanup(self):
        """Clean up log resources"""
        with self._lock:
            # Close global logger
            if self.global_logger:
                for handler in self.global_logger.handlers:
                    handler.close()
                    self.global_logger.removeHandler(handler)
            
            # Close sample loggers
            for logger in self.sample_loggers.values():
                for handler in logger.handlers:
                    handler.close()
                    logger.removeHandler(handler)
            
            self.sample_loggers.clear()


# Global Logger Manager instance
_global_logger_manager: Optional[LoggerManager] = None
_logger_lock = threading.Lock()


def get_logger_manager() -> Optional[LoggerManager]:
    """Get global log manager"""
    global _global_logger_manager
    return _global_logger_manager


def initialize_logger_manager(
    results_dir: Optional[str] = None,
    start: int = 0,
    end: int = -1
) -> LoggerManager:
    """
    Initialize global log manager
    
    Args:
        results_dir: Results directory
        start: Processing start index
        end: Processing end index
        
    Returns:
        LoggerManager: Log manager instance
    """
    global _global_logger_manager
    
    with _logger_lock:
        if _global_logger_manager is None:
            _global_logger_manager = LoggerManager(results_dir, start, end)
    
    return _global_logger_manager


def log_global(message: str):
    """
    Convenience function for recording global log
    
    Args:
        message: Log message
    """
    manager = get_logger_manager()
    if manager:
        manager.log_global(message)
    else:
        print(f"[GLOBAL] {message}")


def log_sample(sample_id: str, message: str):
    """
    Convenience function for recording sample log
    
    Args:
        sample_id: Sample ID
        message: Log message
    """
    manager = get_logger_manager()
    if manager:
        manager.log_sample(sample_id, message)
    else:
        print(f"[{sample_id}] {message}")


def cleanup_logger():
    """Clean up global log manager"""
    global _global_logger_manager
    
    with _logger_lock:
        if _global_logger_manager:
            _global_logger_manager.cleanup()
            _global_logger_manager = None
