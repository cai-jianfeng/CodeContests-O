"""
API Pool Management

Provides thread-safe API endpoint allocation and recycling
"""

import queue
import threading
import requests
from typing import List, Optional, Tuple


class APIPool:
    """
    API Pool
    
    Manages allocation and recycling of Sandbox API endpoints, supports thread-safe concurrent access
    """
    
    def __init__(self, api_paths: List[str]):
        """
        Initialize API pool
        
        Args:
            api_paths: List of API endpoint paths
        """
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._active_count = 0
        
        # Create a session for each API endpoint
        for api_path in api_paths:
            session = requests.Session()
            self._queue.put((api_path, session))
    
    def acquire(self, timeout: float = 0.1) -> Optional[Tuple[str, requests.Session]]:
        """
        Acquire an API endpoint
        
        Args:
            timeout: Wait timeout
            
        Returns:
            Tuple[str, Session]: (API path, HTTP Session), None if timeout
        """
        try:
            item = self._queue.get(timeout=timeout)
            with self._lock:
                self._active_count += 1
            return item
        except queue.Empty:
            return None
    
    def release(self, api_path: str, session: requests.Session):
        """
        Release API endpoint
        
        Args:
            api_path: API path
            session: HTTP Session
        """
        if api_path is not None:
            with self._lock:
                self._active_count -= 1
            self._queue.put((api_path, session))
    
    @property
    def available_count(self) -> int:
        """Get number of available API endpoints"""
        return self._queue.qsize()
    
    @property
    def active_count(self) -> int:
        """Get number of active API endpoints"""
        with self._lock:
            return self._active_count
    
    @property
    def total_count(self) -> int:
        """Get total number of API endpoints"""
        return self.available_count + self.active_count


# Global API pool
_global_api_pool: Optional[APIPool] = None
_global_pool_lock = threading.Lock()


def initialize_api_pool(api_paths: List[str]) -> APIPool:
    """
    Initialize global API pool
    
    Args:
        api_paths: List of API endpoint paths
        
    Returns:
        APIPool: API pool instance
    """
    global _global_api_pool
    
    with _global_pool_lock:
        if _global_api_pool is None:
            _global_api_pool = APIPool(api_paths)
    
    return _global_api_pool


def get_api_pool() -> Optional[APIPool]:
    """Get global API pool"""
    return _global_api_pool


def acquire_api(timeout: float = 0.1) -> Optional[Tuple[str, requests.Session]]:
    """
    Acquire API endpoint from global pool
    
    Args:
        timeout: Wait timeout
        
    Returns:
        Tuple: (API path, Session), None if timeout
    """
    pool = get_api_pool()
    if pool:
        return pool.acquire(timeout)
    return None


def release_api(api_path: str, session: requests.Session):
    """
    Release API endpoint to global pool
    
    Args:
        api_path: API path
        session: HTTP Session
    """
    pool = get_api_pool()
    if pool and api_path is not None:
        pool.release(api_path, session)


def reset_api_pool():
    """Reset global API pool"""
    global _global_api_pool
    
    with _global_pool_lock:
        _global_api_pool = None


class APIPoolContext:
    """
    API Pool Context Manager
    
    Automatically acquire and release API endpoints
    
    Example:
        ```python
        with APIPoolContext() as (api_path, session):
            if api_path:
                response = session.post(api_path + "run_code", json=payload)
        ```
    """
    
    def __init__(self, timeout: float = 5.0, retry_interval: float = 0.1):
        """
        Initialize context manager
        
        Args:
            timeout: Total wait timeout
            retry_interval: Retry interval
        """
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.api_path: Optional[str] = None
        self.session: Optional[requests.Session] = None
    
    def __enter__(self) -> Tuple[Optional[str], Optional[requests.Session]]:
        """Enter context, acquire API endpoint"""
        import time
        
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            result = acquire_api(timeout=self.retry_interval)
            if result:
                self.api_path, self.session = result
                return self.api_path, self.session
        
        return None, None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, release API endpoint"""
        if self.api_path and self.session:
            release_api(self.api_path, self.session)
        return False
