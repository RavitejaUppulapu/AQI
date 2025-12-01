"""
Rate limiting utility
"""
import time
from collections import defaultdict
from typing import Dict, Tuple
from threading import Lock
from config import settings


class RateLimiter:
    """Simple rate limiter using sliding window"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """
        Check if request is allowed
        
        Args:
            identifier: Client identifier (IP address, user ID, etc.)
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, self.max_requests
        
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False, 0
            
            # Add current request
            self.requests[identifier].append(now)
            
            remaining = self.max_requests - len(self.requests[identifier])
            return True, remaining
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        with self.lock:
            if identifier in self.requests:
                del self.requests[identifier]


# Global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_PER_MINUTE,
    window_seconds=60
)



