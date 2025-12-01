"""
Advanced caching utility
"""
import time
from typing import Optional, Any, Dict
from collections import OrderedDict
import threading
from config import settings


class TTLCache:
    """Thread-safe TTL cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of items in cache
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            Cached value or None if expired/not found
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            cache_ttl = ttl or settings.CACHE_TTL
            if time.time() - self.timestamps[key] > cache_ttl:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (LRU)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            return value
    
    def set(self, key: str, value: Any):
        """
        Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Evict oldest if over limit
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


# Global cache instance
cache = TTLCache(max_size=1000)



