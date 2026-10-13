"""In-process caches shared by the views, the forecasting service and the rate limiter.

  LRUCache  bounded, evicts the least recently used entry (O(1) get/set via OrderedDict)
  TTLCache  an LRUCache whose entries also expire a fixed number of seconds after being set

Both are thread-safe, since gunicorn threads and ThreadPoolExecutor workers share them.
Each process has its own copy; use Redis if several processes must share one cache.
"""
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

_MISSING = object()


class Cache(ABC):
    """Interface every cache implements; `get_or_compute` is shared by all of them."""

    @abstractmethod
    def get(self, key, default=None):
        """Cached value for `key`, or `default`."""

    @abstractmethod
    def set(self, key, value):
        """Store `value` under `key`."""

    @abstractmethod
    def __len__(self):
        """Number of live entries."""

    def __contains__(self, key):
        return self.get(key, _MISSING) is not _MISSING

    def get_or_compute(self, key, compute):
        """Cached value for `key`, computing and storing it with `compute()` on a miss.
        Exceptions from `compute` propagate and nothing is cached, so failures are retried."""
        value = self.get(key, _MISSING)
        if value is _MISSING:
            value = compute()
            self.set(key, value)
        return value


class LRUCache(Cache):
    """At most `maxsize` entries; a hit moves the entry to the back, eviction pops the front."""

    def __init__(self, maxsize):
        if maxsize < 1:
            raise ValueError('maxsize must be at least 1')
        self._maxsize = maxsize
        self._data = OrderedDict()
        self._lock = threading.RLock()

    @property
    def maxsize(self):
        return self._maxsize

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def pop(self, key, default=None):
        with self._lock:
            return self._data.pop(key, default)

    def clear(self):
        with self._lock:
            self._data.clear()

    def __len__(self):
        with self._lock:
            return len(self._data)


class TTLCache(LRUCache):
    """LRUCache whose entries expire `ttl` seconds after they were set.
    `clock` is injectable so tests can move time forward without sleeping."""

    def __init__(self, maxsize, ttl, clock=time.monotonic):
        super().__init__(maxsize)
        self._ttl = ttl
        self._clock = clock

    def get(self, key, default=None):
        with self._lock:
            entry = super().get(key, _MISSING)
            if entry is _MISSING:
                return default
            expires, value = entry
            if self._clock() >= expires:
                self.pop(key)
                return default
            return value

    def set(self, key, value):
        super().set(key, (self._clock() + self._ttl, value))

    def __len__(self):
        with self._lock:
            now = self._clock()
            return sum(1 for expires, _ in self._data.values() if now < expires)
