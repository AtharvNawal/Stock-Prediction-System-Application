"""Request rate limiting with the token bucket algorithm.

Each client gets one bucket per policy. A bucket holds up to `capacity` tokens and refills
continuously at `rate` tokens per second; a request spends one token, and is refused with
HTTP 429 when the bucket is empty. Bursts up to `capacity` are allowed, while the long-run
rate can never exceed `rate`. Refill is computed lazily from the time since the last
request, so a bucket needs no timer and costs O(1) time and memory.

Policies (settings.RATE_LIMITS) protect what is expensive or sensitive:
  login/signup POSTs  -> slows password brute forcing and account-creation spam
  forecast            -> each miss trains four neural networks
  peers, API, pages   -> bound the Yahoo Finance traffic one client can cause

Buckets live in a bounded LRUCache, so a flood of spoofed or distinct IPs cannot grow
memory without limit. Like the other caches, limits are per process.
"""
import math
import re
import threading
import time
from dataclasses import dataclass

from django.conf import settings
from django.http import HttpResponse

from .caching import LRUCache


class TokenBucket:
    """Up to `capacity` requests at once, refilled at `rate` tokens per second."""

    def __init__(self, capacity, rate, clock=time.monotonic):
        if capacity < 1 or rate <= 0:
            raise ValueError('capacity must be >= 1 and rate > 0')
        self.capacity = capacity
        self.rate = rate
        self._clock = clock
        self._tokens = float(capacity)
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self):
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    def consume(self, tokens=1):
        """Spend `tokens` if available. Returns True if allowed, False if the bucket is too empty."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def retry_after(self, tokens=1):
        """Seconds until `tokens` would be available."""
        with self._lock:
            self._refill()
            return max(0.0, (tokens - self._tokens) / self.rate)

    @property
    def tokens(self):
        with self._lock:
            self._refill()
            return self._tokens


@dataclass(frozen=True)
class RatePolicy:
    """`requests` per `per_seconds` (bursting up to `requests`) for paths matching `pattern`."""
    name: str
    pattern: str
    requests: int
    per_seconds: float
    methods: tuple = ()  # empty = every method

    def matches(self, request):
        if self.methods and request.method not in self.methods:
            return False
        return re.match(self.pattern, request.path) is not None

    @property
    def rate(self):
        return self.requests / self.per_seconds


class RateLimiter:
    """One TokenBucket per (policy, client), held in a bounded LRU cache."""

    def __init__(self, policies, max_clients=10_000, clock=time.monotonic):
        self.policies = list(policies)
        self._buckets = LRUCache(max_clients)
        self._clock = clock
        self._lock = threading.Lock()

    def policy_for(self, request):
        """First matching policy, so list specific paths before catch-alls."""
        return next((p for p in self.policies if p.matches(request)), None)

    def _bucket(self, policy, client):
        key = (policy.name, client)
        with self._lock:  # get-or-create must be atomic, or two threads could each create a bucket
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(policy.requests, policy.rate, self._clock)
                self._buckets.set(key, bucket)
            return bucket

    def check(self, policy, client):
        """(allowed, seconds to wait before retrying)."""
        bucket = self._bucket(policy, client)
        if bucket.consume():
            return True, 0.0
        return False, bucket.retry_after()


def client_ip(request, trusted_proxies=0):
    """Client address for rate limiting.

    X-Forwarded-For is client-controlled, so only the entries appended by our own proxies are
    trusted: with `trusted_proxies` = N, the client is the Nth address from the right. With no
    trusted proxies the header is ignored, so a client cannot dodge its limit by forging it."""
    remote = request.META.get('REMOTE_ADDR', '')
    if trusted_proxies <= 0:
        return remote
    forwarded = [ip.strip() for ip in request.META.get('HTTP_X_FORWARDED_FOR', '').split(',') if ip.strip()]
    return forwarded[-trusted_proxies] if len(forwarded) >= trusted_proxies else remote


def policies_from_settings():
    return [RatePolicy(**spec) for spec in getattr(settings, 'RATE_LIMITS', [])]


class RateLimitMiddleware:
    """Refuses requests over their policy's rate with 429 Too Many Requests and a Retry-After header."""

    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = getattr(settings, 'RATE_LIMIT_ENABLED', True)
        self.trusted_proxies = getattr(settings, 'RATE_LIMIT_TRUSTED_PROXIES', 0)
        self.limiter = RateLimiter(policies_from_settings())

    def __call__(self, request):
        if self.enabled:
            policy = self.limiter.policy_for(request)
            if policy is not None:
                allowed, wait = self.limiter.check(policy, client_ip(request, self.trusted_proxies))
                if not allowed:
                    return self.too_many_requests(wait)
        return self.get_response(request)

    @staticmethod
    def too_many_requests(wait):
        seconds = max(1, math.ceil(wait))
        response = HttpResponse('Too many requests. Please wait {} second{} and try again.'.format(
            seconds, '' if seconds == 1 else 's'), status=429, content_type='text/plain; charset=utf-8')
        response['Retry-After'] = str(seconds)
        return response
