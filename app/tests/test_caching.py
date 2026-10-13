from django.test import SimpleTestCase

from app.caching import LRUCache, TTLCache


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class LRUCacheTests(SimpleTestCase):
    def test_evicts_least_recently_used(self):
        cache = LRUCache(2)
        cache.set('a', 1)
        cache.set('b', 2)
        cache.get('a')       # 'a' is now the most recent, so 'b' goes first
        cache.set('c', 3)
        self.assertIn('a', cache)
        self.assertNotIn('b', cache)
        self.assertIn('c', cache)
        self.assertEqual(len(cache), 2)

    def test_overwrite_refreshes_recency(self):
        cache = LRUCache(2)
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('a', 10)
        cache.set('c', 3)
        self.assertEqual(cache.get('a'), 10)
        self.assertNotIn('b', cache)

    def test_get_or_compute_computes_once(self):
        cache, calls = LRUCache(4), []
        for _ in range(3):
            value = cache.get_or_compute('k', lambda: calls.append(1) or 42)
        self.assertEqual(value, 42)
        self.assertEqual(len(calls), 1)

    def test_failures_are_not_cached(self):
        cache = LRUCache(4)

        def boom():
            raise RuntimeError('down')

        with self.assertRaises(RuntimeError):
            cache.get_or_compute('k', boom)
        self.assertNotIn('k', cache)
        self.assertEqual(cache.get_or_compute('k', lambda: 'ok'), 'ok')

    def test_caches_falsy_values(self):
        cache = LRUCache(4)
        cache.set('empty', {})
        self.assertIn('empty', cache)
        self.assertEqual(cache.get_or_compute('empty', lambda: 'recomputed'), {})

    def test_rejects_zero_size(self):
        with self.assertRaises(ValueError):
            LRUCache(0)


class TTLCacheTests(SimpleTestCase):
    def test_entries_expire(self):
        clock = FakeClock()
        cache = TTLCache(maxsize=4, ttl=300, clock=clock)
        cache.set('tiles', [1, 2])
        clock.now = 299
        self.assertEqual(cache.get('tiles'), [1, 2])
        clock.now = 300
        self.assertIsNone(cache.get('tiles'))
        self.assertEqual(len(cache), 0)

    def test_recompute_after_expiry(self):
        clock, calls = FakeClock(), []
        cache = TTLCache(maxsize=4, ttl=10, clock=clock)
        cache.get_or_compute('k', lambda: calls.append(1))
        clock.now = 5
        cache.get_or_compute('k', lambda: calls.append(1))
        clock.now = 11
        cache.get_or_compute('k', lambda: calls.append(1))
        self.assertEqual(len(calls), 2)
