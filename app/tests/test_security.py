from django.contrib.auth.models import User
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings

from app.security import RateLimiter, RatePolicy, TokenBucket, client_ip


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TokenBucketTests(SimpleTestCase):
    def test_allows_a_burst_up_to_capacity(self):
        bucket = TokenBucket(capacity=3, rate=1, clock=FakeClock())
        self.assertEqual([bucket.consume() for _ in range(4)], [True, True, True, False])

    def test_refills_at_the_rate(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=2, rate=0.5, clock=clock)  # one token every 2 seconds
        bucket.consume(), bucket.consume()
        self.assertFalse(bucket.consume())
        self.assertAlmostEqual(bucket.retry_after(), 2.0)
        clock.now += 1
        self.assertFalse(bucket.consume())
        clock.now += 1
        self.assertTrue(bucket.consume())

    def test_never_exceeds_capacity(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=2, rate=1, clock=clock)
        clock.now += 3600
        self.assertEqual(bucket.tokens, 2)

    def test_long_run_rate_is_bounded(self):
        clock = FakeClock()
        bucket = TokenBucket(capacity=5, rate=1, clock=clock)
        allowed = 0
        for _ in range(600):  # 10 requests a second for a minute
            allowed += bucket.consume()
            clock.now += 0.1
        self.assertLessEqual(allowed, 5 + 60)  # the burst plus one a second

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            TokenBucket(capacity=0, rate=1)
        with self.assertRaises(ValueError):
            TokenBucket(capacity=1, rate=0)


class RateLimiterTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.policies = [RatePolicy('auth', r'^/accounts/login/$', 2, 60, methods=('POST',)),
                         RatePolicy('pages', r'^/', 100, 60)]

    def test_first_matching_policy_wins(self):
        limiter = RateLimiter(self.policies)
        self.assertEqual(limiter.policy_for(self.factory.post('/accounts/login/')).name, 'auth')
        self.assertEqual(limiter.policy_for(self.factory.get('/accounts/login/')).name, 'pages')

    def test_clients_have_separate_buckets(self):
        limiter, auth = RateLimiter(self.policies, clock=FakeClock()), self.policies[0]
        self.assertTrue(limiter.check(auth, '1.1.1.1')[0])
        self.assertTrue(limiter.check(auth, '1.1.1.1')[0])
        allowed, wait = limiter.check(auth, '1.1.1.1')
        self.assertFalse(allowed)
        self.assertAlmostEqual(wait, 30.0)
        self.assertTrue(limiter.check(auth, '2.2.2.2')[0])

    def test_bucket_memory_is_bounded(self):
        limiter = RateLimiter(self.policies, max_clients=10)
        for i in range(1000):
            limiter.check(self.policies[1], '10.0.{}.{}'.format(i // 256, i % 256))
        self.assertEqual(len(limiter._buckets), 10)


class ClientIpTests(SimpleTestCase):
    def setUp(self):
        self.request = RequestFactory().get('/', REMOTE_ADDR='10.0.0.1', HTTP_X_FORWARDED_FOR='6.6.6.6, 203.0.113.9')

    def test_ignores_forwarded_header_without_trusted_proxies(self):
        self.assertEqual(client_ip(self.request), '10.0.0.1')

    def test_takes_the_address_our_proxy_appended(self):
        # 6.6.6.6 was written by the client itself; only the proxy's entry (rightmost) is trusted
        self.assertEqual(client_ip(self.request, trusted_proxies=1), '203.0.113.9')


class RateLimitMiddlewareTests(TestCase):
    def test_login_brute_force_is_throttled(self):
        User.objects.create_user('ravi', password='Tr1cky-passw0rd')
        codes = [self.client.post('/accounts/login/', {'username': 'ravi', 'password': 'guess{}'.format(i)}).status_code
                 for i in range(6)]
        self.assertEqual(codes, [200] * 5 + [429])
        response = self.client.post('/accounts/login/', {'username': 'ravi', 'password': 'Tr1cky-passw0rd'})
        self.assertEqual(response.status_code, 429)
        self.assertGreater(int(response['Retry-After']), 0)

    def test_other_clients_are_not_affected(self):
        for _ in range(6):
            self.client.post('/accounts/login/', {'username': 'x', 'password': 'y'}, REMOTE_ADDR='192.0.2.1')
        response = self.client.post('/accounts/login/', {'username': 'x', 'password': 'y'}, REMOTE_ADDR='192.0.2.2')
        self.assertEqual(response.status_code, 200)

    def test_reading_the_login_page_is_not_throttled_as_a_login(self):
        codes = {self.client.get('/accounts/login/').status_code for _ in range(10)}
        self.assertEqual(codes, {200})

    @override_settings(RATE_LIMIT_ENABLED=False)
    def test_can_be_disabled(self):
        codes = [self.client.post('/accounts/login/', {'username': 'x', 'password': 'y'}).status_code for _ in range(8)]
        self.assertNotIn(429, codes)


class HardeningTests(TestCase):
    def test_new_passwords_are_hashed_with_argon2(self):
        self.client.post('/accounts/signup/', {'username': 'asha', 'password1': 'Tr1cky-passw0rd',
                                                'password2': 'Tr1cky-passw0rd'})
        self.assertTrue(User.objects.get(username='asha').password.startswith('argon2$argon2id$'))

    def test_old_pbkdf2_hashes_still_work_and_are_upgraded(self):
        from django.contrib.auth.hashers import make_password
        User.objects.create(username='old', password=make_password('Tr1cky-passw0rd', hasher='pbkdf2_sha256'))
        self.assertTrue(self.client.login(username='old', password='Tr1cky-passw0rd'))
        self.assertTrue(User.objects.get(username='old').password.startswith('argon2$'))

    def test_short_passwords_are_rejected(self):
        response = self.client.post('/accounts/signup/', {'username': 'asha', 'password1': 'Xy7#kq2',
                                                          'password2': 'Xy7#kq2'})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='asha').exists())

    def test_security_headers(self):
        response = self.client.get('/history/')
        self.assertEqual(response['X-Frame-Options'], 'DENY')
        self.assertEqual(response['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(response['Referrer-Policy'], 'same-origin')
