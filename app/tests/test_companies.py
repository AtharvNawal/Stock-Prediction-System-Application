import json

from django.test import SimpleTestCase, TestCase

from app.companies import CompanyDirectory, Listing
from app.views import DIRECTORY


class CompanyDirectoryTests(SimpleTestCase):
    def test_resolves_what_users_type(self):
        self.assertEqual(DIRECTORY.resolve('reliance'), 'RELIANCE.NS')  # lower case, no suffix
        self.assertEqual(DIRECTORY.resolve(' TCS '), 'TCS.NS')
        self.assertEqual(DIRECTORY.resolve('BRK.B'), 'BRK-B')             # exchange symbol -> Yahoo symbol
        self.assertEqual(DIRECTORY.resolve('AAPL'), 'AAPL')

    def test_follows_symbol_renames(self):
        self.assertEqual(DIRECTORY.resolve('ZOMATO'), 'ETERNAL.NS')
        self.assertEqual(DIRECTORY.resolve('TATAMOTORS.NS'), 'TMPV.NS')
        self.assertIn('TATAMOTORS.NS', DIRECTORY.old_symbols('TMPV.NS'))

    def test_unknown_symbols_resolve_to_none(self):
        self.assertIsNone(DIRECTORY.resolve('NOT-A-REAL-CO'))
        self.assertIsNone(DIRECTORY.resolve('../../etc/passwd'))

    def test_listing_and_currency(self):
        self.assertEqual(DIRECTORY.listing('M&M.NS'), Listing('M&M', 'NSE & BSE'))
        self.assertEqual(DIRECTORY.listing('AAPL').exchange, 'US')
        self.assertEqual(CompanyDirectory.currency('TCS.NS'), '₹')
        self.assertEqual(CompanyDirectory.currency('AAPL'), '$')

    def test_peers_share_industry_and_country(self):
        peers = DIRECTORY.same_industry('TCS.NS')
        self.assertIn('TCS.NS', peers)
        self.assertTrue(all(DIRECTORY.row(s)['Country'] == 'India' for s in peers))
        self.assertEqual({DIRECTORY.industry(s) for s in peers}, {DIRECTORY.industry('TCS.NS')})

    def test_nifty50_has_fifty_members(self):
        self.assertEqual(len(DIRECTORY.nifty50()), 50)

    def test_suggestions_rank_exact_symbol_first(self):
        results = DIRECTORY.suggest('tcs')
        self.assertEqual(results[0]['symbol'], 'TCS.NS')
        self.assertLessEqual(len(results), 8)
        self.assertEqual(DIRECTORY.suggest('   '), [])


class CompanyEndpointTests(TestCase):
    def test_suggestion_api(self):
        data = json.loads(self.client.get('/api/companies/', {'q': 'reliance'}).content)
        self.assertEqual(data['results'][0]['symbol'], 'RELIANCE.NS')

    def test_ticker_page_lists_every_company(self):
        response = self.client.get('/ticker/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['ticker_list']), len(DIRECTORY))

    def test_invalid_ticker_shows_an_error_without_calling_yahoo(self):
        response = self.client.get('/company/%3Cscript%3E/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('not a valid stock symbol', response.content.decode())
