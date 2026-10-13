"""The companies the app supports, and every lookup on them.

CompanyDirectory wraps app/Data/companies.csv (built by scripts/build_companies.py) and
symbol_changes.csv, so views never touch the underlying DataFrames. It resolves what a
user typed to a Yahoo symbol, describes listings, finds peers and ranks search suggestions.
"""
from typing import NamedTuple

import pandas as pd

SUPPORTED = "companies listed on NSE or BSE, and well-known US companies (S&P 500 and NASDAQ-100)"


class Listing(NamedTuple):
    """Exact symbol as listed on the exchange, and the exchange label, e.g. ('M&M', 'NSE & BSE')."""
    listed: str
    exchange: str


class CompanyDirectory:
    INDIAN_SUFFIXES = ('.NS', '.BO')
    EXCHANGE_RANK = {'NSE': 0, 'US': 1, 'BSE': 2}  # search suggestions: NSE listings first

    def __init__(self, companies, symbol_changes):
        """`companies` is indexed by Yahoo symbol; `symbol_changes` maps old symbol -> current one."""
        self._companies = companies
        self._changes = symbol_changes
        # Exact symbol as listed on the exchange (RELIANCE, M&M, BRK.B) -> Yahoo symbol (RELIANCE.NS, M&M.NS, BRK-B)
        self._listed_to_yahoo = {listed: sym for sym, listed in companies['Listed'].items()
                                 if listed not in companies.index}
        # Inverted index: current symbol -> its old symbols, e.g. TMPV.NS -> [TATAMOTORS.NS, TELCO.NS]
        self._old_symbols = symbol_changes.groupby(symbol_changes).groups
        self._search_index = pd.DataFrame({
            'symbol': companies.index,
            'listed': companies['Listed'].str.upper().values,
            'name': companies['Name'].str.lower().values,
            'exchange_rank': companies['Exchange'].map(self.EXCHANGE_RANK).values,
            # NIFTY 500 / S&P 500 members have an industry: list these better-known companies first
            'minor': (companies['Industry'] == '').values,
        })

    @classmethod
    def from_csv(cls, companies_path='app/Data/companies.csv', changes_path='app/Data/symbol_changes.csv'):
        companies = pd.read_csv(companies_path, index_col='Symbol', keep_default_na=False, dtype=str)
        changes = pd.read_csv(changes_path, index_col='Old')['New']
        return cls(companies, changes)

    # ---------------------------------------------- basic lookups ----------------------------------------------

    def __contains__(self, symbol):
        return symbol in self._companies.index

    def __len__(self):
        return len(self._companies)

    def __iter__(self):
        return iter(self._companies.index)

    def row(self, symbol):
        """All stored fields for a supported symbol (Listed, BSE, BSECode, Name, Exchange, Country, ...)."""
        return self._companies.loc[symbol]

    def name(self, symbol):
        return self._companies.loc[symbol, 'Name']

    def industry(self, symbol):
        return self._companies.loc[symbol, 'Industry']

    def listing(self, symbol):
        row = self._companies.loc[symbol]
        exchange = 'NSE & BSE' if row['Exchange'] == 'NSE' and row['BSE'] else row['Exchange']
        return Listing(row['Listed'], exchange)

    @classmethod
    def is_indian(cls, symbol):
        return symbol.endswith(cls.INDIAN_SUFFIXES)

    @classmethod
    def currency(cls, symbol):
        return '₹' if cls.is_indian(symbol) else '$'

    def old_symbols(self, symbol):
        return tuple(self._old_symbols.get(symbol, ()))

    # ---------------------------------------------- resolution ----------------------------------------------

    def resolve(self, text):
        """Supported Yahoo symbol for what the user typed, e.g. 'reliance' -> 'RELIANCE.NS', else None.
        Tries the text as a Yahoo symbol, as an exchange symbol, then with NSE and BSE suffixes,
        following symbol renames (ZOMATO.NS -> ETERNAL.NS) at each step."""
        text = text.strip().upper()
        for candidate in (text, self._listed_to_yahoo.get(text), text + '.NS', text + '.BO'):
            if candidate is None:
                continue
            candidate = self._changes.get(candidate, candidate)
            if candidate in self:
                return candidate
        return None

    # ---------------------------------------------- groups ----------------------------------------------

    def nifty50(self):
        return self._companies.index[self._companies['Nifty50'] == 'Y'].tolist()

    def same_industry(self, symbol):
        """Symbols in the same industry and country as `symbol` (including it), or [] if it has no industry."""
        industry = self.industry(symbol)
        if not industry:
            return []
        c = self._companies
        return c.index[(c['Industry'] == industry) & (c['Country'] == c.loc[symbol, 'Country'])].tolist()

    def records(self):
        """Symbol, Listed, Name and Exchange of every company, for the ticker table."""
        return [{'Symbol': s, 'Listed': listed, 'Name': self.name(s), 'Exchange': self.listing(s).exchange}
                for s, listed in self._companies['Listed'].items()]

    def describe(self, symbol):
        listed, exchange = self.listing(symbol)
        return {'symbol': symbol, 'listed': listed, 'name': self.name(symbol), 'exchange': exchange}

    # ---------------------------------------------- search ----------------------------------------------

    def suggest(self, query, limit=8):
        """Companies matching `query`, best first. Ranking tiers: exact symbol, symbol prefix,
        name prefix, symbol substring, name substring; ties go to better-known companies,
        then NSE before US before BSE, then alphabetical."""
        query = query.strip()
        if not query:
            return []
        upper, lower = query.upper(), query.lower()
        idx = self._search_index
        rank = pd.Series(99, index=idx.index)
        # assigned weakest first, so a stronger match overwrites a weaker one
        rank[idx['name'].str.contains(lower, regex=False)] = 4
        rank[idx['listed'].str.contains(upper, regex=False)] = 3
        rank[idx['name'].str.startswith(lower)] = 2
        rank[idx['listed'].str.startswith(upper)] = 1
        rank[idx['listed'] == upper] = 0
        hits = idx.assign(rank=rank)[rank < 99].sort_values(['rank', 'minor', 'exchange_rank', 'listed']).head(limit)
        return [self.describe(symbol) for symbol in hits['symbol']]
