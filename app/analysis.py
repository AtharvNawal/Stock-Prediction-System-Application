"""Company fundamentals for the Analysis page, derived from Yahoo Finance statements.

Line items are mapped to the layout Indian investors know from annual reports:
  Operating Profit = Operating Income + Depreciation (i.e. EBITDA before other income)
  Other Income     = Profit before tax - Operating Profit + Interest + Depreciation
  Net Profit       = profit including minority interest (consolidated)
Yahoo keeps ~5 years of annual and ~5 quarters of quarterly data, so tables are shorter
than in a full-history database, and figures can differ slightly from company filings.
"""
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

warnings.simplefilter('ignore', DeprecationWarning)


# ============================================== Helpers ==============================================

def unit_for(symbol):
    """(divisor, label) for money figures: ₹ crore for Indian companies, $ million for US ones."""
    return (1e7, '₹ Cr') if symbol.endswith(('.NS', '.BO')) else (1e6, '$ M')


def row(df, *names):
    """First available row among `names`, as a Series indexed by period (oldest first)."""
    for name in names:
        if name in df.index:
            return df.loc[name].astype(float)
    return pd.Series(np.nan, index=df.columns, dtype=float)


def sort_periods(df):
    return df.reindex(sorted(df.columns), axis=1) if not df.empty else df


def drop_empty(table, key):
    """Drop periods where the table's key row is missing (Yahoo often returns an empty oldest period)."""
    return table.loc[:, table.loc[key].notna()] if key in table.index else table


def ratio(a, b):
    return a / b.where(b != 0)


def cagr(series, years):
    """Compound annual growth from `years` periods back to the latest one (positive values only)."""
    s = series.dropna()
    if len(s) <= years or s.iloc[-1] <= 0 or s.iloc[-1 - years] <= 0:
        return None
    return (s.iloc[-1] / s.iloc[-1 - years]) ** (1 / years) - 1


def price_cagr(close, years):
    start = close.index[-1] - pd.DateOffset(years=years)
    if close.index[0] > start + pd.Timedelta(days=10):
        return None
    first = close[close.index >= start].iloc[0]
    return (close.iloc[-1] / first) ** (1 / years) - 1


# ============================================== Statements ==============================================

def profit_and_loss(inc, cf=None):
    """Screener-style P&L rows (raw currency units) from a Yahoo income statement."""
    inc = sort_periods(inc)
    sales = row(inc, 'Total Revenue', 'Operating Revenue')
    dep = row(inc, 'Reconciled Depreciation', 'Depreciation And Amortization In Income Statement')
    interest = row(inc, 'Interest Expense', 'Interest Expense Non Operating')
    pbt = row(inc, 'Pretax Income')
    tax = row(inc, 'Tax Provision')
    net = row(inc, 'Net Income Including Noncontrolling Interests', 'Net Income')
    operating_income = row(inc, 'Operating Income')
    if dep.notna().any():
        op = operating_income + dep.fillna(0)
        rows = OrderedDict([
            ('Sales', sales),
            ('Expenses', sales - op),
            ('Operating Profit', op),
            ('OPM %', ratio(op, sales) * 100),
            ('Other Income', pbt - op + interest.fillna(0) + dep.fillna(0)),
            ('Interest', interest),
            ('Depreciation', dep),
        ])
    else:
        # Yahoo's quarterly statements have no depreciation, so operating profit is shown after it
        rows = OrderedDict([
            ('Sales', sales),
            ('Expenses incl. Depreciation', sales - operating_income),
            ('Operating Profit after Depreciation', operating_income),
            ('Operating Margin %', ratio(operating_income, sales) * 100),
            ('Other Income', pbt - operating_income + interest.fillna(0)),
            ('Interest', interest),
        ])
    rows.update([
        ('Profit before tax', pbt),
        ('Tax %', ratio(tax, pbt) * 100),
        ('Net Profit', net),
        ('EPS', row(inc, 'Diluted EPS', 'Basic EPS')),
    ])
    if cf is not None:
        dividends = -row(sort_periods(cf), 'Cash Dividends Paid', 'Common Stock Dividend Paid')
        rows['Dividend Payout %'] = (ratio(dividends, net) * 100).reindex(sales.index)
    return drop_empty(pd.DataFrame(rows).T, 'Sales')


def balance_sheet(bs):
    bs = sort_periods(bs)
    capital = row(bs, 'Capital Stock', 'Common Stock')
    equity = row(bs, 'Stockholders Equity')
    debt = row(bs, 'Total Debt').fillna(0)
    total = row(bs, 'Total Assets')
    cwip = row(bs, 'Construction In Progress').fillna(0)
    fixed = row(bs, 'Net PPE').fillna(0) - cwip + row(bs, 'Goodwill And Other Intangible Assets').fillna(0)
    investments = row(bs, 'Investments And Advances', 'Long Term Equity Investment').fillna(0)
    return pd.DataFrame(OrderedDict([
        ('Equity Capital', capital),
        ('Reserves', equity - capital),
        ('Borrowings', debt),
        ('Other Liabilities', total - equity - debt),
        ('Total Liabilities', total),
        ('Fixed Assets', fixed),
        ('CWIP', cwip),
        ('Investments', investments),
        ('Other Assets', total - fixed - cwip - investments),
        ('Total Assets', total),
    ])).T.pipe(drop_empty, 'Total Assets')


def cash_flow(cf, operating_profit):
    cf = sort_periods(cf)
    cfo = row(cf, 'Operating Cash Flow')
    cfi = row(cf, 'Investing Cash Flow')
    cff = row(cf, 'Financing Cash Flow')
    return pd.DataFrame(OrderedDict([
        ('Cash from Operating Activity', cfo),
        ('Cash from Investing Activity', cfi),
        ('Cash from Financing Activity', cff),
        ('Net Cash Flow', cfo + cfi + cff),
        ('Free Cash Flow', row(cf, 'Free Cash Flow')),
        ('CFO/OP %', ratio(cfo, operating_profit.reindex(cfo.index)) * 100),
    ])).T.pipe(drop_empty, 'Cash from Operating Activity')


def ratios(inc, bs):
    inc, bs = sort_periods(inc), sort_periods(bs)
    periods = bs.columns.intersection(inc.columns)
    inc, bs = inc[periods], bs[periods]
    sales = row(inc, 'Total Revenue', 'Operating Revenue')
    cogs = row(inc, 'Cost Of Revenue', 'Reconciled Cost Of Revenue').fillna(sales)
    debtor = ratio(row(bs, 'Accounts Receivable', 'Receivables'), sales) * 365
    inventory = ratio(row(bs, 'Inventory'), cogs) * 365
    payable = ratio(row(bs, 'Accounts Payable', 'Payables'), cogs) * 365
    working_capital = row(bs, 'Current Assets') - row(bs, 'Current Liabilities')
    capital_employed = row(bs, 'Total Assets') - row(bs, 'Current Liabilities')
    return pd.DataFrame(OrderedDict([
        ('Debtor Days', debtor),
        ('Inventory Days', inventory),
        ('Days Payable', payable),
        ('Cash Conversion Cycle', debtor.fillna(0) + inventory.fillna(0) - payable.fillna(0)),
        ('Working Capital Days', ratio(working_capital, sales) * 365),
        ('ROCE %', ratio(row(inc, 'EBIT'), capital_employed) * 100),
    ])).T.pipe(lambda t: t.loc[:, t.drop('Cash Conversion Cycle').notna().any()])


# ============================================== Pros & cons ==============================================

def pros_and_cons(m):
    """Rule-based observations from the computed metrics (like Screener's machine-generated list)."""
    pros, cons = [], []
    if m['debt_to_equity'] is not None:
        if m['debt_to_equity'] < 0.1:
            pros.append('Company is almost debt free.')
        elif m['debt_to_equity'] > 1:
            cons.append('Company has high debt: debt-to-equity is {:.2f}.'.format(m['debt_to_equity']))
    if m['debt_change'] is not None and m['debt_change'] < -0.2:
        pros.append('Company has reduced debt by {:.0f}% over the last {} years.'.format(-m['debt_change'] * 100, m['years']))
    if m['roe_avg'] is not None:
        if m['roe_avg'] > 0.2:
            pros.append('Company has a good return on equity (ROE) of {:.1f}% over the last {} years.'.format(m['roe_avg'] * 100, m['years']))
        elif m['roe_avg'] < 0.1:
            cons.append('Company has a low return on equity of {:.1f}% over the last {} years.'.format(m['roe_avg'] * 100, m['years']))
    if m['profit_cagr'] is not None:
        if m['profit_cagr'] > 0.2:
            pros.append('Company has delivered good profit growth of {:.1f}% a year over the last {} years.'.format(m['profit_cagr'] * 100, m['years']))
    if m['sales_cagr'] is not None and m['sales_cagr'] < 0.1:
        cons.append('Company has delivered poor sales growth of {:.1f}% a year over the last {} years.'.format(m['sales_cagr'] * 100, m['years']))
    if m['payout_avg'] is not None:
        if m['payout_avg'] > 0.25:
            pros.append('Company has been maintaining a healthy dividend payout of {:.1f}%.'.format(m['payout_avg'] * 100))
        elif m['payout_avg'] < 0.15 and m['profit_positive']:
            cons.append('Dividend payout has been low at {:.1f}% of profits over the last {} years.'.format(m['payout_avg'] * 100, m['years']))
    if m['price_to_book'] is not None:
        if m['price_to_book'] > 5:
            cons.append('Stock is trading at {:.1f} times its book value.'.format(m['price_to_book']))
        elif 0 < m['price_to_book'] < 1:
            pros.append('Stock is trading at {:.2f} times its book value.'.format(m['price_to_book']))
    if m['interest_cover'] is not None and m['interest_cover'] < 3:
        cons.append('Interest cover is low: operating profit covers interest only {:.1f} times.'.format(m['interest_cover']))
    if m['cfo_to_op'] is not None and m['cfo_to_op'] > 1:
        pros.append('Profits convert well into cash: operating cash flow was {:.0f}% of operating profit over the last {} years.'.format(m['cfo_to_op'] * 100, m['years']))
    if m['negative_fcf_years'] and m['negative_fcf_years'] >= max(2, m['fcf_years'] - 1):
        cons.append('Free cash flow was negative in {} of the last {} years.'.format(m['negative_fcf_years'], m['fcf_years']))
    return pros, cons


# ============================================== Main entry ==============================================

def _fetch(symbol):
    t = yf.Ticker(symbol)
    return {
        'info': t.info or {},
        'income': t.income_stmt, 'quarterly': t.quarterly_income_stmt,
        'balance': t.balance_sheet, 'cashflow': t.cashflow,
        'prices': t.history(period='max', interval='1d', auto_adjust=False, raise_errors=True),
    }


def build(symbol):
    raw = _fetch(symbol)
    info = raw['info']
    divisor, unit = unit_for(symbol)

    pnl = profit_and_loss(raw['income'], raw['cashflow'])
    quarters = profit_and_loss(raw['quarterly'])
    bal = balance_sheet(raw['balance'])
    cash = cash_flow(raw['cashflow'], pnl.loc['Operating Profit'] if not pnl.empty else pd.Series(dtype=float))
    rat = ratios(raw['income'], raw['balance'])

    # Trailing twelve months from the last four quarters, only if they are consecutive
    q_dates = list(quarters.columns)[-4:]
    ttm = None
    if len(q_dates) == 4 and all(70 <= (q_dates[i + 1] - q_dates[i]).days <= 110 for i in range(3)):
        ttm = quarters[q_dates].loc[['Sales', 'Net Profit']].sum(axis=1)

    prices = raw['prices'].dropna(subset=['Close'])
    close = prices['Close']

    years = max(len(pnl.columns) - 1, 0)
    net = pnl.loc['Net Profit'] if not pnl.empty else pd.Series(dtype=float)
    equity = row(sort_periods(raw['balance']), 'Stockholders Equity').dropna()
    roe = ratio(net.reindex(equity.index), equity).dropna() if not equity.empty else pd.Series(dtype=float)
    payout = pnl.loc['Dividend Payout %'].dropna() / 100 if 'Dividend Payout %' in pnl.index else pd.Series(dtype=float)
    debt = bal.loc['Borrowings'].dropna() if not bal.empty else pd.Series(dtype=float)
    fcf = cash.loc['Free Cash Flow'].dropna() if not cash.empty else pd.Series(dtype=float)
    op_total = pnl.loc['Operating Profit'].reindex(cash.columns).sum() if not cash.empty else 0
    shares = info.get('sharesOutstanding')

    metrics = {
        'years': years,
        'sales_cagr': cagr(pnl.loc['Sales'], years) if years else None,
        'profit_cagr': cagr(net, years) if years else None,
        'roe_avg': float(roe.tail(3).mean()) if len(roe) else None,
        'payout_avg': float(payout.tail(3).mean()) if len(payout) else None,
        'profit_positive': bool(len(net.dropna()) and net.dropna().tail(3).gt(0).all()),
        'debt_to_equity': float(debt.iloc[-1] / equity.dropna().iloc[-1]) if len(debt) and len(equity.dropna()) and equity.dropna().iloc[-1] > 0 else None,
        'debt_change': float(debt.iloc[-1] / debt.iloc[0] - 1) if len(debt) > 1 and debt.iloc[0] > 0 else None,
        'price_to_book': info.get('priceToBook'),
        'interest_cover': float(pnl.loc['Operating Profit'].iloc[-1] / pnl.loc['Interest'].iloc[-1])
                          if not pnl.empty and pnl.loc['Interest'].iloc[-1] > 0 else None,
        'cfo_to_op': float(cash.loc['Cash from Operating Activity'].sum() / op_total) if op_total and op_total > 0 else None,
        'negative_fcf_years': int((fcf < 0).sum()), 'fcf_years': len(fcf),
    }
    pros, cons = pros_and_cons(metrics)

    def pct(v):
        return None if v is None or pd.isna(v) else v * 100

    growth = OrderedDict([
        ('Compounded Sales Growth', [('{} Year{}'.format(n, 's' if n > 1 else ''), pct(cagr(pnl.loc['Sales'], n))) for n in (3, 1) if n <= years]
                                    + ([('TTM', pct(ttm['Sales'] / pnl.loc['Sales'].iloc[-1] - 1))] if ttm is not None else [])),
        ('Compounded Profit Growth', [('{} Year{}'.format(n, 's' if n > 1 else ''), pct(cagr(net, n))) for n in (3, 1) if n <= years]
                                     + ([('TTM', pct(ttm['Net Profit'] / net.iloc[-1] - 1))] if ttm is not None and net.iloc[-1] > 0 else [])),
        ('Stock Price CAGR', [('{} Year{}'.format(n, 's' if n > 1 else ''), pct(price_cagr(close, n))) for n in (10, 5, 3, 1)]),
        ('Return on Equity', [('{} Years'.format(min(3, len(roe))), pct(roe.tail(3).mean()) if len(roe) else None),
                              ('Last Year', pct(roe.iloc[-1]) if len(roe) else None)]),
    ])

    price = info.get('currentPrice') or info.get('regularMarketPrice') or (float(close.iloc[-1]) if len(close) else None)
    dividend_yield = info.get('dividendYield')
    key_ratios = OrderedDict([
        ('Market Cap', ('money', info.get('marketCap'))),
        ('Current Price', ('price', price)),
        ('High / Low', ('range', (info.get('fiftyTwoWeekHigh'), info.get('fiftyTwoWeekLow')))),
        ('Stock P/E', ('number', info.get('trailingPE'))),
        ('Book Value', ('price', info.get('bookValue'))),
        ('Dividend Yield', ('percent', dividend_yield)),  # Yahoo reports this already in %
        ('ROCE', ('percent', rat.loc['ROCE %'].dropna().iloc[-1] if not rat.empty and rat.loc['ROCE %'].notna().any() else None)),
        ('ROE', ('percent', pct(roe.iloc[-1]) if len(roe) else None)),
        ('Price / Book', ('number', info.get('priceToBook'))),
        ('EPS (TTM)', ('price', info.get('trailingEps'))),
        ('Debt / Equity', ('number', metrics['debt_to_equity'])),
        ('Shares Outstanding', ('count', shares)),
    ])

    insiders, institutions = info.get('heldPercentInsiders'), info.get('heldPercentInstitutions')
    holding = None
    if insiders is not None or institutions is not None:
        insiders, institutions = insiders or 0.0, institutions or 0.0
        holding = OrderedDict([('Promoters / Insiders', insiders * 100), ('Institutions', institutions * 100),
                               ('Public & Others', max(0.0, 1 - insiders - institutions) * 100)])

    return {
        'info': info, 'unit': unit, 'divisor': divisor,
        'key_ratios': key_ratios, 'pros': pros, 'cons': cons, 'growth': growth,
        'quarters': quarters, 'pnl': pnl, 'balance': bal, 'cashflow': cash, 'ratios': rat,
        'prices': prices, 'holding': holding,
        'previous_close': info.get('previousClose') or info.get('regularMarketPreviousClose'),
        'price': price,
    }


_cache = OrderedDict()


def company(symbol):
    """Analysis for `symbol`, cached for the day. Raises YFRateLimitError if Yahoo is throttling."""
    key = (symbol, pd.Timestamp.today().date())
    if key not in _cache:
        _cache[key] = build(symbol)
        while len(_cache) > 64:
            _cache.popitem(last=False)
    return _cache[key]


# ============================================== Peers ==============================================

_peer_cache = OrderedDict()


def peer_metrics(symbols):
    """Valuation snapshot for each peer from Yahoo (fetched in parallel, cached for the day)."""
    today = pd.Timestamp.today().date()

    def one(symbol):
        key = (symbol, today)
        if key not in _peer_cache:
            try:
                info = yf.Ticker(symbol).info or {}
            except YFRateLimitError:
                return symbol, None
            _peer_cache[key] = {
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'pe': info.get('trailingPE'),
                'market_cap': info.get('marketCap'),
                'dividend_yield': info.get('dividendYield'),
                'price_to_book': info.get('priceToBook'),
                'year_change': info.get('52WeekChange'),
                'industry': info.get('industry'),
            }
            while len(_peer_cache) > 512:
                _peer_cache.popitem(last=False)
        return symbol, _peer_cache[key]

    with ThreadPoolExecutor(4) as pool:
        return dict(pool.map(one, symbols))
