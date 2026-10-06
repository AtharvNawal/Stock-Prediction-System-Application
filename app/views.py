from django.http import HttpResponse
from django.shortcuts import redirect, render

from plotly.offline import plot
import plotly.graph_objects as go

import pandas as pd
import json
import re
from urllib.parse import quote
import warnings

import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from . import dataset, forecasting


CHART_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS']
# Home page index tiles: (Yahoo symbol, name, value prefix)
INDICES = [('^NSEI', 'NIFTY 50', ''), ('^BSESN', 'SENSEX', ''), ('^NSEBANK', 'NIFTY Bank', ''),
           ('^GSPC', 'S&P 500', ''), ('^IXIC', 'NASDAQ', ''), ('INR=X', 'USD / INR', '₹')]

# The companies the app supports: everything listed on NSE/BSE plus well-known US companies
# (S&P 500 and NASDAQ-100). Rebuild with scripts/build_companies.py.
COMPANIES = pd.read_csv('app/Data/companies.csv', index_col='Symbol', keep_default_na=False, dtype=str)
# Exact symbol as listed on the exchange (RELIANCE, M&M, BRK.B) -> Yahoo symbol (RELIANCE.NS, M&M.NS, BRK-B)
LISTED_TO_YAHOO = {listed: sym for sym, listed in COMPANIES['Listed'].items() if listed not in COMPANIES.index}
# Old NSE symbol -> current one, e.g. TATAMOTORS.NS -> TMPV.NS, ZOMATO.NS -> ETERNAL.NS
SYMBOL_CHANGES = pd.read_csv('app/Data/symbol_changes.csv', index_col='Old')['New']
OLD_SYMBOLS = SYMBOL_CHANGES.groupby(SYMBOL_CHANGES).groups  # current symbol -> its old symbols
SUPPORTED = "companies listed on NSE or BSE, and well-known US companies (S&P 500 and NASDAQ-100)"

# Exchange/Yahoo symbols: letters, digits and . - & ^ = (e.g. AAPL, TMPV.NS, M&M.NS, BRK-B, ^NSEI)
TICKER_PATTERN = re.compile(r'^[A-Z0-9.\-&^=]{1,20}$')

# Chart theme (see static/css/theme.css): white cards, light gridlines and
# low-saturation series colours, so green/red price movement and the purple
# ensemble forecast stand out.
UP, DOWN, HIGHLIGHT, GRID = '#059669', '#dc2626', '#9b30ff', '#e5e7eb'
MUTED_SERIES = ['#6b7280', '#5b9b84', '#c29a4a', '#b9799f', '#7c83c4', '#a8a29e']
PLOT_LAYOUT = dict(
    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', font=dict(color='#4b5563'),
    colorway=MUTED_SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
)


def style_range_buttons(fig):
    fig.update_xaxes(rangeselector_bgcolor='#f3f4f6', rangeselector_activecolor='#e9d5ff',
                     rangeselector_font_color='#1f2937')


def movement(value, pattern):
    """Format a change with ▲/▼ so direction never depends on colour alone; returns (text, css class)."""
    if value is None or pd.isna(value):
        return '-', ''
    if value > 0:
        return '▲ ' + pattern.format(value), 'text-up'
    if value < 0:
        return '▼ ' + pattern.format(abs(value)), 'text-down'
    return pattern.format(value), ''


def resolve_symbol(text):
    """Supported Yahoo symbol for what the user typed, e.g. 'reliance' -> 'RELIANCE.NS', else None."""
    text = text.strip().upper()
    for candidate in (text, LISTED_TO_YAHOO.get(text), text + '.NS', text + '.BO'):
        if candidate is None:
            continue
        candidate = SYMBOL_CHANGES.get(candidate, candidate)
        if candidate in COMPANIES.index:
            return candidate
    return None


def price_history(symbol, period, interval):
    """Yahoo prices for one symbol. Empty if Yahoo has none; raises YFRateLimitError when Yahoo
    is throttling us, so that isn't mistaken for a delisted stock."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, raise_errors=True)
    except YFRateLimitError:
        raise
    except Exception:
        return pd.DataFrame()
    return df.drop(columns=['Dividends', 'Stock Splits', 'Capital Gains'], errors='ignore')


def listing(symbol):
    """(exact listed symbol, exchange label) for a supported Yahoo symbol, e.g. ('M&M', 'NSE & BSE')."""
    row = COMPANIES.loc[symbol]
    exchange = 'NSE & BSE' if row['Exchange'] == 'NSE' and row['BSE'] else row['Exchange']
    return row['Listed'], exchange


def currency_of(symbol):
    return '₹' if symbol.endswith(('.NS', '.BO')) else '$'


def error(request, message, suggestions=None, number_of_days=None):
    return render(request, 'error.html', {
        'message': message,
        'suggestions': suggestions or [],
        'number_of_days': number_of_days,
    })


NAME_NOISE = re.compile(r'\b(common stock|ordinary shares|american depositary shares?|ads|inc|corp|corporation|'
                        r'co|ltd|limited|plc|class [a-z]|holdings?|group)\b\.?', re.IGNORECASE)


def suggest_tickers(ticker):
    """Supported, currently traded tickers for the same company, e.g. TTM (delisted ADR) -> TMPV.NS / TMCV.NS."""
    if ticker in COMPANIES.index:
        name = COMPANIES.loc[ticker, 'Name']
    else:
        name = dataset.us_company_name(ticker) or re.sub(r'\.(NS|BO)$', '', ticker)
    words = NAME_NOISE.sub(' ', re.sub(r'[^\w\s&]', ' ', name)).split()

    for n in (3, 2):
        query = ' '.join(words[:n])
        if not query:
            break
        try:
            quotes = yf.Search(query, max_results=8).quotes
        except Exception:
            return []
        found = [{'symbol': q['symbol'], 'listed': listing(q['symbol'])[0],
                  'name': COMPANIES.loc[q['symbol'], 'Name'], 'exchange': listing(q['symbol'])[1]}
                 for q in quotes
                 if q.get('symbol') != ticker and q.get('symbol') in COMPANIES.index]
        found = list({f['listed'] + f['exchange']: f for f in found}.values())  # Yahoo can return duplicates
        if found:
            return found[:5]
    return []


_market_cache = {}


def cached(key, seconds, fn):
    """Market data for the home page, reused for a few minutes so reloads don't hit Yahoo every time."""
    now = pd.Timestamp.now()
    hit = _market_cache.get(key)
    if hit and (now - hit[0]).total_seconds() < seconds:
        return hit[1]
    value = fn()
    _market_cache[key] = (now, value)
    return value


def sparkline(values, width=120, height=36):
    """Inline SVG polyline points for a small trend line."""
    values = [v for v in values if pd.notna(v)]
    if len(values) < 2:
        return ''
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    step = width / (len(values) - 1)
    return ' '.join('{:.1f},{:.1f}'.format(i * step, height - 2 - (v - lo) / span * (height - 4)) for i, v in enumerate(values))


def _index_tiles():
    closes = yf.download([s for s, _, _ in INDICES], period='1mo', interval='1d', auto_adjust=False,
                         progress=False)['Close']
    tiles = []
    for symbol, name, prefix in INDICES:
        series = closes[symbol].dropna() if symbol in closes else pd.Series(dtype=float)
        if len(series) < 2:
            continue
        change = (series.iloc[-1] / series.iloc[-2] - 1) * 100
        text, cls = movement(change, '{:.2f}%')
        tiles.append({'name': name, 'value': '{}{:,.2f}'.format(prefix, series.iloc[-1]), 'change': text, 'class': cls,
                      'points': sparkline(series.tolist()), 'up': series.iloc[-1] >= series.iloc[0],
                      'month': movement((series.iloc[-1] / series.iloc[0] - 1) * 100, '{:.1f}%')})
    return tiles


def _performance_chart():
    """Top NSE stocks rebased to % return, with 1M / 3M / 6M / 1Y views (absolute prices aren't comparable)."""
    closes = yf.download(CHART_TICKERS, period='1y', interval='1d', auto_adjust=False, progress=False)['Adj Close']
    periods = [('1M', 1), ('3M', 3), ('6M', 6), ('1Y', 12)]
    fig = go.Figure()
    for p, (label, months) in enumerate(periods):
        window = closes[closes.index >= closes.index[-1] - pd.DateOffset(months=months)]
        for symbol in CHART_TICKERS:
            s = window[symbol].dropna()
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=s.index, y=((s / s.iloc[0] - 1) * 100).round(2), name=listing(symbol)[0],
                                     visible=(label == '3M'), legendgroup=symbol, showlegend=True,
                                     hovertemplate='%{y:.1f}%<extra>' + listing(symbol)[0] + '</extra>'))
    per = len(fig.data) // len(periods)
    buttons = [dict(label=label, method='update',
                    args=[{'visible': [i // per == p for i in range(len(fig.data))]}])
               for p, (label, _) in enumerate(periods)]
    fig.update_layout(**PLOT_LAYOUT, height=380, margin=dict(l=40, r=10, t=10, b=30), hovermode='x unified',
                      yaxis_ticksuffix='%', legend=dict(orientation='h', y=-0.15),
                      updatemenus=[dict(type='buttons', direction='right', active=1, x=0, xanchor='left', y=1.12,
                                        buttons=buttons, bgcolor='#f3f4f6', bordercolor='#e5e7eb',
                                        font=dict(color='#1f2937'))])
    fig.add_hline(y=0, line_width=1, line_color='#9ca3af')
    return plot(fig, auto_open=False, output_type='div', config={'displayModeBar': False})


def _nifty_movers():
    members = COMPANIES.index[COMPANIES['Nifty50'] == 'Y'].tolist()
    data = yf.download(members, period='5d', interval='1d', auto_adjust=False, progress=False)
    close, volume = data['Close'], data['Volume']
    rows = []
    for symbol in members:
        c = close[symbol].dropna()
        if len(c) < 2:
            continue
        rows.append({'symbol': symbol, 'listed': listing(symbol)[0], 'name': COMPANIES.loc[symbol, 'Name'],
                     'price': c.iloc[-1], 'change': (c.iloc[-1] / c.iloc[-2] - 1) * 100,
                     'traded': c.iloc[-1] * volume[symbol].reindex(c.index).iloc[-1]})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    def fmt(frame, value):
        return [{'symbol': r.symbol, 'listed': r.listed, 'name': r.name, 'price': '₹{:,.2f}'.format(r.price),
                 'value': value(r)} for r in frame.itertuples()]

    return {
        'gainers': fmt(df.nlargest(5, 'change'), lambda r: movement(r.change, '{:.2f}%')),
        'losers': fmt(df.nsmallest(5, 'change'), lambda r: movement(r.change, '{:.2f}%')),
        'active': fmt(df.nlargest(5, 'traded'), lambda r: ('₹{} Cr'.format(indian_digits(r.traded / 1e7)), '')),
        'advances': int((df['change'] > 0).sum()), 'declines': int((df['change'] < 0).sum()),
    }


# The Home page when Server loads up
def index(request):
    def safe(key, fn, default):
        try:
            return cached(key, 300, fn)
        except Exception:  # Yahoo unavailable or throttling: show the rest of the page
            return default

    return render(request, 'index.html', {
        'today': pd.Timestamp.now().strftime('%A, %d %B %Y'),
        'tiles': safe('tiles', _index_tiles, []),
        'performance': safe('performance', _performance_chart, None),
        'movers': safe('movers', _nifty_movers, {}),
    })


PREDICT_HORIZONS = [(7, '7 days'), (30, '30 days'), (90, '90 days'), (180, '180 days'), (365, '1 year')]
POPULAR = [('RELIANCE.NS', 'Reliance'), ('TCS.NS', 'TCS'), ('HDFCBANK.NS', 'HDFC Bank'), ('INFY.NS', 'Infosys'),
           ('ICICIBANK.NS', 'ICICI Bank'), ('TMPV.NS', 'Tata Motors'), ('ITC.NS', 'ITC'), ('SBIN.NS', 'SBI'),
           ('BHARTIARTL.NS', 'Bharti Airtel'), ('LT.NS', 'L&T'), ('AAPL', 'Apple')]


def search_redirect(request):
    """Old /search/ links: the search dashboard is now the home page."""
    query = request.META.get('QUERY_STRING')
    return redirect('/' + ('?' + query if query else ''))


def search(request):
    return render(request, 'search.html', {
        'horizons': PREDICT_HORIZONS,
        'popular': [(s, name) for s, name in POPULAR if s in COMPANIES.index],
        'company_count': '{:,}'.format(len(COMPANIES)),
    })


def ticker(request):
    # ================================================= Load Ticker Table ================================================
    companies = COMPANIES.reset_index()
    companies['Exchange'] = companies['Symbol'].map(lambda s: listing(s)[1])
    ticker_list = json.loads(companies[['Symbol', 'Listed', 'Name', 'Exchange']].to_json(orient='records'))

    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# Long-term price history from the Kaggle dataset
def history(request):
    typed = request.GET.get('ticker', '').strip().upper()
    if not typed:
        return render(request, 'history.html', {})

    ticker_value = resolve_symbol(typed)
    if ticker_value is None:
        return render(request, 'history.html', {
            'ticker_value': typed,
            'not_found': "{} is not one of the supported {}.".format(typed, SUPPORTED),
        })

    df, exchange = dataset.load_history(ticker_value, OLD_SYMBOLS.get(ticker_value, ()))
    if df is None or df.empty:
        return render(request, 'history.html', {
            'ticker_value': ticker_value,
            'not_found': "{} is not in the historical datasets.".format(ticker_value),
        })
    cur = currency_of(ticker_value)

    fig = go.Figure(go.Scatter(x=df.index, y=df['Adj Close'], name=ticker_value, line=dict(color=HIGHLIGHT)))
    fig.update_layout(title='{} daily adjusted close'.format(listing(ticker_value)[0]),
                      yaxis_title='Stock Price ({} per Share)'.format('INR' if cur == '₹' else 'USD'))
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(**PLOT_LAYOUT)
    style_range_buttons(fig)
    plot_div = plot(fig, auto_open=False, output_type='div')

    first, last = df['Adj Close'].iloc[0], df['Adj Close'].iloc[-1]
    years = (df.index[-1] - df.index[0]).days / 365.25
    total_return = movement((last / first - 1) * 100, '{:,.1f}%')
    cagr = movement(((last / first) ** (1 / years) - 1) * 100, '{:.1f}%') if years >= 1 else ('-', '')
    stats = [  # (label, value, css class)
        ('From', df.index[0].date(), ''),
        ('To', df.index[-1].date(), ''),
        ('Trading Days', '{:,}'.format(len(df)), ''),
        ('All-Time High', '{}{:,.2f} ({})'.format(cur, df['High'].max(), df['High'].idxmax().date()), ''),
        ('All-Time Low', '{}{:,.2f} ({})'.format(cur, df['Low'].min(), df['Low'].idxmin().date()), ''),
        ('Total Return', *total_return),
        ('Annual Growth (CAGR)', *cagr),
        ('Average Daily Volume', '{:,.0f}'.format(df['Volume'].mean()), ''),
    ]

    return render(request, 'history.html', {
        'ticker_value': ticker_value,
        'plot_div': plot_div,
        'listed': listing(ticker_value)[0],
        'info': {'Name': COMPANIES.loc[ticker_value, 'Name'], 'Exchange': listing(ticker_value)[1],
                 'Source': dataset.SOURCES[exchange]},
        'stats': stats,
    })


def predict(request, ticker_value, number_of_days):
    """Old /predict/SYMBOL/DAYS/ links: the forecast is now a section of the company page."""
    days = number_of_days if number_of_days.isdigit() else '30'
    return redirect('/company/{}/?days={}#forecast'.format(quote(ticker_value.upper()), days))


def company_forecast(request, ticker_value):
    """Forecast section of the company page (loaded separately, since training the models takes a while)."""
    symbol = resolve_symbol(ticker_value)
    if symbol is None:
        return HttpResponse('')
    try:
        days = min(max(int(request.GET.get('days', 30)), 1), 365)
    except ValueError:
        days = 30
    try:
        return _forecast(request, symbol, days)
    except YFRateLimitError:
        return render(request, 'company_forecast.html', {
            'message': 'Yahoo Finance is limiting how often we can fetch prices right now. Please try again in a minute.'})


def _forecast(request, symbol, number_of_days):
    listed = listing(symbol)[0]
    cur = currency_of(symbol)
    price_label = 'Stock Price ({} per Share)'.format('INR' if cur == '₹' else 'USD')

    # Most recent trading session at 1-minute resolution. Thinly traded tickers can have no
    # trades yet today, so fetch a few days and keep the latest session that has data.
    df = price_history(symbol, period='5d', interval='1m')
    if df.empty:
        return render(request, 'company_forecast.html', {
            'message': '{} has no recent price data on Yahoo Finance, so it may have been delisted or renamed.'.format(listed)})
    df = df[df.index.date == df.index[-1].date()]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name='market data',
                increasing=dict(line=dict(color=UP), fillcolor='rgba(0,0,0,0)'),
                decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN)))
    # explicit heights: these charts are inserted into the page after it loads, so they can't size from it
    fig.update_layout(title='{} share price on {}'.format(listed, df.index[-1].strftime('%b %d, %Y')),
                      yaxis_title=price_label, height=460)
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(**PLOT_LAYOUT)
    style_range_buttons(fig)
    plot_div = plot(fig, auto_open=False, output_type='div', include_plotlyjs=False)

    # ========================================== Machine Learning ==========================================

    def download_daily(s):
        return price_history(s, period=forecasting.HISTORY_PERIOD, interval='1d')

    try:
        result = forecasting.forecast(symbol, number_of_days, download_daily)
    except forecasting.NotEnoughHistory as e:
        message = '{} only has {} trading day{} of price history on Yahoo Finance, '.format(
            listed, e.days, '' if e.days == 1 else 's')
        if e.max_horizon:
            message += 'which is enough to forecast at most {} days ahead.'.format(e.max_horizon)
        else:
            message += 'which is too short to train the models on. It is probably newly listed, so try again once it has traded for about 18 months.'
        return render(request, 'company_forecast.html', {'message': message, 'max_horizon': e.max_horizon,
                                                         'intraday': plot_div})

    models = result['results']
    for m in models:
        m['change'], m['change_class'] = movement((m['forecast'][-1] / result['last_close'] - 1) * 100, '{:.2f}%')
    ensemble = next(m for m in models if m['name'] == 'Weighted Ensemble')

    # ========================================== Plotting predicted data ======================================

    history = download_daily(symbol)['Adj Close'].iloc[-120:]
    future = pd.bdate_range(history.index[-1] + pd.offsets.BDay(1), periods=number_of_days)

    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=history.index, y=history, name='Actual', line=dict(color='#374151')))
    for m in models:
        if m['weight'] is None:
            continue
        is_ensemble = m['name'] == 'Weighted Ensemble'
        pred_fig.add_trace(go.Scatter(
            x=[history.index[-1], *future], y=[history.iloc[-1], *m['forecast']], name=m['name'],
            line=dict(width=3, color=HIGHLIGHT) if is_ensemble else dict(width=1.5, dash='dot'),
        ))
    pred_fig.update_layout(yaxis_title=price_label, height=460, margin=dict(t=30),
                           legend=dict(orientation='h', y=-0.2), **PLOT_LAYOUT)
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div', include_plotlyjs=False)

    return render(request, 'company_forecast.html', {
        'intraday': plot_div,
        'forecast_chart': plot_div_pred,
        'models': models,
        'ensemble_price': ensemble['forecast'][-1],
        'ensemble_change': (ensemble['change'], ensemble['change_class']),
        'up_rate': result['up_rate'],
        'down_rate': 100 - result['up_rate'],
        'n_test': result['n_test'],
        'number_of_days': number_of_days,
        'listed': listed,
        'currency_symbol': cur,
    })


# ========================================== Company analysis ==========================================

def indian_digits(number):
    """Indian digit grouping: 1648255 -> '16,48,255'."""
    sign, digits = ('-' if number < 0 else ''), str(int(round(abs(number))))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return sign + ','.join([head] + groups + [tail]) if head else sign + ','.join(groups + [tail])


def _amount(value, symbol):
    """Market-cap style amount: ₹ crore (Indian grouping) for Indian companies, $ billions/millions for US ones."""
    if value is None or pd.isna(value):
        return '-'
    if symbol.endswith(('.NS', '.BO')):
        return '₹ {} Cr.'.format(indian_digits(value / 1e7))
    return '$ {:,.2f} B'.format(value / 1e9) if abs(value) >= 1e9 else '$ {:,.0f} M'.format(value / 1e6)


def _key_ratio(kind, value, symbol):
    cur = currency_of(symbol)
    if value is None or (not isinstance(value, tuple) and pd.isna(value)):
        return '-'
    if kind == 'money':
        return _amount(value, symbol)
    if kind == 'price':
        return '{} {:,.2f}'.format(cur, value)
    if kind == 'range':
        high, low = value
        return '{} {:,.0f} / {:,.0f}'.format(cur, high, low) if high and low else '-'
    if kind == 'percent':
        return '{:.2f} %'.format(value)
    if kind == 'count':
        return '{:,.0f}'.format(value)
    return '{:.2f}'.format(value)


def _table(df, divisor):
    """Statement table for the template: headers plus rows of formatted cells."""
    if df is None or df.empty:
        return None
    headers = [c.strftime('%b %Y') for c in df.columns]
    rows = []
    for label, values in df.iterrows():
        cells = []
        for v in values:
            if pd.isna(v):
                cells.append('')
            elif label.endswith('%'):
                cells.append('{:.0f}%'.format(v))
            elif label == 'EPS':
                cells.append('{:,.2f}'.format(v))
            elif label in ('Debtor Days', 'Inventory Days', 'Days Payable', 'Cash Conversion Cycle', 'Working Capital Days'):
                cells.append('{:,.0f}'.format(v))
            else:
                cells.append('{:,.0f}'.format(v / divisor))
        rows.append({'label': label, 'cells': cells,
                     'strong': label in ('Sales', 'Operating Profit', 'Operating Profit after Depreciation', 'Net Profit',
                                         'Total Liabilities', 'Total Assets', 'Net Cash Flow')})
    return {'headers': headers, 'rows': rows}


def _price_chart(prices, listed):
    from plotly.subplots import make_subplots
    close = prices['Close']
    # plain lists (not numpy arrays) so the page script can read values to rescale the axes
    dates = prices.index.tolist()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=dates, y=close.round(2).tolist(), name='Price', line=dict(color=HIGHLIGHT, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close.rolling(50).mean().round(2).tolist(), name='50 DMA',
                             line=dict(color=MUTED_SERIES[2], width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close.rolling(200).mean().round(2).tolist(), name='200 DMA',
                             line=dict(color=MUTED_SERIES[1], width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=prices['Volume'].tolist(), name='Volume', marker_color='#94a3b8',
                         marker_line_width=0), row=2, col=1)
    start = prices.index[-1] - pd.DateOffset(years=1)
    fig.update_layout(**PLOT_LAYOUT, height=520, legend=dict(orientation='h', x=1, xanchor='right', y=1.12), margin=dict(l=50, r=20, t=50, b=30),
                      bargap=0,
                      hovermode='x unified')
    fig.update_xaxes(range=[start, prices.index[-1]], row=2, col=1)
    fig.update_xaxes(row=1, col=1, rangeselector=dict(buttons=[
        dict(count=1, label='1M', step='month', stepmode='backward'),
        dict(count=6, label='6M', step='month', stepmode='backward'),
        dict(count=1, label='1Yr', step='year', stepmode='backward'),
        dict(count=3, label='3Yr', step='year', stepmode='backward'),
        dict(count=5, label='5Yr', step='year', stepmode='backward'),
        dict(count=10, label='10Yr', step='year', stepmode='backward'),
        dict(step='all', label='Max'),
    ]))
    style_range_buttons(fig)
    return plot(fig, auto_open=False, output_type='div')


def company_search(request):
    """/company/?q=... opens that company; plain /company/ goes to the search dashboard."""
    typed = request.GET.get('q', '').strip()
    symbol = resolve_symbol(typed) if typed else None
    if symbol is None:
        return redirect('/' + ('?q=' + quote(typed) if typed else ''))
    return redirect('/company/{}/'.format(quote(symbol)))


def company(request, ticker_value):
    from . import analysis
    days = request.GET.get('days', '30')
    days = int(days) if days.isdigit() and 1 <= int(days) <= 365 else 30
    symbol = resolve_symbol(ticker_value)
    if symbol is None:
        typed = ticker_value.upper()
        if not TICKER_PATTERN.match(typed):
            return error(request, "the ticker you have looked for is not a valid stock symbol.")
        suggestions = suggest_tickers(typed)
        message = "{} is not supported. This app covers {}.".format(typed, SUPPORTED)
        if suggestions:
            message += " Did you mean one of these?"
        return error(request, message, suggestions, days)
    if symbol != ticker_value:
        return redirect('/company/{}/?days={}'.format(quote(symbol), days))

    try:
        data = analysis.company(symbol)
    except YFRateLimitError:
        return error(request, "Yahoo Finance is limiting how often we can fetch data right now. Please try again in a minute.")
    except Exception:
        data = None
    if not data or (data['pnl'].empty and data['prices'].empty):
        return error(request, "we could not find financial data for {} on Yahoo Finance.".format(listing(symbol)[0]))

    listed, exchange = listing(symbol)
    row = COMPANIES.loc[symbol]
    info = data['info']
    change = None
    if data['price'] and data['previous_close']:
        change = movement((data['price'] / data['previous_close'] - 1) * 100, '{:.2f}%')
    divisor = data['divisor']
    pnl = data['pnl']
    links = []
    if info.get('website'):
        links.append(('Website', info['website']))
    if symbol.endswith('.NS'):
        links.append(('NSE: ' + listed, 'https://www.nseindia.com/get-quotes/equity?symbol=' + quote(listed)))
    bse_code = row['BSECode']
    if bse_code:
        links.append(('BSE: ' + bse_code, 'https://www.bseindia.com/stock-share-price/x/{}/{}/'.format(
            quote(row['BSE'] or listed), bse_code)))
    if row['Exchange'] == 'US':
        links.append(('SEC filings', 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=10-K'.format(quote(listed))))

    return render(request, 'company.html', {
        'symbol': symbol, 'listed': listed, 'exchange': exchange,
        'days': days, 'horizons': PREDICT_HORIZONS,
        'name': row['Name'] if exchange == 'BSE' and not info.get('longName') else (info.get('longName') or row['Name']),
        'price': '{} {:,.2f}'.format(currency_of(symbol), data['price']) if data['price'] else '-',
        'change': change,
        'price_date': data['prices'].index[-1].strftime('%d %b') if not data['prices'].empty else '',
        'links': links,
        'about': info.get('longBusinessSummary'),
        'sector': ' · '.join(x for x in (info.get('sector'), info.get('industry'), row['Industry']) if x),
        'key_ratios': [(label, _key_ratio(kind, value, symbol)) for label, (kind, value) in data['key_ratios'].items()],
        'chart': _price_chart(data['prices'], listed) if not data['prices'].empty else None,
        'pros': data['pros'], 'cons': data['cons'],
        'unit': data['unit'],
        'quarters': _table(data['quarters'], divisor),
        'pnl': _table(pnl, divisor),
        'growth': [(title, [(label, movement(v, '{:.0f}%') if v is not None else ('-', '')) for label, v in values])
                   for title, values in data['growth'].items()],
        'balance': _table(data['balance'], divisor),
        'cashflow': _table(data['cashflow'], divisor),
        'ratios': _table(data['ratios'], divisor),
        'holding': data['holding'],
    })


def company_peers(request, ticker_value):
    """Peer comparison table (loaded separately by the company page, since it needs data for many companies)."""
    from . import analysis
    symbol = resolve_symbol(ticker_value)
    if symbol is None:
        return HttpResponse('')
    industry = COMPANIES.loc[symbol, 'Industry']
    if not industry:
        return render(request, 'company_peers.html', {'unavailable': True})
    group = COMPANIES.index[(COMPANIES['Industry'] == industry) & (COMPANIES['Country'] == COMPANIES.loc[symbol, 'Country'])]
    candidates = [symbol] + [s for s in group if s != symbol][:59]
    try:
        metrics = analysis.peer_metrics(candidates)
    except YFRateLimitError:
        return render(request, 'company_peers.html', {'busy': True})
    own = metrics.get(symbol) or {}
    peers = [s for s in candidates[1:] if metrics.get(s)]
    # Prefer companies in the same, narrower Yahoo industry when there are enough of them
    same = [s for s in peers if own.get('industry') and metrics[s]['industry'] == own.get('industry')]
    if len(same) >= 3:
        peers = same
    peers = sorted(peers, key=lambda s: -(metrics[s]['market_cap'] or 0))[:9]

    def cells(m):
        return [
            '{:,.2f}'.format(m['price']) if m['price'] else '-',
            '{:.2f}'.format(m['pe']) if m['pe'] else '-',
            _amount(m['market_cap'], symbol) if m['market_cap'] else '-',
            '{:.2f}'.format(m['dividend_yield']) if m['dividend_yield'] is not None else '-',
            '{:.2f}'.format(m['price_to_book']) if m['price_to_book'] else '-',
        ] + [movement(m['year_change'] * 100, '{:.1f}%') if m['year_change'] is not None else ('-', '')]

    rows = [{'symbol': s, 'listed': listing(s)[0], 'name': COMPANIES.loc[s, 'Name'], 'cells': cells(metrics[s]), 'self': s == symbol}
            for s in [symbol] + peers if metrics.get(s)]
    values = pd.DataFrame([metrics[s] for s in [symbol] + peers if metrics.get(s)])
    median = [
        '{:,.2f}'.format(values['price'].median()) if values['price'].notna().any() else '-',
        '{:.2f}'.format(values['pe'].median()) if values['pe'].notna().any() else '-',
        _amount(values['market_cap'].median(), symbol) if values['market_cap'].notna().any() else '-',
        '{:.2f}'.format(values['dividend_yield'].median()) if values['dividend_yield'].notna().any() else '-',
        '{:.2f}'.format(values['price_to_book'].median()) if values['price_to_book'].notna().any() else '-',
    ] + [movement(values['year_change'].median() * 100, '{:.1f}%') if values['year_change'].notna().any() else ('-', '')]
    return render(request, 'company_peers.html', {
        'rows': rows, 'median': median, 'industry': industry, 'count': len(rows),
        'currency': currency_of(symbol),
    })


# ========================================== Search suggestions ==========================================

_SEARCH_INDEX = pd.DataFrame({
    'symbol': COMPANIES.index,
    'listed': COMPANIES['Listed'].str.upper().values,
    'name': COMPANIES['Name'].str.lower().values,
    'exchange_rank': COMPANIES['Exchange'].map({'NSE': 0, 'US': 1, 'BSE': 2}).values,
    # NIFTY 500 / S&P 500 members have an industry: list these better-known companies first
    'minor': (COMPANIES['Industry'] == '').values,
})


def company_suggest(request):
    """Companies matching what the user has typed, best matches first (used by the search boxes)."""
    from django.http import JsonResponse
    q = request.GET.get('q', '').strip()
    if not q:
        return JsonResponse({'results': []})
    upper, lower = q.upper(), q.lower()
    idx = _SEARCH_INDEX
    rank = pd.Series(99, index=idx.index)
    rank[idx['name'].str.contains(lower, regex=False)] = 4
    rank[idx['listed'].str.contains(upper, regex=False)] = 3
    rank[idx['name'].str.startswith(lower)] = 2
    rank[idx['listed'].str.startswith(upper)] = 1
    rank[idx['listed'] == upper] = 0
    hits = idx.assign(rank=rank)[rank < 99].sort_values(['rank', 'minor', 'exchange_rank', 'listed']).head(8)
    results = []
    for symbol in hits['symbol']:
        listed, exchange = listing(symbol)
        results.append({'symbol': symbol, 'listed': listed, 'name': COMPANIES.loc[symbol, 'Name'], 'exchange': exchange})
    return JsonResponse({'results': results})


# ========================================== Accounts ==========================================

def signup(request):
    """Create an account (stored in Django's auth_user table) and sign the user straight in."""
    from django import forms
    from django.contrib.auth import login
    from django.contrib.auth.forms import UserCreationForm

    class SignupForm(UserCreationForm):
        email = forms.EmailField(required=False, help_text='Optional.')

        class Meta(UserCreationForm.Meta):
            fields = ('username', 'email')

    if request.user.is_authenticated:
        return redirect('/')
    form = SignupForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect(request.GET.get('next') or '/')
    return render(request, 'registration/signup.html', {'form': form})
