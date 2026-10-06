"""Builds app/Data/companies.csv: the companies the app supports.

  India: every company currently listed on NSE (main board + SME, NSE's official lists),
         plus companies trading only on BSE, taken from BSE's official daily bhavcopy
         (price file) for the last couple of weeks. Only equity shares are kept
         (ISIN INE...01...), so funds, ETFs, bonds and other instruments are dropped.
         Companies on both exchanges are listed once, under NSE, matched by ISIN.
         NSE's symbol-change list goes to app/Data/symbol_changes.csv, so old symbols
         (e.g. TATAMOTORS -> TMPV) still resolve. BSE scrip codes come from the
         bhavcopy, and NIFTY 500 members get NSE's industry (used for peer comparison).
  US:    well-known companies only - S&P 500 and NASDAQ-100 constituents.

Run from the project root:  .venv/bin/python scripts/build_companies.py
"""
import io
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path('app/Data/companies.csv')
CHANGES_OUT = Path('app/Data/symbol_changes.csv')
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/154 Safari/537.36',
      'Referer': 'https://www.bseindia.com/'}
NIFTY50 = 'https://archives.nseindia.com/content/indices/ind_nifty50list.csv'  # home page market movers
NIFTY500 = 'https://archives.nseindia.com/content/indices/ind_nifty500list.csv'  # has an Industry column
NSE_SYMBOL_CHANGES = 'https://archives.nseindia.com/content/equities/symbolchange.csv'
NSE_LISTS = ['https://archives.nseindia.com/content/equities/EQUITY_L.csv',
             'https://archives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv']
BSE_BHAVCOPY = 'https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{:%Y%m%d}_F_0000.CSV'
BHAVCOPY_DAYS = 10  # weekdays; thinly traded stocks don't trade every day
SP500 = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
NASDAQ100 = 'https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies'


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()


def is_equity_share(isin):
    # Indian ISINs: IN + issuer type (E = company) + 4-char issuer + 2-digit security type (01 = equity shares)
    return isinstance(isin, str) and isin.startswith('INE') and isin[7:9] == '01'


def nse_companies():
    frames = []
    for url in NSE_LISTS:
        df = pd.read_csv(io.StringIO(fetch(url)))
        df.columns = [c.strip().upper().replace('_', ' ') for c in df.columns]
        frames.append(df[['SYMBOL', 'NAME OF COMPANY', 'ISIN NUMBER']])
    df = pd.concat(frames).drop_duplicates('SYMBOL')
    return pd.DataFrame({'Symbol': df['SYMBOL'] + '.NS', 'Listed': df['SYMBOL'], 'Name': df['NAME OF COMPANY'].str.strip(),
                         'Exchange': 'NSE', 'Country': 'India', 'ISIN': df['ISIN NUMBER'].str.strip()})


def nse_symbol_changes(nse):
    """Old NSE symbol -> current one (following chains like TELCO -> TATAMOTORS -> TMPV)."""
    df = pd.read_csv(io.StringIO(fetch(NSE_SYMBOL_CHANGES)), header=None, usecols=[1, 2, 3],
                     names=['Old', 'New', 'Date'])
    df = df.apply(lambda c: c.str.strip())
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    latest = df.sort_values('Date').groupby('Old')['New'].last().to_dict()

    def current(symbol, seen=()):
        nxt = latest.get(symbol)
        return symbol if nxt is None or nxt in seen else current(nxt, seen + (symbol,))

    listed = {s[:-3] for s in nse.Symbol}
    rows = [{'Old': old + '.NS', 'New': current(old) + '.NS'} for old in latest]
    return pd.DataFrame([r for r in rows if r['New'][:-3] in listed and r['Old'] != r['New']])


def bse_equities():
    frames = []
    for day in pd.bdate_range(end=pd.Timestamp.today(), periods=BHAVCOPY_DAYS + 5)[::-1]:
        try:
            text = fetch(BSE_BHAVCOPY.format(day))
        except urllib.error.HTTPError:
            continue
        if not text.startswith('TradDt'):
            continue  # exchange holiday or not published yet: BSE serves an HTML page instead
        frames.append(pd.read_csv(io.StringIO(text)))
        if len(frames) == BHAVCOPY_DAYS:
            break
    print('BSE bhavcopy days:', len(frames))
    bse = pd.concat(frames)
    bse = bse[(bse['FinInstrmTp'] == 'STK') & bse['ISIN'].map(is_equity_share)]
    bse = bse.drop_duplicates('ISIN')
    print('BSE equity shares traded:', len(bse))
    return bse.assign(TckrSymb=bse['TckrSymb'].str.strip(), BSECode=bse['FinInstrmId'].astype(str))


def bse_only_companies(nse, bse):
    bse = bse[~bse['ISIN'].isin(set(nse['ISIN']))]
    return pd.DataFrame({'Symbol': bse['TckrSymb'] + '.BO', 'Listed': bse['TckrSymb'],
                         'Name': bse['FinInstrmNm'].str.strip(), 'Exchange': 'BSE', 'Country': 'India',
                         'ISIN': bse['ISIN'], 'BSECode': bse['BSECode']})


def us_companies():
    sp = pd.read_html(io.StringIO(fetch(SP500)))[0]
    industry = dict(zip(sp['Symbol'], sp['GICS Sub-Industry']))
    sp = sp[['Symbol', 'Security']]
    nd = next(t for t in pd.read_html(io.StringIO(fetch(NASDAQ100))) if 'Ticker' in t.columns)[['Ticker', 'Company']]
    df = pd.concat([sp.set_axis(['Listed', 'Name'], axis=1), nd.set_axis(['Listed', 'Name'], axis=1)])
    df['Symbol'] = df['Listed'].str.replace('.', '-', regex=False)  # Yahoo style, e.g. BRK.B -> BRK-B
    df = df.drop_duplicates('Symbol')
    df['Exchange'], df['Country'], df['ISIN'] = 'US', 'United States', ''
    df['Industry'] = df['Listed'].map(industry)  # GICS sub-industry, S&P 500 members only
    return df


def main():
    nse = nse_companies()
    print('NSE companies:', len(nse))
    changes = nse_symbol_changes(nse)
    changes.to_csv(CHANGES_OUT, index=False)
    print('NSE symbol changes:', len(changes))
    bse = bse_equities()
    # Companies on both exchanges: remember their BSE symbol too, which can differ from NSE's
    nse['BSE'] = nse['ISIN'].map(bse.set_index('ISIN')['TckrSymb'])
    nse['BSECode'] = nse['ISIN'].map(bse.set_index('ISIN')['BSECode'])
    nifty = pd.read_csv(io.StringIO(fetch(NIFTY500)))
    nse['Industry'] = nse['Listed'].map(dict(zip(nifty['Symbol'].str.strip(), nifty['Industry'].str.strip())))
    print('NIFTY 500 companies with an industry:', nse['Industry'].notna().sum())
    nifty50 = set(pd.read_csv(io.StringIO(fetch(NIFTY50)))['Symbol'].str.strip())
    nse['Nifty50'] = nse['Listed'].isin(nifty50).map({True: 'Y', False: ''})
    print('NIFTY 50 members:', (nse['Nifty50'] == 'Y').sum())
    print('NSE companies also on BSE:', nse['BSE'].notna().sum())
    bse_only = bse_only_companies(nse, bse)
    print('BSE-only companies:', len(bse_only))
    us = us_companies()
    print('US companies:', len(us))
    out = pd.concat([nse, bse_only, us]).drop_duplicates('Symbol').sort_values(['Country', 'Symbol'])
    out = out[['Symbol', 'Listed', 'BSE', 'BSECode', 'Name', 'Exchange', 'Country', 'ISIN', 'Industry', 'Nifty50']]
    out.to_csv(OUT, index=False)
    print('Wrote', len(out), 'companies to', OUT)


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
