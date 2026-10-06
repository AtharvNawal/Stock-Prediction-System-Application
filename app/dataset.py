"""Historical daily prices from two Kaggle datasets, looked up by Yahoo-style symbol.

  India (*.NS / *.BO): chiragb254/indian-stock-market-complete-dataset-2024
      NSE and BSE stocks, 2000 to Dec 2023, one CSV per symbol per year.
  US (plain symbols):  jacksoncrow/stock-market-dataset
      NASDAQ-traded stocks and ETFs, from listing up to 2020-04-01.

Both are several GB, so they are not stored in the repo: kagglehub downloads them once
into its cache. Set INDIA_DATASET_DIR / STOCK_DATASET_DIR to use copies elsewhere.
"""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

US_DATASET = "jacksoncrow/stock-market-dataset"
INDIA_DATASET = "chiragb254/indian-stock-market-complete-dataset-2024"

SOURCES = {
    'NSE': 'Kaggle India dataset',
    'BSE': 'Kaggle India dataset',
    'US': 'Kaggle US dataset',
}


def _download(handle, env_var):
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    import kagglehub
    return Path(kagglehub.dataset_download(handle))


@lru_cache(maxsize=None)
def us_dir():
    return _download(US_DATASET, 'STOCK_DATASET_DIR')


@lru_cache(maxsize=None)
def india_dir():
    return _download(INDIA_DATASET, 'INDIA_DATASET_DIR') / 'comp_stock_data'


@lru_cache(maxsize=None)
def _us_files():
    # Symbol -> csv path. User input is only ever looked up here, never joined into a path.
    files = {}
    for folder in ('etfs', 'stocks'):
        for f in (us_dir() / folder).glob('*.csv'):
            files[f.stem] = f
    return files


@lru_cache(maxsize=None)
def india_symbols(exchange):
    """Symbol -> folder of yearly CSVs, for exchange 'NSE' or 'BSE'."""
    root = india_dir() / 'stock_data_{}'.format(exchange)
    return {d.name: d for d in root.iterdir() if d.is_dir()}


def _locate(symbol):
    """(exchange, source path) for a Yahoo-style symbol, or (None, None)."""
    for suffix, exchange in (('.NS', 'NSE'), ('.BO', 'BSE')):
        if symbol.endswith(suffix):
            folders = india_symbols(exchange)
            base = symbol[:-len(suffix)]
            return exchange, folders.get(base) or folders.get(base.replace('&', 'and'))
    return 'US', _us_files().get(symbol)


def load_history(symbol, aliases=()):
    """(daily OHLCV indexed by Date, exchange) for `symbol`, or (None, exchange) if not in the datasets.
    `aliases` are older symbols for the same company, tried when `symbol` itself is missing."""
    for candidate in (symbol, *aliases):
        exchange, path = _locate(candidate)
        if path is not None:
            break
    if path is None:
        return None, exchange
    files = sorted(path.glob('*.csv')) if path.is_dir() else [path]
    df = pd.concat([pd.read_csv(f, parse_dates=['Date']) for f in files])
    df = df.drop_duplicates('Date').set_index('Date').sort_index()
    return df.dropna(subset=['Adj Close']), exchange


@lru_cache(maxsize=None)
def _us_names():
    meta = pd.read_csv(us_dir() / 'symbols_valid_meta.csv', index_col='Symbol')
    return meta['Security Name'].str.strip()


def us_company_name(symbol):
    """Company name from the US dataset (also covers since-delisted tickers like TTM), or None."""
    return _us_names().get(symbol)
