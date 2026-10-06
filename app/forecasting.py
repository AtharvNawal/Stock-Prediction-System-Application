"""Multi-model N-day price forecasting.

Every model sees the last LOOKBACK trading days of indicator features and predicts
the whole N-day path at once, as cumulative log returns from today's close.
Data is split chronologically into train | val | test, with an N-day gap between
the splits so that no label window overlaps the next split.

  val  -> early stopping and ensemble weights
  test -> the accuracy numbers shown to the user
"""
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge

LOOKBACK = 60
HISTORY_PERIOD = '10y'
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 64

torch.set_num_threads(4)


class NotEnoughHistory(Exception):
    """Raised when a ticker's history is too short for the requested horizon."""
    def __init__(self, days, max_horizon):
        super().__init__(days, max_horizon)
        self.days = days                # trading days of price history available
        self.max_horizon = max_horizon  # largest horizon that would work, 0 if none


# ============================================== Features ==============================================

def make_features(df):
    close = df['Adj Close']
    ret = np.log(close).diff()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / close
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    log_vol = np.log(df['Volume'].replace(0, np.nan)).ffill()

    feats = pd.DataFrame({
        'return': ret,
        'range': np.log(df['High'] / df['Low']),
        'body': np.log(df['Close'] / df['Open']),
        'volume_z': (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std(),
        'ma10_gap': close / close.rolling(10).mean() - 1,
        'ma30_gap': close / close.rolling(30).mean() - 1,
        'rsi': rsi.fillna(50) / 100,
        'macd': macd,
        'macd_signal': macd.ewm(span=9, adjust=False).mean(),
        'volatility': ret.rolling(20).std(),
    }, index=df.index)
    return feats.replace([np.inf, -np.inf], np.nan)


def n_samples(rows, horizon):
    # sample t uses feature rows t-LOOKBACK+1..t and labels t+1..t+horizon
    return rows - LOOKBACK + 1 - horizon


def split_sizes(n, horizon):
    """Train/val/test sizes for n samples (test ~20%, val ~15%), or None if too few to train on."""
    usable = n - 2 * horizon
    n_test, n_val = int(usable * 0.20), int(usable * 0.15)
    n_train = usable - n_test - n_val
    if n_train < 200 or n_val < 30 or n_test < 30:
        return None
    return n_train, n_val, n_test


def max_horizon(rows):
    return next((h for h in range(365, 0, -1) if split_sizes(n_samples(rows, h), h)), 0)


def make_dataset(df, horizon):
    feats = make_features(df)
    valid = feats.notna().all(axis=1)
    feats, close = feats[valid].to_numpy(np.float32), df['Adj Close'][valid].to_numpy(np.float64)

    T = len(close)
    if split_sizes(n_samples(T, horizon), horizon) is None:
        raise NotEnoughHistory(len(df), max_horizon(T))
    log_close = np.log(close)
    starts = np.arange(LOOKBACK - 1, T - horizon)
    X = np.stack([feats[t - LOOKBACK + 1:t + 1] for t in starts])
    Y = np.stack([log_close[t + 1:t + horizon + 1] - log_close[t] for t in starts])
    last_window = feats[-LOOKBACK:]
    return X, Y, last_window, close[-1]


def split(n, horizon):
    """Chronological train | gap | val | gap | test indices."""
    n_train, n_val, n_test = split_sizes(n, horizon)
    train = np.arange(0, n_train)
    val = np.arange(n_train + horizon, n_train + horizon + n_val)
    test = np.arange(n - n_test, n)
    return train, val, test


# ============================================== Models ==============================================

class CNNBiLSTMAttention(nn.Module):
    """Conv layers extract local patterns across indicators, a BiLSTM models the sequence,
    and additive attention picks out the most informative days."""
    def __init__(self, n_features, horizon, channels=32, hidden=32, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, horizon))

    def forward(self, x):                                  # x: (B, L, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)   # (B, L, C)
        h, _ = self.lstm(h)                                # (B, L, 2H)
        weights = torch.softmax(self.attn(h), dim=1)       # (B, L, 1)
        return self.head((weights * h).sum(dim=1))


class CNN1D(nn.Module):
    """Dilated 1D convolutions: fast local pattern detection with a modest receptive field."""
    def __init__(self, n_features, horizon, channels=32, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels, horizon))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))                   # (B, C, L)
        return self.head(torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1))


class AttentionTransformer(nn.Module):
    """Transformer encoder: self-attention over the whole window for long-range dependencies.
    Kept small with dropout and weight decay because it overfits noisy prices easily."""
    def __init__(self, n_features, horizon, d_model=32, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, LOOKBACK, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, horizon))

    def forward(self, x):
        h = self.encoder(self.embed(x) + self.pos)
        return self.head(torch.cat([h.mean(dim=1), h[:, -1]], dim=1))


NEURAL_MODELS = OrderedDict([
    ('CNN + BiLSTM + Attention', (CNNBiLSTMAttention, 1e-4)),
    ('1D CNN', (CNN1D, 1e-4)),
    ('Attention Transformer', (AttentionTransformer, 1e-3)),
])


def train_neural(model_cls, weight_decay, X_train, Y_train, X_val, Y_val):
    torch.manual_seed(0)
    model = model_cls(X_train.shape[2], Y_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.from_numpy(X_train), torch.from_numpy(Y_train)
    Xv, Yv = torch.from_numpy(X_val), torch.from_numpy(Y_val)

    best_loss, best_state, bad_epochs = float('inf'), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for idx in torch.randperm(len(Xt)).split(BATCH_SIZE):
            opt.zero_grad()
            loss_fn(model(Xt[idx]), Yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return lambda X: model(torch.from_numpy(X)).detach().numpy().astype(np.float64)


def train_linear(X_train, Y_train):
    model = Ridge(alpha=10.0).fit(X_train[:, -1, :], Y_train)
    return lambda X: model.predict(X[:, -1, :])


# ============================================== Evaluation ==============================================

def metrics(pred, actual):
    """pred/actual are cumulative log returns (samples, horizon); errors are measured on prices."""
    p, a = np.exp(pred), np.exp(actual)  # price relative to the close on the forecast day
    return {
        'mape': float(np.mean(np.abs(p - a) / a) * 100),
        'final_mape': float(np.mean(np.abs(p[:, -1] - a[:, -1]) / a[:, -1]) * 100),
        'direction': float(np.mean(np.sign(pred[:, -1]) == np.sign(actual[:, -1])) * 100),
    }


def run(df, horizon):
    """Train every model on `df` (daily OHLCV) and forecast `horizon` trading days ahead."""
    X, Y, last_window, last_close = make_dataset(df, horizon)
    train, val, test = split(len(X), horizon)

    # Standardise features with training statistics only, and scale targets to unit variance
    mu = X[train].reshape(-1, X.shape[2]).mean(axis=0)
    sd = X[train].reshape(-1, X.shape[2]).std(axis=0) + 1e-8
    X = ((X - mu) / sd).astype(np.float32)
    last_window = ((last_window - mu) / sd).astype(np.float32)[None]
    y_scale = Y[train].std() + 1e-8
    Ys = (Y / y_scale).astype(np.float32)

    predictors = OrderedDict()
    timings = {}
    for name, (cls, wd) in NEURAL_MODELS.items():
        t0 = time.time()
        predictors[name] = train_neural(cls, wd, X[train], Ys[train], X[val], Ys[val])
        timings[name] = time.time() - t0
    t0 = time.time()
    predictors['Linear Regression'] = train_linear(X[train], Ys[train])
    timings['Linear Regression'] = time.time() - t0

    def predict(fn, idx):
        return fn(X[idx]) * y_scale

    # Ensemble weights: inverse validation MSE, so the models that generalise best count most
    val_mse = {name: np.mean((predict(fn, val) - Y[val]) ** 2) for name, fn in predictors.items()}
    inv = {name: 1 / v for name, v in val_mse.items()}
    weights = {name: v / sum(inv.values()) for name, v in inv.items()}

    def ensemble(rows):
        return sum(weights[name] * fn(rows) for name, fn in predictors.items()) * y_scale

    results = []
    for name, fn in predictors.items():
        results.append({'name': name, 'weight': weights[name] * 100, 'seconds': timings[name],
                        **metrics(predict(fn, test), Y[test]),
                        'forecast': last_close * np.exp(fn(last_window)[0] * y_scale)})
    results.append({'name': 'Weighted Ensemble', 'weight': 100.0, 'seconds': sum(timings.values()),
                    **metrics(ensemble(X[test]), Y[test]),
                    'forecast': last_close * np.exp(ensemble(last_window)[0])})
    results.append({'name': 'Naive baseline (no change)', 'weight': None, 'seconds': 0.0,
                    **metrics(np.zeros_like(Y[test]), Y[test]), 'direction': None,
                    'forecast': np.full(horizon, last_close)})
    return {
        'results': results,
        'last_close': float(last_close),
        # share of test windows where the price actually rose, to put direction accuracy in context
        'up_rate': float(np.mean(Y[test][:, -1] > 0) * 100),
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
    }


_cache = OrderedDict()


def forecast(ticker, horizon, download):
    """Cached per ticker/horizon/day, since training all models takes a while."""
    key = (ticker, horizon, pd.Timestamp.today().date())
    if key not in _cache:
        df = download(ticker)
        out = run(df, horizon)
        out['last_date'] = df.index[-1]
        _cache[key] = out
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return _cache[key]
