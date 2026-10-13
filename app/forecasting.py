"""Multi-model N-day price forecasting.

Every model sees the last LOOKBACK trading days of indicator features and predicts
the whole N-day path at once, as cumulative log returns from today's close.
Data is split chronologically into train | val | test, with an N-day gap between
the splits so that no label window overlaps the next split.

  val  -> early stopping and ensemble weights
  test -> the accuracy numbers shown to the user

Object model:

  Forecaster (abstract)          train() / predict() template, with fit state and timing
    NeuralForecaster             trains any nn.Module network with early stopping
    RidgeForecaster              linear baseline on the last day's features
    NaiveForecaster              "price stays flat", kept out of the ensemble
    WeightedEnsemble             composite of forecasters, weighted by inverse validation MSE
  StandardScaler, TargetScaler   fit on the training split only (no leakage)
  WindowedDataset, ChronologicalSplit
  ForecastPipeline               dataset -> split -> scale -> train -> evaluate -> forecast
  ForecastService                caches pipeline results per ticker, horizon and day
"""
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge

from .caching import LRUCache

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


class NotFittedError(RuntimeError):
    """Raised when predict() is called on a forecaster that has not been trained."""


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
    """Largest horizon (<= 365) that leaves enough samples to train on, or 0.

    Having enough samples is monotonic in the horizon (a longer horizon never leaves more),
    so binary search finds the boundary in ~9 checks instead of up to 365."""
    def fits(h):
        return split_sizes(n_samples(rows, h), h) is not None

    lo, hi = 0, 365  # invariant: lo fits (0 counts as "nothing fits"), everything above hi doesn't
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass(frozen=True)
class WindowedDataset:
    """Sliding windows over the feature rows: X (samples, LOOKBACK, features), Y (samples, horizon)."""
    X: np.ndarray
    Y: np.ndarray
    last_window: np.ndarray  # the most recent LOOKBACK rows, used for the live forecast
    last_close: float

    @classmethod
    def from_prices(cls, df, horizon):
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
        return cls(X, Y, feats[-LOOKBACK:], close[-1])


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological train | gap | val | gap | test indices into the windowed samples."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @classmethod
    def of(cls, n, horizon):
        n_train, n_val, n_test = split_sizes(n, horizon)
        return cls(train=np.arange(0, n_train),
                   val=np.arange(n_train + horizon, n_train + horizon + n_val),
                   test=np.arange(n - n_test, n))


def make_dataset(df, horizon):
    """(X, Y, last_window, last_close); kept for callers of the functional API."""
    d = WindowedDataset.from_prices(df, horizon)
    return d.X, d.Y, d.last_window, d.last_close


def split(n, horizon):
    s = ChronologicalSplit.of(n, horizon)
    return s.train, s.val, s.test


# ============================================== Scaling ==============================================

class StandardScaler:
    """Per-feature standardisation. Fit on training windows only, so no test statistics leak in."""

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)


class TargetScaler:
    """Scales targets to unit variance for training; inverse() brings predictions back to log returns."""

    def fit(self, Y):
        self.scale_ = Y.std() + 1e-8
        return self

    def transform(self, Y):
        return (Y / self.scale_).astype(np.float32)

    def inverse(self, Y_scaled):
        return Y_scaled * self.scale_


# ============================================== Networks ==============================================

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


# name -> (network class, weight decay)
NEURAL_MODELS = OrderedDict([
    ('CNN + BiLSTM + Attention', (CNNBiLSTMAttention, 1e-4)),
    ('1D CNN', (CNN1D, 1e-4)),
    ('Attention Transformer', (AttentionTransformer, 1e-3)),
])


# ============================================== Forecasters ==============================================

@dataclass(frozen=True)
class Batch:
    """Scaled inputs and targets for one split."""
    X: np.ndarray
    Y: np.ndarray


class Forecaster(ABC):
    """A model that maps feature windows to the scaled N-day return path.

    Template method: callers use train() and predict(), which handle timing and the
    fitted check; subclasses only implement _fit() and _predict()."""
    name = 'Forecaster'
    in_ensemble = True        # baselines opt out of the weighted ensemble
    reports_direction = True  # a flat forecast has no direction to score

    def __init__(self):
        self._fitted = False
        self.seconds = 0.0

    @property
    def is_fitted(self):
        return self._fitted

    def train(self, train, val):
        t0 = time.time()
        self._fit(train, val)
        self.seconds = time.time() - t0
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise NotFittedError('{} must be trained before it can predict'.format(self.name))
        return self._predict(X)

    @abstractmethod
    def _fit(self, train, val):
        """Learn from the training batch; the validation batch may be used for early stopping."""

    @abstractmethod
    def _predict(self, X):
        """Predictions (samples, horizon) as float64, in scaled target units."""

    def __repr__(self):
        return '<{} {!r}{}>'.format(type(self).__name__, self.name, '' if self._fitted else ' (untrained)')


class NeuralForecaster(Forecaster):
    """Trains any network class with the (n_features, horizon) constructor, using AdamW,
    gradient clipping and early stopping on validation loss (best weights restored)."""

    def __init__(self, name, network_cls, weight_decay, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 batch_size=BATCH_SIZE, lr=1e-3, seed=0):
        super().__init__()
        self.name = name
        self._network_cls = network_cls
        self._weight_decay = weight_decay
        self._max_epochs = max_epochs
        self._patience = patience
        self._batch_size = batch_size
        self._lr = lr
        self._seed = seed
        self._model = None
        self.epochs_run = 0

    def _fit(self, train, val):
        torch.manual_seed(self._seed)
        model = self._network_cls(train.X.shape[2], train.Y.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        loss_fn = nn.MSELoss()
        Xt, Yt = torch.from_numpy(train.X), torch.from_numpy(train.Y)
        Xv, Yv = torch.from_numpy(val.X), torch.from_numpy(val.Y)

        best_loss, best_state, bad_epochs = float('inf'), None, 0
        for epoch in range(self._max_epochs):
            model.train()
            for idx in torch.randperm(len(Xt)).split(self._batch_size):
                opt.zero_grad()
                loss_fn(model(Xt[idx]), Yt[idx]).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xv), Yv).item()
            self.epochs_run = epoch + 1
            if val_loss < best_loss - 1e-6:
                best_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= self._patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        self._model = model

    def _predict(self, X):
        with torch.no_grad():
            return self._model(torch.from_numpy(X)).numpy().astype(np.float64)


class RidgeForecaster(Forecaster):
    """L2-regularised linear regression on the most recent day's features."""
    name = 'Linear Regression'

    def __init__(self, alpha=10.0):
        super().__init__()
        self._model = Ridge(alpha=alpha)

    def _fit(self, train, val):
        self._model.fit(train.X[:, -1, :], train.Y)

    def _predict(self, X):
        return self._model.predict(X[:, -1, :])


class NaiveForecaster(Forecaster):
    """Baseline that assumes the price stays where it is (zero return on every day)."""
    name = 'Naive baseline (no change)'
    in_ensemble = False
    reports_direction = False

    def _fit(self, train, val):
        self._horizon = train.Y.shape[1]

    def _predict(self, X):
        return np.zeros((len(X), self._horizon))


class WeightedEnsemble(Forecaster):
    """Composite forecaster: trains its members, then weights each by 1 / validation MSE,
    so the models that generalise best count most."""
    name = 'Weighted Ensemble'

    def __init__(self, members):
        super().__init__()
        if not members:
            raise ValueError('an ensemble needs at least one member')
        self.members = list(members)
        self.weights = OrderedDict()

    def _fit(self, train, val):
        inverse_mse = OrderedDict()
        for member in self.members:
            member.train(train, val)
            inverse_mse[member.name] = 1 / np.mean((member.predict(val.X) - val.Y) ** 2)
        total = sum(inverse_mse.values())
        self.weights = OrderedDict((name, v / total) for name, v in inverse_mse.items())

    def _predict(self, X):
        return sum(self.weights[m.name] * m.predict(X) for m in self.members)

    def weight_of(self, member):
        return self.weights[member.name]


def default_models(max_epochs=MAX_EPOCHS):
    """Factory for the forecasters the app compares: three networks, ridge, and the naive baseline."""
    return ([NeuralForecaster(name, cls, wd, max_epochs=max_epochs) for name, (cls, wd) in NEURAL_MODELS.items()]
            + [RidgeForecaster(), NaiveForecaster()])


# ============================================== Evaluation ==============================================

def metrics(pred, actual):
    """pred/actual are cumulative log returns (samples, horizon); errors are measured on prices."""
    p, a = np.exp(pred), np.exp(actual)  # price relative to the close on the forecast day
    return {
        'mape': float(np.mean(np.abs(p - a) / a) * 100),
        'final_mape': float(np.mean(np.abs(p[:, -1] - a[:, -1]) / a[:, -1]) * 100),
        'direction': float(np.mean(np.sign(pred[:, -1]) == np.sign(actual[:, -1])) * 100),
    }


@dataclass
class ModelResult:
    name: str
    weight: object        # % of the ensemble; 100 for the ensemble itself, None for baselines
    seconds: float
    mape: float
    final_mape: float
    direction: object     # None when the model has no direction to score
    forecast: np.ndarray  # predicted price for each of the next `horizon` trading days

    def to_dict(self):
        return asdict(self)


class ForecastPipeline:
    """Windowing -> chronological split -> scaling -> training -> test evaluation -> live forecast."""

    def __init__(self, horizon, model_factory=default_models):
        self.horizon = horizon
        self._model_factory = model_factory

    def run(self, df):
        """Train every model on `df` (daily OHLCV) and forecast `horizon` trading days ahead."""
        data = WindowedDataset.from_prices(df, self.horizon)
        split = ChronologicalSplit.of(len(data.X), self.horizon)

        x_scaler = StandardScaler().fit(data.X[split.train])
        y_scaler = TargetScaler().fit(data.Y[split.train])
        X = x_scaler.transform(data.X)
        Ys = y_scaler.transform(data.Y)
        last_window = x_scaler.transform(data.last_window)[None]
        train, val = Batch(X[split.train], Ys[split.train]), Batch(X[split.val], Ys[split.val])

        models = self._model_factory()
        ensemble = WeightedEnsemble([m for m in models if m.in_ensemble]).train(train, val)
        baselines = [m.train(train, val) for m in models if not m.in_ensemble]

        def weight(model):
            if model is ensemble:
                return 100.0
            return ensemble.weight_of(model) * 100 if model.in_ensemble else None

        results = []
        for model in [*ensemble.members, ensemble, *baselines]:
            scores = metrics(y_scaler.inverse(model.predict(X[split.test])), data.Y[split.test])
            if not model.reports_direction:
                scores['direction'] = None
            forecast = data.last_close * np.exp(y_scaler.inverse(model.predict(last_window))[0])
            results.append(ModelResult(name=model.name, weight=weight(model),
                                       seconds=model.seconds, forecast=forecast, **scores).to_dict())
        return {
            'results': results,
            'last_close': float(data.last_close),
            # share of test windows where the price actually rose, to put direction accuracy in context
            'up_rate': float(np.mean(data.Y[split.test][:, -1] > 0) * 100),
            'n_train': len(split.train), 'n_val': len(split.val), 'n_test': len(split.test),
        }


def run(df, horizon):
    return ForecastPipeline(horizon).run(df)


class ForecastService:
    """Caches pipeline results per (ticker, horizon, day), since training all models takes a while.
    The price download is injected, so the service doesn't depend on any one data source."""

    def __init__(self, cache, pipeline_cls=ForecastPipeline, today=lambda: pd.Timestamp.today().date()):
        self._cache = cache
        self._pipeline_cls = pipeline_cls
        self._today = today

    def forecast(self, ticker, horizon, download):
        def compute():
            df = download(ticker)
            out = self._pipeline_cls(horizon).run(df)
            out['last_date'] = df.index[-1]
            return out
        return self._cache.get_or_compute((ticker, horizon, self._today()), compute)


_service = ForecastService(LRUCache(32))


def forecast(ticker, horizon, download):
    return _service.forecast(ticker, horizon, download)
