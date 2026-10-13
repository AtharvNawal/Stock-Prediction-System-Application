import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from app import forecasting as fc
from app.caching import LRUCache


def synthetic_prices(n=700, seed=1):
    """A random walk with OHLCV columns, like a Yahoo daily download."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({'Open': close * (1 + rng.normal(0, .003, n)), 'High': close * 1.01, 'Low': close * 0.99,
                         'Close': close, 'Adj Close': close, 'Volume': rng.integers(100_000, 1_000_000, n).astype(float)},
                        index=pd.bdate_range('2020-01-01', periods=n))


class ConstantForecaster(fc.Forecaster):
    """Predicts the same scaled return everywhere, so its validation error is known exactly."""
    def __init__(self, name, value):
        super().__init__()
        self.name, self.value = name, value

    def _fit(self, train, val):
        self._horizon = train.Y.shape[1]

    def _predict(self, X):
        return np.full((len(X), self._horizon), self.value)


class SplitTests(SimpleTestCase):
    def test_splits_are_chronological_with_a_horizon_gap(self):
        horizon = 30
        s = fc.ChronologicalSplit.of(1000, horizon)
        self.assertEqual(s.train[0], 0)
        self.assertEqual(s.val[0] - s.train[-1], horizon + 1)  # val labels can't overlap training labels
        self.assertGreaterEqual(s.test[0] - s.val[-1], horizon + 1)
        self.assertEqual(s.test[-1], 999)

    def test_max_horizon_matches_a_linear_scan(self):
        def linear(rows):
            return next((h for h in range(365, 0, -1) if fc.split_sizes(fc.n_samples(rows, h), h)), 0)
        for rows in range(0, 2000, 7):
            self.assertEqual(fc.max_horizon(rows), linear(rows), rows)

    def test_short_history_raises_with_the_largest_usable_horizon(self):
        with self.assertRaises(fc.NotEnoughHistory) as ctx:
            fc.WindowedDataset.from_prices(synthetic_prices(400), horizon=200)
        self.assertGreater(ctx.exception.max_horizon, 0)
        self.assertLess(ctx.exception.max_horizon, 200)


class ScalerTests(SimpleTestCase):
    def test_standard_scaler_uses_only_the_data_it_was_fit_on(self):
        train = np.zeros((10, 5, 2), np.float32)
        train[..., 0], train[..., 1] = 1.0, 3.0
        scaler = fc.StandardScaler().fit(train)
        np.testing.assert_allclose(scaler.mean_, [1.0, 3.0])
        later = np.full((1, 5, 2), 1000.0, np.float32)  # test-period values must not shift the statistics
        scaler.transform(later)
        np.testing.assert_allclose(scaler.mean_, [1.0, 3.0])

    def test_target_scaler_round_trips(self):
        Y = np.random.default_rng(0).normal(0, 0.05, (50, 7))
        scaler = fc.TargetScaler().fit(Y)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(Y)), Y, rtol=1e-6, atol=1e-8)


class ForecasterTests(SimpleTestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.train = fc.Batch(rng.normal(size=(40, 60, 3)).astype(np.float32), np.zeros((40, 5), np.float32))
        self.val = fc.Batch(rng.normal(size=(20, 60, 3)).astype(np.float32), np.zeros((20, 5), np.float32))

    def test_predict_before_train_raises(self):
        with self.assertRaises(fc.NotFittedError):
            fc.RidgeForecaster().predict(self.val.X)

    def test_ensemble_weights_follow_inverse_validation_error(self):
        good, bad = ConstantForecaster('good', 0.1), ConstantForecaster('bad', 0.2)  # MSE 0.01 vs 0.04
        ensemble = fc.WeightedEnsemble([good, bad]).train(self.train, self.val)
        self.assertAlmostEqual(sum(ensemble.weights.values()), 1.0)
        self.assertAlmostEqual(ensemble.weight_of(good), 0.8)
        self.assertAlmostEqual(ensemble.weight_of(bad), 0.2)
        np.testing.assert_allclose(ensemble.predict(self.val.X), 0.8 * 0.1 + 0.2 * 0.2)

    def test_naive_baseline_predicts_no_change_and_stays_out_of_the_ensemble(self):
        naive = fc.NaiveForecaster().train(self.train, self.val)
        self.assertFalse(naive.in_ensemble)
        np.testing.assert_array_equal(naive.predict(self.val.X), np.zeros((20, 5)))

    def test_every_forecaster_is_polymorphic(self):
        for model in fc.default_models(max_epochs=1):
            self.assertIsInstance(model, fc.Forecaster)
            model.train(self.train, self.val)
            self.assertEqual(model.predict(self.val.X).shape, (20, 5))


class PipelineTests(SimpleTestCase):
    def test_end_to_end_report(self):
        report = fc.ForecastPipeline(10, model_factory=lambda: fc.default_models(max_epochs=1)).run(synthetic_prices())
        names = [r['name'] for r in report['results']]
        self.assertEqual(names, ['CNN + BiLSTM + Attention', '1D CNN', 'Attention Transformer', 'Linear Regression',
                                 'Weighted Ensemble', 'Naive baseline (no change)'])
        weights = [r['weight'] for r in report['results'][:4]]
        self.assertAlmostEqual(sum(weights), 100.0)
        for r in report['results']:
            self.assertEqual(len(r['forecast']), 10)
            self.assertTrue(np.all(r['forecast'] > 0))
        naive = report['results'][-1]
        self.assertIsNone(naive['direction'])
        np.testing.assert_allclose(naive['forecast'], report['last_close'])

    def test_service_caches_per_ticker_horizon_and_day(self):
        downloads = []

        class FakePipeline:
            def __init__(self, horizon):
                self.horizon = horizon

            def run(self, df):
                return {'horizon': self.horizon}

        def download(ticker):
            downloads.append(ticker)
            return synthetic_prices(5)

        service = fc.ForecastService(LRUCache(8), pipeline_cls=FakePipeline, today=lambda: '2026-10-13')
        service.forecast('TCS.NS', 30, download)
        service.forecast('TCS.NS', 30, download)
        service.forecast('TCS.NS', 7, download)
        self.assertEqual(downloads, ['TCS.NS', 'TCS.NS'])
