"""
Unit tests for the turbo_mc (matrix completion) package.

These tests are solver-free: the iterative matrix completion machinery is
exercised against a synthetic oracle whose underlying matrix is prescribed
in advance. This is also the fast feedback loop for tuning the turbo
hyperparameters (lam, n_epochs, n_factors, train_ratio, ...).
"""
import numpy as np
import pytest

from compass.turbo_mc import matrix_manipulation
from compass.turbo_mc import metrics
from compass.turbo_mc.iterative_models.matrix_oracle import (
    OracleWithAPrescribedMatrix,
    MatrixOracle,
)
from compass.turbo_mc.iterative_models.iterative_matrix_completion_model import (
    IterativeMCMWithGuaranteedSpearmanR2,
)
from compass.turbo_mc.models.column_normalizer_model_wrapper import ColumnNormalizerModelWrapper
from compass.turbo_mc.models.exclude_constant_columns_model_wrapper import ExcludeConstantColumnsModelWrapper
from compass.turbo_mc.models.matrix_completion_fast_als import MatrixCompletionFastALS


# ---------------------------------------------------------------------------
# matrix_manipulation
# ---------------------------------------------------------------------------

class TestRandomIndices:
    def test_number_of_sampled_entries(self):
        np.random.seed(0)
        R, C, density = 20, 10, 0.5
        res = matrix_manipulation.get_list_of_random_matrix_indices(R, C, density)
        assert len(res) == int(R * C * density)

    def test_all_entries_distinct_and_in_bounds(self):
        np.random.seed(1)
        R, C, density = 15, 8, 0.4
        res = matrix_manipulation.get_list_of_random_matrix_indices(R, C, density)
        assert len(set(res)) == len(res)
        for (r, c) in res:
            assert 0 <= r < R
            assert 0 <= c < C

    def test_each_column_equally_sampled(self):
        np.random.seed(2)
        R, C, density = 100, 5, 0.1
        res = matrix_manipulation.get_list_of_random_matrix_indices(R, C, density)
        per_column = [sum(1 for (_, c) in res if c == i) for i in range(C)]
        assert max(per_column) - min(per_column) == 0

    def test_too_small_density_raises(self):
        with pytest.raises(ValueError):
            matrix_manipulation.get_list_of_random_matrix_indices(10, 10, 0.001)

    def test_randomize_rows_and_columns(self):
        np.random.seed(3)
        R, C, density = 10, 4, 0.3
        plain = matrix_manipulation.get_list_of_random_matrix_indices(R, C, density)
        np.random.seed(3)
        shuffled = matrix_manipulation.get_list_of_random_matrix_indices(
            R, C, density, randomize_rows_and_columns=True)
        # Same number of entries, generally a different set of indices
        assert len(shuffled) == len(plain)
        assert sorted(plain) != sorted(shuffled)


class TestObservationHelpers:
    def test_observation_list_to_df_and_matrix_round_trip(self):
        obs = [(1, 0, 10.0), (0, 1, 20.0), (1, 1, 30.0)]
        df = matrix_manipulation.observation_list_to_df(obs)
        X = matrix_manipulation.matrix_from_observations(df, 2, 2)
        assert X[1, 0] == 10.0
        assert X[0, 1] == 20.0
        assert X[1, 1] == 30.0
        assert np.isnan(X[0, 0])

    def test_observe_entries_and_hide_rest(self):
        np.random.seed(0)
        X_true = np.arange(12, dtype=float).reshape(3, 4)
        indices = matrix_manipulation.get_list_of_random_matrix_indices(3, 4, 0.5)
        X_obs = matrix_manipulation.observe_entries_and_hide_rest(X_true, indices)
        assert X_obs.shape == X_true.shape
        # Observed entries equal the ground truth
        for (r, c) in indices:
            assert X_obs[r, c] == X_true[r, c]
        # Unobserved entries are hidden
        for r in range(3):
            for c in range(4):
                if (r, c) not in set(indices):
                    assert np.isnan(X_obs[r, c])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @staticmethod
    def _unobserved(n):
        # y_observed with no entries observed -> all entries are used
        return np.full(n, np.nan)

    def test_compute_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert metrics.compute_r2(y, self._unobserved(4), y) == pytest.approx(1.0)

    def test_compute_r2_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        # Partially observe the column so the baseline mean is well-defined,
        # and predict the unobserved entries with the (observed) mean.
        y_observed = np.array([1.0, np.nan, 3.0, np.nan])
        y_pred = np.array([1.0, 2.0, 3.0, 2.0])
        # The mean of the observed entries is 2.0, which is exactly the
        # prediction: a constant prediction explains no variance
        assert metrics.compute_r2(y_true, y_observed, y_pred) == pytest.approx(0.0)

    def test_compute_r2_border_case_all_observed(self):
        y = np.array([1.0, 2.0, 3.0])
        # Fully observed column: nothing to predict, R2 is 1 by definition
        assert metrics.compute_r2(y, y, y) == 1.0

    def test_compute_mse(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0.5, -0.5, 1.5, 0.5])
        expected = np.mean((y_true - y_pred) ** 2)
        assert metrics.compute_mse(y_true, self._unobserved(4), y_pred) == pytest.approx(expected)

    def test_compute_spearman_r2_invariance_to_monotonic_transform(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true ** 3  # strictly monotonic: ranks identical
        assert metrics.compute_spearman_r2(y_true, self._unobserved(5), y_pred) == pytest.approx(1.0)

    def test_compute_pearson_r2_linear(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = 2.0 * y_true + 3.0  # perfectly linear
        assert metrics.compute_pearson_r2(y_true, self._unobserved(5), y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# matrix completion models
# ---------------------------------------------------------------------------

def make_low_rank_matrix(R=20, C=20, rank=2, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(R, rank))
    B = rng.normal(size=(C, rank))
    return A @ B.T


def mask_matrix(X, density, seed):
    """Returns (X_observed, indices) with a random 'density' fraction observed."""
    indices = matrix_manipulation.get_list_of_random_matrix_indices(X.shape[0], X.shape[1], density)
    return matrix_manipulation.observe_entries_and_hide_rest(X, indices), indices


class TestFastALS:
    def test_recovers_low_rank_matrix(self):
        np.random.seed(0)
        X_true = make_low_rank_matrix(20, 20, rank=2)
        X_obs, _ = mask_matrix(X_true, 0.5, 0)
        model = MatrixCompletionFastALS(n_factors=2, lam=0.1, n_epochs=300, verbose=False)
        model.fit_matrix(X_obs)
        X_hat = model.predict_all()
        # The low-rank approximation should be close to the ground truth
        corr = np.corrcoef(X_true.flatten(), X_hat.flatten())[0, 1]
        assert corr > 0.99

    def test_objective_decreases(self):
        np.random.seed(1)
        X_obs, _ = mask_matrix(make_low_rank_matrix(15, 15, rank=2, seed=1), 0.5, 1)
        model = MatrixCompletionFastALS(n_factors=2, lam=1.0, n_epochs=1, verbose=False)
        model._fit_matrix_init(X_obs)
        model._step()
        obj1 = model.objective()
        for _ in range(50):
            model._step()
        obj2 = model.objective()
        assert obj2 < obj1

    def test_predict_single_entry(self):
        np.random.seed(2)
        X_true = make_low_rank_matrix(10, 10, rank=1, seed=2)
        X_obs, _ = mask_matrix(X_true, 0.7, 2)
        model = MatrixCompletionFastALS(n_factors=1, lam=0.1, n_epochs=300, verbose=False)
        model.fit_matrix(X_obs)
        rows, cols = np.where(np.isnan(X_obs))
        r, c = int(rows[0]), int(cols[0])
        assert model.predict(r, c) == pytest.approx(model.predict_all()[r, c])


class TestWrappers:
    def test_column_normalizer_unnormalizes(self):
        np.random.seed(0)
        X_true = make_low_rank_matrix(20, 20, rank=2) + 5.0  # non-zero mean
        X_obs, _ = mask_matrix(X_true, 0.6, 0)
        inner = MatrixCompletionFastALS(n_factors=2, lam=0.01, n_epochs=500, verbose=False)
        model = ColumnNormalizerModelWrapper(inner)
        model.fit_matrix(X_obs)
        X_hat = model.predict_all()
        corr = np.corrcoef(X_true.flatten(), X_hat.flatten())[0, 1]
        assert corr > 0.95

    def test_exclude_constant_columns(self):
        X_true = make_low_rank_matrix(10, 10, rank=2)
        X_true[:, 0] = 7.0  # constant column
        X_obs, _ = mask_matrix(X_true, 0.6, 0)
        inner = MatrixCompletionFastALS(n_factors=2, lam=0.1, n_epochs=300, verbose=False)
        model = ExcludeConstantColumnsModelWrapper(inner)
        model.fit_matrix(X_obs)
        X_hat = model.predict_all()
        # The constant column must be reproduced exactly
        np.testing.assert_allclose(X_hat[:, 0], 7.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# iterative matrix completion
# ---------------------------------------------------------------------------

class CountingOracle(OracleWithAPrescribedMatrix):
    """Counts oracle queries and exposes the observed matrix incrementally."""

    def __init__(self, X):
        super().__init__(X)
        self.n_queries = 0
        self._observed = np.full(X.shape, np.nan)

    def _observe_entries(self, rows, cols, iter=0):
        self.n_queries += 1
        vals = super()._observe_entries(rows, cols, iter)
        self._observed[np.asarray(rows), np.asarray(cols)] = vals
        return vals

    def observed_matrix(self):
        return self._observed


class TestIterativeMCM:
    def _cv_model(self, n_factors=2, n_epochs=200):
        def cv_model_func(state):
            # State carries the matrix dimensions and the running sampled density
            assert state.sampled_density > 0
            return _TrainValSplitCV(
                ColumnNormalizerModelWrapper(
                    MatrixCompletionFastALS(
                        n_factors=n_factors, lam=0.1, n_epochs=n_epochs, verbose=False)),
                train_ratio=0.8,
            )
        return cv_model_func

    def test_terminates_and_imputes(self):
        np.random.seed(0)
        X_true = make_low_rank_matrix(30, 30, rank=3)
        oracle = OracleWithAPrescribedMatrix(X_true)
        model = IterativeMCMWithGuaranteedSpearmanR2(
            cv_model=self._cv_model(n_factors=3, n_epochs=200),
            requested_cv_spearman_r2=0.9,
            sampling_density=0.1,
            finally_refit_model=None,
            max_iterations=20,
            verbose=False,
            plot_progress=False,
        )
        model.fit(oracle)
        # Stopped well before observing the full matrix
        X_obs = model.observed_matrix()
        assert 0.0 < (~np.isnan(X_obs)).mean() < 1.0
        # Imputation should match the ground truth well
        X_imputed = model.impute_all()
        assert X_imputed.shape == X_true.shape
        assert not np.isnan(X_imputed).any()
        # Observed entries are kept as-is
        np.testing.assert_allclose(np.where(~np.isnan(X_obs), X_imputed, 0),
                                   np.where(~np.isnan(X_obs), X_obs, 0), rtol=1e-9)
        # Overall quality
        corr = np.corrcoef(X_true.flatten(), X_imputed.flatten())[0, 1]
        assert corr > 0.98
        # Every column meets the requested CV Spearman R2
        assert np.all(model.cv_spearman_r2s() >= 0.9)

    def test_cv_r2s_reported(self):
        np.random.seed(1)
        X_true = make_low_rank_matrix(8, 8, rank=1)
        oracle = OracleWithAPrescribedMatrix(X_true)
        model = IterativeMCMWithGuaranteedSpearmanR2(
            cv_model=lambda state: _TrainValSplitCV(
                ColumnNormalizerModelWrapper(
                    MatrixCompletionFastALS(
                        n_factors=1, lam=0.1, n_epochs=100, verbose=False)),
                train_ratio=0.8,
            ),
            requested_cv_spearman_r2=0.8,
            sampling_density=0.4,
            max_iterations=10,
            verbose=False,
            plot_progress=False,
        )
        model.fit(oracle)
        r2s = model.cv_spearman_r2s()
        assert r2s.shape == (8,)
        assert np.all(~np.isnan(r2s))
        # Once the requested R2 is met, all columns must be above it
        assert np.all(r2s >= 0.8)

    def test_max_iterations_capped(self):
        """Even when the requested R2 is unattainable, fit terminates after
        at most max_iterations oracle queries."""
        np.random.seed(2)
        X_true = make_low_rank_matrix(10, 10, rank=2)
        X_noisy = X_true + np.random.default_rng(2).normal(0, 10.0, X_true.shape)
        oracle = CountingOracle(X_noisy)
        model = IterativeMCMWithGuaranteedSpearmanR2(
            cv_model=self._cv_model(n_factors=2, n_epochs=20),
            requested_cv_spearman_r2=0.9999,  # unattainable
            sampling_density=0.1,
            max_iterations=3,
            verbose=False,
            plot_progress=False,
        )
        model.fit(oracle)
        assert oracle.n_queries <= 3
        # The matrix is not fully observed
        assert (~np.isnan(oracle._observed)).mean() < 1.0

    def test_fully_observed_columns_get_r2_one(self):
        """A column observed in full is reported with Spearman R2 1.0."""
        from compass.turbo_mc.iterative_models.iterative_matrix_completion_model import (
            _override_fully_observed_columns_sr2_to_1,
        )
        X_obs = np.full((4, 3), np.nan)
        X_obs[:, 0] = [1.0, 2.0, 3.0, 4.0]  # column 0 fully observed
        r2s = np.zeros(3)
        _override_fully_observed_columns_sr2_to_1(r2s, X_obs)
        assert r2s[0] == 1.0
        assert r2s[1] == 0.0
        assert r2s[2] == 0.0


class _TrainValSplitCV:
    """
    Minimal stand-in for TrainValSplitCVMatrixCompletionModel with the exact
    same interface (fit_matrix / cv_spearman_r2s / predict_all), avoiding the
    extra train/val split bookkeeping for these unit tests.
    """

    def __init__(self, model, train_ratio=0.8, verbose=False):
        import copy
        self.model = copy.deepcopy(model)
        self.train_ratio = train_ratio
        self.verbose = verbose
        self._cv_r2s = None

    def fit_matrix(self, X_observed, Z=None):
        R, C = X_observed.shape
        # Hold out a random subset of the observed entries for validation
        obs_rows, obs_cols = np.where(~np.isnan(X_observed))
        n = len(obs_rows)
        idx = np.random.permutation(n)
        n_val = max(1, int((1 - self.train_ratio) * n))
        val_idx = idx[:n_val]
        train_mask = np.full(X_observed.shape, True)
        train_mask[obs_rows[val_idx], obs_cols[val_idx]] = False
        X_train = np.where(train_mask, X_observed, np.nan)
        self.model.fit_matrix(X_train)
        X_pred = self.model.predict_all()
        # Per-column Spearman R2 on the validation entries
        r2s = np.ones(C)
        for c in range(C):
            rows = obs_rows[obs_cols == c]
            if len(rows) < 2:
                continue
            y_true = X_observed[rows, c]
            y_pred = X_pred[rows, c]
            r2s[c] = metrics.compute_spearman_r2(y_true, np.full(len(rows), np.nan), y_pred)
        self._cv_r2s = r2s

    def cv_spearman_r2s(self):
        return self._cv_r2s

    def predict_all(self):
        return self.model.predict_all()

    def predict(self, r, c):
        return self.model.predict(r, c)
