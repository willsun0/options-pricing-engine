"""Tests for the neural network volatility surface.

Testing a fitted network needs a different tactic from testing a pricer. There is
no closed form to compare against and the weights are not interpretable, so
asserting on individual parameters is meaningless. What *can* be pinned down:

* **Determinism.** Same seed, same answer, bit for bit. Without this nothing else
  in the file is a test.
* **Structural guarantees.** Positivity of the output, correct shapes, the scaler
  fitted on training rows only, round-tripping through save/load.
* **Measurable claims from the module docstring.** The two substantive ones are
  that a smooth activation yields a smooth surface and that extrapolation
  degrades sharply. Both are asserted here so that if a future change breaks
  them, the docstring stops being a lie.

Training runs here use deliberately small nets and short schedules — these test
the machinery, not the quality of the production fit, which the Phase 5 notebook
reports.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch",
    reason="Phase 5 needs PyTorch: pip install -r requirements.txt",
)

from options_engine.vol_surface.ml_surface import (  # noqa: E402
    FEATURE_NAMES,
    Activation,
    FeatureScaler,
    TrainingConfig,
    VolatilityNet,
    VolatilitySurfaceModel,
    build_features,
    extrapolation_report,
    train_surface_model,
)
from options_engine.vol_surface.svi import SVIParameters  # noqa: E402

# A short schedule that still converges on the smooth synthetic surface below.
# The production defaults train ~20x longer; using them here would make the suite
# take minutes to assert things that are already visible in seconds.
FAST_CONFIG = TrainingConfig(hidden_sizes=(16, 16), epochs=1500, patience=400)


def synthetic_surface(
    n_strikes: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a clean, arbitrage-free surface from three known SVI slices.

    Using SVI to generate the data is deliberate: it is a smooth surface with
    realistic skew and term structure, and it is noiseless, so any error the
    network shows is its own rather than the data's.

    Args:
        n_strikes: Strikes per expiry.

    Returns:
        A ``(features, implied_vols, log_moneyness)`` tuple.
    """
    forward = 100.0
    features, vols, log_moneyness = [], [], []
    for expiry, a in [(0.25, 0.008), (0.50, 0.018), (1.00, 0.040)]:
        k = np.linspace(-0.35, 0.25, n_strikes)
        slice_ = SVIParameters(
            a=a, b=0.09, rho=-0.5, m=0.0, sigma=0.14, time_to_expiry=expiry
        )
        strikes = forward * np.exp(k)
        features.append(
            build_features(strikes, np.full(k.size, expiry), strikes / forward)
        )
        vols.append(slice_.implied_volatility(k))
        log_moneyness.append(k)
    return (
        np.vstack(features),
        np.concatenate(vols),
        np.concatenate(log_moneyness),
    )


@pytest.fixture(scope="module")
def surface_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The synthetic surface, built once for the whole module."""
    return synthetic_surface()


@pytest.fixture(scope="module")
def trained(surface_data) -> tuple[VolatilitySurfaceModel, object]:
    """A model trained once and reused, since training dominates runtime."""
    features, vols, _ = surface_data
    return train_surface_model(features, vols, FAST_CONFIG)


class TestFeatureConstruction:
    """The feature matrix, whose column order everything downstream assumes."""

    def test_columns_are_in_the_declared_order(self) -> None:
        """Columns must match FEATURE_NAMES, or training and prediction disagree."""
        strikes = np.array([90.0, 100.0, 110.0])
        expiries = np.array([0.1, 0.2, 0.3])
        moneyness = np.array([0.9, 1.0, 1.1])

        features = build_features(strikes, expiries, moneyness)
        assert features.shape == (3, len(FEATURE_NAMES))
        np.testing.assert_array_equal(features[:, 0], strikes)
        np.testing.assert_array_equal(features[:, 1], expiries)
        np.testing.assert_array_equal(features[:, 2], moneyness)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            build_features(np.zeros(3), np.zeros(4), np.zeros(3))

    def test_rejects_two_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            build_features(np.zeros((3, 1)), np.zeros(3), np.zeros(3))


class TestFeatureScaler:
    """Standardisation, and the leakage it is easy to introduce."""

    def test_transform_produces_zero_mean_unit_variance(self) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(5.0, 3.0, size=(200, 3))
        scaled = FeatureScaler.fit(features).transform(features)

        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, rtol=1e-12)

    def test_constant_column_does_not_divide_by_zero(self) -> None:
        """A single-expiry chain has a constant column; it must not produce NaN."""
        features = np.column_stack([np.arange(10.0), np.full(10, 0.25), np.arange(10.0)])
        scaled = FeatureScaler.fit(features).transform(features)

        assert np.all(np.isfinite(scaled))
        np.testing.assert_allclose(scaled[:, 1], 0.0)

    def test_scaler_is_fitted_on_training_rows_only(self, surface_data) -> None:
        """Scaler statistics must not reflect the validation rows.

        Fitting the scaler on all the data before splitting is the classic silent
        leak: it never errors and it makes every validation number look slightly
        better than it is. The check is that the model's scaler differs from one
        fitted on the full dataset.
        """
        features, vols, _ = surface_data
        model, _ = train_surface_model(features, vols, FAST_CONFIG)

        full_data_scaler = FeatureScaler.fit(features)
        assert not np.allclose(model.scaler.mean, full_data_scaler.mean)


class TestNetwork:
    """The module itself, before any training."""

    def test_output_shape_is_one_dimensional(self) -> None:
        net = VolatilityNet(n_features=3, hidden_sizes=(8, 8))
        output = net(torch.zeros(7, 3))
        assert output.shape == (7,)

    def test_output_is_always_positive(self) -> None:
        """Softplus guarantees positive volatility even for absurd inputs.

        The extreme values matter: this is exactly the extrapolation regime where
        an unconstrained linear head returns negative volatilities.
        """
        net = VolatilityNet(n_features=3, hidden_sizes=(8, 8))
        extreme = torch.tensor([[-1e4, -1e4, -1e4], [1e4, 1e4, 1e4], [0.0, 0.0, 0.0]])
        with torch.no_grad():
            assert torch.all(net(extreme) > 0.0)

    def test_rejects_empty_or_invalid_hidden_sizes(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            VolatilityNet(hidden_sizes=())
        with pytest.raises(ValueError, match="must be positive"):
            VolatilityNet(hidden_sizes=(8, 0))

    def test_activation_smoothness_flags(self) -> None:
        """is_smooth must single out ReLU, since that flag drives the docstring claim."""
        assert Activation.TANH.is_smooth
        assert Activation.SILU.is_smooth
        assert not Activation.RELU.is_smooth


class TestTraining:
    """Convergence, reproducibility, and early stopping."""

    def test_fits_a_smooth_surface_closely(self, trained) -> None:
        """On noiseless SVI data the network should get within a vol point."""
        model, history = trained
        assert history.best_validation_rmse() < 0.01

    def test_training_loss_decreases(self, trained) -> None:
        _, history = trained
        assert history.train_loss[-1] < history.train_loss[0]

    def test_same_seed_reproduces_predictions_exactly(self, surface_data) -> None:
        """Two runs with the same seed must agree bit for bit.

        Everything else in this file depends on it: a flaky model makes every
        threshold below a coin flip.
        """
        features, vols, _ = surface_data
        first, _ = train_surface_model(features, vols, FAST_CONFIG)
        second, _ = train_surface_model(features, vols, FAST_CONFIG)
        np.testing.assert_array_equal(first.predict(features), second.predict(features))

    def test_different_seeds_give_different_models(self, surface_data) -> None:
        """The seed must actually reach initialisation and the split."""
        features, vols, _ = surface_data
        first, _ = train_surface_model(features, vols, FAST_CONFIG)
        other_config = TrainingConfig(
            hidden_sizes=FAST_CONFIG.hidden_sizes,
            epochs=FAST_CONFIG.epochs,
            patience=FAST_CONFIG.patience,
            seed=99,
        )
        second, _ = train_surface_model(features, vols, other_config)
        assert not np.allclose(first.predict(features), second.predict(features))

    def test_no_early_stop_on_noiseless_data(self, surface_data) -> None:
        """Clean data has nothing to overfit, so validation loss keeps improving.

        Worth asserting rather than treating as a nuisance: it confirms early
        stopping is responding to *overfitting* and not to some artefact of the
        loop. The synthetic surface is exact SVI output, so the network can chase
        it indefinitely and validation loss falls monotonically to ~1e-7.
        """
        features, vols, _ = surface_data
        config = TrainingConfig(hidden_sizes=(16, 16), epochs=800, patience=50, seed=3)
        _, history = train_surface_model(features, vols, config)

        assert not history.stopped_early
        assert history.best_epoch == len(history.train_loss) - 1

    def test_early_stopping_restores_the_best_epoch(self, surface_data) -> None:
        """With noisy data, training must halt early and return the best weights.

        Noise is essential to this test: it is what creates a validation minimum
        to stop at. Without restoring the best state, early stopping would only
        decide *when* to halt and still return weights that had already begun
        memorising the noise — so the assertion recomputes the returned model's
        validation loss and requires it to equal the best one recorded.
        """
        features, vols, _ = surface_data
        rng = np.random.default_rng(11)
        noisy_vols = vols + rng.normal(0.0, 0.01, vols.size)

        config = TrainingConfig(hidden_sizes=(32, 32), epochs=6000, patience=100, seed=3)
        model, history = train_surface_model(features, noisy_vols, config)

        assert history.stopped_early
        # The best epoch must precede the last by roughly the patience window,
        # which is what "stopped after no improvement" means.
        assert history.best_epoch == len(history.train_loss) - 1 - config.patience

        # Recompute the validation loss of the returned weights and compare.
        split_rng = np.random.default_rng(config.seed)
        permutation = split_rng.permutation(features.shape[0])
        n_validation = int(round(config.validation_fraction * features.shape[0]))
        validation_index = permutation[:n_validation]

        errors = model.predict(features[validation_index]) - noisy_vols[validation_index]
        assert float(np.mean(errors**2)) == pytest.approx(
            min(history.validation_loss), rel=1e-5
        )

    def test_training_without_validation_split(self, surface_data) -> None:
        """validation_fraction = 0 must train on everything and not crash."""
        features, vols, _ = surface_data
        config = TrainingConfig(
            hidden_sizes=(16, 16), epochs=300, validation_fraction=0.0, patience=None
        )
        model, history = train_surface_model(features, vols, config)

        assert history.validation_loss == []
        assert np.isnan(history.best_validation_rmse())
        assert model.predict(features).shape == (features.shape[0],)

    def test_predictions_are_positive(self, trained, surface_data) -> None:
        model, _ = trained
        features, _, _ = surface_data
        assert np.all(model.predict(features) > 0.0)


class TestTrainingValidation:
    """Bad input must raise rather than train on nonsense."""

    def test_rejects_one_dimensional_features(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            train_surface_model(np.zeros(10), np.full(10, 0.2), FAST_CONFIG)

    def test_rejects_mismatched_targets(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D of length"):
            train_surface_model(np.zeros((10, 3)), np.full(9, 0.2), FAST_CONFIG)

    def test_rejects_non_finite_data(self) -> None:
        features = np.zeros((10, 3))
        features[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            train_surface_model(features, np.full(10, 0.2), FAST_CONFIG)

    def test_rejects_non_positive_volatility(self) -> None:
        vols = np.full(10, 0.2)
        vols[1] = -0.1
        with pytest.raises(ValueError, match="strictly positive"):
            train_surface_model(np.zeros((10, 3)), vols, FAST_CONFIG)

    def test_rejects_invalid_validation_fraction(self) -> None:
        config = TrainingConfig(validation_fraction=1.0)
        with pytest.raises(ValueError, match=r"validation_fraction must be in \[0, 1\)"):
            train_surface_model(np.zeros((10, 3)), np.full(10, 0.2), config)


class TestPersistence:
    """Saving and loading, including the security posture of the load path."""

    def test_round_trip_preserves_predictions(self, trained, surface_data, tmp_path) -> None:
        model, _ = trained
        features, _, _ = surface_data

        path = tmp_path / "surface.pt"
        model.save(path)
        restored = VolatilitySurfaceModel.load(path)

        np.testing.assert_allclose(restored.predict(features), model.predict(features))
        np.testing.assert_allclose(restored.scaler.mean, model.scaler.mean)
        np.testing.assert_allclose(restored.training_hull, model.training_hull)
        assert restored.config == model.config

    def test_checkpoint_loads_under_weights_only(self, trained, tmp_path) -> None:
        """The payload must contain nothing that requires arbitrary unpickling.

        torch.load defaults to executing pickle, which can run arbitrary code.
        The save format stores only tensors and primitives so the load path can
        pass weights_only=True; this asserts that stays true, since adding one
        convenient dataclass to the payload would silently break it.
        """
        model, _ = trained
        path = tmp_path / "weights_only.pt"
        model.save(path)

        payload = torch.load(path, weights_only=True)
        assert set(payload) >= {"state_dict", "mean", "scale", "training_hull"}


class TestTrainingRange:
    """The bounding box used to flag extrapolation."""

    def test_training_points_are_inside_the_hull(self, trained, surface_data) -> None:
        model, _ = trained
        features, _, _ = surface_data
        assert np.all(model.is_in_training_range(features))

    def test_points_beyond_the_hull_are_flagged(self, trained, surface_data) -> None:
        model, _ = trained
        features, _, _ = surface_data
        beyond = features.max(axis=0) * 2.0 + 1.0
        assert not model.is_in_training_range(beyond[None, :])[0]


class TestDocstringClaims:
    """The two substantive claims the module docstring makes, asserted directly."""

    @staticmethod
    def _curvature(model: VolatilitySurfaceModel, expiry: float = 0.5) -> float:
        """Peak |d^2 sigma / dk^2| along one expiry slice, by finite differences."""
        forward = 100.0
        k = np.linspace(-0.3, 0.2, 400)
        strikes = forward * np.exp(k)
        vols = model.predict(
            build_features(strikes, np.full(k.size, expiry), strikes / forward)
        )
        return float(np.max(np.abs(np.diff(vols, 2)))) / (k[1] - k[0]) ** 2

    def test_relu_is_far_less_smooth_than_tanh(self, surface_data) -> None:
        """The headline architectural claim: ReLU kinks, tanh does not.

        The threshold is deliberately loose (5x) while the measured gap on real
        SPY data is ~15x. The point is to catch a regression that silently swaps
        the default activation, not to pin an exact ratio that depends on the
        seed and the data.
        """
        features, vols, _ = surface_data
        smooth_model, _ = train_surface_model(features, vols, FAST_CONFIG)
        kinked_model, _ = train_surface_model(
            features,
            vols,
            TrainingConfig(
                hidden_sizes=FAST_CONFIG.hidden_sizes,
                epochs=FAST_CONFIG.epochs,
                patience=FAST_CONFIG.patience,
                activation=Activation.RELU,
            ),
        )
        assert self._curvature(kinked_model) > 5.0 * self._curvature(smooth_model)

    def test_extrapolation_is_much_worse_than_interpolation(self, surface_data) -> None:
        """The headline limitation: the wings are not predicted, they are invented.

        Measured on real SPY quotes at 17.8x; asserted at 3x here because the
        synthetic surface is noiseless and smooth, which is the easiest possible
        case for extrapolation. That it still degrades several-fold on easy data
        is the point.
        """
        features, vols, log_moneyness = surface_data
        report = extrapolation_report(features, vols, log_moneyness, config=FAST_CONFIG)

        assert report.n_interior > 0
        assert report.n_exterior > 0
        assert report.degradation_factor > 3.0
        assert report.band[0] < report.band[1]

    def test_extrapolation_report_rejects_invalid_quantile(self, surface_data) -> None:
        features, vols, log_moneyness = surface_data
        with pytest.raises(ValueError, match=r"quantile must be in \(0, 0.5\)"):
            extrapolation_report(
                features, vols, log_moneyness, quantile=0.6, config=FAST_CONFIG
            )
