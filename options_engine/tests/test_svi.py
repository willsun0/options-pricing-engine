"""Tests for the SVI parametric smile fit.

The central difficulty in testing a calibration routine is that there is no
"correct answer" to compare against on real data. The approach here is to
generate quotes *from* a known SVI slice and check the fit recovers it, then to
check the structural properties (asymptotics, arbitrage conditions, invariances)
that must hold for any parameters at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.vol_surface.svi import (
    MIN_TOTAL_VARIANCE,
    SVIFitResult,
    SVIParameters,
    durrleman_function,
    fit_svi_slice,
    fit_svi_surface,
    is_butterfly_free,
    svi_first_derivative,
    svi_second_derivative,
    svi_surface_volatility,
    svi_total_variance,
    worst_calendar_gap,
)

# A well-behaved equity-like slice used as the ground truth throughout: negative
# skew, vertex near the money, one year to expiry, ~20% at-the-money vol.
TRUE_PARAMETERS = SVIParameters(
    a=0.02, b=0.10, rho=-0.40, m=0.00, sigma=0.15, time_to_expiry=1.0
)


@pytest.fixture(scope="module")
def sample_grid() -> np.ndarray:
    """A realistic spread of log-moneyness for a one-year equity slice."""
    return np.linspace(-0.40, 0.25, 40)


class TestTotalVarianceFunction:
    """The functional form itself, independent of any fitting."""

    def test_matches_hand_computed_value(self) -> None:
        """Check w(k) against the formula evaluated by hand at a single point."""
        # w(0.1) = a + b*(rho*(0.1-m) + sqrt((0.1-m)^2 + sigma^2))
        #        = 0.02 + 0.10*(-0.40*0.10 + sqrt(0.01 + 0.0225))
        expected = 0.02 + 0.10 * (-0.40 * 0.10 + np.sqrt(0.10**2 + 0.15**2))
        actual = svi_total_variance(0.1, 0.02, 0.10, -0.40, 0.0, 0.15)
        assert actual == pytest.approx(expected, rel=1e-14)

    def test_minimum_is_at_the_predicted_vertex(self) -> None:
        """The analytic vertex must actually be the minimum of w.

        Differentiating gives k* = m - rho*sigma/sqrt(1-rho^2) with
        w(k*) = a + b*sigma*sqrt(1-rho^2). Both halves are checked: the location
        by a dense scan, the value by the closed form.
        """
        p = TRUE_PARAMETERS
        vertex = p.m - p.rho * p.sigma / np.sqrt(1.0 - p.rho**2)

        grid = np.linspace(vertex - 2.0, vertex + 2.0, 200_001)
        values = p.total_variance(grid)
        assert grid[np.argmin(values)] == pytest.approx(vertex, abs=1e-4)
        assert values.min() == pytest.approx(p.minimum_total_variance(), rel=1e-9)

    def test_wings_are_asymptotically_linear(self) -> None:
        """As |k| grows, w must approach a straight line with slope b(rho +/- 1).

        This is the property Lee's moment formula demands of any admissible
        smile, and it is why SVI is the parametrisation that stuck. Checked far
        out where the sqrt has effectively linearised.
        """
        p = TRUE_PARAMETERS
        far = 5_000.0
        right_slope = (p.total_variance(far + 1.0) - p.total_variance(far)) / 1.0
        left_slope = (p.total_variance(-far + 1.0) - p.total_variance(-far)) / 1.0

        assert right_slope == pytest.approx(p.b * (1.0 + p.rho), abs=1e-6)
        assert left_slope == pytest.approx(p.b * (p.rho - 1.0), abs=1e-6)

    def test_derivatives_match_finite_differences(self, sample_grid: np.ndarray) -> None:
        """The closed-form w' and w'' must agree with numerical differentiation.

        The bumps follow the Phase 1 finite-difference analysis: h ~ eps^(1/3)
        for the first derivative and eps^(1/4) for the second, because the second
        divides by h^2 and so amplifies roundoff far more aggressively.
        """
        args = (TRUE_PARAMETERS.a, TRUE_PARAMETERS.b, TRUE_PARAMETERS.rho,
                TRUE_PARAMETERS.m, TRUE_PARAMETERS.sigma)

        h_first = 1e-6
        numerical_first = (
            svi_total_variance(sample_grid + h_first, *args)
            - svi_total_variance(sample_grid - h_first, *args)
        ) / (2.0 * h_first)
        np.testing.assert_allclose(
            svi_first_derivative(sample_grid, *args), numerical_first, atol=1e-9
        )

        h_second = 1e-4
        numerical_second = (
            svi_total_variance(sample_grid + h_second, *args)
            - 2.0 * svi_total_variance(sample_grid, *args)
            + svi_total_variance(sample_grid - h_second, *args)
        ) / h_second**2
        np.testing.assert_allclose(
            svi_second_derivative(sample_grid, *args), numerical_second, atol=1e-6
        )

    def test_second_derivative_is_strictly_positive(self, sample_grid: np.ndarray) -> None:
        """Raw SVI is convex in total variance whenever b > 0 and sigma > 0."""
        args = (TRUE_PARAMETERS.a, TRUE_PARAMETERS.b, TRUE_PARAMETERS.rho,
                TRUE_PARAMETERS.m, TRUE_PARAMETERS.sigma)
        assert np.all(svi_second_derivative(sample_grid, *args) > 0.0)

    def test_zero_b_gives_a_flat_smile(self, sample_grid: np.ndarray) -> None:
        """b = 0 collapses SVI to constant variance — the Black-Scholes case.

        Worth pinning down explicitly: it is the sanity check that the whole
        Phase 5 exercise is a *generalisation* of Phase 1 rather than a different
        model. Setting the wing slope to zero recovers a horizontal smile.
        """
        flat = SVIParameters(a=0.04, b=0.0, rho=-0.5, m=0.1, sigma=0.2, time_to_expiry=1.0)
        vols = flat.implied_volatility(sample_grid)
        np.testing.assert_allclose(vols, 0.2, rtol=1e-12)


class TestSlopeAndSkew:
    """The at-the-money quantities a trader actually reads off the slice."""

    def test_atm_volatility_matches_direct_evaluation(self) -> None:
        """atm_volatility() is just the slice evaluated at k = 0."""
        p = TRUE_PARAMETERS
        assert p.atm_volatility() == pytest.approx(float(p.implied_volatility(0.0)))
        assert p.atm_volatility() == pytest.approx(
            np.sqrt(p.total_variance(0.0) / p.time_to_expiry)
        )

    def test_atm_skew_matches_finite_difference(self) -> None:
        """The reported skew must be the actual slope of implied vol at k = 0."""
        p = TRUE_PARAMETERS
        h = 1e-6
        numerical = float(p.implied_volatility(h) - p.implied_volatility(-h)) / (2.0 * h)
        assert p.atm_skew() == pytest.approx(numerical, rel=1e-6)

    def test_negative_rho_gives_negative_atm_skew_at_a_centred_vertex(self) -> None:
        """With the vertex at the money, the sign of rho *is* the sign of the skew.

        This is the case where the textbook reading of rho holds. The module
        docstring warns it fails when the fitted vertex leaves the quoted range;
        this test pins the regime where it is safe.
        """
        for rho in (-0.8, -0.4, -0.1):
            p = SVIParameters(a=0.02, b=0.1, rho=rho, m=0.0, sigma=0.15, time_to_expiry=1.0)
            assert p.atm_skew() < 0.0
        for rho in (0.1, 0.4, 0.8):
            p = SVIParameters(a=0.02, b=0.1, rho=rho, m=0.0, sigma=0.15, time_to_expiry=1.0)
            assert p.atm_skew() > 0.0


class TestButterflyCondition:
    """Durrleman's g(k), which decides whether the implied density is valid."""

    def test_well_behaved_slice_is_butterfly_free(self) -> None:
        """A modest equity-like slice should pass across a wide range."""
        assert is_butterfly_free(TRUE_PARAMETERS)
        assert np.all(durrleman_function(TRUE_PARAMETERS, np.linspace(-1.5, 1.5, 601)) >= 0.0)

    def test_excessive_slope_creates_butterfly_arbitrage(self) -> None:
        """Cranking b past the admissible region must be detected.

        A large b with |rho| near 1 makes one wing extremely steep, which is
        exactly the configuration Lee's bound rules out and which produces a
        negative density. If the check cannot catch this, it catches nothing.
        """
        bad = SVIParameters(a=0.01, b=1.2, rho=-0.95, m=0.0, sigma=0.05, time_to_expiry=1.0)
        assert not is_butterfly_free(bad)

    def test_flat_slice_is_butterfly_free(self) -> None:
        """Constant variance is the Black-Scholes case: lognormal, so valid.

        g(k) reduces to 1 - 0 - 0 + 0 = 1 when w' = w'' = 0, so this also pins
        down the normalisation of the Durrleman expression.
        """
        flat = SVIParameters(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2, time_to_expiry=1.0)
        np.testing.assert_allclose(
            durrleman_function(flat, np.linspace(-1.0, 1.0, 51)), 1.0, rtol=1e-12
        )
        assert is_butterfly_free(flat)


class TestSliceCalibration:
    """Fitting: does it recover parameters it should, and fail where it should?"""

    def test_recovers_parameters_from_noiseless_quotes(self, sample_grid: np.ndarray) -> None:
        """Given exact quotes from a known slice, the fit must reproduce it.

        The tolerance is on the fitted *curve*, not on the five parameters. SVI
        is mildly over-parameterised — different (a, b, m) combinations trace
        nearly the same curve over a finite strike range — so demanding parameter
        equality would be testing the optimiser's tie-breaking, not the fit.
        What matters is that the curve matches.
        """
        vols = TRUE_PARAMETERS.implied_volatility(sample_grid)
        result = fit_svi_slice(sample_grid, vols, TRUE_PARAMETERS.time_to_expiry)

        assert result.rmse < 1e-4
        np.testing.assert_allclose(
            result.parameters.implied_volatility(sample_grid), vols, atol=5e-4
        )
        assert result.butterfly_free

    def test_recovers_flat_smile_with_near_zero_slope(self, sample_grid: np.ndarray) -> None:
        """Constant quotes must fit as a flat slice, i.e. b*(...) contributing ~0."""
        vols = np.full_like(sample_grid, 0.20)
        result = fit_svi_slice(sample_grid, vols, 1.0)

        assert result.rmse < 1e-5
        np.testing.assert_allclose(
            result.parameters.implied_volatility(sample_grid), 0.20, atol=1e-4
        )

    def test_fit_is_deterministic(self, sample_grid: np.ndarray) -> None:
        """The multi-start grid is fixed, so repeated fits must be identical.

        This is why the starting points are a hard-coded grid rather than random
        draws: a calibration that returns different parameters on each run is
        impossible to debug and impossible to check into a test suite.
        """
        vols = TRUE_PARAMETERS.implied_volatility(sample_grid) + 0.002 * np.sin(
            10.0 * sample_grid
        )
        first = fit_svi_slice(sample_grid, vols, 1.0)
        second = fit_svi_slice(sample_grid, vols, 1.0)
        assert first.parameters.as_dict() == second.parameters.as_dict()

    def test_noise_degrades_fit_gracefully(self, sample_grid: np.ndarray) -> None:
        """Adding bid-ask-scale noise should raise RMSE to roughly the noise level.

        A five-parameter curve cannot chase 40 independent perturbations, so it
        averages through them — the fit error should land near the noise standard
        deviation rather than near zero (which would mean overfitting) or far
        above it (which would mean the optimiser failed).
        """
        rng = np.random.default_rng(12345)
        noise_scale = 0.004
        vols = TRUE_PARAMETERS.implied_volatility(sample_grid) + rng.normal(
            0.0, noise_scale, sample_grid.size
        )
        result = fit_svi_slice(sample_grid, vols, 1.0)
        assert 0.4 * noise_scale < result.rmse < 1.6 * noise_scale

    def test_weights_shift_the_fit_toward_the_weighted_points(
        self, sample_grid: np.ndarray
    ) -> None:
        """Heavily weighting the left wing must reduce the error there."""
        vols = TRUE_PARAMETERS.implied_volatility(sample_grid) + 0.01 * np.sin(
            8.0 * sample_grid
        )
        left = sample_grid < -0.15

        unweighted = fit_svi_slice(sample_grid, vols, 1.0)
        weights = np.where(left, 100.0, 1.0)
        weighted = fit_svi_slice(sample_grid, vols, 1.0, weights)

        def left_error(result) -> float:
            errors = result.parameters.implied_volatility(sample_grid[left]) - vols[left]
            return float(np.sqrt(np.mean(errors**2)))

        assert left_error(weighted) < left_error(unweighted)

    def test_fitted_slice_respects_the_static_constraints(
        self, sample_grid: np.ndarray
    ) -> None:
        """The constrained optimiser must return parameters inside the valid region."""
        rng = np.random.default_rng(7)
        vols = TRUE_PARAMETERS.implied_volatility(sample_grid) + rng.normal(
            0.0, 0.003, sample_grid.size
        )
        p = fit_svi_slice(sample_grid, vols, 1.0).parameters

        assert p.b >= 0.0
        assert abs(p.rho) < 1.0
        assert p.sigma > 0.0
        assert p.minimum_total_variance() >= -1e-9  # no negative variance
        assert p.b * (1.0 + abs(p.rho)) <= 4.0 / p.time_to_expiry + 1e-9  # Lee's bound

    def test_total_variance_stays_positive_across_the_real_line(
        self, sample_grid: np.ndarray
    ) -> None:
        """A fitted slice must not imply negative variance anywhere, not just on data."""
        vols = TRUE_PARAMETERS.implied_volatility(sample_grid)
        p = fit_svi_slice(sample_grid, vols, 1.0).parameters
        wide = np.linspace(-10.0, 10.0, 5001)
        assert np.all(p.total_variance(wide) >= -MIN_TOTAL_VARIANCE)


class TestSliceCalibrationValidation:
    """Input validation: bad data should raise, not silently produce parameters."""

    def test_rejects_too_few_quotes(self) -> None:
        """Fewer than five quotes cannot identify five parameters."""
        with pytest.raises(ValueError, match="at least 5 quotes"):
            fit_svi_slice(np.array([0.0, 0.1, 0.2, 0.3]), np.full(4, 0.2), 1.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            fit_svi_slice(np.linspace(-0.2, 0.2, 10), np.full(9, 0.2), 1.0)

    def test_rejects_non_positive_expiry(self) -> None:
        with pytest.raises(ValueError, match="time_to_expiry must be positive"):
            fit_svi_slice(np.linspace(-0.2, 0.2, 10), np.full(10, 0.2), 0.0)

    def test_rejects_non_positive_volatility(self) -> None:
        vols = np.full(10, 0.2)
        vols[3] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            fit_svi_slice(np.linspace(-0.2, 0.2, 10), vols, 1.0)

    def test_rejects_non_finite_input(self) -> None:
        vols = np.full(10, 0.2)
        vols[2] = np.nan
        with pytest.raises(ValueError, match="finite"):
            fit_svi_slice(np.linspace(-0.2, 0.2, 10), vols, 1.0)

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            fit_svi_slice(
                np.linspace(-0.2, 0.2, 10),
                np.full(10, 0.2),
                1.0,
                weights=np.concatenate([np.ones(9), [-1.0]]),
            )


class TestSurfaceCalibration:
    """Fitting many slices, and interpolating between them."""

    @staticmethod
    def _synthetic_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build quotes from three slices whose ATM vol rises with expiry."""
        k_all, vol_all, expiry_all = [], [], []
        for expiry, a in [(0.25, 0.008), (0.50, 0.018), (1.00, 0.040)]:
            k = np.linspace(-0.3, 0.2, 25)
            p = SVIParameters(a=a, b=0.09, rho=-0.5, m=0.0, sigma=0.14, time_to_expiry=expiry)
            k_all.append(k)
            vol_all.append(p.implied_volatility(k))
            expiry_all.append(np.full(k.size, expiry))
        return (
            np.concatenate(k_all),
            np.concatenate(vol_all),
            np.concatenate(expiry_all),
        )

    def test_fits_one_slice_per_expiry(self) -> None:
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)

        assert sorted(fits) == [0.25, 0.50, 1.00]
        assert all(fit.rmse < 1e-3 for fit in fits.values())

    def test_skips_slices_with_too_few_quotes(self) -> None:
        """A thin expiry should be dropped, not fitted with three points."""
        k, vols, expiries = self._synthetic_surface()
        k = np.append(k, [0.0, 0.01])
        vols = np.append(vols, [0.2, 0.2])
        expiries = np.append(expiries, [2.0, 2.0])

        fits = fit_svi_surface(k, vols, expiries)
        assert 2.0 not in fits

    def test_rejects_mismatched_surface_input(self) -> None:
        with pytest.raises(ValueError, match="must all match"):
            fit_svi_surface(np.zeros(10), np.full(10, 0.2), np.ones(9))

    def test_interpolation_reproduces_the_fitted_slices(self) -> None:
        """Evaluating at a fitted expiry must return that slice, not a blend."""
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)
        grid = np.linspace(-0.25, 0.15, 20)

        for expiry, fit in fits.items():
            np.testing.assert_allclose(
                svi_surface_volatility(fits, grid, expiry),
                fit.parameters.implied_volatility(grid),
                rtol=1e-12,
            )

    def test_interpolation_is_linear_in_total_variance(self) -> None:
        """Between two slices, w must be the linear blend — not sigma.

        This is the property that makes the interpolation defensible: total
        variance is what accumulates linearly in time. Testing it directly guards
        against the common bug of interpolating volatility instead, which would
        pass a smoke test and be quietly wrong.
        """
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)
        grid = np.linspace(-0.2, 0.1, 15)

        lower, upper = 0.25, 0.50
        target = 0.375  # exactly halfway
        weight = (target - lower) / (upper - lower)

        expected_variance = (1.0 - weight) * fits[lower].parameters.total_variance(grid) + (
            weight * fits[upper].parameters.total_variance(grid)
        )
        actual = svi_surface_volatility(fits, grid, target)
        np.testing.assert_allclose(actual**2 * target, expected_variance, rtol=1e-12)

    def test_extrapolation_holds_volatility_flat(self) -> None:
        """Past the last fitted expiry, the *volatility* is held constant.

        Scaling total variance by T/T_last is the same statement, and it is the
        deliberately dull choice documented in svi_surface_volatility: with no
        quotes beyond the last expiry there is nothing to extrapolate with.
        """
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)
        grid = np.linspace(-0.2, 0.1, 15)

        last = max(fits)
        np.testing.assert_allclose(
            svi_surface_volatility(fits, grid, 3.0),
            svi_surface_volatility(fits, grid, last),
            rtol=1e-12,
        )

    def test_rising_term_structure_has_no_calendar_arbitrage(self) -> None:
        """The synthetic slices have rising ATM variance, so w must not cross."""
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)
        assert worst_calendar_gap(fits, -0.3, 0.2) > 0.0

    def test_detects_crossing_slices(self) -> None:
        """A short expiry with more total variance than a long one must be caught.

        Constructed directly rather than fitted, because a calibration to sane
        data will not produce this — and a check that cannot flag an obviously
        arbitrageable pair is worthless.
        """
        crossing = {
            0.25: SVIFitResult(
                parameters=SVIParameters(
                    a=0.09, b=0.05, rho=-0.3, m=0.0, sigma=0.1, time_to_expiry=0.25
                ),
                rmse=0.0, max_absolute_error=0.0, n_quotes=10,
                butterfly_free=True, butterfly_free_extrapolated=True, success=True,
            ),
            1.00: SVIFitResult(
                parameters=SVIParameters(
                    a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.1, time_to_expiry=1.0
                ),
                rmse=0.0, max_absolute_error=0.0, n_quotes=10,
                butterfly_free=True, butterfly_free_extrapolated=True, success=True,
            ),
        }
        assert worst_calendar_gap(crossing, -0.3, 0.2) < 0.0

    def test_calendar_gap_needs_two_slices(self) -> None:
        """One slice cannot cross anything, so the check must not fail on it."""
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)
        single = {min(fits): fits[min(fits)]}
        assert worst_calendar_gap(single, -0.3, 0.2) == float("inf")

    def test_calendar_gap_rejects_an_inverted_range(self) -> None:
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)
        with pytest.raises(ValueError, match="must be below"):
            worst_calendar_gap(fits, 0.2, -0.3)

    def test_surface_rejects_empty_fits_and_bad_expiry(self) -> None:
        k, vols, expiries = self._synthetic_surface()
        fits = fit_svi_surface(k, vols, expiries)

        with pytest.raises(ValueError, match="fits is empty"):
            svi_surface_volatility({}, np.array([0.0]), 1.0)
        with pytest.raises(ValueError, match="must be positive"):
            svi_surface_volatility(fits, np.array([0.0]), 0.0)
