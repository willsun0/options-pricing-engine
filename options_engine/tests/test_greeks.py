"""Tests for the option Greeks.

The centrepiece is the analytical-vs-numerical comparison: two completely
independent computations of the same derivative. The analytical version comes
from hand-differentiating the price formula; the numerical version only ever
calls the pricer. If they agree across a wide parameter grid, both the algebra
and the price formula are almost certainly right.

The rest of the suite pins down the things that comparison cannot catch — unit
conventions, sign conventions, and the structural identities (gamma and vega
being put-call symmetric, the parity relationships between Greeks).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from options_engine.common import OptionType
from options_engine.greeks import (
    DAYS_PER_YEAR,
    Greeks,
    all_greeks,
    delta,
    gamma,
    numerical_delta,
    numerical_gamma,
    numerical_greeks,
    numerical_rho,
    numerical_theta,
    numerical_vega,
    rho,
    theta,
    vega,
    rho_per_basis_point,
    theta_per_day,
    vega_per_percent,
)
from options_engine.pricing.black_scholes import black_scholes_price

# Grid for the analytical/numerical comparison. Times are kept above ~1 month so
# that the finite differences stay well conditioned: as T -> 0 the price surface
# develops a kink at the strike, gamma blows up, and *any* finite-difference
# scheme degrades. That is a property of the method, not a bug in the pricer, so
# testing it there would be testing the wrong thing.
COMPARISON_GRID = [
    (spot, strike, time, rate, vol, div)
    for spot in (80.0, 100.0, 125.0)
    for strike in (90.0, 100.0, 115.0)
    for time in (0.1, 0.5, 2.0)
    for rate in (0.0, 0.05)
    for vol in (0.15, 0.45)
    for div in (0.0, 0.03)
]

OPTION_TYPES = ["call", "put"]


# ---------------------------------------------------------------------------
# Analytical vs numerical: the core validation
# ---------------------------------------------------------------------------


class TestAnalyticalMatchesNumerical:
    """Hand-differentiated formulas must match finite differences on the pricer."""

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_delta(self, option_type, spot, strike, time, rate, vol, div):
        """Analytical delta matches the central difference on spot."""
        args = (spot, strike, time, rate, vol, option_type, div)
        assert float(delta(*args)) == pytest.approx(
            float(numerical_delta(*args)), abs=1e-7
        )

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_gamma(self, option_type, spot, strike, time, rate, vol, div):
        """Analytical gamma matches the second central difference on spot.

        Looser tolerance than the first-order Greeks by design: a second
        difference divides by h^2, so floating-point cancellation limits it to
        about sqrt(machine epsilon) ~ 1e-8 relative accuracy no matter how well
        the bump is chosen. See the note in greeks.py.
        """
        args = (spot, strike, time, rate, vol, option_type, div)
        assert float(gamma(*args)) == pytest.approx(
            float(numerical_gamma(*args)), rel=1e-5, abs=1e-9
        )

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_vega(self, option_type, spot, strike, time, rate, vol, div):
        """Analytical vega matches the central difference on volatility."""
        args = (spot, strike, time, rate, vol, option_type, div)
        assert float(vega(*args)) == pytest.approx(
            float(numerical_vega(*args)), rel=1e-6, abs=1e-7
        )

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_theta(self, option_type, spot, strike, time, rate, vol, div):
        """Analytical theta matches minus the central difference on time.

        This is the test that catches the classic theta sign error, since a
        flipped sign fails by a factor of -2 rather than by a small amount.
        """
        args = (spot, strike, time, rate, vol, option_type, div)
        assert float(theta(*args)) == pytest.approx(
            float(numerical_theta(*args)), rel=1e-6, abs=1e-6
        )

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_rho(self, option_type, spot, strike, time, rate, vol, div):
        """Analytical rho matches the central difference on the rate."""
        args = (spot, strike, time, rate, vol, option_type, div)
        assert float(rho(*args)) == pytest.approx(
            float(numerical_rho(*args)), rel=1e-6, abs=1e-7
        )

    def test_all_greeks_matches_numerical_greeks(self):
        """The bundled accessors agree field by field."""
        args = (105.0, 100.0, 0.75, 0.04, 0.28, "put", 0.015)
        analytic = all_greeks(*args)
        numeric = numerical_greeks(*args)
        for name, value in analytic.as_dict().items():
            assert float(value) == pytest.approx(
                float(numeric.as_dict()[name]), rel=1e-5, abs=1e-6
            ), f"{name} mismatch"


# ---------------------------------------------------------------------------
# Structural identities
# ---------------------------------------------------------------------------


class TestPutCallRelationships:
    """Greek identities that follow from differentiating put-call parity.

    Parity says C - P = S e^{-qT} - K e^{-rT}. Differentiating both sides with
    respect to each input gives an exact relationship between the call and put
    Greeks, which is a strong independent check on the sign conventions.
    """

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_delta_difference_is_dividend_discount(self, spot, strike, time, rate, vol, div):
        """d/dS of parity: Delta_call - Delta_put = e^{-qT}."""
        call_d = float(delta(spot, strike, time, rate, vol, "call", div))
        put_d = float(delta(spot, strike, time, rate, vol, "put", div))
        assert call_d - put_d == pytest.approx(np.exp(-div * time), abs=1e-12)

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_gamma_is_identical_for_calls_and_puts(self, spot, strike, time, rate, vol, div):
        """d2/dS2 of parity: the right-hand side is linear in S, so gammas match."""
        call_g = float(gamma(spot, strike, time, rate, vol, "call", div))
        put_g = float(gamma(spot, strike, time, rate, vol, "put", div))
        assert call_g == pytest.approx(put_g, abs=1e-15)

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_vega_is_identical_for_calls_and_puts(self, spot, strike, time, rate, vol, div):
        """d/dsigma of parity: the right-hand side has no sigma, so vegas match."""
        call_v = float(vega(spot, strike, time, rate, vol, "call", div))
        put_v = float(vega(spot, strike, time, rate, vol, "put", div))
        assert call_v == pytest.approx(put_v, abs=1e-15)

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_rho_difference_matches_parity(self, spot, strike, time, rate, vol, div):
        """d/dr of parity: Rho_call - Rho_put = K T e^{-rT}."""
        call_r = float(rho(spot, strike, time, rate, vol, "call", div))
        put_r = float(rho(spot, strike, time, rate, vol, "put", div))
        expected = strike * time * np.exp(-rate * time)
        assert call_r - put_r == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_theta_difference_matches_parity(self, spot, strike, time, rate, vol, div):
        """d/dt of parity: Theta_call - Theta_put = q S e^{-qT} - r K e^{-rT}.

        Careful with the sign: theta differentiates with respect to calendar time,
        so it is minus the derivative with respect to T. The right-hand side of
        parity differentiated by T is -q S e^{-qT} + r K e^{-rT}; negating gives
        the expression below.
        """
        call_t = float(theta(spot, strike, time, rate, vol, "call", div))
        put_t = float(theta(spot, strike, time, rate, vol, "put", div))
        expected = div * spot * np.exp(-div * time) - rate * strike * np.exp(-rate * time)
        assert call_t - put_t == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Signs, ranges, and shapes
# ---------------------------------------------------------------------------


class TestSignsAndRanges:
    """Every Greek's sign and admissible range."""

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_call_delta_in_unit_interval(self, spot, strike, time, rate, vol, div):
        """A call delta is a hedge ratio between 0 and e^{-qT} <= 1 shares."""
        value = float(delta(spot, strike, time, rate, vol, "call", div))
        assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_put_delta_in_negative_unit_interval(self, spot, strike, time, rate, vol, div):
        """A put delta is negative: the put gains when the stock falls."""
        value = float(delta(spot, strike, time, rate, vol, "put", div))
        assert -1.0 <= value <= 0.0

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", COMPARISON_GRID)
    def test_gamma_and_vega_are_positive(self, option_type, spot, strike, time, rate, vol, div):
        """Long vanilla options are always long gamma and long vega."""
        args = (spot, strike, time, rate, vol, option_type, div)
        assert float(gamma(*args)) > 0.0
        assert float(vega(*args)) > 0.0

    def test_call_rho_positive_put_rho_negative(self):
        """Higher rates help calls (deferred payment) and hurt puts."""
        assert float(rho(100.0, 100.0, 1.0, 0.05, 0.2, "call")) > 0.0
        assert float(rho(100.0, 100.0, 1.0, 0.05, 0.2, "put")) < 0.0

    def test_theta_is_negative_for_typical_long_options(self):
        """A near-the-money long option decays."""
        assert float(theta(100.0, 100.0, 0.5, 0.05, 0.2, "call")) < 0.0
        assert float(theta(100.0, 100.0, 0.5, 0.05, 0.2, "put")) < 0.0

    def test_deep_itm_european_put_has_positive_theta(self):
        """The documented exception: waiting to collect the strike is costly.

        This is exactly the configuration where early exercise has value, so it is
        the case Phase 2's American put must price above the European one.
        """
        assert float(theta(20.0, 100.0, 1.0, 0.10, 0.15, "put")) > 0.0


class TestGreekLimits:
    """Asymptotic behaviour at the edges of the parameter space."""

    def test_deep_itm_call_delta_approaches_one(self):
        """Certain exercise means the option behaves like the (discounted) stock."""
        value = float(delta(1000.0, 100.0, 0.25, 0.05, 0.2, "call"))
        assert value == pytest.approx(1.0, abs=1e-9)

    def test_deep_otm_call_delta_approaches_zero(self):
        """A hopeless call has no exposure to the underlying."""
        value = float(delta(1.0, 1000.0, 0.25, 0.05, 0.2, "call"))
        assert value == pytest.approx(0.0, abs=1e-9)

    def test_atm_forward_call_delta_is_near_half(self):
        """Struck at the forward, delta is slightly above 0.5.

        Not exactly 0.5: d1 = sigma sqrt(T) / 2 > 0 at the forward, so
        N(d1) > 0.5. The gap is the lognormal skew, and being able to explain why
        "ATM delta is 50" is an approximation rather than an identity is a good
        interview answer.
        """
        time, vol, rate = 1.0, 0.2, 0.03
        forward = 100.0 * np.exp(rate * time)
        value = float(delta(100.0, forward, time, rate, vol, "call"))
        assert 0.5 < value < 0.55

    def test_vega_decays_to_zero_like_sqrt_time(self):
        """No time left means no volatility exposure — but the decay is slow.

        At the money, vega ~ phi(0) * S * sqrt(T), so it vanishes only as sqrt(T),
        not linearly. Concretely, an option one second from expiry still has
        measurably positive vega. Asserting the sqrt scaling is a much stronger
        statement than asserting "close to zero", and it pins down the constant.
        """
        atm_vega_coefficient = float(norm.pdf(0.0))  # phi(0) = 0.3989
        for time in (1e-4, 1e-6, 1e-8):
            expected = atm_vega_coefficient * 100.0 * np.sqrt(time)
            actual = float(vega(100.0, 100.0, time, 0.0, 0.2, "call"))
            assert actual == pytest.approx(expected, rel=1e-6)

        # Quartering T must halve vega, since vega ~ sqrt(T).
        near = float(vega(100.0, 100.0, 4e-8, 0.0, 0.2, "call"))
        nearer = float(vega(100.0, 100.0, 1e-8, 0.0, 0.2, "call"))
        assert near == pytest.approx(2.0 * nearer, rel=1e-6)

    def test_vega_peaks_near_the_money(self):
        """Vega is maximised near the strike and decays in both directions."""
        spots = np.linspace(50.0, 150.0, 501)
        vegas = vega(spots, 100.0, 1.0, 0.0, 0.2, "call")
        peak_spot = spots[int(np.argmax(vegas))]
        assert 95.0 < peak_spot < 105.0

    def test_gamma_peaks_near_the_money(self):
        """Gamma is likewise concentrated around the strike."""
        spots = np.linspace(50.0, 150.0, 501)
        gammas = gamma(spots, 100.0, 0.25, 0.0, 0.2, "call")
        peak_spot = spots[int(np.argmax(gammas))]
        assert 95.0 < peak_spot < 105.0

    def test_gamma_rises_as_expiry_approaches(self):
        """An at-the-money option's gamma grows without bound near expiry.

        This is the practical reason short-dated ATM options are hard to hedge:
        the delta swings violently for small moves in spot.
        """
        far = float(gamma(100.0, 100.0, 1.0, 0.0, 0.2, "call"))
        near = float(gamma(100.0, 100.0, 0.01, 0.0, 0.2, "call"))
        assert near > far * 5.0


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


class TestUnitConversions:
    """The desk-convention helpers, and what they protect against."""

    def test_vega_per_percent_is_hundredth_of_raw(self):
        """One vol point is 0.01 of raw volatility."""
        raw = vega(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert float(vega_per_percent(raw)) == pytest.approx(float(raw) / 100.0)

    def test_theta_per_day_is_raw_over_365(self):
        """Raw theta is annual; desks quote it per calendar day."""
        raw = theta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert float(theta_per_day(raw)) == pytest.approx(float(raw) / DAYS_PER_YEAR)

    def test_theta_per_day_accepts_trading_day_count(self):
        """Some desks decay only on trading days; the helper allows 252."""
        raw = theta(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert float(theta_per_day(raw, days_per_year=252.0)) == pytest.approx(float(raw) / 252.0)

    def test_rho_per_basis_point_is_raw_over_10000(self):
        """A basis point is 0.0001 of a rate."""
        raw = rho(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert float(rho_per_basis_point(raw)) == pytest.approx(float(raw) / 10_000.0)

    def test_converted_theta_predicts_one_day_of_decay(self):
        """Theta per day should approximate the actual price change over a day.

        This is the test that gives the units economic meaning rather than just
        checking arithmetic: reprice one calendar day later and compare.

        Agreement is only approximate, and deliberately so. Theta is the
        instantaneous rate of decay, while the realised change over a whole day
        includes the second-order term (1/2) * d2V/dt2 * dt^2. The ~0.1% gap below
        is that curvature, not an error — which is exactly why desks treat a
        quoted theta as an estimate rather than a promise. The follow-up test
        confirms the gap shrinks with the interval, proving it is truncation.
        """
        spot, strike, time, rate, vol = 100.0, 100.0, 0.5, 0.04, 0.25
        one_day = 1.0 / DAYS_PER_YEAR
        today = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        tomorrow = float(black_scholes_price(spot, strike, time - one_day, rate, vol, "call"))
        predicted = float(theta_per_day(theta(spot, strike, time, rate, vol, "call")))
        assert tomorrow - today == pytest.approx(predicted, rel=2e-3)

    def test_theta_prediction_improves_over_shorter_intervals(self):
        """The gap above is truncation error: it shrinks linearly with the interval.

        Halving the holding period should roughly halve the relative error, which
        confirms the discrepancy is the O(dt^2) curvature term and not a bug in
        theta itself.
        """
        spot, strike, time, rate, vol = 100.0, 100.0, 0.5, 0.04, 0.25
        raw_theta = float(theta(spot, strike, time, rate, vol, "call"))
        today = float(black_scholes_price(spot, strike, time, rate, vol, "call"))

        errors = []
        for fraction in (1.0, 0.5, 0.25):
            dt = fraction / DAYS_PER_YEAR
            later = float(black_scholes_price(spot, strike, time - dt, rate, vol, "call"))
            predicted = raw_theta * dt
            errors.append(abs((later - today) - predicted) / abs(predicted))

        assert errors[0] > errors[1] > errors[2]
        assert errors[1] == pytest.approx(errors[0] / 2.0, rel=0.1)

    def test_converted_vega_predicts_one_point_of_vol(self):
        """Vega per point should approximate repricing at one vol point higher."""
        spot, strike, time, rate, vol = 100.0, 100.0, 1.0, 0.04, 0.25
        base = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        bumped = float(black_scholes_price(spot, strike, time, rate, vol + 0.01, "call"))
        predicted = float(vega_per_percent(vega(spot, strike, time, rate, vol, "call")))
        assert bumped - base == pytest.approx(predicted, rel=1e-2)


# ---------------------------------------------------------------------------
# Interface behaviour
# ---------------------------------------------------------------------------


class TestGreekInterface:
    """Vectorisation, validation, and the Greeks container."""

    def test_greeks_are_vectorised(self):
        """Greeks broadcast over arrays the same way prices do."""
        spots = np.linspace(80.0, 120.0, 9)
        result = all_greeks(spots, 100.0, 1.0, 0.05, 0.2, "call")
        for name, value in result.as_dict().items():
            assert np.asarray(value).shape == spots.shape, f"{name} did not broadcast"

    def test_greeks_dataclass_is_frozen(self):
        """A computed risk snapshot should not be mutated in place."""
        result = all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        with pytest.raises(Exception):
            result.delta = 0.0  # type: ignore[misc]

    def test_greeks_as_dict_has_all_five(self):
        """as_dict exposes exactly the five Greeks, for DataFrame building."""
        result = all_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
        assert set(result.as_dict()) == {"delta", "gamma", "vega", "theta", "rho"}
        assert isinstance(result, Greeks)

    @pytest.mark.parametrize("greek_fn", [delta, gamma, vega, theta, rho])
    def test_greeks_validate_inputs(self, greek_fn):
        """Every Greek rejects out-of-domain inputs, not just the pricer."""
        with pytest.raises(ValueError):
            greek_fn(-100.0, 100.0, 1.0, 0.05, 0.2, "call")

    @pytest.mark.parametrize("greek_fn", [delta, gamma, vega, theta, rho])
    def test_greeks_reject_bad_option_type(self, greek_fn):
        """A typo in option_type fails loudly in the Greeks too."""
        with pytest.raises(ValueError, match="call.*put"):
            greek_fn(100.0, 100.0, 1.0, 0.05, 0.2, "strangle")

    def test_greeks_accept_enum_members(self):
        """OptionType members work interchangeably with strings."""
        with_enum = float(delta(100.0, 100.0, 1.0, 0.05, 0.2, OptionType.PUT))
        with_string = float(delta(100.0, 100.0, 1.0, 0.05, 0.2, "put"))
        assert with_enum == pytest.approx(with_string)

    def test_numerical_greeks_work_on_an_arbitrary_pricer(self):
        """The finite-difference path accepts any pricer with the standard signature.

        Later phases rely on this to get Greeks out of the binomial tree and Monte
        Carlo engines, where no closed form exists. Here we prove the plumbing
        works by passing a deliberately wrapped Black-Scholes.
        """

        def wrapped_pricer(spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield):
            """Stand-in for a model-specific pricer with the standard signature."""
            return black_scholes_price(
                spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
            )

        args = (100.0, 100.0, 1.0, 0.05, 0.2, "call", 0.0)
        via_wrapper = numerical_greeks(*args, price_fn=wrapped_pricer)
        analytic = all_greeks(*args)
        for name, value in via_wrapper.as_dict().items():
            assert float(value) == pytest.approx(
                float(analytic.as_dict()[name]), rel=1e-5, abs=1e-6
            ), f"{name} mismatch"
