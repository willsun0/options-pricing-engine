"""Tests for the closed-form Black-Scholes pricer.

The suite is layered deliberately, weakest assumptions first:

1. **Known values** — match published textbook examples. Catches gross errors.
2. **Put-call parity** — a model-free no-arbitrage identity. Catches sign and
   discounting errors even where both legs are individually wrong.
3. **Arbitrage bounds and monotonicity** — properties any correct pricer must
   have, checked across a wide grid rather than at a single point.
4. **Limits and edge cases** — zero vol, zero time, deep ITM/OTM.
5. **Interface behaviour** — vectorisation, validation, enum/string handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from options_engine.common import OptionType
from options_engine.pricing.black_scholes import (
    black_scholes_price,
    call_price,
    d1_d2,
    forward_price,
    put_price,
)

# A representative grid used by the property-based tests. Spans deep OTM to deep
# ITM, one week to two years, and 5% to 80% vol, including a negative rate.
SPOTS = [50.0, 90.0, 100.0, 110.0, 200.0]
STRIKES = [80.0, 100.0, 125.0]
TIMES = [1.0 / 52.0, 0.25, 1.0, 2.0]
RATES = [-0.005, 0.0, 0.05]
VOLS = [0.05, 0.20, 0.80]
YIELDS = [0.0, 0.03]


def _grid():
    """Yield every combination of the representative parameter grid.

    Yields:
        Tuples of ``(spot, strike, time, rate, vol, dividend_yield)``.
    """
    for s in SPOTS:
        for k in STRIKES:
            for t in TIMES:
                for r in RATES:
                    for v in VOLS:
                        for q in YIELDS:
                            yield s, k, t, r, v, q


# ---------------------------------------------------------------------------
# 1. Known textbook values
# ---------------------------------------------------------------------------


class TestKnownValues:
    """Published examples with independently known answers."""

    def test_hull_example_15_6(self):
        """Hull, Options Futures and Other Derivatives, Example 15.6.

        S=42, K=40, r=10%, sigma=20%, T=0.5 years, no dividends.
        Hull reports call = 4.76, put = 0.81.
        """
        call = float(black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20, "call"))
        put = float(black_scholes_price(42.0, 40.0, 0.5, 0.10, 0.20, "put"))
        assert call == pytest.approx(4.76, abs=0.005)
        assert put == pytest.approx(0.81, abs=0.005)

    def test_hull_example_15_6_intermediate_terms(self):
        """The same example's d1 and d2, which Hull reports as 0.7693 and 0.6278."""
        d1, d2 = d1_d2(42.0, 40.0, 0.5, 0.10, 0.20)
        assert float(d1) == pytest.approx(0.7693, abs=5e-5)
        assert float(d2) == pytest.approx(0.6278, abs=5e-5)

    def test_atm_forward_exact_identity(self):
        """Struck at the forward, C = S e^{-qT} (2 N(sigma sqrt(T) / 2) - 1) exactly.

        Setting K = F kills the log-moneyness term, so d1 = -d2 = sigma sqrt(T)/2.
        The discounted-strike and discounted-spot coefficients then collapse to the
        same quantity S e^{-qT}, leaving S e^{-qT} (N(x) - N(-x)) = S e^{-qT}(2N(x) - 1).
        Holds to machine precision, with no approximation anywhere.
        """
        spot, rate, vol, time, div = 100.0, 0.03, 0.02, 0.25, 0.01
        strike = float(forward_price(spot, time, rate, div))
        x = vol * np.sqrt(time) / 2.0
        expected = spot * np.exp(-div * time) * (2.0 * norm.cdf(x) - 1.0)
        actual = float(black_scholes_price(spot, strike, time, rate, vol, "call", div))
        assert actual == pytest.approx(expected, rel=1e-12)

    def test_atm_forward_rule_of_thumb(self):
        """The trader shortcut: ATM-forward price ~ 0.3989 * S * sigma * sqrt(T).

        Linearising the exact identity above via N(x) ~ 0.5 + phi(0) x gives
        phi(0) * S * sigma * sqrt(T), with phi(0) = 1/sqrt(2 pi) = 0.3989. Note
        there is *no* discount factor: the e^{-rT} on the strike leg cancels
        against the growth in the forward. Getting that wrong is an easy slip, so
        the exact identity above is the real test and this one documents the
        approximation's accuracy.

        The error is O((sigma sqrt(T))^3), so we use a short-dated low-vol case
        where the shortcut is sharp.

        Note that phi(0) is taken from scipy rather than written as the familiar
        0.3989. That literal is itself rounded to 1.1e-5 relative, which is *larger*
        than the 4e-6 approximation error being measured — hardcoding it would mean
        the test was mostly measuring the rounding of its own constant.
        """
        spot, rate, vol, time = 100.0, 0.03, 0.02, 0.25
        strike = float(forward_price(spot, time, rate))
        expected = float(norm.pdf(0.0)) * spot * vol * np.sqrt(time)
        actual = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_zero_rate_zero_dividend_atm_symmetry(self):
        """With r = q = 0 and K = S, a call and a put must be worth the same.

        Parity reduces to C - P = S - K = 0. A pure symmetry check with no
        discounting to hide behind.
        """
        call = float(black_scholes_price(100.0, 100.0, 1.0, 0.0, 0.25, "call"))
        put = float(black_scholes_price(100.0, 100.0, 1.0, 0.0, 0.25, "put"))
        assert call == pytest.approx(put, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Put-call parity
# ---------------------------------------------------------------------------


class TestPutCallParity:
    """C - P = S e^{-qT} - K e^{-rT}.

    This holds by static replication and requires no model at all: buy a call,
    sell a put, and you hold a synthetic forward. If it fails, either a sign or a
    discount factor is wrong, and it will catch that even in cases where the
    individual leg prices look plausible.
    """

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", list(_grid()))
    def test_parity_holds_across_grid(self, spot, strike, time, rate, vol, div):
        """Parity must hold to machine precision at every grid point."""
        call = float(black_scholes_price(spot, strike, time, rate, vol, "call", div))
        put = float(black_scholes_price(spot, strike, time, rate, vol, "put", div))
        expected = spot * np.exp(-div * time) - strike * np.exp(-rate * time)
        assert call - put == pytest.approx(expected, abs=1e-10)

    def test_parity_at_zero_time(self):
        """Parity degenerates to intrinsic difference at expiry: C - P = S - K."""
        call = float(black_scholes_price(107.0, 100.0, 0.0, 0.05, 0.2, "call"))
        put = float(black_scholes_price(107.0, 100.0, 0.0, 0.05, 0.2, "put"))
        assert call - put == pytest.approx(7.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Arbitrage bounds and monotonicity
# ---------------------------------------------------------------------------


class TestArbitrageBounds:
    """No-arbitrage constraints that any correct pricer must satisfy."""

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", list(_grid()))
    def test_call_within_bounds(self, spot, strike, time, rate, vol, div):
        """max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}.

        Lower bound: the call dominates a forward. Upper bound: the call can never
        be worth more than the (dividend-adjusted) stock itself, since its payoff
        is capped by the stock's.
        """
        call = float(black_scholes_price(spot, strike, time, rate, vol, "call", div))
        lower = max(spot * np.exp(-div * time) - strike * np.exp(-rate * time), 0.0)
        upper = spot * np.exp(-div * time)
        assert call >= lower - 1e-10
        assert call <= upper + 1e-10

    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", list(_grid()))
    def test_put_within_bounds(self, spot, strike, time, rate, vol, div):
        """max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}.

        The put's payoff is capped at K (when the stock goes to zero), so its
        value is capped at the present value of K.
        """
        put = float(black_scholes_price(spot, strike, time, rate, vol, "put", div))
        lower = max(strike * np.exp(-rate * time) - spot * np.exp(-div * time), 0.0)
        upper = strike * np.exp(-rate * time)
        assert put >= lower - 1e-10
        assert put <= upper + 1e-10


class TestMonotonicity:
    """Directional properties of the price surface."""

    def test_call_increasing_in_spot(self):
        """Call value rises with spot (delta > 0)."""
        spots = np.linspace(50.0, 150.0, 200)
        prices = black_scholes_price(spots, 100.0, 1.0, 0.05, 0.2, "call")
        assert np.all(np.diff(prices) > 0.0)

    def test_put_decreasing_in_spot(self):
        """Put value falls with spot (delta < 0)."""
        spots = np.linspace(50.0, 150.0, 200)
        prices = black_scholes_price(spots, 100.0, 1.0, 0.05, 0.2, "put")
        assert np.all(np.diff(prices) < 0.0)

    def test_call_decreasing_in_strike(self):
        """Call value falls as the strike rises: you pay more to exercise."""
        strikes = np.linspace(50.0, 150.0, 200)
        prices = black_scholes_price(100.0, strikes, 1.0, 0.05, 0.2, "call")
        assert np.all(np.diff(prices) < 0.0)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_increasing_in_volatility(self, option_type):
        """Both calls and puts gain value with volatility (vega > 0).

        This monotonicity is what makes implied volatility uniquely defined, and
        so is a precondition for the Phase 4 root-finder to be well posed.
        """
        vols = np.linspace(0.01, 2.0, 300)
        prices = black_scholes_price(100.0, 100.0, 1.0, 0.05, vols, option_type)
        assert np.all(np.diff(prices) > 0.0)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_increasing_in_time_when_rate_is_zero(self, option_type):
        """With r = q = 0, more time is unambiguously more valuable.

        The caveat matters: with positive rates a deep in-the-money *European put*
        can lose value as maturity extends, because the holder waits longer to
        collect the strike. So this property is only universal at zero rates.
        """
        times = np.linspace(0.01, 3.0, 300)
        prices = black_scholes_price(100.0, 100.0, times, 0.0, 0.2, option_type)
        assert np.all(np.diff(prices) > 0.0)

    def test_deep_itm_european_put_can_decrease_in_time(self):
        """The exception to the rule above, made explicit.

        A deep ITM European put with a high rate is worth less with more time,
        because the payoff is nearly locked in and discounting it back further
        costs more than the remaining optionality is worth. This is the source of
        early-exercise value for American puts in Phase 2.
        """
        short = float(black_scholes_price(20.0, 100.0, 0.5, 0.10, 0.15, "put"))
        long = float(black_scholes_price(20.0, 100.0, 5.0, 0.10, 0.15, "put"))
        assert long < short


# ---------------------------------------------------------------------------
# 4. Limits and edge cases
# ---------------------------------------------------------------------------


class TestLimitingCases:
    """Degenerate inputs where the price collapses to a known closed form."""

    @pytest.mark.parametrize(
        "spot,strike,option_type,expected",
        [
            (120.0, 100.0, "call", 20.0),
            (80.0, 100.0, "call", 0.0),
            (80.0, 100.0, "put", 20.0),
            (120.0, 100.0, "put", 0.0),
            (100.0, 100.0, "call", 0.0),
        ],
    )
    def test_zero_time_gives_intrinsic_value(self, spot, strike, option_type, expected):
        """At T = 0 the price is the payoff, up to the documented TINY residual.

        The tolerance is not arbitrary. Because ``d1_d2`` clamps T to ``TINY``
        rather than branching, an exactly-at-expiry, exactly-at-the-money option
        retains ~ phi(0) * S * sigma * sqrt(TINY) of time value — about 8e-6 here,
        four orders of magnitude below a one-cent tick. Away from the strike the
        residual is zero to machine precision. See TINY in common.py.
        """
        price = float(black_scholes_price(spot, strike, 0.0, 0.05, 0.2, option_type))
        residual_bound = 0.4 * spot * 0.2 * np.sqrt(1e-12)
        assert price == pytest.approx(expected, abs=residual_bound)

    def test_zero_time_residual_is_exactly_zero_away_from_strike(self):
        """Away from the money the TINY clamp leaves no residual at all."""
        assert float(black_scholes_price(120.0, 100.0, 0.0, 0.05, 0.2, "call")) == 20.0
        assert float(black_scholes_price(80.0, 100.0, 0.0, 0.05, 0.2, "call")) == 0.0

    def test_zero_volatility_gives_discounted_intrinsic(self):
        """With sigma = 0 the stock grows deterministically at r - q.

        The option is then a certain cash flow: max(F - K, 0) discounted, which
        equals max(S e^{-qT} - K e^{-rT}, 0) for a call.
        """
        spot, strike, time, rate, div = 110.0, 100.0, 1.0, 0.05, 0.02
        expected = max(spot * np.exp(-div * time) - strike * np.exp(-rate * time), 0.0)
        actual = float(black_scholes_price(spot, strike, time, rate, 0.0, "call", div))
        assert actual == pytest.approx(expected, abs=1e-8)

    def test_zero_volatility_out_of_the_money_is_worthless(self):
        """With no volatility, an out-of-the-money forward can never come back."""
        price = float(black_scholes_price(50.0, 100.0, 1.0, 0.02, 0.0, "call"))
        assert price == pytest.approx(0.0, abs=1e-10)

    def test_deep_itm_call_approaches_forward_value(self):
        """A deep ITM call is a forward: exercise is effectively certain."""
        price = float(black_scholes_price(1000.0, 100.0, 1.0, 0.05, 0.2, "call"))
        expected = 1000.0 - 100.0 * np.exp(-0.05)
        assert price == pytest.approx(expected, rel=1e-9)

    def test_deep_otm_option_is_negligible(self):
        """A deep OTM call is worth essentially nothing but stays non-negative."""
        price = float(black_scholes_price(1.0, 1000.0, 0.25, 0.05, 0.2, "call"))
        assert 0.0 <= price < 1e-10

    def test_infinite_volatility_limit(self):
        """As sigma -> infinity a call approaches the stock's value.

        d1 -> +inf and d2 -> -inf, so N(d1) -> 1 and N(d2) -> 0, leaving
        S e^{-qT}. Economically: the strike becomes irrelevant when anything can
        happen.
        """
        price = float(black_scholes_price(100.0, 100.0, 1.0, 0.0, 50.0, "call"))
        assert price == pytest.approx(100.0, rel=1e-6)

    def test_negative_rates_are_accepted(self):
        """Negative rates are real and must not be rejected or mishandled."""
        price = float(black_scholes_price(100.0, 100.0, 1.0, -0.01, 0.2, "call"))
        assert price > 0.0
        put = float(black_scholes_price(100.0, 100.0, 1.0, -0.01, 0.2, "put"))
        assert put - price == pytest.approx(100.0 * np.exp(0.01) - 100.0, abs=1e-10)


class TestDividendYield:
    """The Merton continuous-dividend extension."""

    def test_dividends_reduce_call_and_raise_put(self):
        """Paying dividends drags the forward down, hurting calls and helping puts."""
        base_call = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "call", 0.0))
        div_call = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "call", 0.04))
        base_put = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put", 0.0))
        div_put = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put", 0.04))
        assert div_call < base_call
        assert div_put > base_put

    def test_dividend_yield_equals_spot_reduction(self):
        """Pricing with yield q equals pricing a spot pre-discounted by e^{-qT}.

        The formula only ever touches spot as S e^{-qT}, so this identity must
        hold exactly. It confirms q is threaded through every term correctly.
        """
        spot, time, div = 100.0, 2.0, 0.03
        with_yield = float(black_scholes_price(spot, 95.0, time, 0.05, 0.25, "call", div))
        pre_discounted = float(
            black_scholes_price(spot * np.exp(-div * time), 95.0, time, 0.05, 0.25, "call", 0.0)
        )
        assert with_yield == pytest.approx(pre_discounted, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. Interface behaviour
# ---------------------------------------------------------------------------


class TestVectorisation:
    """Broadcasting behaviour, relied on by the surface code in later phases."""

    def test_vectorised_over_strikes(self):
        """An array of strikes returns an array of prices, elementwise correct."""
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = black_scholes_price(100.0, strikes, 1.0, 0.05, 0.2, "call")
        assert prices.shape == strikes.shape
        for i, k in enumerate(strikes):
            assert prices[i] == pytest.approx(
                float(black_scholes_price(100.0, float(k), 1.0, 0.05, 0.2, "call"))
            )

    def test_broadcasting_builds_a_surface(self):
        """Strikes on one axis and expiries on the other produce a 2-D surface."""
        strikes = np.linspace(80.0, 120.0, 5).reshape(-1, 1)
        times = np.linspace(0.1, 2.0, 4).reshape(1, -1)
        prices = black_scholes_price(100.0, strikes, times, 0.05, 0.2, "call")
        assert prices.shape == (5, 4)
        assert np.all(prices >= 0.0)

    def test_vectorised_over_volatility(self):
        """Vol arrays work too, which the implied-vol solver depends on."""
        vols = np.array([0.1, 0.2, 0.3])
        prices = black_scholes_price(100.0, 100.0, 1.0, 0.05, vols, "put")
        assert prices.shape == (3,)
        assert np.all(np.diff(prices) > 0.0)


class TestInputHandling:
    """Argument parsing and validation."""

    @pytest.mark.parametrize("value", ["call", "CALL", "Call", " call ", OptionType.CALL])
    def test_option_type_accepts_strings_and_enum(self, value):
        """Case and whitespace are tolerated; the enum member works directly."""
        price = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, value))
        assert price == pytest.approx(
            float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, OptionType.CALL))
        )

    def test_invalid_option_type_raises(self):
        """A typo in the option type must fail loudly, not silently price a call."""
        with pytest.raises(ValueError, match="call.*put"):
            black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"spot": -100.0}, "spot"),
            ({"spot": 0.0}, "spot"),
            ({"strike": -100.0}, "strike"),
            ({"time_to_expiry": -1.0}, "time_to_expiry"),
            ({"volatility": -0.2}, "volatility"),
            ({"spot": float("nan")}, "spot"),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs, message):
        """Out-of-domain inputs raise ValueError naming the offending argument.

        A negative time to expiry almost always means a date subtraction ran in
        the wrong order — much better caught here than as a silent NaN downstream.
        """
        base = dict(spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.2)
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            black_scholes_price(**base)

    def test_convenience_wrappers_match_main_function(self):
        """call_price and put_price are exactly the dispatching function."""
        args = (100.0, 95.0, 0.5, 0.03, 0.25, 0.01)
        assert float(call_price(*args)) == pytest.approx(
            float(black_scholes_price(100.0, 95.0, 0.5, 0.03, 0.25, "call", 0.01))
        )
        assert float(put_price(*args)) == pytest.approx(
            float(black_scholes_price(100.0, 95.0, 0.5, 0.03, 0.25, "put", 0.01))
        )


class TestForwardPrice:
    """The forward helper used for moneyness in later phases."""

    def test_forward_exceeds_spot_when_rate_exceeds_yield(self):
        """Positive cost of carry pushes the forward above spot."""
        assert float(forward_price(100.0, 1.0, 0.05, 0.01)) > 100.0

    def test_forward_below_spot_when_yield_exceeds_rate(self):
        """A dividend yield above the rate pushes the forward below spot."""
        assert float(forward_price(100.0, 1.0, 0.01, 0.05)) < 100.0

    def test_atm_forward_call_and_put_are_equal(self):
        """Struck at the forward, a call and a put have identical value.

        Parity gives C - P = e^{-rT}(F - K), which is zero when K = F. This is the
        precise sense in which "at the money" should be defined, and it is what
        makes the Phase 4 smile symmetric around zero log-moneyness.
        """
        forward = float(forward_price(100.0, 1.5, 0.04, 0.02))
        call = float(black_scholes_price(100.0, forward, 1.5, 0.04, 0.3, "call", 0.02))
        put = float(black_scholes_price(100.0, forward, 1.5, 0.04, 0.3, "put", 0.02))
        assert call == pytest.approx(put, abs=1e-10)
