"""Tests for the implied volatility solver.

The organising idea is the **round trip**: price an option at a known volatility,
then solve for the volatility that reproduces that price, and check you get the
original number back. Because Phase 1's pricer is independently verified against
textbook values and put-call parity, a successful round trip over a wide grid is
strong evidence the inverse is right too.

Beyond that, the suite pins down the things a round trip cannot see: that the
arbitrage bounds are enforced with useful messages, that the Newton/Brent fallback
actually engages where Newton fails, and that the solver stays accurate in the
wings where the problem is genuinely ill-conditioned.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.common import OptionType
from options_engine.greeks import vega
from options_engine.pricing.black_scholes import black_scholes_price
from options_engine.vol_surface.implied_vol import (
    MAX_VOLATILITY,
    ImpliedVolResult,
    InitialGuess,
    implied_volatility,
    implied_volatility_array,
    initial_guess,
    price_bounds,
)

# Round-trip grid. Volatilities span 1% (a very quiet index) to 300% (a distressed
# single name), and moneyness runs from deep ITM to deep OTM.
ROUND_TRIP_GRID = [
    (spot, strike, time, rate, vol, div)
    for spot in (100.0,)
    for strike in (50.0, 80.0, 95.0, 100.0, 105.0, 130.0, 200.0)
    for time in (0.02, 0.25, 1.0, 3.0)
    for rate in (0.0, 0.05)
    for vol in (0.01, 0.10, 0.30, 0.80, 3.00)
    for div in (0.0, 0.03)
]


class TestRoundTrip:
    """Price at a known vol, invert, recover the vol."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", ROUND_TRIP_GRID)
    def test_recovers_the_input_volatility(self, option_type, spot, strike, time, rate, vol, div):
        """The solver must invert the pricer to the precision float64 allows.

        The tolerance is **derived, not chosen**. Volatility is recovered through
        the price, so the finest volatility difference that can possibly be resolved
        is the one that changes the price by one representable unit:

            delta_sigma ~ (machine epsilon * price) / vega

        For a near-the-money option that is ~1e-15; for a deep in-the-money one,
        where the price is large and vega is minuscule, it can be ~1e-7. Asserting a
        flat ``rel=1e-6`` everywhere would therefore fail on perfectly correct
        answers — as it did during development, on a 130-strike put whose *price
        error was exactly zero* while sigma was off by 1.2e-6.

        The generous 100x safety factor accounts for accumulated rounding in the
        several `norm.cdf` evaluations behind each price.
        """
        price = float(black_scholes_price(spot, strike, time, rate, vol, option_type, div))
        lower, upper = price_bounds(spot, strike, time, rate, option_type, div)
        if not (lower + 1e-14 < price < upper - 1e-14):
            pytest.skip("option value is indistinguishable from its arbitrage bound")

        result = implied_volatility(
            price, spot, strike, time, rate, option_type, div, return_diagnostics=True
        )

        sensitivity = float(vega(spot, strike, time, rate, vol, option_type, div))
        resolvable = 100.0 * np.finfo(float).eps * max(price, 1.0) / max(sensitivity, 1e-300)
        assert result.volatility == pytest.approx(vol, abs=max(resolvable, 1e-10))

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", ROUND_TRIP_GRID)
    def test_reproduces_the_price_regardless_of_conditioning(
        self, option_type, spot, strike, time, rate, vol, div
    ):
        """Whatever the conditioning, the solved vol must reprice to the input.

        This is the solver's actual contract, and unlike volatility accuracy it
        holds everywhere: even where sigma cannot be pinned down, the price can be
        matched to the last representable bit. Separating the two claims is what
        makes the previous test's derived tolerance honest rather than an excuse.
        """
        price = float(black_scholes_price(spot, strike, time, rate, vol, option_type, div))
        lower, upper = price_bounds(spot, strike, time, rate, option_type, div)
        if not (lower + 1e-14 < price < upper - 1e-14):
            pytest.skip("option value is indistinguishable from its arbitrage bound")

        recovered = implied_volatility(price, spot, strike, time, rate, option_type, div)
        reprice = float(black_scholes_price(spot, strike, time, rate, recovered, option_type, div))
        assert reprice == pytest.approx(price, abs=1e-9, rel=1e-11)

    def test_round_trip_through_the_price_is_exact(self):
        """Repricing at the solved vol must return the input price.

        This is the property that actually matters: the solver's contract is
        "reproduce this price", and price error is what the tolerance is applied to.
        """
        for strike in (70.0, 100.0, 140.0):
            target = 12.34 if strike == 70.0 else 3.21
            lower, upper = price_bounds(100.0, strike, 1.0, 0.04, "call", 0.01)
            target = float(np.clip(target, lower + 0.01, upper - 0.01))
            vol = implied_volatility(target, 100.0, strike, 1.0, 0.04, "call", 0.01)
            reprice = float(black_scholes_price(100.0, strike, 1.0, 0.04, vol, "call", 0.01))
            assert reprice == pytest.approx(target, abs=1e-8)

    def test_put_and_call_at_the_same_strike_imply_the_same_vol(self):
        """Put-call parity forces a single implied vol per strike.

        If the call and put prices are consistent with parity — which they are by
        construction here — then inverting either must give the same number. On real
        data this is a *diagnostic*: a large call/put IV gap at one strike means the
        quotes violate parity, i.e. the data is stale, not that vol is ambiguous.
        """
        spot, strike, time, rate, div, vol = 100.0, 110.0, 0.75, 0.05, 0.02, 0.28
        call = float(black_scholes_price(spot, strike, time, rate, vol, "call", div))
        put = float(black_scholes_price(spot, strike, time, rate, vol, "put", div))

        call_iv = implied_volatility(call, spot, strike, time, rate, "call", div)
        put_iv = implied_volatility(put, spot, strike, time, rate, "put", div)
        assert call_iv == pytest.approx(put_iv, abs=1e-9)
        assert call_iv == pytest.approx(vol, abs=1e-9)


class TestArbitrageBounds:
    """A price outside the reachable range has no implied volatility."""

    def test_bounds_are_the_zero_and_infinite_vol_limits(self):
        """price_bounds must match what the pricer actually returns at the limits."""
        args = (100.0, 95.0, 1.0, 0.05)
        lower, upper = price_bounds(*args, "call", 0.02)
        assert float(black_scholes_price(*args, 1e-12, "call", 0.02)) == pytest.approx(lower, abs=1e-6)
        assert float(black_scholes_price(*args, 100.0, "call", 0.02)) == pytest.approx(upper, rel=1e-6)

    def test_price_below_intrinsic_is_rejected(self):
        """Below discounted intrinsic, no volatility works — vol only adds value."""
        lower, _ = price_bounds(100.0, 80.0, 1.0, 0.05, "call")
        with pytest.raises(ValueError, match="at or below the no-arbitrage lower bound"):
            implied_volatility(lower - 0.5, 100.0, 80.0, 1.0, 0.05, "call")

    def test_price_above_the_upper_bound_is_rejected(self):
        """A call can never be worth more than the dividend-discounted stock."""
        _, upper = price_bounds(100.0, 100.0, 1.0, 0.05, "call")
        with pytest.raises(ValueError, match="at or above the no-arbitrage upper bound"):
            implied_volatility(upper + 1.0, 100.0, 100.0, 1.0, 0.05, "call")

    def test_error_message_reports_the_shortfall(self):
        """The message must quantify the violation, not just assert one.

        On real data this is what turns a failure into a diagnosis: a two-cent
        shortfall is a stale quote, a twenty-dollar shortfall is a wrong spot price.
        """
        lower, _ = price_bounds(100.0, 80.0, 1.0, 0.05, "call")
        with pytest.raises(ValueError, match="short by 0.5"):
            implied_volatility(lower - 0.5, 100.0, 80.0, 1.0, 0.05, "call")

    def test_expired_option_is_rejected(self):
        """At T = 0 there is no time value, so no volatility information."""
        with pytest.raises(ValueError, match="time_to_expiry must be positive"):
            implied_volatility(5.0, 100.0, 95.0, 0.0, 0.05, "call")


class TestSolverMechanics:
    """Newton, the Brent fallback, and the starting-point heuristics."""

    def test_newton_handles_ordinary_options_quickly(self):
        """A near-the-money option should converge in a handful of iterations.

        Quadratic convergence roughly doubles the correct digits each step, so
        anything more than about ten iterations means Newton is struggling.
        """
        price = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.25, "call"))
        result = implied_volatility(
            price, 100.0, 100.0, 1.0, 0.05, "call", return_diagnostics=True
        )
        assert isinstance(result, ImpliedVolResult)
        assert result.method == "newton"
        assert result.iterations <= 10
        assert result.price_error < 1e-8

    def test_brent_fallback_produces_the_same_answer_as_newton(self):
        """Force Newton to give up, and Brent must finish the job correctly.

        The fallback is exercised deterministically by capping Newton at a single
        iteration rather than by hunting for an input that breaks it. That is the
        more reliable test: it pins the code path regardless of how the tolerance
        or starting-point heuristics change later.
        """
        checked = 0
        for strike in (70.0, 100.0, 150.0, 220.0):
            for vol in (0.15, 0.30, 1.20):
                price = float(black_scholes_price(100.0, strike, 0.4, 0.05, vol, "call"))
                lower, upper = price_bounds(100.0, strike, 0.4, 0.05, "call")
                if not (lower + 1e-14 < price < upper - 1e-14):
                    continue
                forced = implied_volatility(
                    price, 100.0, strike, 0.4, 0.05, "call",
                    max_iterations=1, return_diagnostics=True,
                )
                assert forced.method == "brent"
                assert forced.volatility == pytest.approx(vol, rel=1e-6, abs=1e-8)
                checked += 1
        assert checked >= 8, "grid degenerated; the fallback was barely exercised"

    def test_newton_carries_the_vast_majority_and_brent_covers_the_rest(self):
        """Newton should be the workhorse, with Brent a genuine safety net.

        Measured over this grid: **93% Newton, 7% Brent**. The Brent cases are deep
        in-the-money short-dated calls, exactly where vega collapses and the Newton
        step blows up — so the fallback is earning its place rather than being dead
        code, and it is not silently absorbing the whole workload either.

        Both halves matter. If Newton's share collapsed, the starting-point
        heuristic or tolerance would have regressed; if Brent were never used, the
        untested branch would rot. And whichever path runs, the answer must be right.
        """
        from collections import Counter

        methods = Counter()
        for strike in (60.0, 100.0, 160.0, 220.0, 300.0):
            for time in (0.02, 0.25, 2.0):
                for vol in (0.05, 0.20, 0.90):
                    price = float(black_scholes_price(100.0, strike, time, 0.05, vol, "call"))
                    lower, upper = price_bounds(100.0, strike, time, 0.05, "call")
                    if not (lower + 1e-14 < price < upper - 1e-14):
                        continue
                    result = implied_volatility(
                        price, 100.0, strike, time, 0.05, "call", return_diagnostics=True
                    )
                    methods[result.method] += 1

                    sensitivity = float(vega(100.0, strike, time, 0.05, vol, "call"))
                    resolvable = 100.0 * np.finfo(float).eps * max(price, 1.0) / max(sensitivity, 1e-300)
                    assert result.volatility == pytest.approx(vol, abs=max(resolvable, 1e-9))

        total = sum(methods.values())
        assert methods["newton"] / total > 0.85, f"Newton share dropped: {methods}"
        assert methods["brent"] > 0, "Brent fallback never exercised; branch is untested"

    @pytest.mark.parametrize("guess", list(InitialGuess))
    def test_both_starting_points_reach_the_same_answer(self, guess):
        """The heuristic changes the path, never the destination."""
        price = float(black_scholes_price(100.0, 110.0, 0.5, 0.04, 0.33, "call"))
        vol = implied_volatility(price, 100.0, 110.0, 0.5, 0.04, "call", guess=guess)
        assert vol == pytest.approx(0.33, abs=1e-7)

    def test_manaster_koenig_handles_the_at_the_money_forward_case(self):
        """At the money forward the raw formula gives zero, which must be handled.

        ``sqrt(|ln(S/K) + (r-q)T| * 2/T)`` vanishes exactly at the forward, and a
        zero starting volatility has zero vega — Newton would divide by zero on its
        first step. The implementation substitutes a neutral 50%.
        """
        spot, time, rate = 100.0, 1.0, 0.05
        forward_strike = spot * np.exp(rate * time)
        guess = initial_guess(
            5.0, spot, forward_strike, time, rate, "call", 0.0, InitialGuess.MANASTER_KOENIG
        )
        assert guess > 0.01

        price = float(black_scholes_price(spot, forward_strike, time, rate, 0.2, "call"))
        assert implied_volatility(price, spot, forward_strike, time, rate, "call") == pytest.approx(
            0.2, abs=1e-8
        )

    def test_initial_guesses_stay_in_range(self):
        """Both heuristics must return something the solver can actually start from."""
        for guess_method in InitialGuess:
            for strike in (40.0, 100.0, 400.0):
                for time in (0.01, 5.0):
                    value = initial_guess(2.0, 100.0, strike, time, 0.05, "call", 0.0, guess_method)
                    assert 0.0 < value <= MAX_VOLATILITY

    def test_very_high_and_very_low_volatilities_are_recoverable(self):
        """The solver must span the whole plausible range, not just typical vols."""
        for vol in (0.005, 0.02, 1.5, 4.0):
            price = float(black_scholes_price(100.0, 100.0, 1.0, 0.03, vol, "call"))
            assert implied_volatility(price, 100.0, 100.0, 1.0, 0.03, "call") == pytest.approx(
                vol, rel=1e-5
            )


class TestConditioning:
    """Where the inverse problem is ill-posed, and why that matters for Phase 4."""

    def test_wing_implied_vol_is_hypersensitive_to_price(self):
        """A one-cent quoting error moves a deep OTM implied vol by vol points.

        This is not a solver defect — it is vega being tiny, so the price-to-vol
        map has an enormous derivative. It is the quantitative justification for
        every liquidity filter in the Phase 4 data cleaning: those quotes cannot
        support a trustworthy implied vol no matter how good the root-finder is.
        """
        spot, strike, time, rate, vol = 100.0, 160.0, 0.1, 0.05, 0.35
        price = float(black_scholes_price(spot, strike, time, rate, vol, "call"))

        base = implied_volatility(price, spot, strike, time, rate, "call")
        bumped = implied_volatility(price + 0.01, spot, strike, time, rate, "call")
        wing_sensitivity = abs(bumped - base)

        # The same one-cent bump at the money moves almost nothing.
        atm_price = float(black_scholes_price(spot, 100.0, time, rate, vol, "call"))
        atm_base = implied_volatility(atm_price, spot, 100.0, time, rate, "call")
        atm_bumped = implied_volatility(atm_price + 0.01, spot, 100.0, time, rate, "call")
        atm_sensitivity = abs(atm_bumped - atm_base)

        assert wing_sensitivity > 20.0 * atm_sensitivity

    def test_solver_stays_accurate_where_it_is_ill_conditioned(self):
        """Ill-conditioned does not mean the solver is allowed to be wrong.

        Given an exact price, the recovered vol must still be right — the
        sensitivity above is about *input* error, not solver error.
        """
        price = float(black_scholes_price(100.0, 200.0, 0.05, 0.05, 0.45, "call"))
        assert implied_volatility(price, 100.0, 200.0, 0.05, 0.05, "call") == pytest.approx(
            0.45, rel=1e-5
        )


class TestArrayInterface:
    """Chain-wide solving, where bad rows must not abort the batch."""

    def test_solves_a_whole_chain(self):
        """Vectorised results must match scalar calls row by row."""
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        times = np.full(5, 0.5)
        vols = np.array([0.35, 0.30, 0.25, 0.27, 0.32])
        prices = np.array(
            [float(black_scholes_price(100.0, k, 0.5, 0.04, v, "call")) for k, v in zip(strikes, vols)]
        )
        recovered = implied_volatility_array(prices, 100.0, strikes, times, 0.04, "call")
        assert np.allclose(recovered, vols, atol=1e-7)

    def test_bad_rows_become_nan_without_aborting(self):
        """One unusable quote must not destroy the rest of the surface.

        Real chains always contain some rows that violate the bounds. Returning NaN
        for those and a number for the rest is the only workable behaviour, and the
        NaN rate doubles as a data quality metric.
        """
        strikes = np.array([90.0, 100.0, 110.0])
        times = np.full(3, 0.5)
        prices = np.array([
            float(black_scholes_price(100.0, 90.0, 0.5, 0.04, 0.3, "call")),
            -5.0,      # nonsensical
            1e9,       # far above the upper bound
        ])
        result = implied_volatility_array(prices, 100.0, strikes, times, 0.04, "call")
        assert np.isfinite(result[0])
        assert np.isnan(result[1]) and np.isnan(result[2])

    def test_per_row_option_types(self):
        """Mixed call/put chains are the normal case, so they must work."""
        strikes = np.array([95.0, 105.0])
        times = np.full(2, 1.0)
        types = np.array(["put", "call"], dtype=object)
        prices = np.array([
            float(black_scholes_price(100.0, 95.0, 1.0, 0.04, 0.22, "put")),
            float(black_scholes_price(100.0, 105.0, 1.0, 0.04, 0.26, "call")),
        ])
        result = implied_volatility_array(prices, 100.0, strikes, times, 0.04, types)
        assert result[0] == pytest.approx(0.22, abs=1e-7)
        assert result[1] == pytest.approx(0.26, abs=1e-7)

    def test_mismatched_shapes_are_rejected(self):
        """A shape mismatch is a caller bug and must not be silently broadcast."""
        with pytest.raises(ValueError, match="must have the same shape"):
            implied_volatility_array(
                np.array([1.0, 2.0]), 100.0, np.array([100.0]), np.array([1.0]), 0.04
            )

    def test_enum_and_string_option_types_agree(self):
        """OptionType members work wherever strings do."""
        strikes, times = np.array([100.0]), np.array([1.0])
        prices = np.array([float(black_scholes_price(100.0, 100.0, 1.0, 0.04, 0.2, "call"))])
        by_enum = implied_volatility_array(prices, 100.0, strikes, times, 0.04, OptionType.CALL)
        by_string = implied_volatility_array(prices, 100.0, strikes, times, 0.04, "call")
        assert by_enum[0] == pytest.approx(by_string[0])

    def test_unknown_guess_method_is_rejected(self):
        """A typo must not silently fall back to a default heuristic."""
        with pytest.raises(ValueError, match="initial_guess must be one of"):
            initial_guess(5.0, 100.0, 100.0, 1.0, 0.05, "call", 0.0, "wishful_thinking")
