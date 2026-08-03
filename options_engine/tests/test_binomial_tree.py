"""Tests for the Cox-Ross-Rubinstein binomial tree.

Structure:

1. **Parameterisation** — u, d, p in isolation, including the arbitrage guard.
2. **Hand-computed trees** — one- and two-step trees worked out longhand in the
   test, so a broken backward induction cannot hide behind a converged limit.
3. **Independent reference** — a deliberately naive nested-loop implementation
   compared against the vectorised one. Different code, same answer.
4. **Convergence to Black-Scholes** — the headline property, including the
   measured O(1/N) rate and the oscillation.
5. **American exercise** — the inequalities that define early-exercise value, and
   the one case where American must equal European.
6. **Boundary and interface**.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.common import ExerciseStyle, OptionType
from options_engine.pricing.binomial_tree import (
    binomial_price,
    binomial_price_averaged,
    crr_parameters,
    early_exercise_boundary,
)
from options_engine.pricing.black_scholes import black_scholes_price

# Grid used for convergence and cross-model comparisons. Kept modest in size
# because each point builds a full tree.
GRID = [
    (spot, strike, time, rate, vol, div)
    for spot in (90.0, 100.0, 115.0)
    for strike in (95.0, 105.0)
    for time in (0.25, 1.0)
    for rate in (0.0, 0.05)
    for vol in (0.20, 0.40)
    for div in (0.0, 0.03)
]


# ---------------------------------------------------------------------------
# 1. CRR parameterisation
# ---------------------------------------------------------------------------


class TestCRRParameters:
    """The per-step constants, checked against their defining conditions."""

    def test_up_and_down_are_reciprocal(self):
        """CRR's defining choice: d = 1/u, which makes the tree recombine."""
        params = crr_parameters(1.0, 0.05, 0.25, 50)
        assert params.up * params.down == pytest.approx(1.0, abs=1e-15)

    def test_up_matches_volatility_scaling(self):
        """u = exp(sigma sqrt(dt)): the sqrt(dt) signature of Brownian motion."""
        time, vol, steps = 2.0, 0.3, 40
        params = crr_parameters(time, 0.04, vol, steps)
        assert params.up == pytest.approx(np.exp(vol * np.sqrt(time / steps)))

    def test_risk_neutral_probability_reproduces_the_forward(self):
        """p is defined by p*u + (1-p)*d = e^{(r-q)dt}.

        This is the condition that makes the discounted price a martingale, so
        checking it directly validates the whole point of p.
        """
        time, rate, vol, div, steps = 1.5, 0.06, 0.22, 0.02, 60
        params = crr_parameters(time, rate, vol, steps, div)
        expected_growth = np.exp((rate - div) * params.dt)
        actual = params.prob_up * params.up + (1.0 - params.prob_up) * params.down
        assert actual == pytest.approx(expected_growth, abs=1e-14)

    def test_probability_is_a_probability(self):
        """p must lie in [0, 1] across sensible parameters."""
        for time in (0.1, 1.0, 5.0):
            for vol in (0.1, 0.3, 0.8):
                for rate in (-0.01, 0.0, 0.08):
                    params = crr_parameters(time, rate, vol, 200)
                    assert 0.0 <= params.prob_up <= 1.0

    def test_variance_matches_to_leading_order(self):
        """One-step log-return variance approaches sigma^2 dt as dt shrinks.

        CRR only matches variance to O(dt), so this is a limit statement rather
        than an identity — the check is that the relative error shrinks with dt.
        """
        time, rate, vol = 1.0, 0.05, 0.25
        errors = []
        for steps in (100, 400, 1600):
            params = crr_parameters(time, rate, vol, steps)
            log_up, log_down = np.log(params.up), np.log(params.down)
            mean = params.prob_up * log_up + (1 - params.prob_up) * log_down
            variance = (
                params.prob_up * log_up**2 + (1 - params.prob_up) * log_down**2 - mean**2
            )
            errors.append(abs(variance - vol**2 * params.dt) / (vol**2 * params.dt))
        assert errors[0] > errors[1] > errors[2]

    def test_too_few_steps_raises_with_actionable_message(self):
        """An invalid p must fail loudly and say how many steps are needed.

        Very low vol against a high rate makes |r-q| sqrt(dt) exceed sigma. A
        silently out-of-range p would produce an arbitrageable tree and quietly
        wrong prices, which is far worse than an exception.
        """
        with pytest.raises(ValueError, match=r"outside \[0, 1\].*use at least"):
            crr_parameters(time_to_expiry=1.0, rate=0.5, volatility=0.01, steps=2)

    def test_suggested_step_count_actually_works(self):
        """The remedy in the error message must genuinely fix the problem."""
        with pytest.raises(ValueError) as excinfo:
            crr_parameters(1.0, 0.5, 0.01, 2)
        suggested = int(str(excinfo.value).split("use at least ")[1].split(" steps")[0])
        params = crr_parameters(1.0, 0.5, 0.01, suggested)
        assert 0.0 <= params.prob_up <= 1.0

    def test_zero_steps_raises(self):
        """A tree needs at least one step."""
        with pytest.raises(ValueError, match="at least 1"):
            crr_parameters(1.0, 0.05, 0.2, 0)


# ---------------------------------------------------------------------------
# 2. Hand-computed trees
# ---------------------------------------------------------------------------


class TestHandComputedTrees:
    """Small trees worked out longhand, so errors cannot hide in a limit.

    A tree with thousands of steps converges to Black-Scholes even if the
    induction has subtle flaws, because the errors average out. Tiny trees do not
    forgive anything.
    """

    def test_single_step_call(self):
        """A one-step tree is just one discounted expectation, computed inline."""
        spot, strike, time, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.20
        params = crr_parameters(time, rate, vol, 1)

        payoff_up = max(spot * params.up - strike, 0.0)
        payoff_down = max(spot * params.down - strike, 0.0)
        expected = params.discount * (
            params.prob_up * payoff_up + (1 - params.prob_up) * payoff_down
        )

        actual = binomial_price(spot, strike, time, rate, vol, "call", steps=1)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_two_step_put(self):
        """A two-step European put, every node written out explicitly."""
        spot, strike, time, rate, vol = 100.0, 105.0, 1.0, 0.04, 0.25
        params = crr_parameters(time, rate, vol, 2)
        u, d, p, disc = params.up, params.down, params.prob_up, params.discount

        # Terminal nodes: uu, ud (== du, since the tree recombines), dd.
        v_uu = max(strike - spot * u * u, 0.0)
        v_ud = max(strike - spot, 0.0)  # u*d = 1 exactly
        v_dd = max(strike - spot * d * d, 0.0)

        # Step 1, then step 0.
        v_u = disc * (p * v_uu + (1 - p) * v_ud)
        v_d = disc * (p * v_ud + (1 - p) * v_dd)
        expected = disc * (p * v_u + (1 - p) * v_d)

        actual = binomial_price(spot, strike, time, rate, vol, "put", steps=2)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_two_step_american_put_takes_early_exercise(self):
        """The same tree with American exercise, checking the max() at each node.

        Chosen so that the intermediate down-node is deep enough in the money that
        exercising there beats waiting — otherwise the test would pass trivially.
        """
        spot, strike, time, rate, vol = 100.0, 130.0, 1.0, 0.10, 0.20
        params = crr_parameters(time, rate, vol, 2)
        u, d, p, disc = params.up, params.down, params.prob_up, params.discount

        v_uu = max(strike - spot * u * u, 0.0)
        v_ud = max(strike - spot, 0.0)
        v_dd = max(strike - spot * d * d, 0.0)

        # At each intermediate node, take the better of holding and exercising.
        v_u = max(disc * (p * v_uu + (1 - p) * v_ud), strike - spot * u)
        v_d = max(disc * (p * v_ud + (1 - p) * v_dd), strike - spot * d)
        expected = max(disc * (p * v_u + (1 - p) * v_d), strike - spot)

        actual = binomial_price(spot, strike, time, rate, vol, "put", steps=2, exercise="american")
        assert actual == pytest.approx(expected, abs=1e-12)

        # Confirm the test is not vacuous: early exercise must actually bind here.
        european = binomial_price(spot, strike, time, rate, vol, "put", steps=2)
        assert actual > european

    def test_recombination_holds_exactly(self):
        """u*d = 1 must hold to machine precision, not just approximately.

        If it drifted, the price ladder used for the American exercise check would
        no longer line up with the nodes reached by the backward induction, and
        every American price would be subtly wrong.
        """
        for steps in (1, 7, 64, 501):
            params = crr_parameters(1.3, 0.03, 0.27, steps)
            assert params.up * params.down == pytest.approx(1.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. Independent reference implementation
# ---------------------------------------------------------------------------


def _reference_binomial(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float,
    steps: int,
    american: bool,
) -> float:
    """Deliberately naive binomial tree, written with explicit Python loops.

    Shares no code with the production implementation: it stores the whole tree in
    nested lists, recomputes each node price with repeated multiplication rather
    than a precomputed ladder, and uses no NumPy at all. Slow and wasteful on
    purpose — that is what makes it a genuine independent check on the vectorised
    slicing tricks in ``binomial_price``.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield.
        steps: Number of time steps.
        american: Whether to allow early exercise.

    Returns:
        The option value at the root of the tree.
    """
    import math

    dt = time_to_expiry / steps
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((rate - dividend_yield) * dt) - d) / (u - d)
    disc = math.exp(-rate * dt)
    sign = 1.0 if option_type == "call" else -1.0

    def payoff(price: float) -> float:
        """Immediate exercise value at a node price."""
        return max(sign * (price - strike), 0.0)

    def node_price(step: int, ups: int) -> float:
        """Node price built by repeated multiplication, not a closed form."""
        price = spot
        for _ in range(ups):
            price *= u
        for _ in range(step - ups):
            price *= d
        return price

    values = [payoff(node_price(steps, j)) for j in range(steps + 1)]

    for step in range(steps - 1, -1, -1):
        new_values = []
        for j in range(step + 1):
            held = disc * (p * values[j + 1] + (1 - p) * values[j])
            if american:
                held = max(held, payoff(node_price(step, j)))
            new_values.append(held)
        values = new_values

    return values[0]


class TestAgainstReferenceImplementation:
    """The vectorised tree must match a naive loop-based one."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("american", [False, True])
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", GRID)
    def test_matches_reference(self, option_type, american, spot, strike, time, rate, vol, div):
        """Two independent implementations agree to floating-point noise."""
        steps = 40
        expected = _reference_binomial(
            spot, strike, time, rate, vol, option_type, div, steps, american
        )
        actual = binomial_price(
            spot, strike, time, rate, vol, option_type, div,
            steps=steps,
            exercise="american" if american else "european",
        )
        assert actual == pytest.approx(expected, rel=1e-11, abs=1e-11)


# ---------------------------------------------------------------------------
# 4. Convergence to Black-Scholes
# ---------------------------------------------------------------------------


class TestConvergenceToBlackScholes:
    """The headline property: the lattice limit is the closed form."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", GRID)
    def test_european_tree_converges(self, option_type, spot, strike, time, rate, vol, div):
        """A 2000-step European tree matches Black-Scholes to about a cent."""
        tree = binomial_price(spot, strike, time, rate, vol, option_type, div, steps=2000)
        exact = float(black_scholes_price(spot, strike, time, rate, vol, option_type, div))
        assert tree == pytest.approx(exact, abs=0.01)

    @pytest.mark.parametrize(
        "spot,strike,time,rate,vol",
        [
            (100.0, 105.0, 1.0, 0.05, 0.25),
            (100.0, 95.0, 0.5, 0.03, 0.20),
            (90.0, 100.0, 2.0, 0.06, 0.35),
            (110.0, 100.0, 0.25, 0.01, 0.15),
        ],
    )
    def test_error_decays_at_first_order(self, spot, strike, time, rate, vol):
        """Measured convergence order is O(1/N), fitted by log-log regression.

        Asserting the *rate* rather than "it gets closer" is what makes this a real
        test of the discretisation. The fit is taken over even ``N`` only: because
        terminal node levels share the parity of ``N``, mixing parities samples two
        different convergence branches and corrupts the slope estimate. Restricting
        to one parity isolates a single smooth branch.

        Measured slopes across these cases run -0.83 to -1.07, so the band below is
        set to match observed behaviour rather than to an idealised -1.
        """
        exact = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        step_counts = np.arange(100, 1001, 2)  # even only
        errors = np.array(
            [
                abs(binomial_price(spot, strike, time, rate, vol, "call", steps=int(n)) - exact)
                for n in step_counts
            ]
        )
        slope = float(np.polyfit(np.log(step_counts), np.log(errors), 1)[0])
        assert -1.3 < slope < -0.7, f"expected O(1/N), fitted exponent {slope:.3f}"

    def test_convergence_is_not_monotone(self):
        """Convergence oscillates rather than descending smoothly.

        A naive expectation is that more steps always means a better price. It does
        not: successive N alternate between two branches, so a larger tree can be
        *less* accurate than a smaller one.
        """
        spot, strike, time, rate, vol = 100.0, 105.0, 1.0, 0.05, 0.25
        exact = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        errors = np.array(
            [
                abs(binomial_price(spot, strike, time, rate, vol, "call", steps=n) - exact)
                for n in range(60, 90)
            ]
        )
        assert np.any(np.diff(errors) > 0), "error never increased; expected oscillation"

    def test_even_and_odd_step_counts_form_separate_branches(self):
        """The oscillation is a parity effect, not random noise.

        Terminal node levels are ``k = -N, -N+2, ..., N``, so k always shares the
        parity of N and the set of nodes available to straddle the strike flips as
        N does. The consequence is that even-N and odd-N errors each vary smoothly
        in N while differing sharply from each other — which is exactly what makes
        averaging consecutive step counts an effective fix.
        """
        spot, strike, time, rate, vol = 100.0, 105.0, 1.0, 0.05, 0.25
        exact = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        errors = {
            n: binomial_price(spot, strike, time, rate, vol, "call", steps=n) - exact
            for n in range(50, 60)
        }
        even = [errors[n] for n in range(50, 60, 2)]
        odd = [errors[n] for n in range(51, 60, 2)]

        # Each branch varies smoothly and monotonically in N, in opposite
        # directions — they are converging towards each other over this window.
        assert all(np.diff(even) < 0), f"even branch not smooth: {even}"
        assert all(np.diff(odd) > 0), f"odd branch not smooth: {odd}"

        # Well away from where they cross, stepping N by one jumps between
        # branches by far more than moving along a branch does. The gap narrows to
        # nothing at the crossing (near N=59 here), so this is asserted at the
        # start of the window rather than across all of it.
        branch_gap = abs(errors[50] - errors[51])
        within_branch_step = abs(errors[50] - errors[52])
        assert branch_gap > 5.0 * within_branch_step

    def test_averaging_reduces_error(self):
        """Averaging V(N) and V(N+1) mixes the two branches and cancels most of it.

        Measured benefit is 2.5-3x, not the order of magnitude one might hope for.
        The window spans N=60..140 deliberately: over a narrow window the branches
        can sit on the same side of the limit, where averaging helps much less, so
        a short window would give an unrepresentative reading in either direction.
        """
        spot, strike, time, rate, vol = 100.0, 105.0, 1.0, 0.05, 0.25
        exact = float(black_scholes_price(spot, strike, time, rate, vol, "call"))

        raw_errors, averaged_errors = [], []
        for steps in range(60, 140):
            raw = binomial_price(spot, strike, time, rate, vol, "call", steps=steps)
            averaged = binomial_price_averaged(spot, strike, time, rate, vol, "call", steps=steps)
            raw_errors.append(abs(raw - exact))
            averaged_errors.append(abs(averaged - exact))

        improvement = float(np.mean(raw_errors) / np.mean(averaged_errors))
        assert improvement > 2.0, f"averaging gained only {improvement:.2f}x"

    def test_richardson_extrapolation_is_not_an_improvement(self):
        """Documents why the textbook O(1/N) fix is deliberately not implemented.

        Richardson extrapolation, ``2*V_{2N} - V_N``, assumes the error has a smooth
        expansion in 1/N. The parity oscillation violates that, and differencing
        across it amplifies rather than cancels: measured error is several times
        *worse* than simply using ``V_{2N}``.

        Asserted as a test so the claim in the module docstring stays honest if the
        implementation ever changes.
        """
        spot, strike, time, rate, vol = 100.0, 105.0, 1.0, 0.05, 0.25
        exact = float(black_scholes_price(spot, strike, time, rate, vol, "call"))
        for steps in (50, 100, 200):
            coarse = binomial_price(spot, strike, time, rate, vol, "call", steps=steps)
            fine = binomial_price(spot, strike, time, rate, vol, "call", steps=2 * steps)
            richardson = 2.0 * fine - coarse
            assert abs(richardson - exact) > abs(fine - exact)

    def test_put_call_parity_holds_on_the_tree(self):
        """European tree prices satisfy parity even before they converge.

        Parity is a statement about the tree's own internal consistency here: both
        legs use the same nodes and the same p, so it holds at *any* step count,
        not just in the limit. A failure would mean the induction itself is broken.
        """
        spot, strike, time, rate, vol, div = 100.0, 110.0, 0.75, 0.05, 0.3, 0.02
        for steps in (3, 17, 200):
            call = binomial_price(spot, strike, time, rate, vol, "call", div, steps=steps)
            put = binomial_price(spot, strike, time, rate, vol, "put", div, steps=steps)
            expected = spot * np.exp(-div * time) - strike * np.exp(-rate * time)
            assert call - put == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. American exercise
# ---------------------------------------------------------------------------


class TestAmericanExercise:
    """The inequalities that define early-exercise value."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", GRID)
    def test_american_is_never_worth_less_than_european(
        self, option_type, spot, strike, time, rate, vol, div
    ):
        """An extra right cannot have negative value."""
        american = binomial_price(
            spot, strike, time, rate, vol, option_type, div, steps=200, exercise="american"
        )
        european = binomial_price(
            spot, strike, time, rate, vol, option_type, div, steps=200, exercise="european"
        )
        assert american >= european - 1e-12

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("spot,strike,time,rate,vol,div", GRID)
    def test_american_never_below_intrinsic(self, option_type, spot, strike, time, rate, vol, div):
        """If it were, you would buy the option and exercise it for a free profit."""
        price = binomial_price(
            spot, strike, time, rate, vol, option_type, div, steps=200, exercise="american"
        )
        sign = 1.0 if option_type == "call" else -1.0
        assert price >= max(sign * (spot - strike), 0.0) - 1e-12

    def test_american_call_equals_european_without_dividends(self):
        """The classic result: never exercise an American call on a non-payer early.

        Exercising early throws away both remaining time value and the interest
        earned by deferring payment of the strike. With no dividend to capture
        there is nothing on the other side of the trade, so the early-exercise
        right is worthless and the two prices coincide.

        This is the single best test that the early-exercise logic is *correct*
        rather than merely *active*: a max() applied too eagerly would push the
        American call above the European one and fail here.
        """
        for spot in (60.0, 100.0, 140.0):
            american = binomial_price(
                spot, 100.0, 1.0, 0.05, 0.25, "call", 0.0, steps=400, exercise="american"
            )
            european = binomial_price(
                spot, 100.0, 1.0, 0.05, 0.25, "call", 0.0, steps=400, exercise="european"
            )
            assert american == pytest.approx(european, abs=1e-12)

    def test_american_call_exceeds_european_with_dividends(self):
        """With a large enough dividend yield, early exercise recaptures income.

        This is the mirror image of the test above and confirms the logic is not
        simply inert for calls.
        """
        american = binomial_price(
            100.0, 70.0, 2.0, 0.02, 0.20, "call", 0.10, steps=400, exercise="american"
        )
        european = binomial_price(
            100.0, 70.0, 2.0, 0.02, 0.20, "call", 0.10, steps=400, exercise="european"
        )
        assert american > european * 1.001

    def test_deep_itm_american_put_exceeds_european(self):
        """Early exercise is genuinely valuable for a deep in-the-money put.

        Phase 1 established that this exact configuration gives a European put
        *positive* theta — it loses value by waiting to collect the strike. The
        American holder simply does not wait. This is the direct continuation of
        that test, one phase later.
        """
        args = (20.0, 100.0, 1.0, 0.10, 0.15)
        american = binomial_price(*args, "put", steps=500, exercise="american")
        european = binomial_price(*args, "put", steps=500, exercise="european")
        assert american > european
        # Deep enough in the money that exercising immediately is optimal, so the
        # value is exactly intrinsic.
        assert american == pytest.approx(80.0, abs=0.01)

    def test_early_exercise_premium_grows_with_rates(self):
        """Higher rates make waiting to collect the strike more costly.

        The premium is driven by the interest forgone on the strike, so it must be
        increasing in r. At r = 0 there is nothing to forgo and the premium
        essentially vanishes.
        """
        premiums = []
        for rate in (0.0, 0.03, 0.08, 0.15):
            american = binomial_price(80.0, 100.0, 1.0, rate, 0.2, "put", steps=400, exercise="american")
            european = binomial_price(80.0, 100.0, 1.0, rate, 0.2, "put", steps=400, exercise="european")
            premiums.append(american - european)
        assert all(np.diff(premiums) > 0)
        assert premiums[0] == pytest.approx(0.0, abs=1e-3)

    def test_american_put_converges_too(self):
        """American prices converge as steps increase, even without a closed form.

        With no exact answer to compare against, convergence is checked as a
        Cauchy property: successive refinements move less and less.
        """
        args = (95.0, 100.0, 1.0, 0.06, 0.25, "put")
        prices = [
            binomial_price(*args, steps=n, exercise="american") for n in (100, 400, 1600, 6400)
        ]
        gaps = [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]
        assert all(np.diff(gaps) < 0), f"refinement gaps not shrinking: {gaps}"


# ---------------------------------------------------------------------------
# 6. Exercise boundary
# ---------------------------------------------------------------------------


class TestEarlyExerciseBoundary:
    """The free boundary that makes American options hard."""

    def test_put_boundary_lies_below_the_strike(self):
        """You only exercise a put once it is in the money."""
        _, boundary = early_exercise_boundary(100.0, 100.0, 1.0, 0.06, 0.25, "put", steps=300)
        assert len(boundary) > 0
        assert np.all(boundary < 100.0)

    def test_put_boundary_decreases_with_time_to_expiry(self):
        """More time left means you demand to be deeper in the money before exercising.

        The boundary starts at the strike at expiry and falls away from it as time
        to expiry grows, because the surrendered optionality is worth more. Checked
        as a trend rather than pointwise, since the lattice resolves the boundary
        only to the nearest node and so returns a staircase.
        """
        times, boundary = early_exercise_boundary(100.0, 100.0, 1.0, 0.06, 0.25, "put", steps=400)
        early_average = float(np.mean(boundary[: len(boundary) // 4]))
        late_average = float(np.mean(boundary[-len(boundary) // 4 :]))
        assert late_average < early_average
        assert np.corrcoef(times, boundary)[0, 1] < -0.8

    def test_boundary_approaches_strike_at_expiry(self):
        """As time to expiry goes to zero the boundary converges to the strike."""
        _, boundary = early_exercise_boundary(100.0, 100.0, 1.0, 0.06, 0.25, "put", steps=600)
        assert boundary[0] == pytest.approx(100.0, rel=0.05)

    def test_exercising_at_the_boundary_matches_the_tree_price(self):
        """Sanity check tying the boundary back to the price.

        At a spot just below the terminal boundary the American put should be worth
        essentially its intrinsic value, since exercise is optimal immediately.
        """
        times, boundary = early_exercise_boundary(100.0, 100.0, 1.0, 0.06, 0.25, "put", steps=500)
        deepest = float(boundary[np.argmax(times)])
        spot = deepest * 0.97
        price = binomial_price(spot, 100.0, 1.0, 0.06, 0.25, "put", steps=500, exercise="american")
        assert price == pytest.approx(100.0 - spot, rel=0.02)


# ---------------------------------------------------------------------------
# 7. Interface and edge cases
# ---------------------------------------------------------------------------


class TestBinomialInterface:
    """Argument handling, degenerate inputs, and integration with the Greeks."""

    def test_zero_time_gives_intrinsic(self):
        """At expiry the tree short-circuits to the payoff."""
        assert binomial_price(120.0, 100.0, 0.0, 0.05, 0.2, "call") == pytest.approx(20.0)
        assert binomial_price(120.0, 100.0, 0.0, 0.05, 0.2, "put") == pytest.approx(0.0)

    def test_zero_volatility_matches_black_scholes(self):
        """With no randomness there is no tree; both models give the certain payoff."""
        args = (110.0, 100.0, 1.0, 0.05, 0.0, "call", 0.02)
        assert binomial_price(*args) == pytest.approx(
            float(black_scholes_price(*args)), abs=1e-10
        )

    def test_zero_volatility_american_put_can_exercise_now(self):
        """A deterministic deep-ITM American put exercises immediately.

        With sigma = 0 the European value is the discounted intrinsic at expiry,
        which is strictly less than intrinsic today when rates are positive. The
        American holder takes the intrinsic value instead.
        """
        american = binomial_price(50.0, 100.0, 1.0, 0.10, 0.0, "put", exercise="american")
        european = binomial_price(50.0, 100.0, 1.0, 0.10, 0.0, "put", exercise="european")
        assert american == pytest.approx(50.0, abs=1e-10)
        assert american > european

    def test_single_step_tree_is_allowed(self):
        """One step is a valid, if wildly inaccurate, tree."""
        assert binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, "call", steps=1) > 0.0

    @pytest.mark.parametrize("value", ["american", "AMERICAN", " American ", ExerciseStyle.AMERICAN])
    def test_exercise_style_accepts_strings_and_enum(self, value):
        """Case and whitespace tolerated, matching OptionType's behaviour."""
        price = binomial_price(90.0, 100.0, 1.0, 0.05, 0.2, "put", steps=50, exercise=value)
        assert price == pytest.approx(
            binomial_price(90.0, 100.0, 1.0, 0.05, 0.2, "put", steps=50, exercise=ExerciseStyle.AMERICAN)
        )

    def test_invalid_exercise_style_raises(self):
        """A typo must not silently fall back to European."""
        with pytest.raises(ValueError, match="european.*american"):
            binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, "call", exercise="bermudan")

    def test_array_spot_raises_a_helpful_error(self):
        """The tree cannot broadcast; the error should say what to do instead."""
        with pytest.raises(ValueError, match="(?i)scalar.*loop over inputs"):
            binomial_price(np.array([90.0, 100.0]), 100.0, 1.0, 0.05, 0.2, "call")

    def test_invalid_inputs_raise(self):
        """Domain validation matches the closed-form pricer."""
        with pytest.raises(ValueError, match="spot"):
            binomial_price(-100.0, 100.0, 1.0, 0.05, 0.2, "call")
        with pytest.raises(ValueError, match="time_to_expiry"):
            binomial_price(100.0, 100.0, -1.0, 0.05, 0.2, "call")

    def test_option_type_enum_works(self):
        """OptionType members are interchangeable with strings here too."""
        assert binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, OptionType.PUT, steps=50) == (
            pytest.approx(binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, "put", steps=50))
        )

    def test_tree_greeks_via_finite_difference(self):
        """The tree plugs into numerical_greeks, giving Greeks with no new algebra.

        This is the payoff of keeping one argument order across every pricer: bind
        the tree-specific options with partial and the Phase 1 machinery works
        unchanged. For a European option the result must match the analytical
        Black-Scholes Greeks.
        """
        from functools import partial

        from options_engine.greeks import all_greeks, numerical_greeks

        tree_pricer = partial(binomial_price, steps=2000, exercise="european")
        args = (100.0, 100.0, 1.0, 0.05, 0.25, "call", 0.0)

        tree = numerical_greeks(*args, price_fn=tree_pricer)
        exact = all_greeks(*args)

        # Tolerances reflect tree discretisation error, not finite-difference
        # error: a 2000-step tree prices to ~1e-3, and delta inherits that.
        assert float(tree.delta) == pytest.approx(float(exact.delta), abs=2e-3)
        assert float(tree.vega) == pytest.approx(float(exact.vega), abs=0.05)
        assert float(tree.rho) == pytest.approx(float(exact.rho), abs=0.05)
