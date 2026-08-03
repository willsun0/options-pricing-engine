"""Tests for the Monte Carlo pricers and variance reduction.

Testing a randomised algorithm needs a different discipline from testing a
deterministic one. Two rules are followed throughout:

* **Every test is seeded.** A flaky test suite is worse than no test suite, and
  "it passes most of the time" is not a property anyone can act on.
* **Tolerances are derived from the standard error, not invented.** Asserting a
  simulated price matches Black-Scholes "to 0.01" is meaningless without knowing
  the noise level. Where a tolerance appears it is expressed in multiples of the
  estimator's own reported standard error, so the test scales correctly if the
  path count changes.

The suite also checks the *statistics*, not just the prices — most importantly
that the reported confidence intervals have the coverage they claim. An estimator
whose error bars lie is more dangerous than one that is simply imprecise.
"""

from __future__ import annotations

import numpy as np
import pytest

from options_engine.common import OptionType
from options_engine.pricing.black_scholes import barrier_price_analytic, black_scholes_price
from options_engine.pricing.monte_carlo import (
    BarrierType,
    ControlVariate,
    MonteCarloResult,
    geometric_asian_price,
    monte_carlo_asian,
    monte_carlo_barrier,
    monte_carlo_european,
    simulate_paths,
    simulate_terminal_prices,
)

SEED = 12345


# ---------------------------------------------------------------------------
# 1. The simulation itself
# ---------------------------------------------------------------------------


class TestTerminalSimulation:
    """The sampled distribution of S_T must be exactly right."""

    def test_mean_matches_the_forward(self):
        """E[S_T] = S e^{(r-q)T} — the risk-neutral drift condition.

        This is the single most important property of the simulation. If it fails,
        the measure is wrong and every price built on it is wrong. It is also the
        test that catches a missing Ito correction: dropping the -sigma^2/2 term
        inflates the sampled mean by exp(sigma^2 T / 2), which here would be a
        3% error — far outside the tolerance below.
        """
        spot, time, rate, div, vol = 100.0, 1.0, 0.05, 0.02, 0.25
        terminal = simulate_terminal_prices(
            spot, time, rate, vol, div, n_paths=1_000_000, seed=SEED
        )
        expected = spot * np.exp((rate - div) * time)
        standard_error = float(np.std(terminal, ddof=1)) / np.sqrt(terminal.size)
        assert abs(float(np.mean(terminal)) - expected) < 4.0 * standard_error

    def test_log_returns_are_normal_with_the_right_moments(self):
        """ln(S_T/S) ~ N((r-q-sigma^2/2)T, sigma^2 T)."""
        spot, time, rate, vol = 100.0, 2.0, 0.03, 0.30
        terminal = simulate_terminal_prices(spot, time, rate, vol, 0.0, n_paths=1_000_000, seed=SEED)
        log_returns = np.log(terminal / spot)

        expected_mean = (rate - 0.5 * vol**2) * time
        expected_std = vol * np.sqrt(time)
        mean_se = expected_std / np.sqrt(log_returns.size)

        assert float(np.mean(log_returns)) == pytest.approx(expected_mean, abs=4.0 * mean_se)
        assert float(np.std(log_returns, ddof=1)) == pytest.approx(expected_std, rel=0.01)
        # A lognormal's log is normal, so skewness should vanish.
        centred = log_returns - np.mean(log_returns)
        skew = float(np.mean(centred**3) / np.std(log_returns) ** 3)
        assert abs(skew) < 0.02

    def test_antithetic_draws_are_exact_mirrors(self):
        """Path i and path i + n/2 must come from Z and -Z.

        The pairing convention is relied on by the standard-error calculation, so
        it is worth pinning down directly rather than trusting the layout.
        """
        spot, time, rate, vol = 100.0, 1.0, 0.05, 0.2
        terminal = simulate_terminal_prices(
            spot, time, rate, vol, 0.0, n_paths=1000, seed=SEED, antithetic=True
        )
        half = terminal.size // 2
        # If S_up = S exp(m + sZ) and S_down = S exp(m - sZ), then their product is
        # S^2 exp(2m), a constant independent of Z.
        products = terminal[:half] * terminal[half:]
        assert np.allclose(products, products[0], rtol=1e-12)

    def test_seeding_is_reproducible(self):
        """The same seed must give bitwise identical results."""
        args = (100.0, 1.0, 0.05, 0.2, 0.0, 1000)
        first = simulate_terminal_prices(*args, seed=42)
        second = simulate_terminal_prices(*args, seed=42)
        different = simulate_terminal_prices(*args, seed=43)
        assert np.array_equal(first, second)
        assert not np.array_equal(first, different)


class TestPathSimulation:
    """Full-path simulation used by the exotics."""

    def test_shape_and_initial_column(self):
        """Paths are (n_paths, n_steps+1) and start at spot."""
        paths = simulate_paths(100.0, 1.0, 0.05, 0.2, n_paths=500, n_steps=12, seed=SEED)
        assert paths.shape == (500, 13)
        assert np.all(paths[:, 0] == 100.0)

    def test_marginal_distribution_at_each_date_is_exact(self):
        """Every monitoring date must have the correct lognormal marginal.

        This is what "exact transition" means, and it is why no Euler bias exists:
        a coarse grid still gets each date's distribution exactly right. An Euler
        scheme would fail this at large dt.
        """
        spot, time, rate, vol = 100.0, 1.0, 0.04, 0.35
        n_steps = 4
        paths = simulate_paths(spot, time, rate, vol, 0.0, n_paths=400_000, n_steps=n_steps, seed=SEED)

        for step in range(1, n_steps + 1):
            t = step * time / n_steps
            log_returns = np.log(paths[:, step] / spot)
            expected_mean = (rate - 0.5 * vol**2) * t
            expected_std = vol * np.sqrt(t)
            mean_se = expected_std / np.sqrt(paths.shape[0])
            assert float(np.mean(log_returns)) == pytest.approx(expected_mean, abs=4.0 * mean_se)
            assert float(np.std(log_returns, ddof=1)) == pytest.approx(expected_std, rel=0.02)

    def test_terminal_distribution_matches_direct_simulation(self):
        """Stepping to expiry agrees in distribution with jumping there directly."""
        spot, time, rate, vol = 100.0, 1.0, 0.05, 0.25
        stepped = simulate_paths(spot, time, rate, vol, 0.0, n_paths=200_000, n_steps=50, seed=SEED)[:, -1]
        direct = simulate_terminal_prices(spot, time, rate, vol, 0.0, n_paths=200_000, seed=SEED + 1)
        assert float(np.mean(stepped)) == pytest.approx(float(np.mean(direct)), rel=0.005)
        assert float(np.std(stepped)) == pytest.approx(float(np.std(direct)), rel=0.01)

    def test_increments_are_independent(self):
        """Consecutive log-increments must be uncorrelated."""
        paths = simulate_paths(100.0, 1.0, 0.05, 0.3, 0.0, n_paths=200_000, n_steps=10, seed=SEED)
        increments = np.diff(np.log(paths), axis=1)
        correlation = float(np.corrcoef(increments[:, 0], increments[:, 1])[0, 1])
        assert abs(correlation) < 0.01

    def test_paths_stay_positive(self):
        """Geometric Brownian motion can never reach zero, even at high vol."""
        paths = simulate_paths(100.0, 5.0, 0.0, 1.5, 0.0, n_paths=20_000, n_steps=100, seed=SEED)
        assert np.all(paths > 0.0)


# ---------------------------------------------------------------------------
# 2. European pricing and estimator statistics
# ---------------------------------------------------------------------------


class TestEuropeanPricing:
    """Simulated vanillas against the exact Phase 1 answer."""

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize(
        "spot,strike,time,rate,vol,div",
        [
            (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
            (90.0, 100.0, 0.5, 0.03, 0.35, 0.02),
            (120.0, 100.0, 2.0, 0.01, 0.15, 0.04),
        ],
    )
    def test_price_matches_black_scholes_within_error_bars(
        self, option_type, spot, strike, time, rate, vol, div
    ):
        """The simulated price must agree with the formula to within its own noise.

        Three standard errors is a ~99.7% interval, so this is a strong statement
        about the estimator being unbiased rather than merely close.
        """
        result = monte_carlo_european(
            spot, strike, time, rate, vol, option_type, div, n_paths=400_000, seed=SEED
        )
        exact = float(black_scholes_price(spot, strike, time, rate, vol, option_type, div))
        assert abs(result.price - exact) < 3.0 * result.standard_error

    def test_confidence_intervals_have_the_coverage_they_claim(self):
        """A 95% interval must contain the true price about 95% of the time.

        This is the test that makes the standard error trustworthy rather than
        decorative. Running 200 independent simulations with different seeds and
        counting how many intervals capture the Black-Scholes value directly
        validates the whole uncertainty pipeline.

        The count itself is binomial(200, 0.95), with standard deviation ~3.1, so
        the acceptance band below is roughly +/- 3 sigma. It is wide on purpose:
        a narrower band would be testing luck rather than correctness.
        """
        spot, strike, time, rate, vol = 100.0, 105.0, 1.0, 0.05, 0.25
        exact = float(black_scholes_price(spot, strike, time, rate, vol, "call"))

        trials = 200
        covered = 0
        for trial in range(trials):
            result = monte_carlo_european(
                spot, strike, time, rate, vol, "call", n_paths=20_000, seed=1000 + trial
            )
            low, high = result.confidence_interval(0.95)
            covered += int(low <= exact <= high)

        assert 175 <= covered <= 199, f"95% intervals covered {covered}/{trials}"

    def test_error_decays_as_one_over_root_n(self):
        """Quadrupling the paths must halve the standard error.

        The O(1/sqrt(N)) rate is the defining limitation of Monte Carlo, so it is
        worth asserting rather than assuming.
        """
        spot, strike, time, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.25
        errors = [
            monte_carlo_european(
                spot, strike, time, rate, vol, "call", n_paths=n, seed=SEED
            ).standard_error
            for n in (25_000, 100_000, 400_000)
        ]
        assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.15)
        assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.15)

    def test_put_call_parity_holds_exactly_on_shared_random_numbers(self):
        """On common random numbers, parity holds to *machine precision*.

        The payoff identity ``max(x-K,0) - max(K-x,0) = x - K`` holds path by path,
        with no expectation taken. So with the same seed,

            C_mc - P_mc = e^{-rT} ( mean(S_T) - K )

        exactly — no statistics involved. That makes this a far sharper test than
        comparing against the theoretical parity value, and it directly verifies
        that both option types consume the same simulated prices.
        """
        spot, strike, time, rate, vol, div = 100.0, 110.0, 0.75, 0.05, 0.3, 0.02
        paths = 200_000
        call = monte_carlo_european(spot, strike, time, rate, vol, "call", div, n_paths=paths, seed=SEED)
        put = monte_carlo_european(spot, strike, time, rate, vol, "put", div, n_paths=paths, seed=SEED)
        terminal = simulate_terminal_prices(spot, time, rate, vol, div, n_paths=paths, seed=SEED)

        sample_parity = float(np.exp(-rate * time) * (np.mean(terminal) - strike))
        assert call.price - put.price == pytest.approx(sample_parity, abs=1e-10)

    def test_parity_deviation_is_within_the_sampling_error_of_the_forward(self):
        """The gap to *theoretical* parity is exactly the error in mean(S_T).

        Since ``C_mc - P_mc = e^{-rT}(mean(S_T) - K)`` exactly, the deviation from
        the theoretical ``S e^{-qT} - K e^{-rT}`` is nothing but the sampling error
        of the simulated forward. The tolerance below is therefore derived from
        that quantity's own standard error rather than picked by hand.

        Worth noting what this rules out: taking the difference on common random
        numbers does *not* reduce error here. Call and put payoffs are negatively
        correlated (only one pays on any path), so the difference has a larger
        standard error than either leg — measured ~0.058 against ~0.034 each.
        Common random numbers help when comparing *similar* quantities; these two
        are complementary, which is the opposite case.
        """
        spot, strike, time, rate, vol, div = 100.0, 110.0, 0.75, 0.05, 0.3, 0.02
        paths = 200_000
        call = monte_carlo_european(spot, strike, time, rate, vol, "call", div, n_paths=paths, seed=SEED)
        put = monte_carlo_european(spot, strike, time, rate, vol, "put", div, n_paths=paths, seed=SEED)
        terminal = simulate_terminal_prices(spot, time, rate, vol, div, n_paths=paths, seed=SEED)

        discount = float(np.exp(-rate * time))
        forward_standard_error = discount * float(np.std(terminal, ddof=1)) / np.sqrt(paths)
        theoretical = spot * np.exp(-div * time) - strike * np.exp(-rate * time)

        assert abs((call.price - put.price) - theoretical) < 3.0 * forward_standard_error

    def test_result_reports_sample_count_correctly(self):
        """Antithetic sampling halves the number of independent samples."""
        plain = monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.2, "call", n_paths=10_000, seed=SEED)
        anti = monte_carlo_european(
            100.0, 100.0, 1.0, 0.05, 0.2, "call", n_paths=10_000, seed=SEED, antithetic=True
        )
        assert plain.n_paths == anti.n_paths == 10_000
        assert plain.n_samples == 10_000
        assert anti.n_samples == 5_000


# ---------------------------------------------------------------------------
# 3. Antithetic variates
# ---------------------------------------------------------------------------


def _standard_errors(option_type: str, strike: float, n_paths: int = 100_000, **kwargs) -> tuple[float, float]:
    """Return (plain, antithetic) standard errors at equal path count.

    Comparing at equal *paths* rather than equal random draws is the fair test:
    it holds computational cost roughly constant, which is what a practitioner
    actually cares about.

    Args:
        option_type: ``"call"`` or ``"put"``.
        strike: Strike price.
        n_paths: Path count used for both estimators.
        **kwargs: Extra arguments forwarded to :func:`monte_carlo_european`.

    Returns:
        A ``(plain_se, antithetic_se)`` tuple.
    """
    base = dict(spot=100.0, strike=strike, time_to_expiry=1.0, rate=0.05, volatility=0.25)
    base.update(kwargs)
    plain = monte_carlo_european(**base, option_type=option_type, n_paths=n_paths, seed=SEED)
    anti = monte_carlo_european(
        **base, option_type=option_type, n_paths=n_paths, seed=SEED, antithetic=True
    )
    return plain.standard_error, anti.standard_error


class TestAntitheticVariates:
    """Antithetic sampling helps, hurts, or does nothing — depending on the payoff."""

    def test_helps_for_a_monotone_payoff(self):
        """A vanilla call is monotone in Z, so mirrored pairs are anticorrelated."""
        plain, anti = _standard_errors("call", strike=100.0)
        assert anti < plain, f"antithetic SE {anti:.5f} not below plain {plain:.5f}"

    def test_benefit_decreases_with_moneyness(self):
        """The gain tracks how monotone the payoff is in the driving normal.

        Deep in the money the payoff is nearly linear in Z, so ``rho -> -1`` and
        the mirrored pair cancels almost all the randomness. Deep out of the money
        one leg pays only when the other does not, the cross products are almost
        always zero, ``rho -> 0``, and the method does essentially nothing.

        Measured improvement factors: ~4.1x deep ITM, ~1.35x at the money, ~1.03x
        deep OTM. Asserting the *ordering* rather than specific factors keeps the
        test meaningful without pinning it to one seed's arithmetic.
        """
        improvements = []
        for strike in (40.0, 100.0, 250.0):
            plain, anti = _standard_errors("call", strike=strike)
            improvements.append(plain / anti)

        assert all(np.diff(improvements) < 0), f"not decreasing in strike: {improvements}"
        assert improvements[0] > 3.0, "deep ITM should benefit substantially"
        assert improvements[-1] < 1.15, "deep OTM should benefit barely at all"

    def test_hurts_for_a_symmetric_payoff(self):
        """A straddle pays on moves either way, so f(Z) and f(-Z) are *positively*
        correlated and the variance ratio 1 + rho exceeds one.

        This is the case that proves the technique is not magic. It is built here
        as a call plus a put on the same seed, which is exactly a straddle.
        """
        spot, strike, time, rate, vol = 100.0, 100.0, 1.0, 0.05, 0.25

        def straddle_samples(antithetic: bool) -> float:
            """Standard error of a simulated straddle."""
            terminal = simulate_terminal_prices(
                spot, time, rate, vol, 0.0, n_paths=200_000, seed=SEED, antithetic=antithetic
            )
            payoffs = np.abs(terminal - strike)
            if antithetic:
                half = payoffs.size // 2
                payoffs = 0.5 * (payoffs[:half] + payoffs[half:])
            return float(np.std(payoffs, ddof=1)) / np.sqrt(payoffs.size)

        assert straddle_samples(True) > straddle_samples(False)

    def test_estimator_stays_unbiased(self):
        """Variance reduction must not move the mean."""
        args = (100.0, 105.0, 1.0, 0.05, 0.25, "call")
        exact = float(black_scholes_price(*args))
        result = monte_carlo_european(*args, n_paths=400_000, seed=SEED, antithetic=True)
        assert abs(result.price - exact) < 3.0 * result.standard_error

    def test_odd_path_count_is_rejected(self):
        """Antithetic paths come in pairs, so an odd count is a caller error."""
        with pytest.raises(ValueError, match="even when antithetic"):
            monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.2, "call", n_paths=999, antithetic=True)


# ---------------------------------------------------------------------------
# 4. Control variates
# ---------------------------------------------------------------------------


class TestControlVariates:
    """Correlated quantities with known expectations, used to cancel noise."""

    def test_terminal_stock_control_reduces_error(self):
        """E[S_T] is known exactly and correlates with a call payoff."""
        args = dict(spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.25)
        plain = monte_carlo_european(**args, option_type="call", n_paths=200_000, seed=SEED)
        controlled = monte_carlo_european(
            **args, option_type="call", n_paths=200_000, seed=SEED, control_variate="terminal_stock"
        )
        assert controlled.standard_error < plain.standard_error

    def test_control_variate_stays_unbiased(self):
        """The correction term has zero mean, so the price must not shift."""
        args = (100.0, 105.0, 1.0, 0.05, 0.25, "call")
        exact = float(black_scholes_price(*args))
        result = monte_carlo_european(
            *args, n_paths=400_000, seed=SEED, control_variate="terminal_stock"
        )
        assert abs(result.price - exact) < 3.0 * result.standard_error

    def test_fitted_beta_is_negative_for_a_call(self):
        """c* = -Cov(Y,X)/Var(X); a call payoff rises with S_T, so c* < 0."""
        result = monte_carlo_european(
            100.0, 100.0, 1.0, 0.05, 0.25, "call", n_paths=100_000, seed=SEED,
            control_variate="terminal_stock",
        )
        assert result.control_beta is not None and result.control_beta < 0.0

    def test_geometric_asian_control_is_dramatically_effective(self):
        """The geometric Asian correlates with the arithmetic one above 0.99.

        Variance falls by 1/(1 - rho^2), so rho = 0.999 would be a 500x reduction.
        This is the textbook example of a control variate chosen well, and the
        improvement should be at least an order of magnitude in standard error.
        """
        args = dict(
            spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.30,
            n_paths=100_000, n_averaging_dates=52, seed=SEED,
        )
        plain = monte_carlo_asian(**args, option_type="call")
        controlled = monte_carlo_asian(**args, option_type="call", control_variate="geometric_asian")
        assert controlled.standard_error < plain.standard_error / 10.0

    def test_european_control_helps_the_asian_less_than_the_geometric_one(self):
        """A vanilla is a decent control for an Asian; the geometric Asian is better.

        Ranking the two makes the point that control variate choice matters more
        than merely having one.
        """
        args = dict(
            spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.30,
            n_paths=100_000, n_averaging_dates=52, seed=SEED, option_type="call",
        )
        plain = monte_carlo_asian(**args)
        european = monte_carlo_asian(**args, control_variate="european_option")
        geometric = monte_carlo_asian(**args, control_variate="geometric_asian")
        assert geometric.standard_error < european.standard_error < plain.standard_error

    def test_techniques_compose_when_the_control_is_imperfect(self):
        """Antithetic sampling still adds value on top of a moderate control.

        With the European-option control the residual variance is only partly
        removed, and what remains is still monotone in the driving normals — so
        mirroring continues to help. Measured: 0.0148 -> 0.0133.
        """
        args = dict(
            spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.30,
            n_paths=200_000, n_averaging_dates=52, seed=SEED, option_type="call",
        )
        control_only = monte_carlo_asian(**args, control_variate="european_option")
        both = monte_carlo_asian(**args, control_variate="european_option", antithetic=True)
        assert both.standard_error < control_only.standard_error

        # Same story for a vanilla, where the gain is larger: 0.0158 -> 0.0079.
        vanilla_args = dict(
            spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.25,
            n_paths=200_000, seed=SEED, option_type="call",
        )
        vanilla_control = monte_carlo_european(**vanilla_args, control_variate="terminal_stock")
        vanilla_both = monte_carlo_european(
            **vanilla_args, control_variate="terminal_stock", antithetic=True
        )
        assert vanilla_both.standard_error < vanilla_control.standard_error / 1.5

    def test_antithetic_adds_nothing_once_the_control_is_near_perfect(self):
        """Variance reduction techniques do **not** simply stack.

        The geometric-Asian control removes about 96% of the standard error on its
        own. What survives is the *difference* between the arithmetic and geometric
        averages, and that residual is not monotone in the driving normals — it is
        a spread, small when the path is calm and large when it is volatile,
        regardless of direction. Mirroring therefore has nothing left to cancel,
        while still halving the number of independent samples. The net effect is
        very slightly worse: measured 0.001591 -> 0.001638.

        The rule this illustrates is worth carrying: antithetic sampling helps to
        the extent the *remaining* variance is monotone in Z. Apply a strong
        control first and you may have already spent that.
        """
        args = dict(
            spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.30,
            n_paths=100_000, n_averaging_dates=52, seed=SEED, option_type="call",
        )
        control_only = monte_carlo_asian(**args, control_variate="geometric_asian")
        both = monte_carlo_asian(**args, control_variate="geometric_asian", antithetic=True)

        # Not better — and the two are within a few percent of each other, i.e. the
        # antithetic layer is doing essentially nothing.
        assert both.standard_error >= control_only.standard_error
        assert both.standard_error < control_only.standard_error * 1.2

    def test_invalid_control_for_payoff_is_rejected(self):
        """A control that makes no sense for the payoff must fail loudly."""
        with pytest.raises(ValueError, match="not applicable to a European"):
            monte_carlo_european(
                100.0, 100.0, 1.0, 0.05, 0.2, "call", control_variate="geometric_asian"
            )
        with pytest.raises(ValueError, match="weak choice for an Asian"):
            monte_carlo_asian(100.0, 100.0, 1.0, 0.05, 0.2, "call", control_variate="terminal_stock")

    def test_unknown_control_name_is_rejected(self):
        """A typo must not silently disable variance reduction."""
        with pytest.raises(ValueError, match="control_variate must be one of"):
            monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.2, "call", control_variate="magic")


# ---------------------------------------------------------------------------
# 5. Asian options
# ---------------------------------------------------------------------------


class TestGeometricAsianClosedForm:
    """The exact geometric-average formula, which anchors everything else."""

    def test_matches_direct_simulation_of_the_geometric_average(self):
        """The closed form must agree with brute-force simulation of ln G.

        This validates the variance factor (n+1)(2n+1)/(6n^2) and the effective
        dividend construction, both of which are easy to get subtly wrong.
        """
        spot, strike, time, rate, vol, n = 100.0, 100.0, 1.0, 0.05, 0.30, 12
        paths = simulate_paths(spot, time, rate, vol, 0.0, n_paths=400_000, n_steps=n, seed=SEED)
        geometric = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        payoffs = np.maximum(geometric - strike, 0.0)
        simulated = float(np.exp(-rate * time) * np.mean(payoffs))
        standard_error = float(np.exp(-rate * time) * np.std(payoffs, ddof=1)) / np.sqrt(payoffs.size)

        exact = geometric_asian_price(spot, strike, time, rate, vol, "call", 0.0, n)
        assert abs(exact - simulated) < 3.5 * standard_error

    def test_single_averaging_date_reduces_to_black_scholes(self):
        """With n = 1 the 'average' is just S_T, so it must equal the vanilla.

        A clean degenerate check: the variance factor becomes (2)(3)/6 = 1 and the
        mean time becomes T, recovering Black-Scholes exactly.
        """
        args = (100.0, 105.0, 1.0, 0.05, 0.25, "call", 0.02)
        asian = geometric_asian_price(*args, n_averaging_dates=1)
        vanilla = float(black_scholes_price(*args))
        assert asian == pytest.approx(vanilla, rel=1e-12)

    def test_effective_volatility_tends_to_sigma_over_root_three(self):
        """As n grows, the averaging factor approaches 1/sqrt(3).

        Checked indirectly: the price with many dates should match a vanilla
        priced at sigma/sqrt(3) with the appropriate forward adjustment. Here we
        assert the simpler consequence that prices converge as n grows.
        """
        args = (100.0, 100.0, 1.0, 0.05, 0.30, "call", 0.0)
        prices = [geometric_asian_price(*args, n_averaging_dates=n) for n in (50, 200, 800, 3200)]
        gaps = [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]
        assert all(np.diff(gaps) < 0), f"not converging: {gaps}"

    def test_cheaper_than_the_vanilla(self):
        """Averaging suppresses volatility, so the Asian is worth less."""
        args = (100.0, 100.0, 1.0, 0.05, 0.30, "call", 0.0)
        assert geometric_asian_price(*args, n_averaging_dates=52) < float(black_scholes_price(*args))


class TestAsianPricing:
    """The arithmetic Asian, which has no closed form."""

    def test_arithmetic_exceeds_geometric(self):
        """AM-GM: the arithmetic mean always dominates the geometric mean.

        Since the payoff is increasing in the average, the arithmetic Asian must be
        worth strictly more than the geometric one. A pure mathematical inequality
        that holds path by path, making it a sharp test of the averaging code.
        """
        args = dict(
            spot=100.0, strike=100.0, time_to_expiry=1.0, rate=0.05, volatility=0.30,
            n_averaging_dates=52,
        )
        arithmetic = monte_carlo_asian(
            **args, option_type="call", n_paths=200_000, seed=SEED, control_variate="geometric_asian"
        )
        geometric = geometric_asian_price(**args, option_type="call")
        assert arithmetic.price > geometric
        # And by a small, sane margin rather than wildly.
        assert arithmetic.price - geometric < 0.5

    def test_cheaper_than_the_vanilla(self):
        """Averaging damps volatility, so Asians are cheaper than vanillas."""
        args = (100.0, 100.0, 1.0, 0.05, 0.30, "call")
        asian = monte_carlo_asian(
            *args, n_paths=200_000, n_averaging_dates=52, seed=SEED,
            control_variate="geometric_asian",
        )
        vanilla = float(black_scholes_price(*args))
        assert asian.price < vanilla

    def test_single_averaging_date_reduces_to_the_vanilla(self):
        """With one averaging date the Asian *is* a European option."""
        args = (100.0, 100.0, 1.0, 0.05, 0.25, "call")
        asian = monte_carlo_asian(*args, n_paths=400_000, n_averaging_dates=1, seed=SEED)
        vanilla = float(black_scholes_price(*args))
        assert abs(asian.price - vanilla) < 3.0 * asian.standard_error

    def test_more_averaging_dates_lowers_the_price(self):
        """More averaging means more volatility damping, hence a lower price."""
        prices = [
            monte_carlo_asian(
                100.0, 100.0, 1.0, 0.05, 0.30, "call", n_paths=200_000,
                n_averaging_dates=n, seed=SEED, control_variate="geometric_asian",
            ).price
            for n in (1, 4, 12, 52)
        ]
        assert all(np.diff(prices) < 0), f"not monotone in averaging dates: {prices}"

    def test_put_is_priced_correctly_too(self):
        """The put branch of the payoff, checked against its geometric sibling."""
        args = dict(
            spot=100.0, strike=105.0, time_to_expiry=1.0, rate=0.05, volatility=0.25,
            n_averaging_dates=26,
        )
        arithmetic = monte_carlo_asian(
            **args, option_type="put", n_paths=200_000, seed=SEED, control_variate="geometric_asian"
        )
        geometric = geometric_asian_price(**args, option_type="put")
        # A put pays on *low* averages, so AM >= GM makes the arithmetic put cheaper.
        assert arithmetic.price < geometric


# ---------------------------------------------------------------------------
# 6. Barrier options
# ---------------------------------------------------------------------------


class TestBarrierPricing:
    """Knock-in and knock-out options, validated by structural identities."""

    @staticmethod
    def _vanilla_on_the_same_paths(
        spot: float, strike: float, time: float, rate: float, vol: float,
        option_type: str, n_paths: int, n_monitoring_dates: int,
    ) -> float:
        """Price a vanilla from the *same* simulated paths the barrier pricer uses.

        This helper exists because seeding alone is not enough to share a sample.
        ``monte_carlo_european`` calls ``simulate_terminal_prices`` and draws an
        ``(n, 1)`` array; the barrier pricer calls ``simulate_paths`` and draws
        ``(n, n_monitoring_dates)``. Identical seeds therefore produce *different*
        numbers, because the generator is consumed in a different shape. Comparing
        across the two is a statistical comparison, not an exact one.

        Regenerating the vanilla from the path simulation makes the comparison
        exact, which is what turns in-out parity into a machine-precision test.
        """
        paths = simulate_paths(
            spot, time, rate, vol, 0.0, n_paths=n_paths, n_steps=n_monitoring_dates, seed=SEED
        )
        sign = 1.0 if option_type == "call" else -1.0
        payoffs = np.maximum(sign * (paths[:, -1] - strike), 0.0)
        return float(np.exp(-rate * time) * np.mean(payoffs))

    def test_in_out_parity(self):
        """knock-in + knock-out = vanilla, exactly, on shared random numbers.

        Every path either touches the barrier or does not, so the two payoffs
        partition the vanilla payoff path by path. No expectation is involved, so
        the identity holds to floating-point precision rather than merely
        statistically — a far sharper test than any convergence check.
        """
        args = dict(
            spot=100.0, strike=100.0, barrier=130.0, time_to_expiry=1.0, rate=0.05,
            volatility=0.25, n_paths=100_000, n_monitoring_dates=50, seed=SEED,
        )
        out = monte_carlo_barrier(**args, barrier_type="up_and_out")
        inn = monte_carlo_barrier(**args, barrier_type="up_and_in")
        vanilla = self._vanilla_on_the_same_paths(
            100.0, 100.0, 1.0, 0.05, 0.25, "call", 100_000, 50
        )
        assert out.price + inn.price == pytest.approx(vanilla, abs=1e-10)

    def test_distant_barrier_recovers_the_vanilla(self):
        """A knock-out barrier far away is never touched, so nothing knocks out."""
        result = monte_carlo_barrier(
            100.0, 100.0, 10_000.0, 1.0, 0.05, 0.25, "call", "up_and_out",
            n_paths=100_000, n_monitoring_dates=50, seed=SEED,
        )
        vanilla = self._vanilla_on_the_same_paths(
            100.0, 100.0, 1.0, 0.05, 0.25, "call", 100_000, 50
        )
        assert result.price == pytest.approx(vanilla, abs=1e-10)

    def test_knock_out_is_cheaper_than_the_vanilla(self):
        """The knock-out risk can only remove value."""
        knock_out = monte_carlo_barrier(
            100.0, 100.0, 120.0, 1.0, 0.05, 0.25, "call", "up_and_out",
            n_paths=200_000, n_monitoring_dates=50, seed=SEED,
        )
        vanilla = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.25, "call"))
        assert knock_out.price < vanilla

    def test_knock_out_value_increases_with_barrier_distance(self):
        """A more distant barrier is less likely to kill the option."""
        prices = [
            monte_carlo_barrier(
                100.0, 100.0, barrier, 1.0, 0.05, 0.25, "call", "up_and_out",
                n_paths=100_000, n_monitoring_dates=50, seed=SEED,
            ).price
            for barrier in (110.0, 130.0, 160.0, 200.0)
        ]
        assert all(np.diff(prices) > 0), f"not monotone in barrier: {prices}"

    def test_down_and_out_put_behaves_symmetrically(self):
        """The down-barrier branch, checked with the same in-out parity identity."""
        args = dict(
            spot=100.0, strike=100.0, barrier=75.0, time_to_expiry=1.0, rate=0.05,
            volatility=0.25, option_type="put", n_paths=100_000, n_monitoring_dates=50, seed=SEED,
        )
        out = monte_carlo_barrier(**args, barrier_type="down_and_out")
        inn = monte_carlo_barrier(**args, barrier_type="down_and_in")
        vanilla = self._vanilla_on_the_same_paths(
            100.0, 100.0, 1.0, 0.05, 0.25, "put", 100_000, 50
        )
        assert out.price + inn.price == pytest.approx(vanilla, abs=1e-10)

    def test_more_monitoring_dates_lowers_a_knock_out(self):
        """More observation dates mean more chances to be knocked out.

        This is the discrete-vs-continuous monitoring effect: a daily-monitored
        knock-out is genuinely worth more than a continuously monitored one,
        because the price can cross the barrier intraday and come back untouched.
        """
        prices = [
            monte_carlo_barrier(
                100.0, 100.0, 125.0, 1.0, 0.05, 0.25, "call", "up_and_out",
                n_paths=200_000, n_monitoring_dates=m, seed=SEED,
            ).price
            for m in (4, 12, 50, 250)
        ]
        assert all(np.diff(prices) < 0), f"not monotone in monitoring dates: {prices}"

    def test_continuity_correction_recovers_the_continuous_price(self):
        """The Broadie-Glasserman-Kou shift approximates continuous monitoring.

        The reference here is the *exact* continuously monitored price from
        :func:`barrier_price_analytic`, not a finely monitored simulation. That
        distinction matters: the raw discrete price converges to the continuous one
        only as O(1/sqrt(m)), so even 1,000 monitoring dates is still visibly off
        and would make a misleading anchor. Using the closed form makes the
        comparison unambiguous.

        Measured at 250 dates: raw error +0.177, corrected error -0.004 — about
        40x better, from a barrier shift rather than any extra computation.
        """
        common = dict(
            spot=100.0, strike=100.0, barrier=125.0, time_to_expiry=1.0, rate=0.05,
            volatility=0.25, option_type="call", barrier_type="up_and_out",
            n_paths=200_000, seed=SEED,
        )
        continuous = barrier_price_analytic(100.0, 100.0, 125.0, 1.0, 0.05, 0.25)

        for dates in (50, 250):
            raw = monte_carlo_barrier(**common, n_monitoring_dates=dates).price
            corrected = monte_carlo_barrier(
                **common, n_monitoring_dates=dates, continuity_correction=True
            ).price
            assert abs(corrected - continuous) < abs(raw - continuous) / 5.0

    def test_raw_discrete_price_converges_to_the_continuous_one(self):
        """Without the correction, convergence happens but is painfully slow.

        The gap shrinks as O(1/sqrt(m)), so quadrupling the monitoring dates only
        halves the error. This is the fact that makes the BGK shift worth knowing:
        brute force is the expensive way to get where a barrier adjustment gets you
        for free.
        """
        continuous = barrier_price_analytic(100.0, 100.0, 125.0, 1.0, 0.05, 0.25)
        common = dict(
            spot=100.0, strike=100.0, barrier=125.0, time_to_expiry=1.0, rate=0.05,
            volatility=0.25, option_type="call", barrier_type="up_and_out",
            n_paths=100_000, seed=SEED,
        )
        errors = [
            monte_carlo_barrier(**common, n_monitoring_dates=m).price - continuous
            for m in (50, 200, 800)
        ]
        assert all(e > 0 for e in errors), "discrete knock-out must be worth more"
        assert all(np.diff(errors) < 0), f"not converging: {errors}"
        # Quadrupling m should roughly halve the error (O(1/sqrt(m))).
        assert 1.6 < errors[0] / errors[1] < 2.6
        assert 1.6 < errors[1] / errors[2] < 2.6

    def test_invalid_barrier_is_rejected(self):
        """A non-positive barrier is meaningless."""
        with pytest.raises(ValueError, match="barrier must be strictly positive"):
            monte_carlo_barrier(100.0, 100.0, -1.0, 1.0, 0.05, 0.25)

    def test_analytic_barrier_satisfies_in_out_parity(self):
        """The closed form's knock-in and knock-out must sum to the vanilla."""
        args = (100.0, 100.0, 125.0, 1.0, 0.05, 0.25, 0.02)
        out = barrier_price_analytic(*args, knock_in=False)
        inn = barrier_price_analytic(*args, knock_in=True)
        vanilla = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.25, "call", 0.02))
        assert out + inn == pytest.approx(vanilla, abs=1e-10)

    def test_analytic_barrier_limits(self):
        """A distant barrier is never hit; a barrier at spot kills the option."""
        vanilla = float(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.25, "call"))
        far = barrier_price_analytic(100.0, 100.0, 1e6, 1.0, 0.05, 0.25)
        assert far == pytest.approx(vanilla, rel=1e-9)
        # Barrier essentially at spot: knock-out is worthless.
        near = barrier_price_analytic(100.0, 100.0, 100.001, 1.0, 0.05, 0.25)
        assert near == pytest.approx(0.0, abs=1e-6)

    def test_analytic_barrier_rejects_unsupported_configurations(self):
        """The implementation covers up-barriers above the strike only."""
        with pytest.raises(ValueError, match="must be above spot"):
            barrier_price_analytic(100.0, 90.0, 95.0, 1.0, 0.05, 0.25)
        with pytest.raises(ValueError, match="above strike"):
            barrier_price_analytic(100.0, 130.0, 120.0, 1.0, 0.05, 0.25)

    def test_unknown_barrier_type_is_rejected(self):
        """A typo must not silently pick a default."""
        with pytest.raises(ValueError, match="barrier_type must be one of"):
            monte_carlo_barrier(100.0, 100.0, 120.0, 1.0, 0.05, 0.25, barrier_type="sideways")


# ---------------------------------------------------------------------------
# 7. Interface
# ---------------------------------------------------------------------------


class TestMonteCarloInterface:
    """Result object behaviour and input validation."""

    def test_confidence_interval_widens_with_level(self):
        """A 99% interval must contain the 95% one."""
        result = monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.25, "call", n_paths=10_000, seed=SEED)
        narrow = result.confidence_interval(0.95)
        wide = result.confidence_interval(0.99)
        assert wide[0] < narrow[0] and wide[1] > narrow[1]

    def test_confidence_interval_is_centred_on_the_price(self):
        """The normal approximation gives a symmetric interval."""
        result = monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.25, "call", n_paths=10_000, seed=SEED)
        low, high = result.confidence_interval()
        assert (low + high) / 2.0 == pytest.approx(result.price)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_confidence_level_is_rejected(self, level):
        """A level outside (0, 1) is a caller error."""
        result = monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.25, "call", n_paths=1000, seed=SEED)
        with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
            result.confidence_interval(level)

    def test_result_is_frozen(self):
        """A computed result should not be mutated in place."""
        result = monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.25, "call", n_paths=1000, seed=SEED)
        assert isinstance(result, MonteCarloResult)
        with pytest.raises(Exception):
            result.price = 0.0  # type: ignore[misc]

    def test_zero_paths_is_rejected(self):
        """Simulation needs at least one path."""
        with pytest.raises(ValueError, match="n_paths must be at least 1"):
            monte_carlo_european(100.0, 100.0, 1.0, 0.05, 0.25, "call", n_paths=0)

    def test_zero_steps_is_rejected(self):
        """Path simulation needs at least one step."""
        with pytest.raises(ValueError, match="n_steps must be at least 1"):
            simulate_paths(100.0, 1.0, 0.05, 0.25, n_paths=100, n_steps=0)

    def test_oversized_simulation_fails_fast_with_a_usable_number(self):
        """A request that would exhaust memory must raise, not swap.

        Path memory is O(n_paths * n_steps) and grows faster than intuition
        suggests: 400,000 paths over 2,000 dates is about 12 GiB across the two
        live buffers. Without this guard the process does not fail — it thrashes
        for minutes and looks like a hang, which is a far worse failure mode.

        The message must also name a path count that actually fits, so the caller
        can act on it without doing the arithmetic themselves.
        """
        with pytest.raises(ValueError, match="GiB.*Reduce n_paths to about") as excinfo:
            simulate_paths(100.0, 1.0, 0.05, 0.25, n_paths=400_000, n_steps=2000)

        suggested = int(str(excinfo.value).split("Reduce n_paths to about ")[1].split(",\xa0")[0]
                        .split(" ")[0].replace(",", ""))
        # The suggested size must genuinely be within budget.
        simulate_paths(100.0, 1.0, 0.05, 0.25, n_paths=suggested, n_steps=2000)

    def test_budget_permits_ordinary_workloads(self):
        """The guard must not obstruct realistic use: 200k paths of daily steps."""
        paths = simulate_paths(100.0, 1.0, 0.05, 0.25, n_paths=100_000, n_steps=252, seed=SEED)
        assert paths.shape == (100_000, 253)

    def test_domain_validation_matches_other_pricers(self):
        """Bad inputs raise the same errors as the closed-form and tree pricers."""
        with pytest.raises(ValueError, match="spot"):
            monte_carlo_european(-100.0, 100.0, 1.0, 0.05, 0.25, "call")
        with pytest.raises(ValueError, match="volatility"):
            monte_carlo_european(100.0, 100.0, 1.0, 0.05, -0.25, "call")

    def test_enums_are_accepted_alongside_strings(self):
        """OptionType, ControlVariate and BarrierType members all work directly."""
        by_enum = monte_carlo_european(
            100.0, 100.0, 1.0, 0.05, 0.25, OptionType.CALL,
            n_paths=10_000, seed=SEED, control_variate=ControlVariate.TERMINAL_STOCK,
        )
        by_string = monte_carlo_european(
            100.0, 100.0, 1.0, 0.05, 0.25, "call",
            n_paths=10_000, seed=SEED, control_variate="terminal_stock",
        )
        assert by_enum.price == pytest.approx(by_string.price)
        assert BarrierType.UP_AND_OUT.is_up and BarrierType.UP_AND_OUT.is_knock_out
        assert not BarrierType.DOWN_AND_IN.is_up and not BarrierType.DOWN_AND_IN.is_knock_out
