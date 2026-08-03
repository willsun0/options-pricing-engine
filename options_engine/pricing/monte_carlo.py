r"""Monte Carlo pricing with variance reduction, plus Asian and barrier exotics.

===============================================================================
WHY SIMULATE AT ALL
===============================================================================

Black-Scholes gives a formula; the binomial tree handles early exercise. Both run
out of road when the payoff depends on the *whole path* (an average, a barrier
touch) or on several underlyings at once. Monte Carlo does not care: if you can
simulate the path and evaluate the payoff, you can price it.

The price is a discounted risk-neutral expectation,

    V_0 = e^{-rT} E^Q[ payoff ]

and an expectation is an integral, and Monte Carlo integrates by averaging samples.
That is the entire idea. Everything else is about making the average converge
faster.

-------------------------------------------------------------------------------
THE CONVERGENCE RATE, AND WHY IT IS BOTH BAD AND GOOD
-------------------------------------------------------------------------------

By the central limit theorem, the standard error of the estimate is

    SE = sigma_payoff / sqrt(N)

**This is slow.** Cutting the error in half needs 4x the paths; one more decimal
digit needs 100x. Against the binomial tree's O(1/N) from Phase 2, Monte Carlo
looks terrible for a vanilla European option — and it is. Nobody prices a vanilla
by simulation when a formula exists.

**But the rate is independent of dimension.** A lattice or PDE grid costs
O(steps^d) in d underlyings and becomes hopeless past three or four. Monte Carlo's
sqrt(N) does not know or care what d is. That single property is why every desk
runs simulations for basket, spread, and path-dependent products.

Since the O(1/sqrt(N)) exponent cannot be improved by cleverness, the only lever
is the numerator: shrink ``sigma_payoff``. That is what variance reduction means,
and it is the substance of this module.

-------------------------------------------------------------------------------
NO DISCRETISATION BIAS HERE
-------------------------------------------------------------------------------

Many Monte Carlo write-ups reach for an Euler scheme,
``S_{t+dt} = S_t (1 + r dt + sigma sqrt(dt) Z)``, which introduces O(dt) bias on
top of the sampling error. **None of that is necessary for geometric Brownian
motion**, because the SDE has an exact solution:

    S_T = S_0 exp( (r - q - sigma^2/2) T + sigma sqrt(T) Z )

and, just as importantly, an exact *transition* between any two dates. So we can
jump straight to expiry for a European payoff, and step exactly from one
monitoring date to the next for a path-dependent one. Every price here is
unbiased: the only error is statistical, and the reported standard error
genuinely bounds it.

The one exception is a modelling choice rather than a numerical error: a barrier
monitored at discrete dates is a *different contract* from one monitored
continuously. See :func:`monte_carlo_barrier`.

===============================================================================
VARIANCE REDUCTION
===============================================================================

-------------------------------------------------------------------------------
1. Antithetic variates
-------------------------------------------------------------------------------

For every normal draw ``Z``, also use ``-Z``. Both are valid samples (the normal
is symmetric), so the estimator stays unbiased.

**Why it works.** Average the payoffs of a mirrored pair,
``Y_pair = (f(Z) + f(-Z)) / 2``. Then

    Var(Y_pair) = (1/2) [ Var(f) + Cov(f(Z), f(-Z)) ]

Comparing at *equal computational cost* (same number of payoff evaluations), the
variance ratio against plain Monte Carlo is

    Var_antithetic / Var_plain = 1 + rho,    rho = Corr(f(Z), f(-Z))

So the benefit is entirely determined by ``rho``:

* ``f`` **monotone** in ``Z`` (a vanilla call or put): ``rho < 0``, and the method
  wins. Deep in the money, ``f`` is nearly linear in ``Z``, ``rho -> -1``, and the
  variance nearly vanishes.
* ``f`` **symmetric** in ``Z`` (a straddle, which pays on moves in either
  direction): ``rho > 0``, and antithetic sampling is actively **harmful**. It is
  not a free win, and both cases are asserted in the tests.
* Deep **out of the money**: one leg of the pair is in the money only when the
  other is not, so the products are almost always zero, ``rho -> 0``, and the
  method does close to nothing.

The intuition: mirroring cancels the sampling error in the *mean* of Z. It helps
exactly to the extent that the payoff responds monotonically to that mean.

-------------------------------------------------------------------------------
2. Control variates
-------------------------------------------------------------------------------

Suppose alongside the payoff ``Y`` we can compute a correlated quantity ``X``
whose expectation ``E[X]`` we know *exactly*. Then for any constant ``c``,

    Y* = Y + c (X - E[X])

has the same expectation as ``Y`` — the added term is mean-zero — but different
variance. Minimising over ``c`` gives

    c* = -Cov(Y, X) / Var(X),   and   Var(Y*) = Var(Y) (1 - rho^2)

with ``rho = Corr(X, Y)``. The variance reduction factor is ``1 / (1 - rho^2)``,
which is dramatic when the control is highly correlated: rho = 0.99 gives a 50x
variance reduction, i.e. the same accuracy from 50x fewer paths.

The intuition is simple bookkeeping. We know the exact answer for ``X``, so we can
see the sampling error the random draws produced *on the control*. If ``Y`` moves
with ``X``, the same draws made a similar error on ``Y``, so we subtract off the
part we can measure.

Controls used here, in increasing order of usefulness:

* **Terminal stock** (``E[S_T] = S e^{(r-q)T}``) for a European option. Modest, but
  it demonstrates the machinery on a case where the true answer is known.
* **Black-Scholes vanilla** for an Asian or barrier option. This is the classic
  application: the exotic has no formula, but a vanilla on the same terminal price
  does, and the two are strongly correlated.
* **Geometric-average Asian** for an arithmetic Asian. Best of all — the geometric
  average is lognormal so it has an exact closed form (:func:`geometric_asian_price`),
  and it correlates with the arithmetic average at rho > 0.99. This is the standard
  textbook example of a control variate done right.

**One caveat, honestly stated.** Estimating ``c*`` from the same sample used for
the price makes the estimator very slightly biased, because ``c*`` and the payoffs
are then correlated. The bias is O(1/N) while the standard error is O(1/sqrt(N)),
so it is dominated and vanishes faster than the noise. Purists use a separate
pilot run; :func:`monte_carlo_european` and friends accept the O(1/N) bias, which
is standard practice.

-------------------------------------------------------------------------------
A NOTE ON STANDARD ERRORS WITH ANTITHETIC SAMPLING
-------------------------------------------------------------------------------

Antithetic paths are **not independent** — that is the entire point of them. So
the standard error must be computed over *pair averages*, not over the individual
paths. Treating 2N mirrored paths as 2N independent samples produces a confidence
interval that is simply wrong (usually too narrow, since rho < 0 is the case you
were hoping for). Every estimator here reduces to "effective samples" first and
computes moments from those; see :func:`_summarise`.

References: Glasserman, *Monte Carlo Methods in Financial Engineering*, ch. 4;
Kemna & Vorst (1990) for the geometric Asian control; Broadie, Glasserman & Kou
(1997) for the discrete-barrier continuity correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from options_engine.common import OptionType, intrinsic_value, validate_inputs
from options_engine.pricing.black_scholes import black_scholes_price

__all__ = [
    "MonteCarloResult",
    "ControlVariate",
    "BarrierType",
    "simulate_terminal_prices",
    "simulate_paths",
    "monte_carlo_european",
    "monte_carlo_asian",
    "monte_carlo_barrier",
    "geometric_asian_price",
    "BROADIE_GLASSERMAN_KOU_CONSTANT",
    "MEMORY_BUDGET_BYTES",
]

# Ceiling on the estimated peak allocation for a single path simulation. Two GiB is
# generous for a laptop and small enough to catch the accidental 6 GiB request
# before it starts swapping. Raise it deliberately if you have the headroom.
MEMORY_BUDGET_BYTES: int = 2 * 2**30

# Broadie-Glasserman-Kou (1997) continuity correction constant, beta = -zeta(1/2)/sqrt(2 pi).
# Shifting a discretely monitored barrier by exp(+/- beta sigma sqrt(dt)) makes the
# discrete price approximate the continuous one to O(1/sqrt(m)). See monte_carlo_barrier.
BROADIE_GLASSERMAN_KOU_CONSTANT: float = 0.5826


class ControlVariate(str, Enum):
    """Which control variate to use, if any.

    See the module docstring for what each control is and why it helps.
    """

    NONE = "none"
    TERMINAL_STOCK = "terminal_stock"
    EUROPEAN_OPTION = "european_option"
    GEOMETRIC_ASIAN = "geometric_asian"

    @classmethod
    def parse(cls, value: Union[str, "ControlVariate"]) -> "ControlVariate":
        """Coerce a string or enum member to a :class:`ControlVariate`.

        Args:
            value: A control variate name (any case) or enum member.

        Returns:
            The corresponding :class:`ControlVariate` member.

        Raises:
            ValueError: If ``value`` is not a recognised control variate.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(
                f"control_variate must be one of {valid}, got {value!r}"
            ) from None


class BarrierType(str, Enum):
    """Barrier direction and knock behaviour."""

    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"

    @classmethod
    def parse(cls, value: Union[str, "BarrierType"]) -> "BarrierType":
        """Coerce a string or enum member to a :class:`BarrierType`.

        Args:
            value: A barrier type name (any case) or enum member.

        Returns:
            The corresponding :class:`BarrierType` member.

        Raises:
            ValueError: If ``value`` is not a recognised barrier type.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(f"barrier_type must be one of {valid}, got {value!r}") from None

    @property
    def is_up(self) -> bool:
        """Whether the barrier is breached by the price moving *up*."""
        return self in (BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN)

    @property
    def is_knock_out(self) -> bool:
        """Whether breaching the barrier *cancels* the option."""
        return self in (BarrierType.UP_AND_OUT, BarrierType.DOWN_AND_OUT)


@dataclass(frozen=True)
class MonteCarloResult:
    """A Monte Carlo price together with its statistical uncertainty.

    A simulated price without a standard error is not a result, it is a rumour —
    so the two always travel together.

    Attributes:
        price: The estimated present value.
        standard_error: Standard error of that estimate. Computed over *effective*
            samples (antithetic pairs count once), never over raw paths.
        n_paths: Total number of simulated paths.
        n_samples: Number of independent samples the moments were computed from.
            Equals ``n_paths`` without antithetic sampling and ``n_paths / 2`` with it.
        control_beta: The fitted control-variate coefficient ``c*``, or ``None`` if
            no control was used. Useful for diagnosing how much the control helped.
    """

    price: float
    standard_error: float
    n_paths: int
    n_samples: int
    control_beta: float | None = None

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return a normal-approximation confidence interval for the price.

        The interval is symmetric and based on the central limit theorem, which is
        an excellent approximation at the path counts used here even though option
        payoffs are strongly non-normal (they are kinked and non-negative).

        Args:
            level: Confidence level, e.g. 0.95 for a 95% interval.

        Returns:
            A ``(lower, upper)`` tuple.

        Raises:
            ValueError: If ``level`` is not strictly between 0 and 1.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = float(norm.ppf(0.5 + level / 2.0))
        margin = z * self.standard_error
        return self.price - margin, self.price + margin

    def __repr__(self) -> str:
        return (
            f"MonteCarloResult(price={self.price:.6f}, se={self.standard_error:.6f}, "
            f"n_paths={self.n_paths})"
        )


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------


def _make_normals(
    n_samples: int, n_steps: int, rng: np.random.Generator, antithetic: bool
) -> NDArray[np.float64]:
    """Draw standard normals, mirrored into antithetic pairs if requested.

    Args:
        n_samples: Number of independent samples (antithetic *pairs* if enabled).
        n_steps: Number of time steps per sample.
        rng: NumPy random generator.
        antithetic: Whether to append the negated draws.

    Returns:
        An array of shape ``(n_paths, n_steps)`` where ``n_paths`` is ``n_samples``
        without antithetic sampling and ``2 * n_samples`` with it. When antithetic,
        row ``i`` and row ``i + n_samples`` are exact mirror images — the layout the
        rest of this module relies on for pairing.
    """
    draws = rng.standard_normal((n_samples, n_steps))
    if antithetic:
        # Stacking rather than interleaving keeps the pairing rule trivial:
        # row i pairs with row i + n_samples.
        return np.concatenate([draws, -draws], axis=0)
    return draws


def simulate_terminal_prices(
    spot: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Simulate terminal underlying prices ``S_T`` directly, with no time stepping.

    For a payoff that depends only on ``S_T`` there is no reason to walk the path.
    Geometric Brownian motion has an exact solution, so we sample the terminal
    price in a single draw:

        S_T = S_0 exp( (r - q - sigma^2/2) T + sigma sqrt(T) Z )

    That makes this estimator completely free of discretisation bias, unlike an
    Euler scheme. Note the ``-sigma^2/2``: it is the Ito correction from Phase 1,
    and dropping it is the classic simulation bug — prices come out systematically
    too high because the sampled mean of ``S_T`` exceeds the forward.

    Args:
        spot: Current price of the underlying. Must be positive.
        time_to_expiry: Time to expiry in years. Must be non-negative.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal. Must be non-negative.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of paths. Must be even if ``antithetic`` is set.
        seed: Seed for reproducibility. ``None`` draws from OS entropy.
        antithetic: Whether to use antithetic sampling. Row ``i`` then mirrors row
            ``i + n_paths // 2``.

    Returns:
        An array of ``n_paths`` terminal prices.

    Raises:
        ValueError: If inputs are out of domain, ``n_paths`` is not positive, or
            ``antithetic`` is set with an odd ``n_paths``.
    """
    validate_inputs(spot, spot, time_to_expiry, volatility)
    n_samples = _resolve_sample_count(n_paths, antithetic)

    rng = np.random.default_rng(seed)
    normals = _make_normals(n_samples, 1, rng, antithetic)[:, 0]

    drift = (rate - dividend_yield - 0.5 * volatility**2) * time_to_expiry
    diffusion = volatility * np.sqrt(time_to_expiry) * normals
    return spot * np.exp(drift + diffusion)


def simulate_paths(
    spot: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 52,
    seed: int | None = None,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Simulate full price paths on an evenly spaced monitoring grid.

    Uses the **exact** lognormal transition between consecutive dates, not an
    Euler approximation. Geometric Brownian motion is one of the few models where
    that is possible, and it means the simulated dates are distributed exactly
    right no matter how coarse the grid — there is no ``O(dt)`` bias to worry about.

    The path is built by exponentiating a cumulative sum of independent normal
    increments, which is both the natural way to write it and numerically better
    behaved than repeated multiplication (errors add rather than compound).

    Args:
        spot: Current price of the underlying. Must be positive.
        time_to_expiry: Time to expiry in years. Must be positive.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal. Must be non-negative.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of paths. Must be even if ``antithetic`` is set.
        n_steps: Number of monitoring dates (excluding time zero).
        seed: Seed for reproducibility.
        antithetic: Whether to use antithetic sampling.

    Returns:
        An array of shape ``(n_paths, n_steps + 1)``. Column 0 is ``spot`` for every
        path; column ``j`` is the price at time ``j * T / n_steps``.

    Raises:
        ValueError: If inputs are out of domain, ``n_steps`` is not positive, or the
            request would exceed :data:`MEMORY_BUDGET_BYTES`.

    Note:
        Memory scales as ``O(n_paths * n_steps)``, which is easy to underestimate:
        400,000 paths over 2,000 monitoring dates is 6.4 GB *per array*. The
        arithmetic below is done in place to keep the peak near two arrays rather
        than four, and :func:`_check_memory_budget` refuses requests that would
        thrash before they start. A production engine would instead process paths
        in batches, accumulating payoff moments as it goes; that is the right fix
        if you need it, and it is left out here because it buys nothing pedagogically
        and costs readability.
    """
    validate_inputs(spot, spot, time_to_expiry, volatility)
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    n_samples = _resolve_sample_count(n_paths, antithetic)
    _check_memory_budget(n_paths, n_steps)

    rng = np.random.default_rng(seed)
    # `values` starts as standard normals and is transformed in place into log
    # increments, then cumulative log returns, then price ratios. Reusing the one
    # buffer halves peak memory; the comments below track what it holds.
    values = _make_normals(n_samples, n_steps, rng, antithetic)

    dt = time_to_expiry / n_steps
    drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)

    values *= diffusion
    values += drift  # now: per-step log increments
    np.cumsum(values, axis=1, out=values)  # now: cumulative log return to each date
    np.exp(values, out=values)  # now: price ratio S_t / S_0

    paths = np.empty((values.shape[0], n_steps + 1), dtype=float)
    paths[:, 0] = spot
    np.multiply(values, spot, out=paths[:, 1:])
    return paths


def _check_memory_budget(n_paths: int, n_steps: int) -> None:
    """Refuse path simulations that would exhaust memory.

    Path storage is ``O(n_paths * n_steps)`` and the numbers get large faster than
    intuition suggests: 400,000 paths over 2,000 monitoring dates is 6.4 GB. Without
    a guard, such a request does not fail — it swaps, and the process crawls for
    many minutes looking like a hang. Failing immediately with the actual figure and
    a suggested remedy is far kinder.

    Args:
        n_paths: Number of paths requested.
        n_steps: Number of time steps requested.

    Raises:
        ValueError: If the estimated peak allocation exceeds
            :data:`MEMORY_BUDGET_BYTES`.
    """
    # Two live float64 buffers: the working array and the output paths.
    estimated_bytes = 2 * n_paths * (n_steps + 1) * 8
    if estimated_bytes > MEMORY_BUDGET_BYTES:
        affordable_paths = int(MEMORY_BUDGET_BYTES / (2 * (n_steps + 1) * 8))
        raise ValueError(
            f"simulating {n_paths:,} paths x {n_steps:,} steps needs about "
            f"{estimated_bytes / 2**30:.1f} GiB, over the "
            f"{MEMORY_BUDGET_BYTES / 2**30:.1f} GiB budget. Reduce n_paths to about "
            f"{affordable_paths:,}, reduce n_steps, or raise MEMORY_BUDGET_BYTES if "
            f"you really do have the memory."
        )


def _resolve_sample_count(n_paths: int, antithetic: bool) -> int:
    """Validate ``n_paths`` and return the number of independent samples.

    Args:
        n_paths: Requested total number of paths.
        antithetic: Whether antithetic sampling is enabled.

    Returns:
        The number of independent draws: ``n_paths`` normally, ``n_paths // 2``
        when antithetic (each draw yielding a mirrored pair).

    Raises:
        ValueError: If ``n_paths`` is not positive, or is odd while antithetic.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError(
                f"n_paths must be even when antithetic=True (paths come in mirrored "
                f"pairs), got {n_paths}"
            )
        return n_paths // 2
    return n_paths


# ---------------------------------------------------------------------------
# Estimator assembly
# ---------------------------------------------------------------------------


def _fold_antithetic(values: NDArray[np.float64], antithetic: bool) -> NDArray[np.float64]:
    """Collapse mirrored path pairs into one effective sample each.

    This is the step that makes the reported standard error correct. Antithetic
    paths are deliberately negatively correlated, so treating them as independent
    samples would misstate the uncertainty. Averaging each pair first yields
    genuinely independent samples, and every moment is computed from those.

    Args:
        values: Per-path values of shape ``(n_paths,)``. When antithetic, rows
            ``i`` and ``i + n_paths // 2`` are the mirrored pair.
        antithetic: Whether the values came from antithetic sampling.

    Returns:
        Effective samples: the input unchanged, or the pairwise means.
    """
    if not antithetic:
        return values
    half = values.shape[0] // 2
    return 0.5 * (values[:half] + values[half:])


def _apply_control_variate(
    payoffs: NDArray[np.float64],
    controls: NDArray[np.float64],
    control_expectation: float,
) -> tuple[NDArray[np.float64], float]:
    """Adjust payoffs using a control variate with a known expectation.

    Computes ``c* = -Cov(Y, X) / Var(X)`` and returns ``Y + c*(X - E[X])``, which
    has the same mean as ``Y`` but variance scaled by ``(1 - rho^2)``.

    Args:
        payoffs: Per-sample payoff values ``Y``.
        controls: Per-sample control values ``X``, same shape as ``payoffs``.
        control_expectation: The exact, analytically known ``E[X]``.

    Returns:
        A ``(adjusted_payoffs, beta)`` tuple, where ``beta`` is the fitted ``c*``.
    """
    control_variance = float(np.var(controls, ddof=1))
    if control_variance <= 0.0:
        # A degenerate control carries no information; leave the payoffs alone
        # rather than dividing by zero.
        return payoffs, 0.0

    covariance = float(np.cov(payoffs, controls, ddof=1)[0, 1])
    beta = -covariance / control_variance
    return payoffs + beta * (controls - control_expectation), beta


def _summarise(
    payoffs: NDArray[np.float64],
    discount: float,
    n_paths: int,
    antithetic: bool,
    controls: NDArray[np.float64] | None = None,
    control_expectation: float | None = None,
) -> MonteCarloResult:
    """Turn per-path payoffs into a discounted price with a standard error.

    Ordering matters: antithetic pairs are folded into effective samples *before*
    the control variate is fitted, so the regression sees independent observations
    and the resulting standard error is computed from independent samples too.

    Args:
        payoffs: Undiscounted per-path payoffs.
        discount: Discount factor ``e^{-rT}``.
        n_paths: Total number of paths simulated.
        antithetic: Whether antithetic sampling was used.
        controls: Optional per-path control values.
        control_expectation: Exact expectation of the control, required if
            ``controls`` is given.

    Returns:
        The assembled :class:`MonteCarloResult`.
    """
    samples = _fold_antithetic(payoffs, antithetic)
    beta: float | None = None

    if controls is not None:
        assert control_expectation is not None, "control_expectation required with controls"
        control_samples = _fold_antithetic(controls, antithetic)
        samples, beta = _apply_control_variate(samples, control_samples, control_expectation)

    n_samples = samples.shape[0]
    price = discount * float(np.mean(samples))
    # ddof=1 for the unbiased sample variance; with one sample the SE is undefined.
    standard_error = (
        discount * float(np.std(samples, ddof=1)) / np.sqrt(n_samples) if n_samples > 1 else float("nan")
    )
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        n_paths=n_paths,
        n_samples=n_samples,
        control_beta=beta,
    )


# ---------------------------------------------------------------------------
# European
# ---------------------------------------------------------------------------


def monte_carlo_european(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: str | ControlVariate = ControlVariate.NONE,
) -> MonteCarloResult:
    """Price a European option by simulating terminal prices.

    Pricing a vanilla by simulation is deliberately redundant — Phase 1 already
    solves it exactly — and that redundancy is the point: with the true answer
    known, the estimator's bias and the honesty of its standard error can both be
    verified, which is impossible for the exotics further down.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of simulated paths.
        seed: Seed for reproducibility.
        antithetic: Whether to use antithetic sampling. Helps for vanillas, whose
            payoff is monotone in the driving normal.
        control_variate: ``"none"`` or ``"terminal_stock"``. The terminal-stock
            control uses the exactly known ``E[S_T] = S e^{(r-q)T}``.

    Returns:
        A :class:`MonteCarloResult`.

    Raises:
        ValueError: If any input is out of domain or the control variate is not
            valid for a European payoff.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    opt = OptionType.parse(option_type)
    control = ControlVariate.parse(control_variate)
    if control not in (ControlVariate.NONE, ControlVariate.TERMINAL_STOCK):
        raise ValueError(
            f"control_variate {control.value!r} is not applicable to a European "
            f"option; use 'none' or 'terminal_stock'"
        )

    terminal = simulate_terminal_prices(
        spot, time_to_expiry, rate, volatility, dividend_yield, n_paths, seed, antithetic
    )
    payoffs = intrinsic_value(terminal, strike, opt)
    discount = float(np.exp(-rate * time_to_expiry))

    controls = None
    control_expectation = None
    if control is ControlVariate.TERMINAL_STOCK:
        controls = terminal
        control_expectation = spot * float(np.exp((rate - dividend_yield) * time_to_expiry))

    return _summarise(payoffs, discount, n_paths, antithetic, controls, control_expectation)


# ---------------------------------------------------------------------------
# Asian
# ---------------------------------------------------------------------------


def geometric_asian_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    n_averaging_dates: int = 52,
) -> float:
    r"""Price a discretely monitored **geometric**-average Asian option in closed form.

    The arithmetic average of lognormals is not lognormal, which is exactly why the
    arithmetic Asian has no formula. The *geometric* average is:

        ln G = (1/n) sum_i ln S_{t_i}

    is a sum of normals, hence normal, hence ``G`` is lognormal. That makes the
    option a Black-Scholes problem with a modified volatility and forward.

    With evenly spaced dates ``t_i = iT/n``:

        E[ln G]   = ln S + (r - q - sigma^2/2) * T(n+1)/(2n)
        Var[ln G] = sigma^2 T (n+1)(2n+1) / (6n^2)

    using ``sum_i sum_j min(i, j) = n(n+1)(2n+1)/6``.

    Rather than re-deriving a pricing formula, we map onto the Phase 1 pricer: the
    Black-Scholes function is fully determined by the forward and the total
    variance, so feeding it ``sigma_G`` together with a synthetic dividend yield
    that reproduces ``E[G]`` gives exactly ``e^{-rT} E[(G-K)^+]``. Reusing
    thoroughly tested code beats writing a second one.

    Note ``sigma_G < sigma``: averaging suppresses volatility, which is the general
    reason Asian options are cheaper than their vanilla equivalents. As
    ``n -> infinity`` the factor tends to ``1/sqrt(3)``.

    This function exists for two reasons: it validates the Monte Carlo machinery
    against an exact answer, and it is the best available control variate for the
    arithmetic Asian (Kemna & Vorst, 1990).

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years. Must be positive.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        n_averaging_dates: Number of evenly spaced averaging dates.

    Returns:
        The exact present value of the geometric-average Asian option.

    Raises:
        ValueError: If any input is out of domain or ``n_averaging_dates`` < 1.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    opt = OptionType.parse(option_type)
    if n_averaging_dates < 1:
        raise ValueError(f"n_averaging_dates must be at least 1, got {n_averaging_dates}")

    n = n_averaging_dates
    mean_time = time_to_expiry * (n + 1) / (2 * n)
    variance_factor = (n + 1) * (2 * n + 1) / (6 * n**2)

    log_mean = np.log(spot) + (rate - dividend_yield - 0.5 * volatility**2) * mean_time
    log_variance = volatility**2 * time_to_expiry * variance_factor
    effective_volatility = float(np.sqrt(log_variance / time_to_expiry))

    # E[G] = exp(mean + variance/2) for a lognormal. Express it as a forward so the
    # Black-Scholes pricer can consume it via a synthetic dividend yield:
    #   F = S e^{(r - q_eff) T}  =>  q_eff = r - ln(F / S) / T
    expected_average = float(np.exp(log_mean + 0.5 * log_variance))
    effective_dividend = rate - np.log(expected_average / spot) / time_to_expiry

    return float(
        black_scholes_price(
            spot, strike, time_to_expiry, rate, effective_volatility, opt, effective_dividend
        )
    )


def monte_carlo_asian(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    n_paths: int = 50_000,
    n_averaging_dates: int = 52,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: str | ControlVariate = ControlVariate.NONE,
) -> MonteCarloResult:
    """Price an arithmetic-average Asian option by simulation.

    The payoff is ``max(w (A - K), 0)`` where ``A`` is the arithmetic mean of the
    underlying over ``n_averaging_dates`` evenly spaced dates (excluding time
    zero, following market convention).

    There is no closed form, because a sum of lognormals is not lognormal. That is
    what makes this a genuine Monte Carlo problem rather than an exercise.

    Averaging damps the effective volatility, so an Asian option is worth less than
    the corresponding vanilla — a property the tests assert. Averaging is also what
    makes these contracts popular for hedging commodity and FX exposures, where the
    business risk really is an average price over a period, and it makes the payoff
    much harder to manipulate near expiry.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of simulated paths.
        n_averaging_dates: Number of averaging dates.
        seed: Seed for reproducibility.
        antithetic: Whether to use antithetic sampling.
        control_variate: ``"none"``, ``"european_option"`` (Black-Scholes vanilla on
            the terminal price), or ``"geometric_asian"`` (the exact geometric
            Asian — by far the most effective, correlating above 0.99).

    Returns:
        A :class:`MonteCarloResult`.

    Raises:
        ValueError: If any input is out of domain or the control is not applicable.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    opt = OptionType.parse(option_type)
    control = ControlVariate.parse(control_variate)
    if control is ControlVariate.TERMINAL_STOCK:
        raise ValueError(
            "control_variate 'terminal_stock' is a weak choice for an Asian option; "
            "use 'geometric_asian' or 'european_option'"
        )

    paths = simulate_paths(
        spot, time_to_expiry, rate, volatility, dividend_yield,
        n_paths, n_averaging_dates, seed, antithetic,
    )
    # Exclude the known initial price from the average: market convention is to
    # average observed fixings, and including a constant would only dilute the
    # payoff's sensitivity.
    monitored = paths[:, 1:]
    arithmetic_average = np.mean(monitored, axis=1)
    payoffs = intrinsic_value(arithmetic_average, strike, opt)
    discount = float(np.exp(-rate * time_to_expiry))

    controls = None
    control_expectation = None
    if control is ControlVariate.GEOMETRIC_ASIAN:
        # Geometric mean via logs: numerically safer than taking an n-th root of a
        # product, which would underflow for long averaging windows.
        geometric_average = np.exp(np.mean(np.log(monitored), axis=1))
        controls = intrinsic_value(geometric_average, strike, opt)
        control_expectation = geometric_asian_price(
            spot, strike, time_to_expiry, rate, volatility, opt, dividend_yield, n_averaging_dates
        ) / discount  # undiscounted, to match the raw payoffs
    elif control is ControlVariate.EUROPEAN_OPTION:
        controls = intrinsic_value(paths[:, -1], strike, opt)
        control_expectation = float(
            black_scholes_price(spot, strike, time_to_expiry, rate, volatility, opt, dividend_yield)
        ) / discount

    return _summarise(payoffs, discount, n_paths, antithetic, controls, control_expectation)


# ---------------------------------------------------------------------------
# Barrier
# ---------------------------------------------------------------------------


def monte_carlo_barrier(
    spot: float,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str | OptionType = OptionType.CALL,
    barrier_type: str | BarrierType = BarrierType.UP_AND_OUT,
    dividend_yield: float = 0.0,
    n_paths: int = 50_000,
    n_monitoring_dates: int = 52,
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: str | ControlVariate = ControlVariate.NONE,
    continuity_correction: bool = False,
) -> MonteCarloResult:
    """Price a discretely monitored knock-in or knock-out barrier option.

    A knock-out option is cancelled if the underlying ever touches the barrier; a
    knock-in only comes alive if it does. They are cheaper than vanillas, which is
    the entire commercial point.

    **Discrete vs continuous monitoring is a real distinction, not an approximation
    error.** A contract monitored at daily closes is genuinely different from one
    monitored continuously: the price can cross the barrier intraday and return
    without triggering. Discrete monitoring therefore makes knock-outs *more*
    valuable (fewer chances to be killed) and knock-ins *less* valuable. This
    simulation prices the discrete contract exactly — it is not converging to
    the continuous price and it should not.

    If you *do* want the continuous price, the gap closes only as ``O(1/sqrt(m))``
    in the number of monitoring dates, which is painfully slow. Broadie, Glasserman
    and Kou (1997) showed the fix is a barrier shift rather than more dates: move
    the barrier by ``exp(+/- 0.5826 sigma sqrt(dt))``, away from the spot for a
    discrete contract, towards it to approximate a continuous one. Setting
    ``continuity_correction=True`` applies the latter. The constant is
    ``-zeta(1/2)/sqrt(2 pi)`` and comes from the expected overshoot of a random walk
    past a level — a genuinely surprising result worth knowing.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        barrier: Barrier level. Must be positive.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        barrier_type: One of ``"up_and_out"``, ``"up_and_in"``, ``"down_and_out"``,
            ``"down_and_in"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        n_paths: Number of simulated paths.
        n_monitoring_dates: Number of barrier observation dates.
        seed: Seed for reproducibility.
        antithetic: Whether to use antithetic sampling. Note that barrier payoffs
            are discontinuous, so this helps less than for a vanilla.
        control_variate: ``"none"`` or ``"european_option"``.
        continuity_correction: Whether to apply the Broadie-Glasserman-Kou barrier
            shift so the discrete simulation approximates the *continuously*
            monitored price.

    Returns:
        A :class:`MonteCarloResult`.

    Raises:
        ValueError: If any input is out of domain or the control is not applicable.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    if barrier <= 0.0:
        raise ValueError(f"barrier must be strictly positive, got {barrier}")
    opt = OptionType.parse(option_type)
    barrier_spec = BarrierType.parse(barrier_type)
    control = ControlVariate.parse(control_variate)
    if control not in (ControlVariate.NONE, ControlVariate.EUROPEAN_OPTION):
        raise ValueError(
            f"control_variate {control.value!r} is not applicable to a barrier "
            f"option; use 'none' or 'european_option'"
        )

    effective_barrier = barrier
    if continuity_correction:
        dt = time_to_expiry / n_monitoring_dates
        shift = float(np.exp(BROADIE_GLASSERMAN_KOU_CONSTANT * volatility * np.sqrt(dt)))
        # Move the barrier *towards* spot so the coarse grid knocks out about as
        # often as continuous monitoring would.
        effective_barrier = barrier / shift if barrier_spec.is_up else barrier * shift

    paths = simulate_paths(
        spot, time_to_expiry, rate, volatility, dividend_yield,
        n_paths, n_monitoring_dates, seed, antithetic,
    )
    monitored = paths[:, 1:]

    if barrier_spec.is_up:
        breached = np.any(monitored >= effective_barrier, axis=1)
    else:
        breached = np.any(monitored <= effective_barrier, axis=1)

    alive = ~breached if barrier_spec.is_knock_out else breached
    payoffs = np.where(alive, intrinsic_value(paths[:, -1], strike, opt), 0.0)
    discount = float(np.exp(-rate * time_to_expiry))

    controls = None
    control_expectation = None
    if control is ControlVariate.EUROPEAN_OPTION:
        controls = intrinsic_value(paths[:, -1], strike, opt)
        control_expectation = float(
            black_scholes_price(spot, strike, time_to_expiry, rate, volatility, opt, dividend_yield)
        ) / discount

    return _summarise(payoffs, discount, n_paths, antithetic, controls, control_expectation)
