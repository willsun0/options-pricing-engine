r"""SVI: the classical parametric smile, and the baseline the neural net must beat.

===============================================================================
WHY PARAMETRISE THE SMILE AT ALL
===============================================================================

Phase 4 produced a cloud of points: one implied volatility per traded strike and
expiry. That cloud is not yet a *surface*. Three things are missing.

1. **Interpolation.** Traders need a volatility at strikes that are not listed —
   for an over-the-counter option, or to reprice a book on a finer grid than the
   exchange quotes.
2. **Smoothing.** Individual quotes carry bid-ask noise. Phase 4 showed the raw
   smile is visibly ragged in the wings, where a one-cent price change moves the
   implied vol by a point.
3. **No-arbitrage structure.** An arbitrary interpolant through noisy points can
   easily imply a *negative probability density*. That is not a cosmetic problem:
   the second derivative of call price with respect to strike is, by Breeden and
   Litzenberger (1978), the risk-neutral density

       q(K) = e^{rT} d^2C/dK^2

   so a surface that bends the wrong way says the market assigns negative
   probability to an outcome. Any model calibrated to it will produce nonsense.

A parametric form solves all three at once: five numbers per expiry replace a
hundred noisy quotes, the function is smooth by construction, and the parameter
region that is arbitrage-free is *known in closed form*.

-------------------------------------------------------------------------------
THE RAW SVI PARAMETRISATION (GATHERAL, 2004)
-------------------------------------------------------------------------------

SVI = "stochastic volatility inspired". It models one expiry slice at a time, and
it models **total implied variance** rather than implied volatility:

    w(k) = sigma_BS(k)^2 * T          (total variance)
    k    = ln(K / F)                  (log-moneyness against the forward)

The raw form is

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

with five parameters, each of which has a reading a trader would recognise:

    a      vertical level      — shifts the whole slice up or down.
    b      slope / wing angle  — larger b means steeper wings on both sides.
    rho    skew, in [-1, 1]    — tilts left versus right.
    m      horizontal shift    — where the minimum sits in log-moneyness.
    sigma  curvature / ATM     — how rounded the vertex is. sigma -> 0 gives a
                                 kink; large sigma gives a broad smooth bowl.

One caveat about reading ``rho`` as "the skew", because it bites on real data.
``rho`` tilts the curve *relative to its vertex at k = m*. When the fitted vertex
lands outside the quoted strike range — which happens on short-dated equity
slices, where every listed strike sits on the downward-sloping left wing — the
data only ever sees one branch of the hyperbola, and ``rho`` can come out
positive while the fitted smile still falls monotonically across every observed
strike. The parameters are not wrong; ``rho`` simply is not the quantity a trader
means by skew. That quantity is the at-the-money slope ``d(sigma)/dk`` at k = 0,
which :meth:`SVIParameters.atm_skew` reports directly and which is negative on
this data as expected.

Two properties explain why this particular functional form, out of the many that
could fit a smile, is the one that stuck:

* **The wings are asymptotically linear in k.** As k -> +/- infinity,
  w(k) ~ b(rho +/- 1)(k - m) + const. Roger Lee's moment formula (2004) proves
  total variance *must* grow at most linearly in |k| for any model with the
  relevant moments finite. So SVI has exactly the right tail behaviour — it is
  not a curve that happens to fit, it is a curve with the correct asymptotics.

* **The no-arbitrage region is characterisable.** The conditions below are
  checkable inequalities on five numbers, not a search over function space.

-------------------------------------------------------------------------------
WHY TOTAL VARIANCE AND NOT VOLATILITY
-------------------------------------------------------------------------------

Total variance is the natural coordinate because it is what *adds* across time.
Under Black-Scholes with a term structure of vol, variance accumulates linearly:
w(T2) - w(T1) is the variance earned between the two dates. Two consequences:

* Calendar-spread arbitrage has a trivial statement in w: slices must not cross,
  ``w(k, T1) <= w(k, T2)`` for ``T1 < T2``. In volatility units the same
  condition is a messy inequality involving both T's.
* Interpolating *between* expiries is linear in w, not in sigma. Interpolating
  volatility linearly in T is a classic and silent mistake.

-------------------------------------------------------------------------------
THE NO-ARBITRAGE CONDITIONS
-------------------------------------------------------------------------------

**Static (parameter) conditions** — cheap, enforced during the fit:

    b >= 0                              wings point upwards
    |rho| < 1                           otherwise one wing is flat or inverted
    sigma > 0                           positive curvature scale
    a + b*sigma*sqrt(1 - rho^2) >= 0    the minimum of w is non-negative,
                                        i.e. no negative variance anywhere
    b * (1 + |rho|) <= 4 / T            Lee's slope bound: the steeper wing
                                        cannot exceed 2 in w-per-k, or call
                                        spreads are arbitrageable

**Butterfly (density) condition** — Durrleman's function must be non-negative:

    g(k) = (1 - k w'(k) / (2 w(k)))^2 - (w'(k)^2 / 4) (1/w(k) + 1/4) + w''(k)/2

``g(k) >= 0`` for all k is exactly the statement that the implied risk-neutral
density is non-negative. It is checked numerically after fitting rather than
imposed as a constraint, because it is a condition on a *function*, not on the
parameters directly; :func:`is_butterfly_free` reports it and
:func:`fit_svi_slice` records it in the fit result.

-------------------------------------------------------------------------------
HOW THE FIT IS DONE HERE
-------------------------------------------------------------------------------

The objective is squared error in **implied volatility units**, not in total
variance. That is a deliberate departure from the classical presentation, and the
reason is comparability: Phase 5 puts SVI head to head with a neural network, and
the network is trained on volatility. Scoring them on different objectives and
then comparing the scores would be meaningless. Volatility is also the unit a
trader reads, so an RMSE of "0.4 vol points" is directly interpretable.

The optimiser is SLSQP, because the two non-box constraints above (positive
minimum variance, Lee's slope bound) are genuine inequality constraints that a
box-bounded least-squares routine cannot express. SLSQP is run from a small
deterministic grid of starting points, because the SVI objective is **not
convex** — the ``(m, sigma)`` pair in particular admits local minima where the
vertex latches onto the wrong part of the smile. A fixed grid keeps the fit
reproducible, which a random multi-start would not.

The production-grade alternative is Zeliade's *quasi-explicit* calibration
(2009), which observes that for fixed ``(m, sigma)`` the problem in
``(a, b, rho)`` is a **constrained linear** least squares with a closed-form
solution — reducing a non-convex 5D search to a well-behaved 2D one. It is
faster and more robust. It is not implemented here because the two-stage change
of variables costs a page of algebra to explain, and this fit converges reliably
on the data at hand; the multi-start grid is the readable version of the same
insurance.

References: Gatheral, *A parsimonious arbitrage-free implied volatility
parameterization* (2004); Gatheral & Jacquier, *Arbitrage-free SVI volatility
surfaces* (2014); Lee, *The moment formula for implied volatility at extreme
strikes* (2004); Durrleman (2010); Zeliade Systems, *Quasi-explicit calibration
of Gatheral's SVI model* (2009); Breeden & Litzenberger (1978).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

__all__ = [
    "SVIParameters",
    "SVIFitResult",
    "svi_total_variance",
    "svi_first_derivative",
    "svi_second_derivative",
    "durrleman_function",
    "is_butterfly_free",
    "fit_svi_slice",
    "fit_svi_surface",
    "svi_surface_volatility",
    "worst_calendar_gap",
    "MIN_TOTAL_VARIANCE",
    "MAX_ABSOLUTE_RHO",
]

# Floor for total variance when converting back to volatility. The static
# constraints only guarantee w >= 0, and w = 0 exactly is attainable at the
# vertex; dividing by sqrt(0) would produce a NaN mid-optimisation and abort an
# otherwise healthy fit. 1e-12 in total variance is 1e-6 vol on a one-year
# option — far below a quote's resolution, so it never distorts a real fit.
MIN_TOTAL_VARIANCE: float = 1e-12

# |rho| = 1 makes one wing exactly flat, which is a degenerate limit rather than
# a usable fit, so the box excludes it strictly.
MAX_ABSOLUTE_RHO: float = 0.999


def svi_total_variance(
    log_moneyness: NDArray[np.float64] | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Evaluate the raw SVI total variance ``w(k)``.

    Args:
        log_moneyness: Log-moneyness ``k = ln(K/F)``, scalar or array.
        a: Vertical level parameter.
        b: Wing slope parameter, non-negative.
        rho: Skew parameter in ``(-1, 1)``.
        m: Horizontal shift of the vertex.
        sigma: Curvature scale, positive.

    Returns:
        Total implied variance ``w(k) = sigma_BS(k)^2 * T``.
    """
    k = np.asarray(log_moneyness, dtype=float)
    centred = k - m
    return a + b * (rho * centred + np.sqrt(centred * centred + sigma * sigma))


def svi_first_derivative(
    log_moneyness: NDArray[np.float64] | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Evaluate ``dw/dk`` for raw SVI.

    The derivative is available in closed form, which matters: the butterfly
    condition needs ``w'`` and ``w''``, and computing them by finite differences
    would make the arbitrage check depend on an arbitrary bump size.

    Args:
        log_moneyness: Log-moneyness ``k``, scalar or array.
        a: Unused; present so the signature matches the other SVI functions.
        b: Wing slope parameter.
        rho: Skew parameter.
        m: Horizontal shift.
        sigma: Curvature scale.

    Returns:
        ``dw/dk`` at each ``k``.
    """
    del a  # w' does not depend on the vertical level.
    k = np.asarray(log_moneyness, dtype=float)
    centred = k - m
    return b * (rho + centred / np.sqrt(centred * centred + sigma * sigma))


def svi_second_derivative(
    log_moneyness: NDArray[np.float64] | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Evaluate ``d^2w/dk^2`` for raw SVI.

    Strictly positive whenever ``b > 0`` and ``sigma > 0``: raw SVI is convex in
    total variance by construction. Convexity in ``w`` is necessary but *not*
    sufficient for a non-negative density, which is why
    :func:`durrleman_function` exists.

    Args:
        log_moneyness: Log-moneyness ``k``, scalar or array.
        a: Unused; present for signature symmetry.
        b: Wing slope parameter.
        rho: Unused; the curvature is independent of the skew.
        m: Horizontal shift.
        sigma: Curvature scale.

    Returns:
        ``d^2w/dk^2`` at each ``k``.
    """
    del a, rho
    k = np.asarray(log_moneyness, dtype=float)
    centred = k - m
    return b * sigma * sigma / np.power(centred * centred + sigma * sigma, 1.5)


@dataclass(frozen=True)
class SVIParameters:
    """One calibrated SVI slice: five parameters plus the expiry they belong to.

    The expiry is carried alongside the parameters rather than passed separately
    because total variance is meaningless without it — converting ``w`` back to a
    quoted volatility requires dividing by ``T``, and a slice detached from its
    ``T`` invites exactly that mistake.

    Attributes:
        a: Vertical level.
        b: Wing slope, non-negative.
        rho: Skew in ``(-1, 1)``; negative for equity index smiles.
        m: Horizontal shift of the vertex.
        sigma: Curvature scale, positive.
        time_to_expiry: Expiry of this slice in years.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    time_to_expiry: float

    def total_variance(
        self, log_moneyness: NDArray[np.float64] | float
    ) -> NDArray[np.float64]:
        """Total implied variance ``w(k)`` for this slice.

        Args:
            log_moneyness: Log-moneyness ``k = ln(K/F)``.

        Returns:
            Total variance at each ``k``.
        """
        return svi_total_variance(log_moneyness, self.a, self.b, self.rho, self.m, self.sigma)

    def implied_volatility(
        self, log_moneyness: NDArray[np.float64] | float
    ) -> NDArray[np.float64]:
        """Black-Scholes implied volatility implied by this slice.

        Args:
            log_moneyness: Log-moneyness ``k = ln(K/F)``.

        Returns:
            Implied volatility ``sqrt(w(k) / T)`` at each ``k``.
        """
        variance = np.maximum(self.total_variance(log_moneyness), MIN_TOTAL_VARIANCE)
        return np.sqrt(variance / self.time_to_expiry)

    def minimum_total_variance(self) -> float:
        """Return the smallest total variance the slice attains.

        Differentiating ``w`` and setting it to zero gives the vertex at
        ``k* = m - rho*sigma/sqrt(1 - rho^2)``, where
        ``w(k*) = a + b*sigma*sqrt(1 - rho^2)``. This is the quantity that must
        stay non-negative for the slice to be free of negative variance.

        Returns:
            The minimum of ``w`` over all ``k``.
        """
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    def atm_volatility(self) -> float:
        """Implied volatility at the forward, ``k = 0``.

        Returns:
            The at-the-money-forward implied volatility.
        """
        return float(self.implied_volatility(0.0))

    def atm_skew(self) -> float:
        """At-the-money skew ``d(sigma)/dk`` at ``k = 0``.

        This, not ``rho``, is what a trader means by "the skew" — the rate at
        which quoted volatility falls as you move up in strike. Differentiating
        ``sigma = sqrt(w/T)`` gives ``dsigma/dk = w'(k) / (2 sqrt(w(k) T))``.
        Equity index smiles give a negative value.

        Returns:
            The slope of implied volatility in log-moneyness at the forward.
        """
        w = max(float(self.total_variance(0.0)), MIN_TOTAL_VARIANCE)
        w_prime = float(
            svi_first_derivative(0.0, self.a, self.b, self.rho, self.m, self.sigma)
        )
        return w_prime / (2.0 * np.sqrt(w * self.time_to_expiry))

    def as_dict(self) -> dict[str, float]:
        """Return the parameters as a plain dictionary, for tabulating fits.

        Returns:
            Mapping of parameter name to value, including ``time_to_expiry``.
        """
        return {
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma,
            "time_to_expiry": self.time_to_expiry,
        }


def durrleman_function(
    parameters: SVIParameters, log_moneyness: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    r"""Evaluate Durrleman's function ``g(k)``, whose sign is the density's sign.

    .. math::

        g(k) = \left(1 - \frac{k\,w'(k)}{2 w(k)}\right)^2
               - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac14\right)
               + \frac{w''(k)}{2}

    ``g(k) >= 0`` everywhere is equivalent to the risk-neutral density implied by
    the slice being non-negative — that is, to the absence of butterfly
    arbitrage. Where ``g`` dips below zero, a butterfly spread centred on that
    strike has negative cost and non-negative payoff.

    Args:
        parameters: The calibrated slice.
        log_moneyness: Log-moneyness values at which to evaluate ``g``.

    Returns:
        ``g(k)`` at each ``k``. Negative entries indicate butterfly arbitrage.
    """
    k = np.asarray(log_moneyness, dtype=float)
    p = parameters
    w = np.maximum(svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma), MIN_TOTAL_VARIANCE)
    w_prime = svi_first_derivative(k, p.a, p.b, p.rho, p.m, p.sigma)
    w_double_prime = svi_second_derivative(k, p.a, p.b, p.rho, p.m, p.sigma)

    return (
        (1.0 - k * w_prime / (2.0 * w)) ** 2
        - 0.25 * w_prime**2 * (1.0 / w + 0.25)
        + 0.5 * w_double_prime
    )


def is_butterfly_free(
    parameters: SVIParameters,
    k_min: float = -1.5,
    k_max: float = 1.5,
    n_points: int = 601,
) -> bool:
    """Check numerically whether a slice admits butterfly arbitrage.

    The check is a dense grid scan rather than an analytic condition. Gatheral
    and Jacquier (2014) give closed-form sufficient conditions, but they are
    sufficient rather than necessary, so they reject slices that are in fact
    fine. Scanning ``g`` on a fine grid is exact up to the grid resolution and is
    trivial to explain — the right trade at this scale.

    The default range covers roughly ``exp(+/-1.5)``, i.e. strikes from a
    quarter to four and a half times the forward, well beyond any listed strike.

    Args:
        parameters: The calibrated slice.
        k_min: Lower end of the log-moneyness scan.
        k_max: Upper end of the log-moneyness scan.
        n_points: Number of grid points in the scan.

    Returns:
        ``True`` if ``g(k) >= 0`` across the whole grid.
    """
    grid = np.linspace(k_min, k_max, n_points)
    return bool(np.all(durrleman_function(parameters, grid) >= 0.0))


@dataclass(frozen=True)
class SVIFitResult:
    """The outcome of calibrating one slice, with the diagnostics to judge it.

    Attributes:
        parameters: The fitted slice.
        rmse: Root mean squared error in implied volatility units (0.004 = 0.4
            vol points).
        max_absolute_error: Largest single-quote error, in volatility units.
        n_quotes: Number of quotes the slice was fitted to.
        butterfly_free: Whether ``g(k) >= 0`` across the range of log-moneyness
            actually quoted. This is the claim the data supports.
        butterfly_free_extrapolated: Whether ``g(k) >= 0`` across a much wider
            range than was quoted. Reported separately because it fails on real
            SPY slices while ``butterfly_free`` passes — the arbitrage appears
            beyond the last listed strike, where the fit is extrapolating and no
            quote constrains it. See the Phase 5 discussion: this is the
            parametric model's version of the extrapolation risk that the neural
            network is usually accused of.
        success: Whether the optimiser reported convergence from at least one
            starting point.
    """

    parameters: SVIParameters
    rmse: float
    max_absolute_error: float
    n_quotes: int
    butterfly_free: bool
    butterfly_free_extrapolated: bool
    success: bool


# Deterministic multi-start grid over the two parameters the objective is least
# convex in. m is the vertex location, which can latch onto the wrong side of the
# smile; sigma is the curvature scale, which trades off against b. The remaining
# three are started from data-driven values, so nine starts suffice.
_M_STARTS: tuple[float, ...] = (-0.10, 0.0, 0.10)
_SIGMA_STARTS: tuple[float, ...] = (0.05, 0.20, 0.50)


def fit_svi_slice(
    log_moneyness: NDArray[np.float64],
    implied_vols: NDArray[np.float64],
    time_to_expiry: float,
    weights: NDArray[np.float64] | None = None,
) -> SVIFitResult:
    """Calibrate one raw SVI slice to a set of quotes at a single expiry.

    Minimises the (optionally weighted) mean squared error in implied volatility
    subject to the static no-arbitrage constraints, using SLSQP from a fixed grid
    of starting points and keeping the best result.

    Args:
        log_moneyness: Log-moneyness ``k = ln(K/F)`` of each quote, shape ``(n,)``.
        implied_vols: Implied volatility of each quote, shape ``(n,)``.
        time_to_expiry: Expiry of the slice in years, strictly positive.
        weights: Optional non-negative weights, shape ``(n,)``. Useful for
            down-weighting wide-spread wing quotes. Defaults to equal weights.

    Returns:
        An :class:`SVIFitResult` carrying the parameters and fit diagnostics.

    Raises:
        ValueError: If the inputs are mismatched, too few to identify five
            parameters, non-positive in expiry or volatility, or if the weights
            are negative.
    """
    k = np.asarray(log_moneyness, dtype=float)
    vols = np.asarray(implied_vols, dtype=float)

    if k.shape != vols.shape:
        raise ValueError(f"log_moneyness {k.shape} and implied_vols {vols.shape} must match")
    if k.ndim != 1:
        raise ValueError(f"expected 1-D inputs, got shape {k.shape}")
    if k.size < 5:
        raise ValueError(f"need at least 5 quotes to identify 5 SVI parameters, got {k.size}")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(vols)):
        raise ValueError("log_moneyness and implied_vols must be finite")
    if time_to_expiry <= 0.0:
        raise ValueError(f"time_to_expiry must be positive, got {time_to_expiry}")
    if np.any(vols <= 0.0):
        raise ValueError("implied_vols must be strictly positive")

    if weights is None:
        weight_array = np.ones_like(vols)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != vols.shape:
            raise ValueError(f"weights {weight_array.shape} must match quotes {vols.shape}")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")
    weight_array = weight_array / weight_array.sum()

    target_variance = vols**2 * time_to_expiry

    def objective(theta: NDArray[np.float64]) -> float:
        """Weighted mean squared error in volatility units."""
        variance = np.maximum(svi_total_variance(k, *theta), MIN_TOTAL_VARIANCE)
        model_vols = np.sqrt(variance / time_to_expiry)
        return float(np.sum(weight_array * (model_vols - vols) ** 2))

    # --- Constraints -------------------------------------------------------
    # SLSQP treats "ineq" constraints as fun(theta) >= 0.
    constraints = (
        # Minimum total variance is non-negative: a + b*sigma*sqrt(1 - rho^2) >= 0.
        {
            "type": "ineq",
            "fun": lambda t: t[0] + t[1] * t[4] * np.sqrt(max(1.0 - t[2] ** 2, 0.0)),
        },
        # Lee's slope bound: the steeper wing satisfies b(1 + |rho|) <= 4/T.
        {
            "type": "ineq",
            "fun": lambda t: 4.0 / time_to_expiry - t[1] * (1.0 + abs(t[2])),
        },
    )

    # Box bounds are scaled to the data rather than fixed, so a two-week slice
    # (tiny total variance) and a one-year slice are both comfortably inside.
    max_variance = float(target_variance.max())
    bounds = (
        (-4.0 * max_variance, 4.0 * max_variance),          # a
        (0.0, 4.0 / time_to_expiry),                        # b, capped by Lee's bound
        (-MAX_ABSOLUTE_RHO, MAX_ABSOLUTE_RHO),              # rho
        (float(k.min()) - 1.0, float(k.max()) + 1.0),       # m
        (1e-4, 2.0),                                        # sigma
    )

    # Data-driven starts for the three parameters the objective is well behaved
    # in: put the level at the observed minimum variance, take the slope from the
    # observed spread of variance across the strike range, and start the skew at
    # the equity-typical negative value.
    variance_span = float(target_variance.max() - target_variance.min())
    k_span = max(float(k.max() - k.min()), 1e-3)
    b_start = float(np.clip(variance_span / k_span, 1e-4, 4.0 / time_to_expiry))
    a_start = float(target_variance.min())

    best_result = None
    best_objective = np.inf
    any_success = False

    for m_start in _M_STARTS:
        for sigma_start in _SIGMA_STARTS:
            start = np.array([a_start, b_start, -0.5, m_start, sigma_start], dtype=float)
            start = np.clip(start, [low for low, _ in bounds], [high for _, high in bounds])
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            any_success = any_success or bool(result.success)
            # Keep the lowest objective even from a run flagged unsuccessful:
            # SLSQP reports failure on iteration limits it was already close to
            # converging within, and the diagnostics below judge the fit anyway.
            if result.fun < best_objective:
                best_objective = float(result.fun)
                best_result = result

    assert best_result is not None  # the grid is non-empty, so a best always exists
    parameters = SVIParameters(*(float(x) for x in best_result.x), time_to_expiry=time_to_expiry)

    errors = parameters.implied_volatility(k) - vols
    return SVIFitResult(
        parameters=parameters,
        rmse=float(np.sqrt(np.mean(errors**2))),
        max_absolute_error=float(np.max(np.abs(errors))),
        n_quotes=int(k.size),
        butterfly_free=is_butterfly_free(parameters, float(k.min()), float(k.max())),
        butterfly_free_extrapolated=is_butterfly_free(parameters),
        success=any_success,
    )


def fit_svi_surface(
    log_moneyness: NDArray[np.float64],
    implied_vols: NDArray[np.float64],
    times_to_expiry: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    min_quotes_per_slice: int = 5,
) -> dict[float, SVIFitResult]:
    """Fit one SVI slice per distinct expiry present in the data.

    Slices are fitted **independently**. That is the standard first pass and it
    is what makes SVI so tractable, but it does not by itself rule out calendar
    arbitrage: two independently fitted slices can cross. Gatheral and Jacquier's
    surface SVI (SSVI) ties the slices together through a common ATM variance
    curve to prevent exactly that. Here the slices are fitted separately and the
    crossing is *checked* — Phase 5's figure plots the fitted total variance
    curves so any crossing is visible rather than assumed away.

    Args:
        log_moneyness: Log-moneyness of every quote, shape ``(n,)``.
        implied_vols: Implied volatility of every quote, shape ``(n,)``.
        times_to_expiry: Expiry of every quote in years, shape ``(n,)``.
        weights: Optional per-quote weights, shape ``(n,)``.
        min_quotes_per_slice: Skip expiries with fewer quotes than this. Five is
            the identifiability floor; slices near it are fitted but should be
            read with suspicion.

    Returns:
        Mapping from time to expiry to its fit result, in ascending expiry order.

    Raises:
        ValueError: If the input arrays have mismatched shapes.
    """
    k = np.asarray(log_moneyness, dtype=float)
    vols = np.asarray(implied_vols, dtype=float)
    expiries = np.asarray(times_to_expiry, dtype=float)

    if not (k.shape == vols.shape == expiries.shape):
        raise ValueError(
            f"log_moneyness {k.shape}, implied_vols {vols.shape} and "
            f"times_to_expiry {expiries.shape} must all match"
        )

    fits: dict[float, SVIFitResult] = {}
    for expiry in sorted(np.unique(expiries)):
        rows = expiries == expiry
        if int(rows.sum()) < min_quotes_per_slice:
            continue
        slice_weights = None if weights is None else np.asarray(weights, dtype=float)[rows]
        fits[float(expiry)] = fit_svi_slice(k[rows], vols[rows], float(expiry), slice_weights)
    return fits


def worst_calendar_gap(
    fits: dict[float, SVIFitResult],
    k_min: float,
    k_max: float,
    n_points: int = 2001,
) -> float:
    """Return the worst calendar-spread violation across consecutive slices.

    Total variance must be non-decreasing in expiry at every log-moneyness:
    ``w(k, T_long) >= w(k, T_short)``. If it is not, a calendar spread — long the
    far-dated option, short the near-dated one at the same strike — has negative
    cost and non-negative payoff.

    Because :func:`fit_svi_surface` fits each slice independently, nothing in the
    calibration enforces this. It has to be checked, which is what this is for.

    The scan range matters and should be chosen deliberately. Over the region
    where *both* slices have quotes, the fits on the Phase 4 SPY data are clean;
    widening the scan past the last listed strike produces violations. Passing
    the quoted range and the extrapolated range separately makes that distinction
    explicit rather than reporting a single misleading verdict.

    Args:
        fits: Slice fits keyed by time to expiry.
        k_min: Lower end of the log-moneyness scan.
        k_max: Upper end of the log-moneyness scan.
        n_points: Grid resolution of the scan.

    Returns:
        The minimum of ``w(k, T_long) - w(k, T_short)`` over all consecutive
        pairs and all grid points. Non-negative means no calendar arbitrage;
        the more negative, the worse the violation. Returns ``inf`` when there
        are fewer than two slices to compare.

    Raises:
        ValueError: If ``k_min`` is not below ``k_max``.
    """
    if k_min >= k_max:
        raise ValueError(f"k_min {k_min} must be below k_max {k_max}")

    expiries = sorted(fits)
    if len(expiries) < 2:
        return float("inf")

    grid = np.linspace(k_min, k_max, n_points)
    gaps = [
        fits[longer].parameters.total_variance(grid)
        - fits[shorter].parameters.total_variance(grid)
        for shorter, longer in zip(expiries, expiries[1:])
    ]
    return float(np.min(gaps))


def svi_surface_volatility(
    fits: dict[float, SVIFitResult],
    log_moneyness: NDArray[np.float64] | float,
    time_to_expiry: float,
) -> NDArray[np.float64]:
    """Evaluate the fitted surface at an arbitrary expiry, interpolating in ``w``.

    Interpolation between slices is **linear in total variance**, which is the
    only defensible choice: total variance is what accumulates linearly in time,
    so linear interpolation in ``w`` corresponds to a piecewise-constant forward
    variance between the two expiries. Interpolating implied volatility linearly
    in ``T`` instead would imply a forward variance that is wrong by a factor
    involving the ratio of the expiries — a mistake that shows up as a kinked
    term structure.

    Outside the fitted expiry range the nearest slice's total variance is scaled
    by ``T / T_nearest``, i.e. the volatility is held flat. This is deliberately
    the dullest possible extrapolation: with nothing observed beyond the last
    expiry there is no information to extrapolate *with*, and a flat vol at least
    cannot cross the last fitted slice.

    Args:
        fits: Slice fits keyed by time to expiry, as returned by
            :func:`fit_svi_surface`.
        log_moneyness: Log-moneyness values to evaluate at.
        time_to_expiry: Expiry in years, which need not be one of the fitted ones.

    Returns:
        Implied volatility at each ``k`` for the requested expiry.

    Raises:
        ValueError: If ``fits`` is empty or ``time_to_expiry`` is not positive.
    """
    if not fits:
        raise ValueError("fits is empty; nothing to interpolate")
    if time_to_expiry <= 0.0:
        raise ValueError(f"time_to_expiry must be positive, got {time_to_expiry}")

    k = np.asarray(log_moneyness, dtype=float)
    expiries = sorted(fits)

    if time_to_expiry <= expiries[0]:
        variance = fits[expiries[0]].parameters.total_variance(k) * (
            time_to_expiry / expiries[0]
        )
    elif time_to_expiry >= expiries[-1]:
        variance = fits[expiries[-1]].parameters.total_variance(k) * (
            time_to_expiry / expiries[-1]
        )
    else:
        upper_index = int(np.searchsorted(expiries, time_to_expiry, side="left"))
        lower_expiry, upper_expiry = expiries[upper_index - 1], expiries[upper_index]
        span = upper_expiry - lower_expiry
        weight = (time_to_expiry - lower_expiry) / span
        variance = (1.0 - weight) * fits[lower_expiry].parameters.total_variance(k) + (
            weight * fits[upper_expiry].parameters.total_variance(k)
        )

    return np.sqrt(np.maximum(variance, MIN_TOTAL_VARIANCE) / time_to_expiry)
