r"""Implied volatility: inverting Black-Scholes for the volatility input.

===============================================================================
WHAT IMPLIED VOLATILITY ACTUALLY IS
===============================================================================

Every input to Black-Scholes is observable except one. Spot, strike, expiry and
rates are all quoted; volatility is not. So practitioners run the formula
backwards: given the *market price*, find the sigma that reproduces it.

    Find sigma such that   BS(S, K, T, r, sigma, q) = market price

This is worth being precise about, because the phrase invites two
misinterpretations:

* **It is not a forecast of volatility.** It is the number that makes a
  particular model reproduce a particular price. If the model is wrong — and
  Phase 4 shows it visibly is — implied volatility absorbs everything the model
  gets wrong, not just volatility.

* **It is a quoting convention, not a belief.** Traders quote in vol because it
  strips out the mechanical dependence on spot, strike and expiry, making
  contracts comparable. Saying "that option is 18 vol" is a normalisation, in the
  same way bond traders quote yield rather than price. Nobody believes the stock
  has a different volatility depending on which strike you look at — yet the
  implied vols differ across strikes, and that discrepancy *is* the smile.

The famous summary: implied volatility is "the wrong number to put into the wrong
formula to get the right price."

-------------------------------------------------------------------------------
WHY THE PROBLEM IS WELL POSED
-------------------------------------------------------------------------------

Inverting a function is only sensible if it is monotone. It is: vega is strictly
positive for any option with time left, so price rises strictly with sigma. Between
the two limits,

    sigma -> 0:        price -> max(w(F - K), 0) * e^{-rT}   (discounted intrinsic)
    sigma -> infinity: price -> S e^{-qT} (call)  or  K e^{-rT} (put)

price sweeps monotonically through every value in between. So a solution exists and
is unique **exactly when the market price lies strictly inside those bounds**. If it
does not, no volatility can reproduce it — the quote violates static arbitrage, or,
far more often, the data is bad. :func:`implied_volatility` checks the bounds first
and says which one failed, rather than letting a root-finder wander.

-------------------------------------------------------------------------------
THE ALGORITHM: NEWTON FIRST, BRENT AS BACKSTOP
-------------------------------------------------------------------------------

**Newton-Raphson** is the obvious choice because we already have vega in closed
form from Phase 1, so each step is nearly free:

    sigma_{n+1} = sigma_n - (BS(sigma_n) - target) / vega(sigma_n)

It converges quadratically — roughly doubling the number of correct digits per
iteration — and typically needs three or four steps.

**But it is not safe on its own.** Vega collapses towards zero for deep in- or
out-of-the-money options and as expiry approaches. Dividing by a near-zero vega
produces an enormous step, and Newton happily jumps to a negative or absurd
volatility and diverges. This is not a rare corner case: it is precisely the wing
strikes that Phase 4 needs in order to plot a smile.

So the solver is a hybrid, which is what any production implementation does:

1. Check the arbitrage bounds. If the price is outside them, stop and say so.
2. Run Newton from a good starting point, rejecting any step that leaves the
   valid range.
3. If Newton fails to converge, fall back to **Brent's method**, which is
   guaranteed to converge given a bracketing interval and is nearly as fast in
   practice (it blends bisection's safety with the speed of secant and inverse
   quadratic interpolation).

Bisection alone would also be safe, but converges linearly and would need ~50
iterations for the same accuracy Brent reaches in ~8.

-------------------------------------------------------------------------------
THE STARTING POINT MATTERS
-------------------------------------------------------------------------------

Newton's basin of attraction is not the whole positive line, so the initial guess
does real work. Two standard choices are implemented in :func:`initial_guess`:

* **Brenner-Subrahmanyam (1988)**: inverting the at-the-money approximation
  ``C ~ 0.3989 S sigma sqrt(T)`` from Phase 1 gives
  ``sigma_0 ~ sqrt(2 pi / T) * C / S``. Excellent near the money, poor in the wings.

* **Manaster-Koenig (1982)**: ``sigma_0 = sqrt(|ln(S/K) + (r-q)T| * 2/T)``. This is
  the volatility at which the option has *maximum vega* for the given moneyness,
  and Manaster and Koenig proved Newton converges monotonically from it. It is the
  safer choice away from the money, so it is the default.

-------------------------------------------------------------------------------
CONVERGE ON PRICE, NOT ON VOLATILITY
-------------------------------------------------------------------------------

The tolerance is applied to the *pricing error*, not to the change in sigma. The
reason is conditioning: where vega is small, a tiny price error corresponds to an
enormous volatility error, and a sigma-based tolerance would declare victory while
the price is still visibly wrong. Pricing error is the quantity we actually care
about matching, and it degrades gracefully.

The flip side is worth stating plainly, because it drives all of Phase 4's data
cleaning: **where vega is small, implied volatility is badly determined by
construction**. A one-cent change in a deep out-of-the-money quote can move its
implied vol by several points. That is not solver error — it is the inverse problem
being ill-conditioned, and the only fix is to not trust those quotes.

References: Manaster & Koenig (1982); Brenner & Subrahmanyam (1988); Brent (1973);
Jaeckel, *By Implication* (2006), for a genuinely state-of-the-art treatment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from options_engine.common import OptionType, validate_inputs
from options_engine.greeks import vega
from options_engine.pricing.black_scholes import black_scholes_price

__all__ = [
    "ImpliedVolResult",
    "InitialGuess",
    "implied_volatility",
    "implied_volatility_array",
    "initial_guess",
    "price_bounds",
    "MIN_VOLATILITY",
    "MAX_VOLATILITY",
    "RELATIVE_PRICE_TOLERANCE",
    "ABSOLUTE_PRICE_FLOOR",
    "BRENT_MAX_ITERATIONS",
]

# Search bracket for the root-finder. The lower bound is not zero because a zero-vol
# price is the boundary case rather than an interior root; the upper bound of 500%
# is far above anything a traded equity option implies (even meme-stock weeklies
# top out near 300%), so hitting it signals bad data rather than a real quote.
MIN_VOLATILITY: float = 1e-8
MAX_VOLATILITY: float = 5.0

# Convergence tolerance, in price units, applied as
#
#     tolerance = max(ABSOLUTE_FLOOR, RELATIVE_TOLERANCE * time_value)
#
# WHY IT SCALES, AND WHY WITH *TIME VALUE* RATHER THAN PRICE.
#
# A single fixed tolerance cannot serve both ends of a real option chain. A flat
# 1e-8 sounds strict — a millionth of a tick — but a deep out-of-the-money option
# can be *worth* 1e-9, and a tolerance larger than the price itself is satisfied by
# almost any volatility. Measured on a 200-strike 7-day call worth 1.2e-9: a flat
# 1e-8 tolerance returns sigma = 0.830 against a true 0.800 while reporting success.
#
# Scaling with the price fixes the out-of-the-money end but not the in-the-money
# one. A 50-strike call on a 100 spot is worth ~50, of which the time value might be
# 1e-9; a tolerance of 1e-12 * 50 = 5e-11 is still fifty times the entire quantity
# volatility controls. **Only the time value responds to sigma** — the intrinsic part
# is fixed — so that is the correct scale, and using it makes the tolerance behave
# identically at both wings. For an out-of-the-money option the two definitions
# coincide, since intrinsic value is zero.
#
# The absolute floor exists because a purely relative tolerance would demand
# impossible precision on a near-zero time value, far below what double precision
# can represent in the price computation itself.
RELATIVE_PRICE_TOLERANCE: float = 1e-12
ABSOLUTE_PRICE_FLOOR: float = 1e-15

# Retained for callers who want to pass an explicit absolute tolerance.
DEFAULT_PRICE_TOLERANCE: float | None = None

DEFAULT_MAX_ITERATIONS: int = 100

# Brent gets its own, fixed iteration budget rather than sharing Newton's.
# Sharing them is a subtle trap: `max_iterations` means "how long to persevere with
# Newton before giving up", so a caller lowering it to fail over quickly would also
# be crippling the fallback that is supposed to rescue them. Brent is guaranteed to
# converge on a bracketed root and needs well under 100 iterations for this bracket,
# so a generous constant is both safe and simple.
BRENT_MAX_ITERATIONS: int = 200


def _resolve_tolerance(
    market_price: float, lower_bound: float, tolerance: float | None
) -> float:
    """Return the price tolerance to converge to for a given quote.

    Args:
        market_price: The observed price being inverted.
        lower_bound: The zero-volatility price, i.e. discounted intrinsic value.
            ``market_price - lower_bound`` is the time value, the only part of the
            price that volatility moves.
        tolerance: An explicit absolute tolerance, or ``None`` to derive one that
            scales with time value (see the constants above for why that matters).

    Returns:
        The absolute price tolerance to use.
    """
    if tolerance is not None:
        return tolerance
    time_value = abs(market_price - lower_bound)
    return max(ABSOLUTE_PRICE_FLOOR, RELATIVE_PRICE_TOLERANCE * time_value)


class InitialGuess(str, Enum):
    """Which starting-point heuristic to hand Newton-Raphson.

    See the module docstring for what each one is and when it works.
    """

    MANASTER_KOENIG = "manaster_koenig"
    BRENNER_SUBRAHMANYAM = "brenner_subrahmanyam"

    @classmethod
    def parse(cls, value: Union[str, "InitialGuess"]) -> "InitialGuess":
        """Coerce a string or enum member to an :class:`InitialGuess`.

        Args:
            value: A heuristic name (any case) or enum member.

        Returns:
            The corresponding :class:`InitialGuess` member.

        Raises:
            ValueError: If ``value`` is not a recognised heuristic.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(f"initial_guess must be one of {valid}, got {value!r}") from None


@dataclass(frozen=True)
class ImpliedVolResult:
    """An implied volatility together with how it was obtained.

    The diagnostics are not decoration. When a surface looks wrong, the first
    question is always whether the solver struggled, and ``method`` plus
    ``price_error`` answer it immediately.

    Attributes:
        volatility: The implied volatility, as a decimal (0.20 means 20%).
        iterations: Number of iterations used.
        price_error: ``|BS(volatility) - market_price|`` at the solution. Should be
            at or below the requested tolerance.
        method: ``"newton"`` if Newton-Raphson converged, ``"brent"`` if the
            fallback was needed.
    """

    volatility: float
    iterations: int
    price_error: float
    method: str


def price_bounds(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return the no-arbitrage price range spanned by volatility in (0, infinity).

    Because price is strictly increasing in volatility, these are exactly the
    prices for which an implied volatility exists. A quote outside them cannot be
    inverted at any sigma.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        A ``(lower, upper)`` tuple. ``lower`` is the discounted intrinsic value
        (the zero-volatility limit); ``upper`` is ``S e^{-qT}`` for a call or
        ``K e^{-rT}`` for a put (the infinite-volatility limit).
    """
    opt = OptionType.parse(option_type)
    spot_discounted = spot * np.exp(-dividend_yield * time_to_expiry)
    strike_discounted = strike * np.exp(-rate * time_to_expiry)

    lower = max(opt.sign * (spot_discounted - strike_discounted), 0.0)
    upper = spot_discounted if opt is OptionType.CALL else strike_discounted
    return float(lower), float(upper)


def initial_guess(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    method: str | InitialGuess = InitialGuess.MANASTER_KOENIG,
) -> float:
    """Compute a starting volatility for Newton-Raphson.

    See the module docstring for the derivation and trade-offs of each heuristic.

    Args:
        market_price: Observed option price.
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years. Must be positive.
        rate: Continuously compounded risk-free rate.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        method: Which heuristic to use.

    Returns:
        A starting volatility, clipped into ``[MIN_VOLATILITY, MAX_VOLATILITY]``.
    """
    guess_method = InitialGuess.parse(method)

    if guess_method is InitialGuess.BRENNER_SUBRAHMANYAM:
        # Invert C ~ phi(0) * S * sigma * sqrt(T) from Phase 1's ATM rule of thumb.
        raw = np.sqrt(2.0 * np.pi / time_to_expiry) * market_price / spot
    else:
        # Manaster-Koenig: the volatility that maximises vega at this moneyness,
        # from which Newton is proven to converge monotonically.
        log_moneyness = np.log(spot / strike) + (rate - dividend_yield) * time_to_expiry
        raw = np.sqrt(abs(log_moneyness) * 2.0 / time_to_expiry)
        # Exactly at the money forward the expression vanishes, which would give a
        # zero starting point and a zero vega. Fall back to a neutral 50%.
        if raw < 1e-4:
            raw = 0.5

    return float(np.clip(raw, MIN_VOLATILITY, MAX_VOLATILITY))


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    tolerance: float | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    guess: str | InitialGuess = InitialGuess.MANASTER_KOENIG,
    return_diagnostics: bool = False,
) -> float | ImpliedVolResult:
    """Solve for the volatility that reproduces an observed option price.

    Newton-Raphson with an analytic vega, falling back to Brent's method when
    Newton misbehaves. See the module docstring for why the hybrid is necessary
    rather than merely tidy.

    Args:
        market_price: Observed option price. Must lie strictly inside the
            no-arbitrage bounds from :func:`price_bounds`.
        spot: Current price of the underlying. Must be positive.
        strike: Strike price. Must be positive.
        time_to_expiry: Time to expiry in years. Must be positive — an expired
            option carries no volatility information.
        rate: Continuously compounded risk-free rate.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        tolerance: Absolute convergence tolerance in *price* units, not volatility
            units. Leave as ``None`` to derive one that scales with the quote —
            strongly recommended, since a fixed tolerance is meaningless for deep
            out-of-the-money options worth less than the tolerance itself.
        max_iterations: Cap on Newton iterations before falling back to Brent. Does
            not restrict Brent, which has its own budget — see
            :data:`BRENT_MAX_ITERATIONS`.
        guess: Which starting-point heuristic to use.
        return_diagnostics: If True, return an :class:`ImpliedVolResult` instead of
            a bare float.

    Returns:
        The implied volatility as a decimal, or an :class:`ImpliedVolResult` if
        ``return_diagnostics`` is set.

    Raises:
        ValueError: If inputs are out of domain, if ``time_to_expiry`` is zero, or
            if ``market_price`` lies outside the no-arbitrage bounds (the message
            says which bound and by how much).
        RuntimeError: If both Newton and Brent fail, which should not happen for a
            price inside the bounds and indicates a genuine bug.

    Examples:
        Round-tripping a known volatility recovers it exactly::

            >>> from options_engine.pricing.black_scholes import black_scholes_price
            >>> price = float(black_scholes_price(100, 105, 1.0, 0.05, 0.23, "call"))
            >>> round(implied_volatility(price, 100, 105, 1.0, 0.05, "call"), 10)
            0.23
    """
    validate_inputs(spot, strike, time_to_expiry, 0.1)  # vol is the unknown here
    opt = OptionType.parse(option_type)

    if time_to_expiry <= 0.0:
        raise ValueError(
            "time_to_expiry must be positive; an expired option has no time value "
            "and therefore carries no volatility information"
        )

    lower, upper = price_bounds(spot, strike, time_to_expiry, rate, opt, dividend_yield)
    price_tolerance = _resolve_tolerance(market_price, lower, tolerance)
    if market_price <= lower:
        raise ValueError(
            f"market_price {market_price:.6f} is at or below the no-arbitrage lower "
            f"bound {lower:.6f} (discounted intrinsic value), short by "
            f"{lower - market_price:.6f}. No volatility can produce it: at sigma -> 0 "
            f"the price already equals the bound. Usually stale or crossed quotes."
        )
    if market_price >= upper:
        raise ValueError(
            f"market_price {market_price:.6f} is at or above the no-arbitrage upper "
            f"bound {upper:.6f}, exceeding it by {market_price - upper:.6f}. No "
            f"volatility can produce it: even sigma -> infinity only approaches the bound."
        )

    def pricing_error(sigma: float) -> float:
        """Model price minus market price at the given volatility."""
        return (
            float(
                black_scholes_price(
                    spot, strike, time_to_expiry, rate, sigma, opt, dividend_yield
                )
            )
            - market_price
        )

    # --- Attempt 1: Newton-Raphson ---
    sigma = initial_guess(
        market_price, spot, strike, time_to_expiry, rate, opt, dividend_yield, guess
    )
    newton_iterations = 0

    for newton_iterations in range(1, max_iterations + 1):
        error = pricing_error(sigma)
        if abs(error) < price_tolerance:
            if return_diagnostics:
                return ImpliedVolResult(sigma, newton_iterations, abs(error), "newton")
            return sigma

        slope = float(vega(spot, strike, time_to_expiry, rate, sigma, opt, dividend_yield))
        if slope < 1e-12:
            # Vega has collapsed; the Newton step is meaningless. Hand over to Brent
            # rather than taking a wild jump.
            break

        step = error / slope
        candidate = sigma - step
        if not MIN_VOLATILITY < candidate < MAX_VOLATILITY or not np.isfinite(candidate):
            # Newton has left the valid range, the classic failure in the wings.
            break
        sigma = candidate
    else:
        # Loop ran to max_iterations without converging.
        pass

    # --- Attempt 2: Brent's method, guaranteed given a bracket ---
    # The bounds check above guarantees a sign change across the full range, so a
    # bracket always exists here.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brent_sigma, results = brentq(
                pricing_error,
                MIN_VOLATILITY,
                MAX_VOLATILITY,
                xtol=1e-14,
                rtol=1e-15,
                maxiter=BRENT_MAX_ITERATIONS,
                full_output=True,
            )
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"both Newton and Brent failed for market_price={market_price}, "
            f"spot={spot}, strike={strike}, T={time_to_expiry}: {exc}"
        ) from exc

    final_error = abs(pricing_error(brent_sigma))
    if return_diagnostics:
        return ImpliedVolResult(
            float(brent_sigma), int(results.iterations), final_error, "brent"
        )
    return float(brent_sigma)


def implied_volatility_array(
    market_prices: NDArray[np.float64],
    spot: float,
    strikes: NDArray[np.float64],
    times_to_expiry: NDArray[np.float64],
    rate: float,
    option_types: NDArray[np.str_] | str | OptionType = OptionType.CALL,
    dividend_yield: float = 0.0,
    tolerance: float | None = None,
) -> NDArray[np.float64]:
    """Solve for implied volatility across a whole option chain.

    Rows that cannot be inverted yield ``NaN`` rather than raising. That is the
    right behaviour here: a real chain always contains some unusable quotes, and
    one bad strike should not abort an entire surface. Filtering those quotes out
    beforehand is the job of the Phase 4 data cleaning, and the ``NaN`` count is a
    useful check that the cleaning did its job.

    Args:
        market_prices: Observed prices, shape ``(n,)``.
        spot: Current price of the underlying.
        strikes: Strike prices, shape ``(n,)``.
        times_to_expiry: Times to expiry in years, shape ``(n,)``.
        rate: Continuously compounded risk-free rate.
        option_types: Either a single type applied to every row, or an array of
            ``"call"``/``"put"`` strings of shape ``(n,)``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        tolerance: Absolute convergence tolerance in price units, or ``None`` to
            scale it with each quote (recommended).

    Returns:
        Implied volatilities of shape ``(n,)``, with ``NaN`` where no solution
        exists.

    Raises:
        ValueError: If the input arrays have mismatched lengths.
    """
    prices = np.asarray(market_prices, dtype=float)
    strike_array = np.asarray(strikes, dtype=float)
    time_array = np.asarray(times_to_expiry, dtype=float)

    if not (prices.shape == strike_array.shape == time_array.shape):
        raise ValueError(
            f"market_prices {prices.shape}, strikes {strike_array.shape} and "
            f"times_to_expiry {time_array.shape} must have the same shape"
        )

    if isinstance(option_types, (str, OptionType)):
        type_array = np.full(prices.shape, OptionType.parse(option_types).value, dtype=object)
    else:
        type_array = np.asarray(option_types, dtype=object)
        if type_array.shape != prices.shape:
            raise ValueError(
                f"option_types {type_array.shape} must match market_prices {prices.shape}"
            )

    result = np.full(prices.shape, np.nan, dtype=float)
    for i in range(prices.size):
        try:
            result[i] = implied_volatility(
                float(prices[i]),
                spot,
                float(strike_array[i]),
                float(time_array[i]),
                rate,
                str(type_array[i]),
                dividend_yield,
                tolerance,
            )
        except (ValueError, RuntimeError):
            # Leave NaN. The caller inspects the NaN rate to judge data quality.
            continue
    return result
