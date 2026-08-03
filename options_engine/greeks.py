r"""Option sensitivities (Greeks), both analytical and by finite differences.

Every Greek here is a partial derivative of the Black-Scholes price, and each one
is provided twice:

* **Analytical** — differentiate the closed-form price by hand. Exact and fast.
* **Numerical** — bump an input and re-price. Slower and slightly inexact, but it
  works on *any* pricer, including the binomial tree and Monte Carlo engines in
  later phases where no closed form exists.

The two agreeing to ~6 decimal places is the strongest single test in this
project: it independently validates the price formula, the derivative algebra,
and the finite-difference machinery all at once.

-------------------------------------------------------------------------------
UNITS — read this before using any number below
-------------------------------------------------------------------------------

Every function returns the **raw mathematical derivative**. Trading desks quote
several of these rescaled, and silently mixing the two conventions is the most
common bug in Greeks code, so nothing is rescaled implicitly here.

    Greek   Raw derivative      Desk convention        Convert with
    -----   -----------------   --------------------   --------------------------
    Delta   dV/dS               same                   --
    Gamma   d2V/dS2             same                   --
    Vega    dV/dsigma           per 1 vol point (1%)   vega_per_percent()
    Theta   dV/dt  (per year)   per calendar day       theta_per_day()
    Rho     dV/dr               per 1 basis point      rho_per_basis_point()

So a raw vega of 16.2 means "the option gains 16.2 if vol goes from 20% to 120%",
which is the linear approximation, not a realistic move. Desks quote 0.162: the
gain for 20% -> 21%. Use the helpers rather than dividing by 100 inline.

Sign conventions:

* **Theta** is dV/dt where t is *calendar time moving forward*, i.e. the negative
  of the derivative with respect to time-to-expiry. It is normally negative: a
  long option decays. Sign errors here are endemic; the tests pin it down.
* **Delta** for a put is negative, in [-1, 0].
* **Gamma** and **vega** are identical for calls and puts at the same strike —
  a direct consequence of put-call parity, since the parity difference
  ``S e^{-qT} - K e^{-rT}`` is linear in S and free of sigma, so it vanishes under
  the second S-derivative and the sigma-derivative. The tests assert this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from options_engine.common import TINY, Numeric, OptionType, validate_inputs
from options_engine.pricing.black_scholes import black_scholes_price, d1_d2

__all__ = [
    "Greeks",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "all_greeks",
    "numerical_delta",
    "numerical_gamma",
    "numerical_vega",
    "numerical_theta",
    "numerical_rho",
    "numerical_greeks",
    "vega_per_percent",
    "theta_per_day",
    "rho_per_basis_point",
    "DAYS_PER_YEAR",
]

# Calendar days, not trading days. Theta measures the passage of *wall-clock* time
# to expiry, and an option expiring in 30 calendar days decays over 30 days whether
# or not the market is open. (Practitioners who mark decay only on trading days use
# 252 here; that is a business-calendar adjustment, not a change to the math.)
DAYS_PER_YEAR: float = 365.0


@dataclass(frozen=True)
class Greeks:
    """The five first- and second-order Black-Scholes sensitivities.

    All values are raw derivatives in the units documented in this module's
    docstring. Frozen because a computed risk snapshot should not be mutated in
    place — recompute it instead.

    Attributes:
        delta: dV/dS, change in value per unit change in spot.
        gamma: d2V/dS2, change in delta per unit change in spot.
        vega: dV/dsigma, per 1.00 (100 vol points) change in volatility.
        theta: dV/dt, per year of calendar time elapsed. Usually negative.
        rho: dV/dr, per 1.00 (10,000 bp) change in the risk-free rate.
    """

    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    theta: NDArray[np.float64]
    rho: NDArray[np.float64]

    def as_dict(self) -> dict[str, NDArray[np.float64]]:
        """Return the Greeks as a plain dict, convenient for DataFrame building.

        Returns:
            Mapping from Greek name to value.
        """
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


# ---------------------------------------------------------------------------
# Analytical Greeks
# ---------------------------------------------------------------------------


def delta(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    r"""Compute delta: the first derivative of price with respect to spot.

    ``Delta_call = e^{-qT} N(d1)``,  ``Delta_put = -e^{-qT} N(-d1)``

    Deriving this is less work than it looks. Differentiating
    ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)`` by the product rule gives three
    terms, and the two involving ``dN/dS`` cancel exactly:

        S e^{-qT} phi(d1) * dd1/dS  -  K e^{-rT} phi(d2) * dd2/dS  =  0

    because ``dd1/dS == dd2/dS`` (they differ by a constant) and the identity
    ``S e^{-qT} phi(d1) == K e^{-rT} phi(d2)`` holds at the d1/d2 definitions.
    That leaves only ``e^{-qT} N(d1)``. The same cancellation is why vega and
    gamma come out so clean, and it is worth being able to reproduce on a
    whiteboard.

    Interpretation: the number of shares to hold to hedge one option, and roughly
    (but not exactly) the risk-neutral probability of finishing in the money.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        Delta, in [0, 1] for calls and [-1, 0] for puts.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    w = OptionType.parse(option_type).sign
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)
    discount = np.exp(
        -np.asarray(dividend_yield, dtype=float)
        * np.asarray(time_to_expiry, dtype=float)
    )
    return w * discount * norm.cdf(w * d1)


def gamma(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    r"""Compute gamma: the second derivative of price with respect to spot.

    ``Gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))``

    Identical for calls and puts (see the module docstring). ``option_type`` is
    accepted only so that every Greek in this module has the same signature and
    can be swapped in generically; it does not affect the result.

    Interpretation: how fast your hedge goes stale. Gamma peaks near the money
    and near expiry, which is exactly when a delta hedge needs rebalancing most
    often — the practical reason short-dated at-the-money options are expensive
    to hedge.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: Ignored; present for signature symmetry.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        Gamma, always non-negative for a long vanilla option.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    OptionType.parse(option_type)  # validate even though the result is symmetric
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    spot_arr = np.asarray(spot, dtype=float)
    time_arr = np.maximum(np.asarray(time_to_expiry, dtype=float), TINY)
    vol_arr = np.maximum(np.asarray(volatility, dtype=float), TINY)
    discount = np.exp(-np.asarray(dividend_yield, dtype=float) * time_arr)

    return discount * norm.pdf(d1) / (spot_arr * vol_arr * np.sqrt(time_arr))


def vega(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    r"""Compute vega: the derivative of price with respect to volatility.

    ``Vega = S e^{-qT} phi(d1) sqrt(T)``, per **1.00** change in volatility.
    Use :func:`vega_per_percent` for the desk convention.

    Identical for calls and puts. ``option_type`` is accepted for signature
    symmetry only.

    Note that "vega" is not a Greek letter — it is trader slang that stuck. Some
    texts write it as kappa.

    Interpretation: vega is largest for at-the-money options with long maturities,
    and decays to zero at expiry (there is no volatility left to be uncertain
    about). Phase 4 inverts this function to back out implied volatility, which is
    only well posed *because* vega is strictly positive: price is monotone in vol.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: Ignored; present for signature symmetry.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        Vega per 1.00 change in volatility, always non-negative.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    OptionType.parse(option_type)
    d1, _ = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    spot_arr = np.asarray(spot, dtype=float)
    time_arr = np.maximum(np.asarray(time_to_expiry, dtype=float), TINY)
    discount = np.exp(-np.asarray(dividend_yield, dtype=float) * time_arr)

    return spot_arr * discount * norm.pdf(d1) * np.sqrt(time_arr)


def theta(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    r"""Compute theta: the derivative of price with respect to calendar time.

    ``Theta_call = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)``
    ``Theta_put  = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)``

    Per **year**; use :func:`theta_per_day` for the desk convention.

    The three terms have distinct economic meanings, and reading them off is a
    good way to explain theta:

    1. The shared first term is pure time-value decay — the option's optionality
       shrinking as there is less time for the stock to move. Always negative.
    2. The ``q`` term is the dividend the option holder forgoes (or, for a put,
       benefits from).
    3. The ``r`` term is the financing carry on the strike you will pay or receive.

    Sign: negative for almost all long options. The notable exception is a deep
    in-the-money European put, where term 3 dominates — exercising early would
    let you collect the strike and earn interest on it, but a European holder
    cannot, so waiting is costly and theta turns positive. This is precisely the
    situation that makes an *American* put worth more than a European one, which
    is what Phase 2's early-exercise logic has to capture.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        Theta per year, usually negative for long options.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    w = OptionType.parse(option_type).sign
    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    spot_arr = np.asarray(spot, dtype=float)
    strike_arr = np.asarray(strike, dtype=float)
    time_arr = np.maximum(np.asarray(time_to_expiry, dtype=float), TINY)
    vol_arr = np.maximum(np.asarray(volatility, dtype=float), TINY)
    rate_arr = np.asarray(rate, dtype=float)
    div_arr = np.asarray(dividend_yield, dtype=float)

    spot_discounted = spot_arr * np.exp(-div_arr * time_arr)
    strike_discounted = strike_arr * np.exp(-rate_arr * time_arr)

    time_decay = -spot_discounted * norm.pdf(d1) * vol_arr / (2.0 * np.sqrt(time_arr))
    dividend_carry = w * div_arr * spot_discounted * norm.cdf(w * d1)
    financing_carry = -w * rate_arr * strike_discounted * norm.cdf(w * d2)

    return time_decay + dividend_carry + financing_carry


def rho(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    r"""Compute rho: the derivative of price with respect to the risk-free rate.

    ``Rho_call = K T e^{-rT} N(d2)``,  ``Rho_put = -K T e^{-rT} N(-d2)``

    Per **1.00** change in the rate; use :func:`rho_per_basis_point` for the desk
    convention.

    Only the ``K e^{-rT}`` discount factor carries an explicit ``r``, and the
    ``dN/dr`` terms cancel by the same identity used in :func:`delta`, leaving
    just the derivative of that discount factor times ``N(d2)``.

    Interpretation: a call is implicitly a leveraged long position — you defer
    paying the strike — so higher rates make that deferral more valuable and rho
    is positive. A put is the mirror image. Rho is the least-watched Greek for
    short-dated equity options (the ``T`` factor makes it small) and the most
    important one for long-dated products like LEAPS.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        Rho per 1.00 change in the rate; positive for calls, negative for puts.
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    w = OptionType.parse(option_type).sign
    _, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    strike_arr = np.asarray(strike, dtype=float)
    time_arr = np.asarray(time_to_expiry, dtype=float)
    rate_arr = np.asarray(rate, dtype=float)

    return w * strike_arr * time_arr * np.exp(-rate_arr * time_arr) * norm.cdf(w * d2)


def all_greeks(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> Greeks:
    """Compute all five analytical Greeks in one call.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        A :class:`Greeks` dataclass of raw derivatives.
    """
    args = (spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield)
    return Greeks(
        delta=delta(*args),
        gamma=gamma(*args),
        vega=vega(*args),
        theta=theta(*args),
        rho=rho(*args),
    )


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


def vega_per_percent(raw_vega: Numeric) -> NDArray[np.float64]:
    """Convert raw vega to the desk convention: P&L per 1 volatility point.

    Args:
        raw_vega: Vega per 1.00 change in volatility, as returned by :func:`vega`.

    Returns:
        Vega per 0.01 (one percentage point) change in volatility.
    """
    return np.asarray(raw_vega, dtype=float) / 100.0


def theta_per_day(raw_theta: Numeric, days_per_year: float = DAYS_PER_YEAR) -> NDArray[np.float64]:
    """Convert raw theta to the desk convention: P&L per calendar day elapsed.

    Args:
        raw_theta: Theta per year, as returned by :func:`theta`.
        days_per_year: Days to divide by. Defaults to 365 calendar days; pass 252
            if your desk marks decay only on trading days.

    Returns:
        Theta per day, usually a small negative number.
    """
    return np.asarray(raw_theta, dtype=float) / days_per_year


def rho_per_basis_point(raw_rho: Numeric) -> NDArray[np.float64]:
    """Convert raw rho to the desk convention: P&L per basis point of rate move.

    Args:
        raw_rho: Rho per 1.00 change in the rate, as returned by :func:`rho`.

    Returns:
        Rho per 0.0001 (one basis point) change in the rate.
    """
    return np.asarray(raw_rho, dtype=float) / 10_000.0


# ---------------------------------------------------------------------------
# Numerical (finite-difference) Greeks
# ---------------------------------------------------------------------------
#
# CHOOSING THE BUMP SIZE
#
# A central difference has two competing error sources:
#
#   truncation error ~ O(h^2)      -- from dropping higher Taylor terms
#   roundoff error   ~ O(eps / h)  -- from cancellation in (V(x+h) - V(x-h))
#
# Total error is minimised where they balance. Setting h^2 = eps/h gives
# h* ~ eps^(1/3) ~ 6e-6 for float64, with achievable accuracy ~ eps^(2/3) ~ 4e-11.
#
# For gamma the second difference divides by h^2, so cancellation bites harder:
# roundoff ~ eps/h^2 balances against h^2 at h* ~ eps^(1/4) ~ 1.2e-4, giving
# accuracy only ~ eps^(1/2) ~ 1e-8. This is why gamma is tested to a looser
# tolerance than the first-order Greeks — it is a genuine property of the method,
# not sloppiness in the implementation.
#
# The two optima differ by two orders of magnitude, so gamma gets its own bump
# constant rather than sharing delta's. Using one bump for both would force a
# compromise that is wrong for at least one of them.
#
# These defaults were confirmed empirically, not just derived: sweeping the bump
# over the test grid shows first-order errors falling cleanly as h^2 down to
# h ~ 1e-6 (delta ~4e-11, vega ~1e-10, theta/rho ~2e-8) and rising again below it
# as roundoff takes over — exactly the predicted V-shape. Gamma's error curve
# bottoms out near 1e-4, as predicted by eps^(1/4).
#
# Spot bumps are *relative* (h * S) because spot can be 5 or 5000 and a fixed
# absolute bump would be badly scaled at one end; volatility, rate, and time bumps
# are *absolute* because those inputs are all naturally O(0.01 - 1) and a relative
# bump would vanish near zero.

_REL_SPOT_BUMP: float = 1e-6  # first-order (delta): h* ~ eps^(1/3)
_REL_SPOT_BUMP_GAMMA: float = 1e-4  # second-order (gamma): h* ~ eps^(1/4)
_ABS_VOL_BUMP: float = 1e-6
_ABS_RATE_BUMP: float = 1e-6
_ABS_TIME_BUMP: float = 1e-6


PriceFunction = Callable[..., NDArray[np.float64]]


def _price(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType,
    dividend_yield: Numeric,
    price_fn: PriceFunction,
) -> NDArray[np.float64]:
    """Call a pricer with the engine's standard positional argument order.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield.
        price_fn: The pricing function to call.

    Returns:
        The option price returned by ``price_fn``.
    """
    return np.asarray(
        price_fn(
            spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield
        ),
        dtype=float,
    )


def numerical_delta(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    price_fn: PriceFunction = black_scholes_price,
    relative_bump: float = _REL_SPOT_BUMP,
) -> NDArray[np.float64]:
    """Estimate delta by central difference on spot.

    ``(V(S+h) - V(S-h)) / (2h)`` with ``h = relative_bump * S``.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        price_fn: Pricer to differentiate. Defaults to Black-Scholes; pass a tree
            or Monte Carlo pricer to get Greeks for those models.
        relative_bump: Spot bump as a fraction of spot. See the note above on
            choosing this.

    Returns:
        The finite-difference estimate of delta.
    """
    spot_arr = np.asarray(spot, dtype=float)
    h = relative_bump * spot_arr
    common = (strike, time_to_expiry, rate, volatility, option_type, dividend_yield, price_fn)
    up = _price(spot_arr + h, *common)
    down = _price(spot_arr - h, *common)
    return (up - down) / (2.0 * h)


def numerical_gamma(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    price_fn: PriceFunction = black_scholes_price,
    relative_bump: float = _REL_SPOT_BUMP_GAMMA,
) -> NDArray[np.float64]:
    """Estimate gamma by second central difference on spot.

    ``(V(S+h) - 2 V(S) + V(S-h)) / h^2`` with ``h = relative_bump * S``.

    Note the much larger default bump than :func:`numerical_delta` uses. That is
    deliberate: dividing by ``h^2`` amplifies floating-point cancellation, so the
    optimal bump for a second difference is ~100x larger than for a first one.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        price_fn: Pricer to differentiate. Defaults to Black-Scholes.
        relative_bump: Spot bump as a fraction of spot.

    Returns:
        The finite-difference estimate of gamma. Expect ~1e-8 relative accuracy
        at best; see the note above on why second differences lose precision.
    """
    spot_arr = np.asarray(spot, dtype=float)
    h = relative_bump * spot_arr
    common = (strike, time_to_expiry, rate, volatility, option_type, dividend_yield, price_fn)
    up = _price(spot_arr + h, *common)
    mid = _price(spot_arr, *common)
    down = _price(spot_arr - h, *common)
    return (up - 2.0 * mid + down) / (h**2)


def numerical_vega(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    price_fn: PriceFunction = black_scholes_price,
    bump: float = _ABS_VOL_BUMP,
) -> NDArray[np.float64]:
    """Estimate vega by central difference on volatility.

    Returns the raw derivative per 1.00 of volatility, matching :func:`vega`.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        price_fn: Pricer to differentiate. Defaults to Black-Scholes.
        bump: Absolute volatility bump.

    Returns:
        The finite-difference estimate of vega.
    """
    vol_arr = np.asarray(volatility, dtype=float)
    up = _price(spot, strike, time_to_expiry, rate, vol_arr + bump, option_type, dividend_yield, price_fn)
    down = _price(spot, strike, time_to_expiry, rate, vol_arr - bump, option_type, dividend_yield, price_fn)
    return (up - down) / (2.0 * bump)


def numerical_theta(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    price_fn: PriceFunction = black_scholes_price,
    bump: float = _ABS_TIME_BUMP,
) -> NDArray[np.float64]:
    """Estimate theta by central difference on time to expiry.

    Note the leading minus sign: theta is the derivative with respect to calendar
    time moving *forward*, whereas we can only bump time *to expiry*. One day
    passing means ``T`` decreases, so ``dV/dt = -dV/dT``. Getting this backwards
    is the classic theta sign bug.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years. Must exceed ``bump``.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        price_fn: Pricer to differentiate. Defaults to Black-Scholes.
        bump: Absolute time bump in years.

    Returns:
        The finite-difference estimate of theta, per year.
    """
    time_arr = np.asarray(time_to_expiry, dtype=float)
    up = _price(spot, strike, time_arr + bump, rate, volatility, option_type, dividend_yield, price_fn)
    down = _price(spot, strike, np.maximum(time_arr - bump, 0.0), rate, volatility, option_type, dividend_yield, price_fn)
    return -(up - down) / (2.0 * bump)


def numerical_rho(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    price_fn: PriceFunction = black_scholes_price,
    bump: float = _ABS_RATE_BUMP,
) -> NDArray[np.float64]:
    """Estimate rho by central difference on the risk-free rate.

    Returns the raw derivative per 1.00 of rate, matching :func:`rho`.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        price_fn: Pricer to differentiate. Defaults to Black-Scholes.
        bump: Absolute rate bump.

    Returns:
        The finite-difference estimate of rho.
    """
    rate_arr = np.asarray(rate, dtype=float)
    up = _price(spot, strike, time_to_expiry, rate_arr + bump, volatility, option_type, dividend_yield, price_fn)
    down = _price(spot, strike, time_to_expiry, rate_arr - bump, volatility, option_type, dividend_yield, price_fn)
    return (up - down) / (2.0 * bump)


def numerical_greeks(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    price_fn: PriceFunction = black_scholes_price,
) -> Greeks:
    """Compute all five Greeks by finite differences on an arbitrary pricer.

    This is the model-agnostic path: pass a binomial or Monte Carlo pricer with
    the standard signature and you get its Greeks without any new derivations.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        price_fn: Pricer to differentiate. Must accept
            ``(spot, strike, time_to_expiry, rate, volatility, option_type,
            dividend_yield)`` positionally. Defaults to Black-Scholes.

    Returns:
        A :class:`Greeks` dataclass of finite-difference estimates.
    """
    args = (spot, strike, time_to_expiry, rate, volatility, option_type, dividend_yield, price_fn)
    return Greeks(
        delta=numerical_delta(*args),
        gamma=numerical_gamma(*args),
        vega=numerical_vega(*args),
        theta=numerical_theta(*args),
        rho=numerical_rho(*args),
    )
