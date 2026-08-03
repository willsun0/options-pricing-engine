r"""Closed-form Black-Scholes-Merton pricing for European options.

===============================================================================
WHERE THE FORMULA COMES FROM
===============================================================================

This is the derivation I'd give at a whiteboard. Three steps: model the stock,
argue that risk preferences drop out, then evaluate one integral.

-------------------------------------------------------------------------------
Step 1. The model: geometric Brownian motion
-------------------------------------------------------------------------------

Assume the underlying follows

    dS_t = (mu - q) * S_t * dt + sigma * S_t * dW_t

where mu is the expected total return, q is a *continuous* dividend yield paid
out of the stock (so the price drifts down by q relative to total return), sigma
is constant volatility, and W_t is a standard Brownian motion.

The key structural assumption is that returns, not prices, are what scale: the
dt and dW terms are both proportional to S_t. That's why the stock can never go
negative, and why the terminal distribution ends up lognormal rather than normal.

Applying Ito's lemma to x_t = ln(S_t) — and this is where the famous "extra"
term appears, because Ito's lemma carries a second-order piece that ordinary
calculus does not:

    d(ln S) = (1/S) dS - (1/2) (1/S^2) (dS)^2
            = (mu - q - sigma^2/2) dt + sigma dW

The -sigma^2/2 is *not* a modelling choice; it falls out of (dS)^2 = sigma^2 S^2 dt.
It is the reason the median of S_T sits below its mean, and it is the single most
common thing people fumble in an interview. Integrating from 0 to T:

    ln(S_T) = ln(S_0) + (mu - q - sigma^2/2) T + sigma * sqrt(T) * Z,   Z ~ N(0,1)

So ln(S_T) is normal, i.e. S_T is *lognormal*.

-------------------------------------------------------------------------------
Step 2. Risk-neutral valuation: why mu disappears
-------------------------------------------------------------------------------

Black and Scholes' insight: build a portfolio long one option and short delta
shares. Choosing delta = dV/dS cancels the dW term, so over an instant the
portfolio is *riskless*. No-arbitrage forces it to earn the risk-free rate r,
which gives the Black-Scholes PDE:

    dV/dt + (r - q) S dV/dS + (1/2) sigma^2 S^2 d2V/dS2 - r V = 0

Notice what is absent: mu. The hedging argument removed it. Two investors who
violently disagree about the stock's expected return must still agree on the
option's price, because either could run the hedge against the other.

Equivalently (Feynman-Kac), the PDE's solution is a discounted expectation taken
under a measure where the drift is r - q rather than mu - q:

    V_0 = exp(-rT) * E^Q[ payoff(S_T) ]

This is *not* a claim that investors are risk-neutral. It is a change of measure:
a pricing device that happens to give the same answer as the hedge. Under Q,

    S_T = S_0 * exp( (r - q - sigma^2/2) T + sigma sqrt(T) Z )

-------------------------------------------------------------------------------
Step 3. Evaluating the expectation
-------------------------------------------------------------------------------

For a call, payoff = max(S_T - K, 0), so

    C = exp(-rT) * E^Q[ (S_T - K) * 1{S_T > K} ]
      = exp(-rT) * ( E^Q[S_T * 1{S_T > K}] - K * Q(S_T > K) )

Second term first. S_T > K iff

    Z > [ ln(K/S_0) - (r - q - sigma^2/2) T ] / (sigma sqrt(T))  ==  -d2

so by symmetry of the standard normal, Q(S_T > K) = N(d2), with

    d2 = [ ln(S_0/K) + (r - q - sigma^2/2) T ] / (sigma sqrt(T))

First term: substitute S_T and complete the square inside the Gaussian integral.
The exp(sigma sqrt(T) z) factor shifts the mean of the normal density by
sigma*sqrt(T), which is exactly what turns d2 into d1:

    E^Q[S_T 1{S_T > K}] = S_0 exp((r - q) T) N(d1),   d1 = d2 + sigma sqrt(T)

Putting the pieces together and cancelling exp(-rT) against exp(rT):

    C = S_0 exp(-qT) N(d1) - K exp(-rT) N(d2)
    P = K exp(-rT) N(-d2) - S_0 exp(-qT) N(-d1)          [same argument, or parity]

    d1 = [ ln(S_0/K) + (r - q + sigma^2/2) T ] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

-------------------------------------------------------------------------------
How to read the two terms
-------------------------------------------------------------------------------

* N(d2) is the risk-neutral probability the option finishes in the money. So
  K exp(-rT) N(d2) is "what you expect to pay for the shares, discounted".

* N(d1) is *not* a probability of anything under Q. It is the same event's
  probability measured under a different numeraire (the stock instead of the
  bank account) — that is, probability re-weighted by how much stock you own in
  each state. It is also exactly dC/dS, the hedge ratio. S_0 exp(-qT) N(d1) is
  "what you expect to receive in shares, discounted".

  Price = PV(what you get) - PV(what you pay), each weighted by the probability
  of the exercise event under its natural numeraire.

* exp(-qT) on the stock term: holding the option does not entitle you to the
  dividends the stock pays before expiry, so the relevant quantity is the stock
  net of that income stream. Setting q = 0 recovers the original 1973 formula;
  the q term is Merton's 1973 extension. For an index like SPY (yield ~1.2%),
  ignoring q biases call prices high and put prices low.

-------------------------------------------------------------------------------
What the model assumes, and what breaks in practice
-------------------------------------------------------------------------------

Constant sigma, constant r, no transaction costs, continuous frictionless
trading, no jumps, European exercise. The first assumption is the one the market
visibly rejects: if it held, every strike would imply the same volatility. It
does not — see the volatility smile in Phase 4.

References: Black & Scholes (1973); Merton (1973); Hull, *Options, Futures and
Other Derivatives*, ch. 15.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from options_engine.common import TINY, Numeric, OptionType, validate_inputs

__all__ = [
    "d1_d2",
    "black_scholes_price",
    "call_price",
    "put_price",
    "forward_price",
    "barrier_price_analytic",
]


def d1_d2(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    dividend_yield: Numeric = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the d1 and d2 terms of the Black-Scholes formula.

    Factored out because the price, every Greek, and the implied-vol solver all
    need them; computing them in one place keeps the sign conventions consistent.

    Degenerate inputs (``time_to_expiry == 0`` or ``volatility == 0``) are handled
    by clamping to :data:`~options_engine.common.TINY` rather than branching. The
    formula's own limit then does the right thing: d1 and d2 blow up to +/-infinity
    with the sign of the log-moneyness, the normal CDFs saturate at 1 or 0, and the
    price collapses to discounted intrinsic value. See TINY's docstring.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate (0.05 means 5%).
        volatility: Annualised volatility as a decimal (0.20 means 20%).
        dividend_yield: Continuous dividend yield as a decimal. Defaults to 0.0.

    Returns:
        A ``(d1, d2)`` tuple of arrays broadcast to the shape of the inputs.
    """
    spot_arr = np.asarray(spot, dtype=float)
    strike_arr = np.asarray(strike, dtype=float)
    time_arr = np.maximum(np.asarray(time_to_expiry, dtype=float), TINY)
    vol_arr = np.maximum(np.asarray(volatility, dtype=float), TINY)

    vol_sqrt_t = vol_arr * np.sqrt(time_arr)
    d1 = (
        np.log(spot_arr / strike_arr)
        + (rate - dividend_yield + 0.5 * vol_arr**2) * time_arr
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def forward_price(
    spot: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    """Compute the forward price of the underlying for delivery at expiry.

    ``F = S * exp((r - q) * T)``. This is the risk-neutral expectation of ``S_T``,
    and it is the quantity that actually determines moneyness — an option is
    "at the money forward" when ``K == F``, which is not the same as ``K == S``.
    Phase 4 uses this when building the volatility smile.

    Args:
        spot: Current price of the underlying.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        The forward price, broadcast to the shape of the inputs.
    """
    return np.asarray(spot, dtype=float) * np.exp(
        (np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float))
        * np.asarray(time_to_expiry, dtype=float)
    )


def black_scholes_price(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    option_type: str | OptionType = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    """Price a European option with the closed-form Black-Scholes-Merton formula.

    ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)``
    ``P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)``

    Both cases are written as a single expression using ``w = +1`` for a call and
    ``w = -1`` for a put, since ``N(-x) = 1 - N(x)`` makes the two formulas mirror
    images. One code path means one place for a sign error to hide.

    All arguments broadcast, so you can price an entire strike ladder or an entire
    surface in one vectorised call.

    Args:
        spot: Current price of the underlying. Must be positive.
        strike: Strike price. Must be positive.
        time_to_expiry: Time to expiry in years. Must be non-negative; ``0``
            returns intrinsic value.
        rate: Continuously compounded risk-free rate (0.05 means 5%). May be
            negative.
        volatility: Annualised volatility as a decimal (0.20 means 20%). Must be
            non-negative; ``0`` returns discounted intrinsic value.
        option_type: ``"call"`` or ``"put"``. Defaults to a call.
        dividend_yield: Continuous dividend yield as a decimal. Defaults to 0.0.

    Returns:
        The option's present value, broadcast to the shape of the inputs. Returns
        a 0-d array for scalar inputs; wrap in ``float()`` if you need a scalar.

    Raises:
        ValueError: If any input is outside its permitted domain (see
            :func:`~options_engine.common.validate_inputs`).

    Examples:
        Hull, *Options, Futures and Other Derivatives*, Example 15.6::

            >>> round(float(black_scholes_price(42, 40, 0.5, 0.10, 0.20, "call")), 2)
            4.76
            >>> round(float(black_scholes_price(42, 40, 0.5, 0.10, 0.20, "put")), 2)
            0.81
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    opt = OptionType.parse(option_type)
    w = opt.sign

    d1, d2 = d1_d2(spot, strike, time_to_expiry, rate, volatility, dividend_yield)

    discounted_spot = np.asarray(spot, dtype=float) * np.exp(
        -np.asarray(dividend_yield, dtype=float)
        * np.asarray(time_to_expiry, dtype=float)
    )
    discounted_strike = np.asarray(strike, dtype=float) * np.exp(
        -np.asarray(rate, dtype=float) * np.asarray(time_to_expiry, dtype=float)
    )

    return w * (discounted_spot * norm.cdf(w * d1) - discounted_strike * norm.cdf(w * d2))


def call_price(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    """Price a European call. Thin wrapper over :func:`black_scholes_price`.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        The call's present value.
    """
    return black_scholes_price(
        spot, strike, time_to_expiry, rate, volatility, OptionType.CALL, dividend_yield
    )


def put_price(
    spot: Numeric,
    strike: Numeric,
    time_to_expiry: Numeric,
    rate: Numeric,
    volatility: Numeric,
    dividend_yield: Numeric = 0.0,
) -> NDArray[np.float64]:
    """Price a European put. Thin wrapper over :func:`black_scholes_price`.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.

    Returns:
        The put's present value.
    """
    return black_scholes_price(
        spot, strike, time_to_expiry, rate, volatility, OptionType.PUT, dividend_yield
    )


def barrier_price_analytic(
    spot: float,
    strike: float,
    barrier: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    knock_in: bool = False,
) -> float:
    r"""Price a **continuously monitored** up-and-out or up-and-in call in closed form.

    Restricted to up-barriers on calls with ``barrier > strike``, which is the
    case Phase 3 needs as a validation anchor. The general Reiner-Rubinstein
    taxonomy has eight variants with different sub-cases; implementing all of them
    would add a great deal of sign-error surface for no pedagogical gain, so this
    deliberately covers one and validates it thoroughly.

    **Where it comes from.** The barrier makes the payoff depend on the running
    maximum, not just ``S_T``. The trick is the *reflection principle*: for driftless
    Brownian motion, every path that touches level ``H`` and finishes at ``x`` can
    be mirrored about ``H`` into a path finishing at ``2H - x``. That turns "the
    probability of touching and finishing somewhere" into an ordinary Gaussian
    probability evaluated at a reflected point. Geometric Brownian motion has drift,
    which is why the reflection picks up the correction factor ``(H/S)^{2 lambda}``
    with ``lambda = (r - q + sigma^2/2) / sigma^2`` — a Girsanov reweighting of the
    reflected paths.

    Everything else follows from in-out parity: a knock-in plus a knock-out equals
    a vanilla, since every path does exactly one of the two.

    **This is the continuous-monitoring price.** A contract monitored at discrete
    dates is a *different, more valuable* contract for a knock-out, and the gap
    closes only as ``O(1/sqrt(m))``. See
    :func:`~options_engine.pricing.monte_carlo.monte_carlo_barrier`.

    Args:
        spot: Current price of the underlying. Must be positive and below
            ``barrier`` (otherwise the option has already knocked out or in).
        strike: Strike price. Must be positive and below ``barrier``.
        barrier: Up-barrier level.
        time_to_expiry: Time to expiry in years. Must be positive.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility as a decimal. Must be positive.
        dividend_yield: Continuous dividend yield. Defaults to 0.0.
        knock_in: Return the up-and-in price instead of the up-and-out price.

    Returns:
        The present value of the barrier call under continuous monitoring.

    Raises:
        ValueError: If inputs are out of domain, or if ``barrier`` is not strictly
            above both ``spot`` and ``strike``.

    Examples:
        >>> round(barrier_price_analytic(100, 100, 125, 1.0, 0.05, 0.25), 4)
        1.354
    """
    validate_inputs(spot, strike, time_to_expiry, volatility)
    if barrier <= spot:
        raise ValueError(
            f"barrier ({barrier}) must be above spot ({spot}); an up-barrier at or "
            f"below spot has already been breached"
        )
    if barrier <= strike:
        raise ValueError(
            f"this implementation requires barrier ({barrier}) above strike "
            f"({strike}); the barrier <= strike case uses a different formula"
        )

    vol_sqrt_t = volatility * np.sqrt(time_to_expiry)
    # lambda is the Girsanov reweighting exponent for reflected paths.
    lam = (rate - dividend_yield + 0.5 * volatility**2) / volatility**2

    x1 = np.log(spot / barrier) / vol_sqrt_t + lam * vol_sqrt_t
    y1 = np.log(barrier / spot) / vol_sqrt_t + lam * vol_sqrt_t
    # H^2 / (S K) is the reflected strike: the barrier mirrors S about H.
    y = np.log(barrier**2 / (spot * strike)) / vol_sqrt_t + lam * vol_sqrt_t

    spot_discounted = spot * np.exp(-dividend_yield * time_to_expiry)
    strike_discounted = strike * np.exp(-rate * time_to_expiry)

    up_and_in = (
        spot_discounted * norm.cdf(x1)
        - strike_discounted * norm.cdf(x1 - vol_sqrt_t)
        - spot_discounted * (barrier / spot) ** (2.0 * lam)
        * (norm.cdf(-y) - norm.cdf(-y1))
        + strike_discounted * (barrier / spot) ** (2.0 * lam - 2.0)
        * (norm.cdf(-y + vol_sqrt_t) - norm.cdf(-y1 + vol_sqrt_t))
    )

    if knock_in:
        return float(up_and_in)

    vanilla = float(
        black_scholes_price(
            spot, strike, time_to_expiry, rate, volatility, OptionType.CALL, dividend_yield
        )
    )
    return vanilla - float(up_and_in)
