"""Shared types, constants, and input validation used across the pricing engine.

Keeping these in one place means every pricer (analytic, tree, Monte Carlo)
speaks the same vocabulary and rejects the same bad inputs the same way.
"""

from __future__ import annotations

from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# A pricing input may be a plain float or a NumPy array (so we can price a whole
# strike ladder in one call). This alias documents that intent at every callsite.
Numeric = Union[float, NDArray[np.float64]]

# Smallest value we substitute for a degenerate T or sigma.
#
# Why substitute instead of branching: the Black-Scholes formula converges to the
# correct limit (discounted intrinsic value) as T -> 0 or sigma -> 0, because
# d1, d2 -> +/-infinity and the normal CDFs saturate at 0 or 1. Clamping to a tiny
# positive number therefore gives the right answer *and* keeps the code branch-free,
# which matters because these functions are vectorised over NumPy arrays.
#
# 1e-16 would underflow sqrt to ~1e-8 and still work, but 1e-12 keeps d1 comfortably
# inside float64 range (|d1| ~ 1e6 rather than ~1e8) while being far smaller than any
# real expiry (one second is ~3e-8 years) or any real volatility.
#
# The clamp is not free, and the size of the residual is worth knowing: at exactly
# T = 0 and exactly at the money, the formula returns the tiny remaining time value
# rather than a clean zero. That residual is bounded by the at-the-money rule of
# thumb, ~ phi(0) * S * sigma * sqrt(TINY) = 0.4 * S * sigma * 1e-6, which for a $100
# underlying at 20% vol is about 8e-6 — four orders of magnitude below a one-cent
# tick, and it decays to zero as S moves away from K. Accepting a sub-tick error at a
# single measure-zero point is a good trade for keeping every pricer branch-free and
# fully vectorised.
TINY: float = 1e-12


class OptionType(str, Enum):
    """Whether a contract is a call or a put.

    Subclasses ``str`` so that ``OptionType.CALL == "call"`` is True. Callers can
    pass either the enum member or the bare string ``"call"`` / ``"put"``, which
    keeps notebook and test code readable without giving up the enum's safety.
    """

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: Union[str, "OptionType"]) -> "OptionType":
        """Coerce a string or enum member to an :class:`OptionType`.

        Args:
            value: ``"call"``, ``"put"`` (any case), or an :class:`OptionType`.

        Returns:
            The corresponding :class:`OptionType` member.

        Raises:
            ValueError: If ``value`` is not a recognised option type.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise ValueError(
                f"option_type must be 'call' or 'put', got {value!r}"
            ) from None

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put.

        Several formulas (payoff, put-call parity, the "unified" Black-Scholes
        expression) differ only by this sign, so exposing it lets us write one
        formula instead of two near-identical branches.
        """
        return 1 if self is OptionType.CALL else -1


class ExerciseStyle(str, Enum):
    """When the holder is allowed to exercise.

    European options may only be exercised at expiry; American options may be
    exercised at any time up to it. That extra right can never be worth less than
    nothing, so an American option is always worth at least its European twin —
    an inequality the tests assert directly.

    Like :class:`OptionType`, subclasses ``str`` so bare strings work as arguments.
    """

    EUROPEAN = "european"
    AMERICAN = "american"

    @classmethod
    def parse(cls, value: Union[str, "ExerciseStyle"]) -> "ExerciseStyle":
        """Coerce a string or enum member to an :class:`ExerciseStyle`.

        Args:
            value: ``"european"``, ``"american"`` (any case), or an
                :class:`ExerciseStyle`.

        Returns:
            The corresponding :class:`ExerciseStyle` member.

        Raises:
            ValueError: If ``value`` is not a recognised exercise style.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise ValueError(
                f"exercise must be 'european' or 'american', got {value!r}"
            ) from None


def intrinsic_value(
    spot: ArrayLike, strike: ArrayLike, option_type: OptionType
) -> NDArray[np.float64]:
    """Compute the payoff from exercising immediately.

    ``max(S - K, 0)`` for a call and ``max(K - S, 0)`` for a put, written as one
    expression using the option's sign. This is both the terminal condition for a
    lattice and the exercise value tested at every American node, so it lives here
    rather than being reimplemented in each pricer.

    Args:
        spot: Current price of the underlying.
        strike: Strike price.
        option_type: Whether the contract is a call or a put.

    Returns:
        The immediate exercise value, broadcast to the shape of the inputs.
    """
    sign = option_type.sign
    return np.maximum(
        sign * (np.asarray(spot, dtype=float) - np.asarray(strike, dtype=float)), 0.0
    )


def validate_inputs(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    volatility: ArrayLike,
) -> None:
    """Reject economically meaningless pricing inputs.

    We allow ``time_to_expiry == 0`` and ``volatility == 0`` (both are well-defined
    limits that return intrinsic value) but reject negatives, which are always a
    caller bug — most often a date subtraction done in the wrong order.

    Interest rates and dividend yields are deliberately *not* validated: negative
    rates are real (EUR and JPY have traded there for years) and the formulas
    handle them without modification.

    Args:
        spot: Current price of the underlying.
        strike: Strike price of the option.
        time_to_expiry: Time to expiry in years.
        volatility: Annualised volatility as a decimal (0.20 means 20%).

    Raises:
        ValueError: If any input is outside its permitted domain, or if any input
            is NaN.
    """
    checks: list[tuple[str, ArrayLike, bool]] = [
        # (name, value, must_be_strictly_positive)
        ("spot", spot, True),
        ("strike", strike, True),
        ("time_to_expiry", time_to_expiry, False),
        ("volatility", volatility, False),
    ]
    for name, value, strictly_positive in checks:
        array = np.asarray(value, dtype=float)
        if np.any(np.isnan(array)):
            raise ValueError(f"{name} contains NaN")
        if strictly_positive:
            if np.any(array <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
        elif np.any(array < 0.0):
            raise ValueError(f"{name} must be non-negative")
