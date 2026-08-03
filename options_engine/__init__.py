"""A readable options pricing engine: analytics, lattices, simulation, and vol surfaces."""

from options_engine.common import ExerciseStyle, OptionType
from options_engine.greeks import Greeks, all_greeks, numerical_greeks
from options_engine.pricing.binomial_tree import (
    binomial_price,
    binomial_price_averaged,
    early_exercise_boundary,
)
from options_engine.pricing.monte_carlo import (
    BarrierType,
    ControlVariate,
    MonteCarloResult,
    geometric_asian_price,
    monte_carlo_asian,
    monte_carlo_barrier,
    monte_carlo_european,
)
from options_engine.pricing.black_scholes import (
    barrier_price_analytic,
    black_scholes_price,
    call_price,
    forward_price,
    put_price,
)

# Phase 4/5 surface tools. ml_surface is *not* imported here: it requires torch,
# and this module must stay importable with only the core numeric stack.
from options_engine.vol_surface.implied_vol import (
    implied_volatility,
    implied_volatility_array,
)
from options_engine.vol_surface.svi import (
    SVIParameters,
    fit_svi_slice,
    fit_svi_surface,
    svi_surface_volatility,
)

__version__ = "0.1.0"

__all__ = [
    "OptionType",
    "ExerciseStyle",
    "black_scholes_price",
    "call_price",
    "put_price",
    "forward_price",
    "binomial_price",
    "binomial_price_averaged",
    "early_exercise_boundary",
    "MonteCarloResult",
    "ControlVariate",
    "BarrierType",
    "monte_carlo_european",
    "monte_carlo_asian",
    "monte_carlo_barrier",
    "geometric_asian_price",
    "barrier_price_analytic",
    "Greeks",
    "all_greeks",
    "numerical_greeks",
    "implied_volatility",
    "implied_volatility_array",
    "SVIParameters",
    "fit_svi_slice",
    "fit_svi_surface",
    "svi_surface_volatility",
    "__version__",
]
