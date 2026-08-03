"""Implied volatility, parametric smile fits, and the ML surface model.

``ml_surface`` is deliberately **not** re-exported here. It imports ``torch``,
which is a large optional dependency, and pulling it in at package import time
would make ``import options_engine`` fail for anyone who only wants Phases 1-4.
Import it explicitly instead::

    from options_engine.vol_surface.ml_surface import train_surface_model
"""

from options_engine.vol_surface.implied_vol import (
    ImpliedVolResult,
    InitialGuess,
    implied_volatility,
    implied_volatility_array,
    price_bounds,
)
from options_engine.vol_surface.svi import (
    SVIFitResult,
    SVIParameters,
    durrleman_function,
    fit_svi_slice,
    fit_svi_surface,
    is_butterfly_free,
    svi_surface_volatility,
    svi_total_variance,
)

__all__ = [
    "ImpliedVolResult",
    "InitialGuess",
    "implied_volatility",
    "implied_volatility_array",
    "price_bounds",
    "SVIParameters",
    "SVIFitResult",
    "fit_svi_slice",
    "fit_svi_surface",
    "svi_surface_volatility",
    "svi_total_variance",
    "durrleman_function",
    "is_butterfly_free",
]
