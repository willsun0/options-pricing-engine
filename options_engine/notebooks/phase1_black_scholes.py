"""Phase 1 figures: Black-Scholes prices, Greeks, and validation.

Run from the project root::

    python options_engine/notebooks/phase1_black_scholes.py

Produces two figures in ``plots/``:

* ``phase1_greeks.png`` — price and all five Greeks against spot, at three
  expiries. The shapes here are the ones worth being able to sketch from memory.
* ``phase1_analytical_vs_numerical.png`` — the error between the hand-derived
  Greeks and finite differences, which is the visual form of the main test.

Written as a plain script rather than a notebook so it runs in CI and diffs
cleanly in git. The section structure maps one-to-one onto notebook cells if you
want to paste it into Jupyter for exploration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: this script only writes files
import matplotlib.pyplot as plt
import numpy as np

from options_engine.greeks import (
    delta,
    gamma,
    numerical_delta,
    numerical_gamma,
    numerical_rho,
    numerical_theta,
    numerical_vega,
    rho,
    theta,
    theta_per_day,
    vega,
    vega_per_percent,
)
from options_engine.pricing.black_scholes import black_scholes_price

PLOT_DIR = Path(__file__).resolve().parents[2] / "plots"

# A conventional set of parameters: $100 stock, $100 strike, 5% rates, 20% vol.
STRIKE = 100.0
RATE = 0.05
VOLATILITY = 0.20
DIVIDEND_YIELD = 0.0
EXPIRIES = [(1.0 / 12.0, "1 month"), (0.5, "6 months"), (2.0, "2 years")]
SPOTS = np.linspace(50.0, 150.0, 400)


def plot_price_and_greeks(output_path: Path) -> None:
    """Plot price and all five Greeks against spot at three expiries.

    Greeks are shown in desk units (theta per day, vega per vol point) because
    those are the numbers a trader would actually recognise on a risk report.

    Args:
        output_path: File to write the figure to.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Black-Scholes call: price and Greeks vs spot "
        f"(K={STRIKE:.0f}, r={RATE:.0%}, $\\sigma$={VOLATILITY:.0%})",
        fontsize=14,
    )

    panels = [
        ("Price", lambda t: black_scholes_price(SPOTS, STRIKE, t, RATE, VOLATILITY, "call", DIVIDEND_YIELD), "$"),
        ("Delta", lambda t: delta(SPOTS, STRIKE, t, RATE, VOLATILITY, "call", DIVIDEND_YIELD), "shares per option"),
        ("Gamma", lambda t: gamma(SPOTS, STRIKE, t, RATE, VOLATILITY, "call", DIVIDEND_YIELD), "delta per $1"),
        ("Vega", lambda t: vega_per_percent(vega(SPOTS, STRIKE, t, RATE, VOLATILITY, "call", DIVIDEND_YIELD)), "$ per vol point"),
        ("Theta", lambda t: theta_per_day(theta(SPOTS, STRIKE, t, RATE, VOLATILITY, "call", DIVIDEND_YIELD)), "$ per day"),
        ("Rho", lambda t: rho(SPOTS, STRIKE, t, RATE, VOLATILITY, "call", DIVIDEND_YIELD) / 100.0, "$ per 1% rate move"),
    ]

    for ax, (name, compute, ylabel) in zip(axes.flat, panels):
        for time, label in EXPIRIES:
            ax.plot(SPOTS, compute(time), label=label, linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel("Spot")
        ax.set_ylabel(ylabel)
        ax.axvline(STRIKE, color="grey", linestyle=":", linewidth=1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Overlay intrinsic value on the price panel: the gap between the curve and
    # the kinked payoff is the option's time value, which is the whole product.
    axes.flat[0].plot(
        SPOTS,
        np.maximum(SPOTS - STRIKE, 0.0),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="intrinsic",
    )
    axes.flat[0].legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_analytical_vs_numerical(output_path: Path) -> None:
    """Plot the error between analytical and finite-difference Greeks.

    This is the picture behind the headline test. Errors sit at the 1e-9 level or
    below for the first-order Greeks; gamma is visibly noisier because a second
    difference divides by h^2 and so loses roughly half the available precision.

    Args:
        output_path: File to write the figure to.
    """
    time = 0.5
    common = (STRIKE, time, RATE, VOLATILITY, "call", DIVIDEND_YIELD)

    comparisons = [
        ("Delta", delta(SPOTS, *common), numerical_delta(SPOTS, *common)),
        ("Gamma", gamma(SPOTS, *common), numerical_gamma(SPOTS, *common)),
        ("Vega", vega(SPOTS, *common), numerical_vega(SPOTS, *common)),
        ("Theta", theta(SPOTS, *common), numerical_theta(SPOTS, *common)),
        ("Rho", rho(SPOTS, *common), numerical_rho(SPOTS, *common)),
    ]

    fig, (ax_values, ax_errors) = plt.subplots(1, 2, figsize=(14, 5.5))

    for name, analytic, _ in comparisons:
        ax_values.plot(SPOTS, np.asarray(analytic), label=name, linewidth=1.6)
    ax_values.set_title("Analytical Greeks (raw units, 6-month call)")
    ax_values.set_xlabel("Spot")
    ax_values.set_ylabel("Value")
    ax_values.axvline(STRIKE, color="grey", linestyle=":", linewidth=1)
    ax_values.grid(alpha=0.3)
    ax_values.legend(fontsize=9)

    for name, analytic, numeric in comparisons:
        absolute_error = np.abs(np.asarray(analytic) - np.asarray(numeric))
        # Clip at the float64 floor so the log axis stays readable where the two
        # methods agree to the last bit.
        ax_errors.semilogy(SPOTS, np.maximum(absolute_error, 1e-18), label=name, linewidth=1.4)
    ax_errors.set_title("Absolute error: analytical vs finite difference")
    ax_errors.set_xlabel("Spot")
    ax_errors.set_ylabel("Absolute error (log scale)")
    ax_errors.grid(alpha=0.3, which="both")
    ax_errors.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Generate every Phase 1 figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    greeks_path = PLOT_DIR / "phase1_greeks.png"
    validation_path = PLOT_DIR / "phase1_analytical_vs_numerical.png"

    plot_price_and_greeks(greeks_path)
    plot_analytical_vs_numerical(validation_path)

    print(f"wrote {greeks_path}")
    print(f"wrote {validation_path}")


if __name__ == "__main__":
    main()
