"""Phase 2 figures: binomial tree convergence and American early exercise.

Run from the project root::

    python options_engine/notebooks/phase2_binomial_tree.py

Produces three figures in ``plots/``:

* ``phase2_convergence.png`` — tree price against step count, converging to
  Black-Scholes, with the even/odd branch structure made explicit.
* ``phase2_convergence_rate.png`` — the same error on log-log axes, showing the
  fitted O(1/N) slope.
* ``phase2_american_early_exercise.png`` — where American puts are worth more
  than European ones, and the free boundary that says when to exercise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from options_engine.pricing.binomial_tree import (
    binomial_price,
    binomial_price_averaged,
    early_exercise_boundary,
)
from options_engine.pricing.black_scholes import black_scholes_price

PLOT_DIR = Path(__file__).resolve().parents[2] / "plots"

# Reference contract. The strike is deliberately *not* equal to spot: an ATM
# strike sits on a node for some step counts and the oscillation looks different
# (and misleadingly tidy).
SPOT = 100.0
STRIKE = 105.0
TIME = 1.0
RATE = 0.05
VOLATILITY = 0.25
DIVIDEND_YIELD = 0.0


def plot_convergence(output_path: Path) -> None:
    """Plot tree price vs step count against the Black-Scholes limit.

    The left panel shows the raw sawtooth over a small range of steps, with even
    and odd step counts coloured separately to show they trace two distinct
    smooth branches. The right panel shows raw vs averaged over a wider range.

    Args:
        output_path: File to write the figure to.
    """
    exact = float(black_scholes_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD))

    fig, (ax_zoom, ax_wide) = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- Left: the parity structure, zoomed in ---
    zoom_steps = np.arange(20, 81)
    zoom_prices = np.array(
        [binomial_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD, steps=int(n))
         for n in zoom_steps]
    )
    even = zoom_steps % 2 == 0

    ax_zoom.plot(zoom_steps, zoom_prices, color="lightgrey", linewidth=1, zorder=1)
    ax_zoom.scatter(zoom_steps[even], zoom_prices[even], s=22, label="even steps", zorder=3)
    ax_zoom.scatter(zoom_steps[~even], zoom_prices[~even], s=22, label="odd steps", zorder=3)
    ax_zoom.axhline(exact, color="black", linestyle="--", linewidth=1.4, label="Black-Scholes", zorder=2)
    ax_zoom.set_title("Convergence oscillates: even and odd steps\ntrace two separate branches")
    ax_zoom.set_xlabel("Number of time steps (N)")
    ax_zoom.set_ylabel("Call price")
    ax_zoom.grid(alpha=0.3)
    ax_zoom.legend(fontsize=9)

    # --- Right: raw vs averaged, wider range ---
    wide_steps = np.arange(10, 401)
    raw = np.array(
        [binomial_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD, steps=int(n))
         for n in wide_steps]
    )
    averaged = np.array(
        [binomial_price_averaged(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD, steps=int(n))
         for n in wide_steps]
    )

    ax_wide.plot(wide_steps, raw, linewidth=0.9, alpha=0.75, label="CRR tree")
    ax_wide.plot(wide_steps, averaged, linewidth=1.4, label="averaged (N, N+1)")
    ax_wide.axhline(exact, color="black", linestyle="--", linewidth=1.4, label="Black-Scholes")
    ax_wide.set_title("Averaging consecutive step counts damps the oscillation")
    ax_wide.set_xlabel("Number of time steps (N)")
    ax_wide.set_ylabel("Call price")
    ax_wide.set_ylim(exact - 0.12, exact + 0.12)
    ax_wide.grid(alpha=0.3)
    ax_wide.legend(fontsize=9)

    fig.suptitle(
        f"CRR binomial convergence to Black-Scholes "
        f"(S={SPOT:.0f}, K={STRIKE:.0f}, T={TIME:.0f}y, r={RATE:.0%}, $\\sigma$={VOLATILITY:.0%})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_convergence_rate(output_path: Path) -> None:
    """Plot absolute error on log-log axes with the fitted convergence order.

    A straight line on log-log axes means a power law; its slope is the order of
    convergence. Fitting over even step counts only isolates one branch — mixing
    parities would corrupt the estimate.

    Args:
        output_path: File to write the figure to.
    """
    exact = float(black_scholes_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD))

    fig, ax = plt.subplots(figsize=(8.5, 6))

    all_steps = np.arange(20, 1001)
    all_errors = np.abs(
        np.array([binomial_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD, steps=int(n))
                  for n in all_steps]) - exact
    )
    ax.loglog(all_steps, all_errors, linewidth=0.6, alpha=0.4, color="grey", label="all N (both branches)")

    even_steps = np.arange(20, 1001, 2)
    even_errors = np.abs(
        np.array([binomial_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD, steps=int(n))
                  for n in even_steps]) - exact
    )
    ax.loglog(even_steps, even_errors, linewidth=1.4, label="even N only")

    slope, intercept = np.polyfit(np.log(even_steps), np.log(even_errors), 1)
    ax.loglog(
        even_steps,
        np.exp(intercept) * even_steps**slope,
        linestyle="--",
        color="black",
        linewidth=1.6,
        label=f"fitted slope = {slope:.2f}",
    )
    # O(1/N) guide line, anchored to the fit so it is visually comparable.
    ax.loglog(
        even_steps,
        np.exp(intercept) * even_steps**-1.0,
        linestyle=":",
        color="crimson",
        linewidth=1.6,
        label="reference slope $-1$  (i.e. $O(1/N)$)",
    )

    ax.set_title("Convergence order: error vs number of steps")
    ax.set_xlabel("Number of time steps (N)")
    ax.set_ylabel("|tree price $-$ Black-Scholes|")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_american_early_exercise(output_path: Path) -> None:
    """Plot the American early-exercise premium and the free boundary.

    Uses a high rate and a put, since that is where early exercise actually has
    value on a non-dividend-paying underlying.

    Args:
        output_path: File to write the figure to.
    """
    strike, time, rate, vol = 100.0, 1.0, 0.08, 0.25
    spots = np.linspace(50.0, 150.0, 120)
    steps = 600

    american = np.array(
        [binomial_price(s, strike, time, rate, vol, "put", steps=steps, exercise="american") for s in spots]
    )
    european = np.array(
        [binomial_price(s, strike, time, rate, vol, "put", steps=steps, exercise="european") for s in spots]
    )
    intrinsic = np.maximum(strike - spots, 0.0)

    fig, (ax_price, ax_premium, ax_boundary) = plt.subplots(1, 3, figsize=(16.5, 5.4))

    ax_price.plot(spots, american, linewidth=1.9, label="American put")
    ax_price.plot(spots, european, linewidth=1.9, linestyle="--", label="European put")
    ax_price.plot(spots, intrinsic, color="black", linewidth=1.2, linestyle=":", label="intrinsic")
    ax_price.set_title("American vs European put")
    ax_price.set_xlabel("Spot")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.3)
    ax_price.legend(fontsize=9)

    ax_premium.plot(spots, american - european, linewidth=1.9, color="crimson")
    ax_premium.set_title("Early-exercise premium\n(American $-$ European)")
    ax_premium.set_xlabel("Spot")
    ax_premium.set_ylabel("Premium")
    ax_premium.axvline(strike, color="grey", linestyle=":", linewidth=1)
    ax_premium.grid(alpha=0.3)

    times, boundary = early_exercise_boundary(
        strike, strike, time, rate, vol, "put", steps=steps
    )
    ax_boundary.plot(times, boundary, linewidth=1.8)
    ax_boundary.axhline(strike, color="grey", linestyle="--", linewidth=1.2, label=f"strike = {strike:.0f}")
    ax_boundary.fill_between(times, 0, boundary, alpha=0.18, label="exercise immediately")
    ax_boundary.set_title("Early-exercise boundary\n(exercise below the curve)")
    ax_boundary.set_xlabel("Time to expiry (years)")
    ax_boundary.set_ylabel("Critical spot")
    ax_boundary.set_ylim(boundary.min() * 0.95, strike * 1.05)
    ax_boundary.grid(alpha=0.3)
    ax_boundary.legend(fontsize=9)

    fig.suptitle(
        f"American put early exercise (K={strike:.0f}, T={time:.0f}y, r={rate:.0%}, $\\sigma$={vol:.0%})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Generate every Phase 2 figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (PLOT_DIR / "phase2_convergence.png", plot_convergence),
        (PLOT_DIR / "phase2_convergence_rate.png", plot_convergence_rate),
        (PLOT_DIR / "phase2_american_early_exercise.png", plot_american_early_exercise),
    ]
    for path, plot_fn in targets:
        plot_fn(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
