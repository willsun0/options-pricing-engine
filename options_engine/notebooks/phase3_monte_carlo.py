"""Phase 3 figures: Monte Carlo convergence and variance reduction.

Run from the project root::

    python options_engine/notebooks/phase3_monte_carlo.py

Produces three figures in ``plots/``:

* ``phase3_convergence.png`` — price and error vs path count, with the
  O(1/sqrt(N)) rate shown explicitly.
* ``phase3_variance_reduction.png`` — standard error of each technique on the same
  axes, plus the case where antithetic sampling makes things *worse*.
* ``phase3_exotics.png`` — Asian and barrier prices against their vanilla
  equivalents, showing what the path dependence actually does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from options_engine.pricing.black_scholes import barrier_price_analytic, black_scholes_price
from options_engine.pricing.monte_carlo import (
    geometric_asian_price,
    monte_carlo_asian,
    monte_carlo_barrier,
    monte_carlo_european,
    simulate_terminal_prices,
)

PLOT_DIR = Path(__file__).resolve().parents[2] / "plots"

SPOT = 100.0
STRIKE = 100.0
TIME = 1.0
RATE = 0.05
VOLATILITY = 0.30
DIVIDEND_YIELD = 0.0
SEED = 20240101


def plot_convergence(output_path: Path) -> None:
    """Plot Monte Carlo price and error against path count.

    The left panel shows prices with their 95% confidence bands narrowing towards
    the exact Black-Scholes value. The right panel puts absolute error on log-log
    axes against an O(1/sqrt(N)) reference.

    Args:
        output_path: File to write the figure to.
    """
    exact = float(black_scholes_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD))
    path_counts = np.unique(np.logspace(2.5, 6.0, 24).astype(int))

    prices, errors = [], []
    for n in path_counts:
        result = monte_carlo_european(
            SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD,
            n_paths=int(n), seed=SEED,
        )
        prices.append(result.price)
        errors.append(result.standard_error)

    prices = np.array(prices)
    errors = np.array(errors)

    fig, (ax_price, ax_error) = plt.subplots(1, 2, figsize=(15, 5.5))

    ax_price.fill_between(
        path_counts, prices - 1.96 * errors, prices + 1.96 * errors,
        alpha=0.25, label="95% confidence band",
    )
    ax_price.plot(path_counts, prices, marker="o", markersize=3.5, linewidth=1.2, label="Monte Carlo")
    ax_price.axhline(exact, color="black", linestyle="--", linewidth=1.4, label="Black-Scholes")
    ax_price.set_xscale("log")
    ax_price.set_title("Price converges, and the error bars know it")
    ax_price.set_xlabel("Number of paths (N)")
    ax_price.set_ylabel("Call price")
    ax_price.grid(alpha=0.3)
    ax_price.legend(fontsize=9)

    absolute_error = np.abs(prices - exact)
    ax_error.loglog(path_counts, absolute_error, marker="o", markersize=3.5,
                    linewidth=1.1, label="|MC $-$ Black-Scholes|")
    ax_error.loglog(path_counts, errors, linewidth=1.8, label="reported standard error")
    reference = errors[0] * np.sqrt(path_counts[0]) / np.sqrt(path_counts)
    ax_error.loglog(path_counts, reference, linestyle=":", color="crimson", linewidth=1.8,
                    label=r"reference $O(1/\sqrt{N})$")
    ax_error.set_title("Error decays as $1/\\sqrt{N}$\n(100x more paths for one more digit)")
    ax_error.set_xlabel("Number of paths (N)")
    ax_error.set_ylabel("Error")
    ax_error.grid(alpha=0.3, which="both")
    ax_error.legend(fontsize=9)

    fig.suptitle(
        f"Monte Carlo convergence, European call "
        f"(S={SPOT:.0f}, K={STRIKE:.0f}, T={TIME:.0f}y, r={RATE:.0%}, $\\sigma$={VOLATILITY:.0%})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_variance_reduction(output_path: Path) -> None:
    """Plot standard error for each variance reduction technique.

    Three panels: vanilla European (where antithetic helps), the Asian option
    (where the geometric control variate is spectacular), and a straddle (where
    antithetic sampling actively hurts).

    Args:
        output_path: File to write the figure to.
    """
    path_counts = np.unique(np.logspace(3, 5.7, 16).astype(int))
    fig, (ax_vanilla, ax_asian, ax_straddle) = plt.subplots(1, 3, figsize=(17, 5.4))

    # --- Panel 1: European call ---
    variants = [
        ("plain", dict()),
        ("antithetic", dict(antithetic=True)),
        ("control variate", dict(control_variate="terminal_stock")),
        ("both", dict(antithetic=True, control_variate="terminal_stock")),
    ]
    for label, kwargs in variants:
        errors = [
            monte_carlo_european(
                SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD,
                n_paths=int(n) - int(n) % 2, seed=SEED, **kwargs,
            ).standard_error
            for n in path_counts
        ]
        ax_vanilla.loglog(path_counts, errors, marker="o", markersize=3, linewidth=1.4, label=label)
    ax_vanilla.set_title("European call\n(all slopes are $-1/2$: only the level moves)")
    ax_vanilla.set_xlabel("Number of paths (N)")
    ax_vanilla.set_ylabel("Standard error")
    ax_vanilla.grid(alpha=0.3, which="both")
    ax_vanilla.legend(fontsize=8.5)

    # --- Panel 2: Asian call ---
    asian_variants = [
        ("plain", dict()),
        ("antithetic", dict(antithetic=True)),
        ("geometric control", dict(control_variate="geometric_asian")),
        ("both", dict(antithetic=True, control_variate="geometric_asian")),
    ]
    for label, kwargs in asian_variants:
        errors = [
            monte_carlo_asian(
                SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD,
                n_paths=int(n) - int(n) % 2, n_averaging_dates=52, seed=SEED, **kwargs,
            ).standard_error
            for n in path_counts
        ]
        ax_asian.loglog(path_counts, errors, marker="o", markersize=3, linewidth=1.4, label=label)
    ax_asian.set_title("Arithmetic Asian call\n(geometric control: 25x less error = 600x less variance)")
    ax_asian.set_xlabel("Number of paths (N)")
    ax_asian.set_ylabel("Standard error")
    ax_asian.grid(alpha=0.3, which="both")
    ax_asian.legend(fontsize=8.5)

    # --- Panel 3: the counterexample ---
    def straddle_standard_error(n_paths: int, antithetic: bool) -> float:
        """Standard error of a simulated straddle, |S_T - K|."""
        terminal = simulate_terminal_prices(
            SPOT, TIME, RATE, VOLATILITY, DIVIDEND_YIELD,
            n_paths=n_paths, seed=SEED, antithetic=antithetic,
        )
        payoffs = np.abs(terminal - STRIKE)
        if antithetic:
            half = payoffs.size // 2
            payoffs = 0.5 * (payoffs[:half] + payoffs[half:])
        return float(np.std(payoffs, ddof=1)) / np.sqrt(payoffs.size)

    for label, antithetic in [("plain", False), ("antithetic", True)]:
        errors = [straddle_standard_error(int(n) - int(n) % 2, antithetic) for n in path_counts]
        ax_straddle.loglog(path_counts, errors, marker="o", markersize=3, linewidth=1.4, label=label)
    ax_straddle.set_title("Straddle $|S_T - K|$\n(antithetic HURTS: payoff is symmetric)")
    ax_straddle.set_xlabel("Number of paths (N)")
    ax_straddle.set_ylabel("Standard error")
    ax_straddle.grid(alpha=0.3, which="both")
    ax_straddle.legend(fontsize=8.5)

    fig.suptitle("Variance reduction: what works, and what does not", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_exotics(output_path: Path) -> None:
    """Plot Asian and barrier prices against their vanilla equivalents.

    Args:
        output_path: File to write the figure to.
    """
    fig, (ax_asian, ax_barrier, ax_monitoring) = plt.subplots(1, 3, figsize=(17, 5.4))

    # --- Asian vs vanilla across strikes ---
    strikes = np.linspace(70.0, 130.0, 25)
    vanilla = np.array(
        [float(black_scholes_price(SPOT, k, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD)) for k in strikes]
    )
    asian = np.array(
        [
            monte_carlo_asian(
                SPOT, k, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD,
                n_paths=200_000, n_averaging_dates=52, seed=SEED,
                antithetic=True, control_variate="geometric_asian",
            ).price
            for k in strikes
        ]
    )
    geometric = np.array(
        [geometric_asian_price(SPOT, k, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD, 52) for k in strikes]
    )
    ax_asian.plot(strikes, vanilla, linewidth=1.9, label="European (Black-Scholes)")
    ax_asian.plot(strikes, asian, linewidth=1.9, label="arithmetic Asian (MC)")
    ax_asian.plot(strikes, geometric, linewidth=1.4, linestyle="--", label="geometric Asian (exact)")
    ax_asian.axvline(SPOT, color="grey", linestyle=":", linewidth=1)
    ax_asian.set_title("Asian options are cheaper\n(averaging damps volatility)")
    ax_asian.set_xlabel("Strike")
    ax_asian.set_ylabel("Price")
    ax_asian.grid(alpha=0.3)
    ax_asian.legend(fontsize=8.5)

    # --- Barrier vs vanilla across barrier levels ---
    barriers = np.linspace(105.0, 220.0, 25)
    vanilla_atm = float(black_scholes_price(SPOT, STRIKE, TIME, RATE, VOLATILITY, "call", DIVIDEND_YIELD))
    knock_out = np.array(
        [
            monte_carlo_barrier(
                SPOT, STRIKE, float(b), TIME, RATE, VOLATILITY, "call", "up_and_out",
                DIVIDEND_YIELD, n_paths=150_000, n_monitoring_dates=250, seed=SEED,
            ).price
            for b in barriers
        ]
    )
    knock_in = np.array(
        [
            monte_carlo_barrier(
                SPOT, STRIKE, float(b), TIME, RATE, VOLATILITY, "call", "up_and_in",
                DIVIDEND_YIELD, n_paths=150_000, n_monitoring_dates=250, seed=SEED,
            ).price
            for b in barriers
        ]
    )
    ax_barrier.plot(barriers, knock_out, linewidth=1.9, label="up-and-out call")
    ax_barrier.plot(barriers, knock_in, linewidth=1.9, label="up-and-in call")
    ax_barrier.plot(barriers, knock_out + knock_in, linewidth=1.2, linestyle=":", color="black",
                    label="in + out (= vanilla)")
    ax_barrier.axhline(vanilla_atm, color="grey", linestyle="--", linewidth=1.2, label="vanilla call")
    ax_barrier.set_title("In-out parity holds exactly\n(every path knocks out or does not)")
    ax_barrier.set_xlabel("Barrier level")
    ax_barrier.set_ylabel("Price")
    ax_barrier.grid(alpha=0.3)
    ax_barrier.legend(fontsize=8.5)

    # --- Discrete vs continuous monitoring ---
    monitoring_counts = np.unique(np.logspace(0.5, 2.9, 14).astype(int))
    raw = np.array(
        [
            monte_carlo_barrier(
                SPOT, STRIKE, 125.0, TIME, RATE, VOLATILITY, "call", "up_and_out",
                DIVIDEND_YIELD, n_paths=150_000, n_monitoring_dates=int(m), seed=SEED,
            ).price
            for m in monitoring_counts
        ]
    )
    corrected = np.array(
        [
            monte_carlo_barrier(
                SPOT, STRIKE, 125.0, TIME, RATE, VOLATILITY, "call", "up_and_out",
                DIVIDEND_YIELD, n_paths=150_000, n_monitoring_dates=int(m), seed=SEED,
                continuity_correction=True,
            ).price
            for m in monitoring_counts
        ]
    )
    ax_monitoring.semilogx(monitoring_counts, raw, marker="o", markersize=4, linewidth=1.6,
                           label="discrete monitoring")
    ax_monitoring.semilogx(monitoring_counts, corrected, marker="s", markersize=4, linewidth=1.6,
                           label="with BGK correction")
    continuous = barrier_price_analytic(SPOT, STRIKE, 125.0, TIME, RATE, VOLATILITY, DIVIDEND_YIELD)
    ax_monitoring.axhline(continuous, color="black", linestyle="--", linewidth=1.4,
                          label="exact continuous price")
    ax_monitoring.set_title("Discrete monitoring is worth more\n(BGK shift lands on the exact price)")
    ax_monitoring.set_xlabel("Monitoring dates")
    ax_monitoring.set_ylabel("Up-and-out call price")
    ax_monitoring.grid(alpha=0.3, which="both")
    ax_monitoring.legend(fontsize=8.5)

    fig.suptitle("Path-dependent payoffs: Asian and barrier options", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Generate every Phase 3 figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (PLOT_DIR / "phase3_convergence.png", plot_convergence),
        (PLOT_DIR / "phase3_variance_reduction.png", plot_variance_reduction),
        (PLOT_DIR / "phase3_exotics.png", plot_exotics),
    ]
    for path, plot_fn in targets:
        plot_fn(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
