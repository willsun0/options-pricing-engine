"""Phase 4 figures: real market data, implied volatility, and the smile.

Run from the project root::

    python options_engine/notebooks/phase4_market_comparison.py            # cached
    python options_engine/notebooks/phase4_market_comparison.py --fetch    # live

By default this reads the most recent cached snapshot so the figures are
reproducible. Pass ``--fetch`` to pull a fresh chain from yfinance first.

Produces four figures in ``plots/``:

* ``phase4_smile.png`` — the volatility smile at each expiry, the headline result.
* ``phase4_surface.png`` — the same data as a surface and a term structure.
* ``phase4_model_vs_market.png`` — what constant-volatility Black-Scholes actually
  does to prices when you calibrate it at the money.
* ``phase4_data_quality.png`` — what the cleaning filters remove, and why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_engine.data.market_data import (
    MarketSnapshot,
    clean_option_chain,
    fetch_option_chain,
    latest_cached_snapshot,
)
from options_engine.pricing.black_scholes import black_scholes_price
from options_engine.vol_surface.implied_vol import implied_volatility_array

PLOT_DIR = Path(__file__).resolve().parents[2] / "plots"


def load_data(fetch: bool) -> tuple[MarketSnapshot, pd.DataFrame]:
    """Load a snapshot and attach implied volatilities.

    Args:
        fetch: Whether to pull fresh data from yfinance rather than using the cache.

    Returns:
        A ``(snapshot, chain)`` tuple where ``chain`` is cleaned and carries an
        ``iv`` column.
    """
    snapshot = fetch_option_chain("SPY") if fetch else latest_cached_snapshot("SPY")
    chain = clean_option_chain(snapshot)
    chain = chain.assign(
        iv=implied_volatility_array(
            chain["mid"].values, snapshot.spot, chain["strike"].values,
            chain["time_to_expiry"].values, snapshot.rate,
            chain["option_type"].values, snapshot.dividend_yield,
        )
    ).dropna(subset=["iv"])
    return snapshot, chain


def _atm_volatility(group: pd.DataFrame) -> float:
    """Return the implied volatility of the strike closest to the forward.

    Args:
        group: Rows for a single expiry, carrying ``moneyness`` and ``iv``.

    Returns:
        The at-the-money-forward implied volatility.
    """
    return float(group.iloc[(group["moneyness"] - 1.0).abs().argmin()]["iv"])


def plot_smile(snapshot: MarketSnapshot, chain: pd.DataFrame, output_path: Path) -> None:
    """Plot the implied volatility smile for each expiry.

    Args:
        snapshot: The market snapshot, used for the title.
        chain: Cleaned chain with an ``iv`` column.
        output_path: File to write the figure to.
    """
    fig, (ax_smile, ax_normalised) = plt.subplots(1, 2, figsize=(15.5, 6))
    expiries = sorted(chain["days_to_expiry"].unique())
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(expiries)))

    for colour, days in zip(colours, expiries):
        group = chain[chain["days_to_expiry"] == days].sort_values("moneyness")
        if len(group) < 5:
            continue
        ax_smile.plot(
            group["moneyness"], group["iv"] * 100,
            marker="o", markersize=2.5, linewidth=1.5, color=colour, label=f"{days}d",
        )

        # Normalising the x-axis by sigma*sqrt(T) collapses the curves towards each
        # other, because it measures moneyness in standard deviations of the
        # terminal distribution rather than in percent. What remains after that
        # collapse is genuine term structure of *shape*, not just of scale.
        atm = _atm_volatility(group)
        standardised = group["log_moneyness"] / (atm * np.sqrt(group["time_to_expiry"]))
        ax_normalised.plot(
            standardised, group["iv"] / atm,
            marker="o", markersize=2.5, linewidth=1.5, color=colour, label=f"{days}d",
        )

    ax_smile.axvline(1.0, color="grey", linestyle=":", linewidth=1.2)
    ax_smile.set_title("The volatility smile\n(Black-Scholes predicts a horizontal line)")
    ax_smile.set_xlabel("Moneyness  $K/F$")
    ax_smile.set_ylabel("Implied volatility (%)")
    ax_smile.grid(alpha=0.3)
    ax_smile.legend(fontsize=8, title="expiry", ncol=2)

    ax_normalised.axvline(0.0, color="grey", linestyle=":", linewidth=1.2)
    ax_normalised.set_title(
        "Same data, rescaled by $\\sigma\\sqrt{T}$\n(curves collapse: skew decays like $1/\\sqrt{T}$)"
    )
    ax_normalised.set_xlabel("Standardised moneyness  $\\ln(K/F) / (\\sigma\\sqrt{T})$")
    ax_normalised.set_ylabel("Implied vol / ATM vol")
    ax_normalised.grid(alpha=0.3)
    ax_normalised.legend(fontsize=8, title="expiry", ncol=2)

    fig.suptitle(
        f"{snapshot.ticker} implied volatility, spot {snapshot.spot:.2f}, "
        f"{snapshot.as_of:%Y-%m-%d}  (r={snapshot.rate:.2%}, q={snapshot.dividend_yield:.2%})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_surface(snapshot: MarketSnapshot, chain: pd.DataFrame, output_path: Path) -> None:
    """Plot the volatility surface and its term structure.

    Args:
        snapshot: The market snapshot.
        chain: Cleaned chain with an ``iv`` column.
        output_path: File to write the figure to.
    """
    fig = plt.figure(figsize=(16.5, 5.6))
    ax_surface = fig.add_subplot(1, 3, 1, projection="3d")
    ax_term = fig.add_subplot(1, 3, 2)
    ax_skew = fig.add_subplot(1, 3, 3)

    scatter = ax_surface.scatter(
        chain["moneyness"], chain["days_to_expiry"], chain["iv"] * 100,
        c=chain["iv"] * 100, cmap="viridis", s=7,
    )
    ax_surface.set_xlabel("Moneyness $K/F$", fontsize=9)
    ax_surface.set_ylabel("Days to expiry", fontsize=9)
    ax_surface.set_zlabel("Implied vol (%)", fontsize=9)
    ax_surface.set_title("The volatility surface")
    ax_surface.view_init(elev=22, azim=-128)
    fig.colorbar(scatter, ax=ax_surface, shrink=0.55, pad=0.09)

    # --- Term structure of at-the-money volatility ---
    summary = []
    for days, group in chain.groupby("days_to_expiry"):
        if len(group) < 8:
            continue
        wing = group[(group["moneyness"] > 0.85) & (group["moneyness"] < 1.05)]
        slope = (
            float(np.polyfit(wing["log_moneyness"], wing["iv"], 1)[0]) if len(wing) >= 5 else np.nan
        )
        summary.append({"days": days, "atm": _atm_volatility(group), "slope": slope})
    summary = pd.DataFrame(summary).sort_values("days")

    ax_term.plot(summary["days"], summary["atm"] * 100, marker="o", linewidth=1.8)
    ax_term.set_title("ATM volatility term structure\n(upward sloping in calm markets)")
    ax_term.set_xlabel("Days to expiry")
    ax_term.set_ylabel("ATM implied volatility (%)")
    ax_term.set_xscale("log")
    ax_term.grid(alpha=0.3, which="both")

    # --- Skew decay, against a 1/sqrt(T) reference ---
    valid = summary.dropna(subset=["slope"])
    ax_skew.plot(valid["days"], -valid["slope"], marker="o", linewidth=1.8, label="measured skew")
    reference = -valid["slope"].iloc[0] * np.sqrt(valid["days"].iloc[0] / valid["days"])
    ax_skew.plot(
        valid["days"], reference, linestyle="--", color="crimson", linewidth=1.6,
        label=r"$1/\sqrt{T}$ reference",
    )
    ax_skew.set_title("Skew decays with maturity\n(central limit theorem at work)")
    ax_skew.set_xlabel("Days to expiry")
    ax_skew.set_ylabel(r"$-\partial\sigma_{\rm imp}/\partial\ln(K/F)$")
    ax_skew.set_xscale("log")
    ax_skew.set_yscale("log")
    ax_skew.grid(alpha=0.3, which="both")
    ax_skew.legend(fontsize=9)

    fig.suptitle(f"{snapshot.ticker} volatility surface, {snapshot.as_of:%Y-%m-%d}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_model_vs_market(
    snapshot: MarketSnapshot, chain: pd.DataFrame, output_path: Path
) -> None:
    """Show what constant-volatility Black-Scholes does to prices.

    Calibrating a single volatility at the money — which is what "using
    Black-Scholes" means in practice — and then pricing every other strike with it
    is the concrete version of the model's failure. The mispricing is largest
    exactly where crash protection is bought.

    Args:
        snapshot: The market snapshot.
        chain: Cleaned chain with an ``iv`` column.
        output_path: File to write the figure to.
    """
    expiries = sorted(chain["days_to_expiry"].unique())
    target = expiries[len(expiries) // 2]
    group = chain[chain["days_to_expiry"] == target].sort_values("strike").copy()

    atm_vol = _atm_volatility(group)
    group["model"] = [
        float(black_scholes_price(
            snapshot.spot, row.strike, row.time_to_expiry, snapshot.rate,
            atm_vol, row.option_type, snapshot.dividend_yield,
        ))
        for row in group.itertuples()
    ]
    group["error"] = group["model"] - group["mid"]
    group["relative_error"] = group["error"] / group["mid"]

    fig, (ax_price, ax_error, ax_relative) = plt.subplots(1, 3, figsize=(16.5, 5.4))

    for option_type, marker in (("put", "v"), ("call", "^")):
        subset = group[group["option_type"] == option_type]
        ax_price.plot(subset["moneyness"], subset["mid"], marker=marker, markersize=4,
                      linewidth=1.6, label=f"market {option_type}")
        ax_price.plot(subset["moneyness"], subset["model"], marker=marker, markersize=3,
                      linestyle="--", linewidth=1.4, label=f"BS @ {atm_vol:.1%} {option_type}")
    ax_price.set_yscale("log")
    ax_price.set_title(f"Market vs constant-vol model ({target}d)\ncalibrated at the money")
    ax_price.set_xlabel("Moneyness $K/F$")
    ax_price.set_ylabel("Option price (log scale)")
    ax_price.grid(alpha=0.3, which="both")
    ax_price.legend(fontsize=8)

    ax_error.axhline(0.0, color="black", linewidth=1)
    ax_error.plot(group["moneyness"], group["error"], marker="o", markersize=3.5,
                  linewidth=1.6, color="crimson")
    ax_error.fill_between(group["moneyness"], -group["spread"] / 2, group["spread"] / 2,
                          alpha=0.3, color="grey", label="bid-ask half-spread")
    ax_error.set_title("Pricing error in dollars\n(errors dwarf the bid-ask spread)")
    ax_error.set_xlabel("Moneyness $K/F$")
    ax_error.set_ylabel("Model $-$ market ($)")
    ax_error.grid(alpha=0.3)
    ax_error.legend(fontsize=9)

    ax_relative.axhline(0.0, color="black", linewidth=1)
    ax_relative.plot(group["moneyness"], group["relative_error"] * 100, marker="o",
                     markersize=3.5, linewidth=1.6, color="darkorange")
    ax_relative.set_title("Relative error\n(model badly underprices the crash wing)")
    ax_relative.set_xlabel("Moneyness $K/F$")
    ax_relative.set_ylabel("Model / market $-$ 1 (%)")
    ax_relative.grid(alpha=0.3)

    fig.suptitle(
        f"{snapshot.ticker}: one volatility cannot price every strike "
        f"({target} days to expiry)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_data_quality(snapshot: MarketSnapshot, output_path: Path) -> None:
    """Show what the cleaning filters remove and why it matters.

    Args:
        snapshot: The raw market snapshot.
        output_path: File to write the figure to.
    """
    raw = snapshot.chain
    permissive = clean_option_chain(
        snapshot, max_relative_spread=1e9, min_price=0.0,
        min_open_interest=0, otm_only=False,
    )
    strict = clean_option_chain(snapshot)

    fig, (ax_funnel, ax_spread, ax_stale) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # --- Filter funnel ---
    stages = {
        "raw chain": len(raw),
        "tradeable quote": len(permissive),
        "+ liquidity\n& spread": len(clean_option_chain(snapshot, otm_only=False)),
        "+ OTM only": len(strict),
    }
    bars = ax_funnel.bar(range(len(stages)), list(stages.values()), color="steelblue")
    ax_funnel.set_xticks(range(len(stages)))
    ax_funnel.set_xticklabels(list(stages), fontsize=8)
    ax_funnel.bar_label(bars, fontsize=9)
    ax_funnel.set_title(f"Cleaning funnel\n({len(strict)/len(raw):.0%} of quotes survive)")
    ax_funnel.set_ylabel("Contracts")
    ax_funnel.grid(alpha=0.3, axis="y")

    # --- Relative spread distribution ---
    ax_spread.hist(permissive["relative_spread"].clip(0, 1.5), bins=60, color="steelblue")
    ax_spread.axvline(0.25, color="crimson", linestyle="--", linewidth=1.8, label="filter at 25%")
    ax_spread.set_yscale("log")
    ax_spread.set_title("Relative bid-ask spread\n(a wide market means the mid is a guess)")
    ax_spread.set_xlabel("(ask $-$ bid) / mid")
    ax_spread.set_ylabel("Contracts (log scale)")
    ax_spread.grid(alpha=0.3)
    ax_spread.legend(fontsize=9)

    # --- Staleness of lastPrice vs mid ---
    if "lastPrice" in raw.columns:
        usable = raw[(raw["bid"] > 0) & (raw["ask"] >= raw["bid"])].copy()
        usable["mid"] = 0.5 * (usable["bid"] + usable["ask"])
        usable = usable[usable["mid"] > 0.10]
        staleness = (usable["lastPrice"] - usable["mid"]) / usable["mid"]
        ax_stale.hist(staleness.clip(-1, 1) * 100, bins=70, color="darkorange")
        ax_stale.axvline(0.0, color="black", linewidth=1.4)
        share = float((staleness.abs() > 0.10).mean())
        ax_stale.set_yscale("log")
        ax_stale.set_title(
            f"`lastPrice` vs mid\n({share:.0%} of quotes stale by more than 10%)"
        )
        ax_stale.set_xlabel("lastPrice / mid $-$ 1 (%)")
        ax_stale.set_ylabel("Contracts (log scale)")
        ax_stale.grid(alpha=0.3)

    fig.suptitle(
        f"{snapshot.ticker} data quality: why cleaning is most of the work", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Generate every Phase 4 figure."""
    fetch = "--fetch" in sys.argv
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot, chain = load_data(fetch)
    print(f"{snapshot}\nusable quotes with implied vol: {len(chain)}")

    targets = [
        (PLOT_DIR / "phase4_smile.png", lambda p: plot_smile(snapshot, chain, p)),
        (PLOT_DIR / "phase4_surface.png", lambda p: plot_surface(snapshot, chain, p)),
        (PLOT_DIR / "phase4_model_vs_market.png", lambda p: plot_model_vs_market(snapshot, chain, p)),
        (PLOT_DIR / "phase4_data_quality.png", lambda p: plot_data_quality(snapshot, p)),
    ]
    for path, plot_fn in targets:
        plot_fn(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
