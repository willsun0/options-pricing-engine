"""Phase 5 figures: SVI against a neural network on the SPY volatility surface.

Run from the project root::

    python options_engine/notebooks/phase5_ml_vol_surface.py            # cached data
    python options_engine/notebooks/phase5_ml_vol_surface.py --fetch    # live data

Requires torch (``pip install -r requirements.txt``). Takes roughly a minute:
most of it is the several networks trained for the activation comparison.

Produces five figures in ``plots/``:

* ``phase5_svi_fit.png`` — the parametric fit slice by slice, plus the fitted
  total variance curves that reveal whether the slices cross.
* ``phase5_ml_vs_svi.png`` — the two models on the same smiles, with residuals.
* ``phase5_surface.png`` — both surfaces on a dense grid, and their difference.
* ``phase5_smoothness.png`` — why the activation matters: tanh against ReLU, in
  volatility and in curvature.
* ``phase5_extrapolation.png`` — the honest limitation, for both models.
"""

from __future__ import annotations

import argparse
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
from options_engine.vol_surface.implied_vol import implied_volatility_array
from options_engine.vol_surface.ml_surface import (
    Activation,
    TrainingConfig,
    VolatilitySurfaceModel,
    build_features,
    extrapolation_report,
    train_surface_model,
)
from options_engine.vol_surface.svi import (
    SVIFitResult,
    fit_svi_slice,
    fit_svi_surface,
    worst_calendar_gap,
)

PLOT_DIR = Path(__file__).resolve().parents[2] / "plots"

# Colour convention held across every Phase 5 figure: market quotes in grey,
# the parametric model in blue, the network in orange.
MARKET_COLOUR = "0.35"
SVI_COLOUR = "#1f77b4"
ML_COLOUR = "#ff7f0e"


def load_data(fetch: bool) -> tuple[MarketSnapshot, pd.DataFrame]:
    """Load a cleaned chain with implied volatilities attached.

    Args:
        fetch: Whether to pull a fresh chain from yfinance rather than the cache.

    Returns:
        A ``(snapshot, chain)`` tuple where ``chain`` carries an ``iv`` column.
    """
    snapshot = fetch_option_chain("SPY") if fetch else latest_cached_snapshot("SPY")
    chain = clean_option_chain(snapshot)
    chain = chain.assign(
        iv=implied_volatility_array(
            chain["mid"].values, snapshot.spot, chain["strike"].values,
            chain["time_to_expiry"].values, snapshot.rate,
            chain["option_type"].values, snapshot.dividend_yield,
        )
    ).dropna(subset=["iv"]).reset_index(drop=True)
    return snapshot, chain


def fit_both_models(
    chain: pd.DataFrame,
) -> tuple[dict[float, SVIFitResult], VolatilitySurfaceModel, object]:
    """Calibrate SVI slice by slice and train the network on the whole surface.

    Args:
        chain: Cleaned chain with an ``iv`` column.

    Returns:
        A ``(svi_fits, model, history)`` tuple.
    """
    svi_fits = fit_svi_surface(
        chain["log_moneyness"].values, chain["iv"].values, chain["time_to_expiry"].values
    )
    features = build_features(
        chain["strike"].values, chain["time_to_expiry"].values, chain["moneyness"].values
    )
    model, history = train_surface_model(features, chain["iv"].values)
    return svi_fits, model, history


def _predict_ml(
    model: VolatilitySurfaceModel,
    strikes: np.ndarray,
    time_to_expiry: float,
    forward: float,
) -> np.ndarray:
    """Evaluate the network along one expiry slice.

    Args:
        model: The trained surface model.
        strikes: Strikes to evaluate at.
        time_to_expiry: Expiry in years.
        forward: Forward price for that expiry, used to form moneyness.

    Returns:
        Predicted implied volatilities.
    """
    return model.predict(
        build_features(strikes, np.full(strikes.size, time_to_expiry), strikes / forward)
    )


def plot_svi_fit(
    snapshot: MarketSnapshot,
    chain: pd.DataFrame,
    svi_fits: dict[float, SVIFitResult],
    output_path: Path,
) -> None:
    """Plot the SVI fit per expiry and the fitted total variance curves.

    The right-hand panel is the calendar-arbitrage check. Slices are fitted
    independently, so nothing in the calibration prevents a shorter expiry's
    total variance from exceeding a longer one's — which would be an arbitrage.
    Plotting w rather than sigma makes the condition visual: the curves must not
    cross.

    Args:
        snapshot: The market snapshot, used for the title.
        chain: Cleaned chain with an ``iv`` column.
        svi_fits: Fitted slices keyed by time to expiry.
        output_path: File to write the figure to.
    """
    fig, (ax_fit, ax_variance) = plt.subplots(1, 2, figsize=(15.5, 6))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(svi_fits)))

    for colour, (expiry, fit) in zip(colours, sorted(svi_fits.items())):
        rows = chain[chain["time_to_expiry"] == expiry].sort_values("log_moneyness")
        days = int(round(expiry * 365))
        grid = np.linspace(rows["log_moneyness"].min(), rows["log_moneyness"].max(), 200)

        ax_fit.scatter(
            rows["log_moneyness"], rows["iv"] * 100,
            s=7, color=colour, alpha=0.45, edgecolors="none",
        )
        ax_fit.plot(
            grid, fit.parameters.implied_volatility(grid) * 100,
            color=colour, linewidth=1.8,
            label=f"{days}d  rmse {fit.rmse * 100:.2f}vp",
        )

        wide = np.linspace(-0.8, 0.4, 300)
        ax_variance.plot(
            wide, fit.parameters.total_variance(wide),
            color=colour, linewidth=1.8, label=f"{days}d",
        )

    # Mark where quotes actually exist. Without this the crossings on the right
    # of the variance panel look like a calibration failure; with it they are
    # visibly confined to the extrapolated region, which is the real finding.
    quoted_low = float(chain["log_moneyness"].min())
    quoted_high = float(chain["log_moneyness"].max())
    ax_variance.axvspan(
        quoted_low, quoted_high, color="grey", alpha=0.10, label="quoted range"
    )

    ax_fit.set_title(
        "SVI fitted slice by slice\n(points: market quotes, lines: 5-parameter fit)"
    )
    ax_fit.set_xlabel("Log-moneyness  $k = \\ln(K/F)$")
    ax_fit.set_ylabel("Implied volatility (%)")
    ax_fit.grid(alpha=0.3)
    ax_fit.legend(fontsize=8, title="expiry")

    ax_variance.set_title(
        "Fitted total variance $w(k) = \\sigma^2 T$\n"
        "(crossing = calendar arbitrage; note it only happens off-data)"
    )
    ax_variance.set_xlabel("Log-moneyness  $k$")
    ax_variance.set_ylabel("Total variance  $w$")
    ax_variance.grid(alpha=0.3)
    ax_variance.legend(fontsize=8, title="expiry")

    fig.suptitle(
        f"{snapshot.ticker} SVI calibration, {snapshot.as_of:%Y-%m-%d}  "
        f"(spot {snapshot.spot:.2f}, {len(chain)} quotes, "
        f"{5 * len(svi_fits)} parameters total)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_ml_vs_svi(
    snapshot: MarketSnapshot,
    chain: pd.DataFrame,
    svi_fits: dict[float, SVIFitResult],
    model: VolatilitySurfaceModel,
    output_path: Path,
) -> None:
    """Overlay both models on every smile, with residuals underneath.

    The residual row is the informative one. Both curves look identical at the
    scale of the smile itself, and only the residuals show where each model is
    systematically wrong — SVI in the wings it cannot bend enough to reach, the
    network wherever its shared parameters must compromise between expiries.

    Args:
        snapshot: The market snapshot, used for the title.
        chain: Cleaned chain with an ``iv`` column.
        svi_fits: Fitted SVI slices.
        model: The trained network.
        output_path: File to write the figure to.
    """
    expiries = sorted(svi_fits)
    n_columns = len(expiries)
    fig, axes = plt.subplots(
        2, n_columns, figsize=(3.5 * n_columns, 7.5), sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    for column, expiry in enumerate(expiries):
        rows = chain[chain["time_to_expiry"] == expiry].sort_values("log_moneyness")
        forward = float(rows["forward"].iloc[0])
        days = int(round(expiry * 365))

        grid_k = np.linspace(rows["log_moneyness"].min(), rows["log_moneyness"].max(), 200)
        grid_strikes = forward * np.exp(grid_k)
        svi_curve = svi_fits[expiry].parameters.implied_volatility(grid_k)
        ml_curve = _predict_ml(model, grid_strikes, expiry, forward)

        ax = axes[0, column]
        ax.scatter(
            rows["log_moneyness"], rows["iv"] * 100,
            s=9, color=MARKET_COLOUR, alpha=0.5, edgecolors="none", label="market",
        )
        ax.plot(grid_k, svi_curve * 100, color=SVI_COLOUR, linewidth=1.8, label="SVI")
        ax.plot(
            grid_k, ml_curve * 100, color=ML_COLOUR, linewidth=1.8,
            linestyle="--", label="network",
        )
        ax.set_title(f"{days} days", fontsize=11)
        ax.grid(alpha=0.3)
        if column == 0:
            ax.set_ylabel("Implied volatility (%)")
            ax.legend(fontsize=8)

        svi_residual = (
            svi_fits[expiry].parameters.implied_volatility(rows["log_moneyness"].values)
            - rows["iv"].values
        )
        ml_residual = (
            _predict_ml(model, rows["strike"].values, expiry, forward) - rows["iv"].values
        )

        ax_residual = axes[1, column]
        ax_residual.axhline(0.0, color="black", linewidth=0.8)
        ax_residual.scatter(
            rows["log_moneyness"], svi_residual * 100,
            s=8, color=SVI_COLOUR, alpha=0.6, edgecolors="none", label="SVI",
        )
        ax_residual.scatter(
            rows["log_moneyness"], ml_residual * 100,
            s=8, color=ML_COLOUR, alpha=0.6, edgecolors="none", label="network",
        )
        ax_residual.grid(alpha=0.3)
        ax_residual.set_xlabel("$k = \\ln(K/F)$")
        if column == 0:
            ax_residual.set_ylabel("Model $-$ market (vol pts)")
            ax_residual.legend(fontsize=8)

    fig.suptitle(
        f"{snapshot.ticker} {snapshot.as_of:%Y-%m-%d}: five parameters per slice "
        f"against one network for the whole surface",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_surfaces(
    snapshot: MarketSnapshot,
    chain: pd.DataFrame,
    svi_fits: dict[float, SVIFitResult],
    model: VolatilitySurfaceModel,
    output_path: Path,
) -> None:
    """Render both fitted surfaces on a dense grid, plus their difference.

    The difference panel answers the question the smile plots cannot: the two
    models agree closely where quotes are dense and diverge where they are not,
    which is a direct picture of how much of each surface is data and how much
    is the functional form talking.

    Args:
        snapshot: The market snapshot, used for the title.
        chain: Cleaned chain, used for the quote scatter and grid extent.
        svi_fits: Fitted SVI slices.
        model: The trained network.
        output_path: File to write the figure to.
    """
    expiries = np.array(sorted(svi_fits))
    k_grid = np.linspace(chain["log_moneyness"].min(), chain["log_moneyness"].max(), 120)

    svi_surface = np.empty((expiries.size, k_grid.size))
    ml_surface = np.empty_like(svi_surface)
    for row, expiry in enumerate(expiries):
        forward = float(chain[chain["time_to_expiry"] == expiry]["forward"].iloc[0])
        svi_surface[row] = svi_fits[expiry].parameters.implied_volatility(k_grid)
        ml_surface[row] = _predict_ml(model, forward * np.exp(k_grid), expiry, forward)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    extent = [k_grid.min(), k_grid.max(), 0, expiries.size]
    shared = {"aspect": "auto", "origin": "lower", "extent": extent}

    # One colour scale across both panels. Letting each autoscale would render
    # the two surfaces in different colours for the same volatility, which is
    # exactly the comparison these panels exist to support.
    low = min(svi_surface.min(), ml_surface.min()) * 100
    high = max(svi_surface.max(), ml_surface.max()) * 100
    levels = np.linspace(low, high, 21)

    for ax, surface, title in (
        (axes[0], svi_surface, "SVI  (30 parameters, 6 independent slices)"),
        (axes[1], ml_surface, "Network  (4.5k parameters, one joint fit)"),
    ):
        image = ax.imshow(surface * 100, cmap="viridis", vmin=low, vmax=high, **shared)
        ax.contour(
            surface * 100, levels=levels, colors="white", linewidths=0.4, alpha=0.5,
            extent=extent, origin="lower",
        )
        ax.set_title(title, fontsize=11)
        fig.colorbar(image, ax=ax, label="implied vol (%)")

    difference = (ml_surface - svi_surface) * 100
    limit = float(np.abs(difference).max())
    image = axes[2].imshow(
        difference, cmap="RdBu_r", vmin=-limit, vmax=limit, **shared
    )
    axes[2].set_title("Network $-$ SVI\n(largest where quotes are sparsest)", fontsize=11)
    fig.colorbar(image, ax=axes[2], label="difference (vol pts)")

    # Show where the quotes actually are, so the divergence can be read against
    # data density rather than guessed at.
    expiry_to_row = {expiry: index + 0.5 for index, expiry in enumerate(expiries)}
    for ax in axes:
        ax.scatter(
            chain["log_moneyness"],
            chain["time_to_expiry"].map(expiry_to_row),
            s=3, color="black", alpha=0.35, edgecolors="none",
        )
        ax.set_yticks(np.arange(expiries.size) + 0.5)
        ax.set_yticklabels([f"{int(round(t * 365))}d" for t in expiries])
        ax.set_xlabel("Log-moneyness  $k$")
    axes[0].set_ylabel("Expiry")

    fig.suptitle(
        f"{snapshot.ticker} fitted volatility surfaces, {snapshot.as_of:%Y-%m-%d} "
        "(black dots: actual quotes)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_smoothness(
    chain: pd.DataFrame, output_path: Path
) -> dict[str, tuple[float, float]]:
    """Compare activations on fit quality and on curvature.

    This is the figure that justifies the architecture. The left panel shows all
    three activations fitting the same slice almost identically; the right panel
    shows their second derivatives, where ReLU's piecewise-linear structure turns
    into spikes. Since the risk-neutral density is the second derivative of price
    in strike, the right panel is the one that decides whether the surface is
    usable — and it is invisible in every fit-quality metric.

    Args:
        chain: Cleaned chain with an ``iv`` column.
        output_path: File to write the figure to.

    Returns:
        Mapping from activation name to ``(validation_rmse, peak_curvature)``.
    """
    features = build_features(
        chain["strike"].values, chain["time_to_expiry"].values, chain["moneyness"].values
    )
    targets = chain["iv"].values

    # The busiest slice, so the comparison is made where the data is densest and
    # any wiggling is the model's doing rather than the data's.
    expiry = chain["time_to_expiry"].value_counts().idxmax()
    rows = chain[chain["time_to_expiry"] == expiry]
    forward = float(rows["forward"].iloc[0])
    days = int(round(expiry * 365))

    k_grid = np.linspace(rows["log_moneyness"].min(), rows["log_moneyness"].max(), 500)
    strikes = forward * np.exp(k_grid)
    step = k_grid[1] - k_grid[0]

    fig, (ax_fit, ax_curvature) = plt.subplots(1, 2, figsize=(15.5, 5.5))
    ax_fit.scatter(
        rows["log_moneyness"], rows["iv"] * 100,
        s=10, color=MARKET_COLOUR, alpha=0.45, edgecolors="none", label="market",
    )

    summary: dict[str, tuple[float, float]] = {}
    styles = {
        Activation.TANH: ("#1f77b4", "-"),
        Activation.RELU: ("#d62728", "-"),
        Activation.SILU: ("#2ca02c", "--"),
    }

    for activation, (colour, linestyle) in styles.items():
        model, history = train_surface_model(
            features, targets, TrainingConfig(activation=activation)
        )
        curve = _predict_ml(model, strikes, expiry, forward)
        curvature = np.diff(curve, 2) / step**2
        rmse = history.best_validation_rmse()
        peak = float(np.max(np.abs(curvature)))
        summary[activation.value] = (rmse, peak)

        ax_fit.plot(
            k_grid, curve * 100, color=colour, linestyle=linestyle, linewidth=1.6,
            label=f"{activation.value}  (val rmse {rmse * 100:.2f}vp)",
        )
        ax_curvature.plot(
            k_grid[1:-1], curvature, color=colour, linestyle=linestyle, linewidth=1.3,
            label=f"{activation.value}  (peak {peak:.0f})",
        )

    ax_fit.set_title(
        f"All three fit the {days}-day smile about equally well\n"
        "(RMSE cannot tell them apart)"
    )
    ax_fit.set_xlabel("Log-moneyness  $k$")
    ax_fit.set_ylabel("Implied volatility (%)")
    ax_fit.grid(alpha=0.3)
    ax_fit.legend(fontsize=8)

    ax_curvature.axhline(0.0, color="black", linewidth=0.8)
    ax_curvature.set_title(
        "Their second derivatives are not comparable at all\n"
        "$d^2\\sigma/dk^2$ — and the density depends on this"
    )
    ax_curvature.set_xlabel("Log-moneyness  $k$")
    ax_curvature.set_ylabel("$d^2\\sigma/dk^2$")
    ax_curvature.grid(alpha=0.3)
    ax_curvature.legend(fontsize=8)

    fig.suptitle(
        "Why the activation must be smooth: ReLU's kinks are invisible in the fit "
        "and glaring in the curvature",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_extrapolation(
    chain: pd.DataFrame, output_path: Path
) -> tuple[object, float, float]:
    """Train both models on the central band only, then show them in the wings.

    A random train/validation split flatters both models, because every held-out
    quote has training quotes on either side. Withholding the wings entirely is
    the test that matters for the use case people actually reach for a fitted
    surface for: pricing a strike nobody trades.

    Args:
        chain: Cleaned chain with an ``iv`` column.
        output_path: File to write the figure to.

    Returns:
        A ``(network_report, svi_interior_rmse, svi_exterior_rmse)`` tuple.
    """
    features = build_features(
        chain["strike"].values, chain["time_to_expiry"].values, chain["moneyness"].values
    )
    targets = chain["iv"].values
    k = chain["log_moneyness"].values

    report = extrapolation_report(features, targets, k)
    low, high = report.band
    interior = (k >= low) & (k <= high)

    # Retrain the network on the interior so the plotted curve is the same model
    # the report scored.
    model, _ = train_surface_model(features[interior], targets[interior])

    # SVI on the same restricted data, slice by slice.
    svi_interior_errors, svi_exterior_errors = [], []
    band_fits: dict[float, SVIFitResult] = {}
    for expiry in sorted(chain["time_to_expiry"].unique()):
        rows = chain["time_to_expiry"].values == expiry
        train_rows = rows & interior
        test_rows = rows & ~interior
        if train_rows.sum() < 5 or test_rows.sum() == 0:
            continue
        fit = fit_svi_slice(k[train_rows], targets[train_rows], float(expiry))
        band_fits[float(expiry)] = fit
        svi_interior_errors.append(
            fit.parameters.implied_volatility(k[train_rows]) - targets[train_rows]
        )
        svi_exterior_errors.append(
            fit.parameters.implied_volatility(k[test_rows]) - targets[test_rows]
        )

    svi_interior_rmse = float(np.sqrt(np.mean(np.concatenate(svi_interior_errors) ** 2)))
    svi_exterior_rmse = float(np.sqrt(np.mean(np.concatenate(svi_exterior_errors) ** 2)))

    # Plot the busiest expiry that took part in the band fit.
    expiry = max(band_fits, key=lambda t: (chain["time_to_expiry"] == t).sum())
    rows = chain[chain["time_to_expiry"] == expiry].sort_values("log_moneyness")
    forward = float(rows["forward"].iloc[0])
    days = int(round(expiry * 365))

    grid_k = np.linspace(rows["log_moneyness"].min(), rows["log_moneyness"].max(), 300)
    grid_strikes = forward * np.exp(grid_k)

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(15.5, 5.5), gridspec_kw={"width_ratios": [1.7, 1]}
    )

    in_band = (rows["log_moneyness"] >= low) & (rows["log_moneyness"] <= high)
    ax_curve.scatter(
        rows.loc[in_band, "log_moneyness"], rows.loc[in_band, "iv"] * 100,
        s=14, color=MARKET_COLOUR, alpha=0.6, edgecolors="none", label="training quotes",
    )
    ax_curve.scatter(
        rows.loc[~in_band, "log_moneyness"], rows.loc[~in_band, "iv"] * 100,
        s=16, facecolors="none", edgecolors="crimson", linewidths=0.9,
        label="withheld quotes",
    )
    ax_curve.plot(
        grid_k, band_fits[expiry].parameters.implied_volatility(grid_k) * 100,
        color=SVI_COLOUR, linewidth=1.8, label="SVI",
    )
    ax_curve.plot(
        grid_k, _predict_ml(model, grid_strikes, expiry, forward) * 100,
        color=ML_COLOUR, linewidth=1.8, linestyle="--", label="network",
    )
    ax_curve.axvspan(low, high, color="grey", alpha=0.12)
    ax_curve.axvline(low, color="grey", linestyle=":", linewidth=1.2)
    ax_curve.axvline(high, color="grey", linestyle=":", linewidth=1.2)
    ax_curve.set_title(
        f"{days}-day smile: both models trained only on the shaded band\n"
        "(outside it, neither is using data)"
    )
    ax_curve.set_xlabel("Log-moneyness  $k$")
    ax_curve.set_ylabel("Implied volatility (%)")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(fontsize=8)

    positions = np.arange(2)
    width = 0.36
    ax_bar.bar(
        positions - width / 2,
        [svi_interior_rmse * 100, svi_exterior_rmse * 100],
        width, color=SVI_COLOUR, label="SVI",
    )
    ax_bar.bar(
        positions + width / 2,
        [report.interior_rmse * 100, report.exterior_rmse * 100],
        width, color=ML_COLOUR, label="network",
    )
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(["inside band\n(interpolation)", "in the wings\n(extrapolation)"])
    ax_bar.set_ylabel("RMSE (vol points)")
    ax_bar.set_title(
        f"Both degrade by roughly 20x\n"
        f"(network {report.degradation_factor:.0f}x, "
        f"SVI {svi_exterior_rmse / svi_interior_rmse:.0f}x)"
    )
    ax_bar.grid(alpha=0.3, axis="y")
    ax_bar.legend(fontsize=9)

    fig.suptitle(
        "Interpolation is easy, extrapolation is not — for either model", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return report, svi_interior_rmse, svi_exterior_rmse


def main() -> None:
    """Fit both models, write every Phase 5 figure, and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="pull a fresh chain from yfinance"
    )
    arguments = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot, chain = load_data(arguments.fetch)
    print(f"{snapshot!r}")
    print(f"usable implied vols: {len(chain)} across {chain['expiry'].nunique()} expiries\n")

    svi_fits, model, history = fit_both_models(chain)

    print("SVI slices")
    print(
        f"{'expiry':>8} {'quotes':>7} {'rmse':>8} {'atm vol':>8} {'atm skew':>9} "
        f"{'no butterfly':>13} {'   extrapolated':>15}"
    )
    for expiry, fit in sorted(svi_fits.items()):
        p = fit.parameters
        print(
            f"{int(round(expiry * 365)):>7}d {fit.n_quotes:>7} "
            f"{fit.rmse * 100:>7.3f}vp {p.atm_volatility():>8.4f} {p.atm_skew():>9.4f} "
            f"{str(fit.butterfly_free):>13} {str(fit.butterfly_free_extrapolated):>15}"
        )

    # Calendar check, run twice on purpose. Slices are fitted independently, so
    # nothing prevents them crossing; the interesting question is *where*.
    print("\nCalendar-spread check (min of w_long - w_short; negative = arbitrage)")
    for expiry, longer in zip(sorted(svi_fits), sorted(svi_fits)[1:]):
        rows_short = chain[chain["time_to_expiry"] == expiry]
        rows_long = chain[chain["time_to_expiry"] == longer]
        overlap_low = max(rows_short["log_moneyness"].min(), rows_long["log_moneyness"].min())
        overlap_high = min(rows_short["log_moneyness"].max(), rows_long["log_moneyness"].max())
        pair = {expiry: svi_fits[expiry], longer: svi_fits[longer]}
        print(
            f"  {int(round(expiry * 365)):>4}d vs {int(round(longer * 365)):>4}d  "
            f"quoted overlap {worst_calendar_gap(pair, overlap_low, overlap_high):+.5f}   "
            f"extrapolated {worst_calendar_gap(pair, -0.8, 0.4):+.5f}"
        )

    svi_weighted = np.sqrt(
        sum(fit.rmse**2 * fit.n_quotes for fit in svi_fits.values())
        / sum(fit.n_quotes for fit in svi_fits.values())
    )
    n_parameters = sum(p.numel() for p in model.net.parameters())
    print(
        f"\nSVI     : {svi_weighted * 100:.3f} vp in-sample over all quotes, "
        f"{5 * len(svi_fits)} parameters"
    )
    print(
        f"network : {history.best_validation_rmse() * 100:.3f} vp held-out, "
        f"{n_parameters} parameters, best epoch {history.best_epoch}"
    )

    plot_svi_fit(snapshot, chain, svi_fits, PLOT_DIR / "phase5_svi_fit.png")
    plot_ml_vs_svi(snapshot, chain, svi_fits, model, PLOT_DIR / "phase5_ml_vs_svi.png")
    plot_surfaces(snapshot, chain, svi_fits, model, PLOT_DIR / "phase5_surface.png")

    print("\nActivation comparison (fit quality vs curvature)")
    smoothness = plot_smoothness(chain, PLOT_DIR / "phase5_smoothness.png")
    for name, (rmse, peak) in smoothness.items():
        print(f"  {name:>5}: val rmse {rmse * 100:.3f}vp   peak |d2sigma/dk2| {peak:8.1f}")

    print("\nInterpolation vs extrapolation")
    report, svi_interior, svi_exterior = plot_extrapolation(
        chain, PLOT_DIR / "phase5_extrapolation.png"
    )
    print(f"  band: k in [{report.band[0]:+.3f}, {report.band[1]:+.3f}]")
    print(
        f"  network: {report.interior_rmse * 100:.3f}vp inside -> "
        f"{report.exterior_rmse * 100:.3f}vp outside  ({report.degradation_factor:.1f}x)"
    )
    print(
        f"  SVI    : {svi_interior * 100:.3f}vp inside -> "
        f"{svi_exterior * 100:.3f}vp outside  ({svi_exterior / svi_interior:.1f}x)"
    )

    print(f"\nfigures written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
