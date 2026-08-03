r"""Fetching and cleaning real option chain data.

===============================================================================
WHY THIS MODULE IS LONGER THAN THE PRICERS
===============================================================================

Phases 1-3 assume clean inputs. Real option chains are not clean, and the gap
between "yfinance returned a DataFrame" and "these are prices you can calibrate
to" is where most of the actual work in derivatives data sits. Nearly every
surprising result in an options project traces back to a data problem rather than
a modelling one, so the filters here are documented individually with the failure
each one prevents.

-------------------------------------------------------------------------------
THE PROBLEMS WITH RAW CHAIN DATA
-------------------------------------------------------------------------------

**1. `lastPrice` is a trap.** It is the last price at which the contract traded,
which for an illiquid strike may have been hours or days ago, at a completely
different spot level. In the SPY snapshot used to build this project, a 630-strike
call showed ``lastPrice = 107.38`` against a bid/ask of 124.62/127.43 — stale by
seventeen dollars. Using it would imply a nonsensically low volatility, or fall
below intrinsic value and admit no implied vol at all. **Always use the mid of the
current bid and ask**, which reflects where the market is *now*.

**2. Zero bids mean the option is uninvestable.** A zero bid says no one will buy
at any positive price. The mid is then meaningless (it is half the ask), and the
implied vol computed from it is fiction.

**3. Wide spreads mean the "price" is a guess.** A 0.05/0.60 market has a mid of
0.325, but the true value could be anywhere in that range. Since the mid carries a
+/-85% uncertainty, so does any vol derived from it. Filtering on *relative* spread
removes these.

**4. Deep in-the-money options are mostly intrinsic value.** A call struck at 400
with spot at 756 is worth ~356, of which perhaps 0.20 is time value. Implied
volatility is recovered from that 0.20 alone, so a one-cent quoting error moves the
implied vol enormously. This is not solver weakness — vega is genuinely tiny there,
so the inverse problem is ill-conditioned. **The standard fix is to use only
out-of-the-money options**, which is what desks quote and what this module keeps by
default. Put-call parity means nothing is lost: the OTM put and the ITM call at the
same strike carry identical information, and the OTM one carries it better.

**5. Very short expiries misbehave.** Options with days to go have tiny vega, are
dominated by pinning and gamma effects, and often trade at a tick regardless of
theory. A minimum maturity filter removes a large source of noise.

-------------------------------------------------------------------------------
THE RATE AND DIVIDEND INPUTS
-------------------------------------------------------------------------------

Both are needed and both are easy to get subtly wrong.

**Risk-free rate.** Taken from ``^IRX``, the 13-week Treasury bill yield, quoted in
percent (so 3.70 means 3.70%). Two caveats worth knowing: bill yields are quoted on
a discount basis rather than as continuously compounded rates, and a 13-week rate is
not the right discount rate for a two-year option. Both effects are small at current
levels — a few basis points — relative to the bid-ask noise in the option quotes, so
a single flat rate is used and the simplification is stated rather than hidden. A
production system would bootstrap a proper curve and interpolate to each expiry.

**Dividend yield.** Computed from the *actual* trailing twelve months of dividends
divided by spot, not from ``yfinance``'s ``info`` dictionary. That field is a units
trap: for SPY, ``info["dividendYield"]`` returns ``1.01`` (a percentage) while
``info["yield"]`` returns ``0.0101`` (a decimal), and picking the wrong one is a
hundred-fold error in an input that directly shifts the forward. Summing the actual
dividend payments is unambiguous and checkable.

For SPY the yield is around 1%, which sounds ignorable but is not: it shifts the
forward by roughly 1% per year of maturity, which is comparable to the entire
bid-ask spread on a liquid option and would visibly tilt the smile.

-------------------------------------------------------------------------------
CACHING
-------------------------------------------------------------------------------

Every fetch is written to disk as CSV, and analysis reads from that snapshot. This
is not an optimisation:

* **Reproducibility.** Option chains change every second. Without a snapshot the
  figures in this repository could never be regenerated, and no result would be
  checkable.
* **Tests must not need the network.** A test suite that fails when an API rate
  limits, or when run on a plane, is not a test suite. The Phase 4 tests run
  entirely against a committed fixture; the one test that does hit the network is
  marked and skipped by default.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "MarketSnapshot",
    "fetch_option_chain",
    "clean_option_chain",
    "load_snapshot",
    "save_snapshot",
    "DEFAULT_CACHE_DIR",
    "DAYS_PER_YEAR",
]

DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parents[2] / "data_cache"

# Calendar days, matching the theta convention in greeks.py. An option expiring in
# 30 days decays over 30 calendar days whether or not the market is open.
DAYS_PER_YEAR: float = 365.0


@dataclass(frozen=True)
class MarketSnapshot:
    """A point-in-time option chain together with the inputs needed to price it.

    Bundling the chain with its spot, rate, yield and timestamp is deliberate: an
    option chain divorced from the spot price it was observed against is not
    analysable, and mismatching them by even a few minutes visibly distorts a
    smile. Keeping them in one immutable object makes that mistake hard to make.

    Attributes:
        ticker: Underlying symbol.
        spot: Underlying price at the time of the snapshot.
        rate: Continuously compounded risk-free rate as a decimal.
        dividend_yield: Continuous dividend yield as a decimal.
        as_of: Timestamp of the snapshot.
        chain: One row per contract. See :func:`clean_option_chain` for columns.
    """

    ticker: str
    spot: float
    rate: float
    dividend_yield: float
    as_of: dt.datetime
    chain: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"MarketSnapshot(ticker={self.ticker!r}, spot={self.spot:.2f}, "
            f"rate={self.rate:.4f}, dividend_yield={self.dividend_yield:.4f}, "
            f"as_of={self.as_of:%Y-%m-%d %H:%M}, contracts={len(self.chain)})"
        )

    def expiries(self) -> list[float]:
        """Return the distinct times to expiry present in the chain, ascending.

        Returns:
            Sorted list of times to expiry in years.
        """
        return sorted(self.chain["time_to_expiry"].unique().tolist())


def _fetch_risk_free_rate(fallback: float = 0.04) -> float:
    """Fetch the 13-week Treasury bill yield as a decimal.

    Args:
        fallback: Rate to use if the fetch fails, so a data outage degrades to a
            reasonable constant rather than an exception.

    Returns:
        The risk-free rate as a decimal (0.037 for 3.7%).
    """
    import yfinance as yf

    try:
        history = yf.Ticker("^IRX").history(period="5d")
        if len(history) == 0:
            return fallback
        # ^IRX is quoted in percent: 3.70 means 3.70%.
        return float(history["Close"].iloc[-1]) / 100.0
    except Exception:
        return fallback


def _fetch_dividend_yield(ticker: str, spot: float, fallback: float = 0.0) -> float:
    """Compute a trailing twelve-month dividend yield from actual payments.

    Deliberately avoids ``yfinance``'s ``info`` fields, which disagree with each
    other on units (``dividendYield`` is a percentage, ``yield`` is a decimal).
    Summing observed payments is unambiguous.

    Args:
        ticker: Underlying symbol.
        spot: Current underlying price, used as the denominator.
        fallback: Yield to use if no dividend history is available.

    Returns:
        The continuous dividend yield as a decimal.
    """
    import yfinance as yf

    try:
        dividends = yf.Ticker(ticker).dividends
        if len(dividends) == 0:
            return fallback
        cutoff = dividends.index.max() - pd.Timedelta(days=365)
        trailing = dividends[dividends.index > cutoff].sum()
        return float(trailing) / spot
    except Exception:
        return fallback


# Target horizons for expiry selection, in calendar days: roughly one week, two
# weeks, one month, two months, a quarter, half a year, and a year. Chosen to be
# spread across the term structure rather than clustered, because SPY lists daily
# expiries and simply taking "the next eight" would sample only the front month —
# giving a detailed smile but no term structure at all.
DEFAULT_TARGET_DAYS: tuple[int, ...] = (7, 14, 30, 60, 91, 182, 365)


def fetch_option_chain(
    ticker: str = "SPY",
    target_days: tuple[int, ...] = DEFAULT_TARGET_DAYS,
    min_days_to_expiry: int = 5,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> MarketSnapshot:
    """Download a live option chain and its pricing inputs.

    Requires network access. The result is cached to ``cache_dir`` so that analysis
    and figures are reproducible from a fixed snapshot; see the module docstring.

    Expiries are chosen as the listed dates nearest each entry in ``target_days``,
    rather than the first N available. On a name like SPY with daily expiries, the
    naive approach returns eight dates inside a single month — plenty of strikes but
    no term structure, which is half of what a volatility surface is.

    Args:
        ticker: Underlying symbol. SPY is a good default — it is the most liquid
            option market in the world, so quotes are tight and the smile is clean.
        target_days: Approximate horizons to sample, in calendar days.
        min_days_to_expiry: Skip expiries closer than this. Very short-dated options
            have negligible vega and are dominated by pinning effects.
        cache_dir: Directory to write the snapshot to, or ``None`` to skip caching.

    Returns:
        A :class:`MarketSnapshot` with the *raw* chain. Call
        :func:`clean_option_chain` before computing implied volatilities.

    Raises:
        RuntimeError: If the underlying price or the option chain cannot be
            retrieved.
    """
    import yfinance as yf

    underlying = yf.Ticker(ticker)

    history = underlying.history(period="1d")
    if len(history) == 0:
        raise RuntimeError(f"could not fetch a spot price for {ticker!r}")
    spot = float(history["Close"].iloc[-1])

    available = underlying.options
    if not available:
        raise RuntimeError(f"no option expiries available for {ticker!r}")

    as_of = dt.datetime.now()
    today = as_of.date()

    eligible = {
        expiry: (dt.date.fromisoformat(expiry) - today).days for expiry in available
    }
    eligible = {e: d for e, d in eligible.items() if d >= min_days_to_expiry}

    if not eligible:
        raise RuntimeError(
            f"no expiries at least {min_days_to_expiry} days out for {ticker!r}"
        )

    # For each target horizon take the closest listed expiry, de-duplicating: two
    # nearby targets may resolve to the same date when listings are sparse.
    selected_set = {
        min(eligible, key=lambda expiry: abs(eligible[expiry] - target))
        for target in target_days
    }
    selected = sorted(selected_set, key=lambda expiry: eligible[expiry])

    frames: list[pd.DataFrame] = []
    for expiry in selected:
        chain = underlying.option_chain(expiry)
        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            frame = table.copy()
            frame["option_type"] = option_type
            frame["expiry"] = expiry
            frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)

    # Time to expiry in years. Expiries are dates, so this is measured to the start
    # of the expiry day; the resulting half-day error is immaterial next to the
    # bid-ask noise, but it is a real approximation and worth naming.
    expiry_dates = pd.to_datetime(raw["expiry"]).dt.date
    raw["days_to_expiry"] = [(e - today).days for e in expiry_dates]
    raw["time_to_expiry"] = raw["days_to_expiry"] / DAYS_PER_YEAR

    snapshot = MarketSnapshot(
        ticker=ticker,
        spot=spot,
        rate=_fetch_risk_free_rate(),
        dividend_yield=_fetch_dividend_yield(ticker, spot),
        as_of=as_of,
        chain=raw,
    )

    if cache_dir is not None:
        save_snapshot(snapshot, cache_dir)

    return snapshot


def clean_option_chain(
    snapshot: MarketSnapshot,
    max_relative_spread: float = 0.25,
    min_price: float = 0.05,
    min_open_interest: int = 10,
    otm_only: bool = True,
    min_time_to_expiry: float = 7.0 / DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Filter a raw chain down to quotes worth calibrating to.

    Each filter targets a specific failure mode documented in the module docstring.
    The returned frame carries a ``mid`` column built from bid and ask — never from
    ``lastPrice``, which is frequently stale by hours.

    Args:
        snapshot: The raw snapshot to clean.
        max_relative_spread: Drop quotes whose ``(ask - bid) / mid`` exceeds this.
            0.25 is permissive enough to keep the wings on a liquid name while
            removing quotes where the mid is essentially a guess.
        min_price: Drop quotes with a mid below this. Sub-nickel options are all
            quantisation noise: one tick is a large fraction of the price.
        min_open_interest: Drop contracts with less open interest than this, as a
            proxy for whether anyone actually trades them.
        otm_only: Keep only out-of-the-money contracts (calls above the forward,
            puts below). Strongly recommended — see the module docstring on why
            deep ITM options give ill-conditioned implied vols.
        min_time_to_expiry: Drop expiries closer than this, in years.

    Returns:
        A cleaned DataFrame with columns ``option_type``, ``strike``, ``expiry``,
        ``days_to_expiry``, ``time_to_expiry``, ``bid``, ``ask``, ``mid``,
        ``spread``, ``relative_spread``, ``volume``, ``open_interest``,
        ``moneyness``, ``log_moneyness``, and ``forward``.
    """
    frame = snapshot.chain.copy()

    # Normalise column names coming from yfinance.
    frame = frame.rename(columns={"openInterest": "open_interest"})
    for column in ("bid", "ask", "volume", "open_interest"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["volume"] = frame["volume"].fillna(0.0)
    frame["open_interest"] = frame["open_interest"].fillna(0.0)

    # --- Price construction: mid, never last ---
    frame = frame[(frame["bid"] > 0.0) & (frame["ask"] > 0.0)]
    frame = frame[frame["ask"] >= frame["bid"]]  # drop crossed quotes
    frame["mid"] = 0.5 * (frame["bid"] + frame["ask"])
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["relative_spread"] = frame["spread"] / frame["mid"]

    # --- Liquidity and quality filters ---
    frame = frame[frame["mid"] >= min_price]
    frame = frame[frame["relative_spread"] <= max_relative_spread]
    frame = frame[frame["open_interest"] >= min_open_interest]
    frame = frame[frame["time_to_expiry"] >= min_time_to_expiry]

    # --- Moneyness, measured against the forward rather than spot ---
    # The forward is the correct reference: an option is at the money when K = F,
    # not when K = S. Using spot would tilt the whole smile by the cost of carry.
    frame["forward"] = snapshot.spot * np.exp(
        (snapshot.rate - snapshot.dividend_yield) * frame["time_to_expiry"]
    )
    frame["moneyness"] = frame["strike"] / frame["forward"]
    frame["log_moneyness"] = np.log(frame["moneyness"])

    if otm_only:
        is_otm_call = (frame["option_type"] == "call") & (frame["strike"] >= frame["forward"])
        is_otm_put = (frame["option_type"] == "put") & (frame["strike"] < frame["forward"])
        frame = frame[is_otm_call | is_otm_put]

    columns = [
        "option_type", "strike", "expiry", "days_to_expiry", "time_to_expiry",
        "bid", "ask", "mid", "spread", "relative_spread", "volume", "open_interest",
        "forward", "moneyness", "log_moneyness",
    ]
    if "impliedVolatility" in frame.columns:
        # Keep the vendor's own implied vol as an independent cross-check.
        frame = frame.rename(columns={"impliedVolatility": "vendor_iv"})
        columns.append("vendor_iv")

    return frame[columns].sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)


def save_snapshot(snapshot: MarketSnapshot, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Write a snapshot to disk as CSV plus a metadata sidecar.

    CSV rather than parquet so the committed fixture is human-readable and diffs
    meaningfully in git — worth more than the space saving at this size.

    Args:
        snapshot: The snapshot to persist.
        cache_dir: Directory to write into. Created if absent.

    Returns:
        Path to the written chain CSV.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{snapshot.ticker}_{snapshot.as_of:%Y%m%d_%H%M}"

    chain_path = cache_dir / f"{stem}_chain.csv"
    snapshot.chain.to_csv(chain_path, index=False)

    metadata = pd.DataFrame(
        [{
            "ticker": snapshot.ticker,
            "spot": snapshot.spot,
            "rate": snapshot.rate,
            "dividend_yield": snapshot.dividend_yield,
            "as_of": snapshot.as_of.isoformat(),
        }]
    )
    metadata.to_csv(cache_dir / f"{stem}_meta.csv", index=False)
    return chain_path


def load_snapshot(chain_path: Path) -> MarketSnapshot:
    """Load a snapshot previously written by :func:`save_snapshot`.

    Args:
        chain_path: Path to the ``*_chain.csv`` file. The matching ``*_meta.csv``
            is inferred by replacing the suffix.

    Returns:
        The reconstructed :class:`MarketSnapshot`.

    Raises:
        FileNotFoundError: If either the chain or its metadata sidecar is missing.
    """
    chain_path = Path(chain_path)
    meta_path = chain_path.with_name(chain_path.name.replace("_chain.csv", "_meta.csv"))
    if not chain_path.exists():
        raise FileNotFoundError(f"chain file not found: {chain_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata sidecar not found: {meta_path}")

    metadata = pd.read_csv(meta_path).iloc[0]
    return MarketSnapshot(
        ticker=str(metadata["ticker"]),
        spot=float(metadata["spot"]),
        rate=float(metadata["rate"]),
        dividend_yield=float(metadata["dividend_yield"]),
        as_of=dt.datetime.fromisoformat(str(metadata["as_of"])),
        chain=pd.read_csv(chain_path),
    )


def latest_cached_snapshot(
    ticker: str = "SPY", cache_dir: Path = DEFAULT_CACHE_DIR
) -> MarketSnapshot:
    """Load the most recent cached snapshot for a ticker.

    Args:
        ticker: Underlying symbol.
        cache_dir: Directory to search.

    Returns:
        The newest matching :class:`MarketSnapshot`.

    Raises:
        FileNotFoundError: If no snapshot exists for that ticker.
    """
    candidates = sorted(Path(cache_dir).glob(f"{ticker}_*_chain.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"no cached snapshot for {ticker!r} in {cache_dir}. "
            f"Run fetch_option_chain({ticker!r}) first (requires network access)."
        )
    return load_snapshot(candidates[-1])
