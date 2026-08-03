"""Tests for market data loading and cleaning.

**These tests do not touch the network.** They run against a committed snapshot of
a real SPY chain in ``fixtures/``, which keeps them deterministic and runnable
offline. A test suite that fails on a plane, or when a vendor rate-limits, is not
a test suite.

The one test that does hit the network is marked ``network`` and skipped by
default. Run it with ``pytest -m network`` when you want to confirm the live path
still works after a yfinance upgrade.

The fixture deliberately retains the *messy* rows — zero bids, wide spreads, thin
open interest — because those are precisely what the cleaning code exists to
remove. A fixture of only well-behaved quotes would test nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from options_engine.data.market_data import (
    DAYS_PER_YEAR,
    MarketSnapshot,
    clean_option_chain,
    load_snapshot,
    save_snapshot,
)
from options_engine.vol_surface.implied_vol import implied_volatility_array

FIXTURE = Path(__file__).parent / "fixtures" / "SPY_fixture_chain.csv"


@pytest.fixture(scope="module")
def snapshot() -> MarketSnapshot:
    """Load the committed SPY snapshot."""
    return load_snapshot(FIXTURE)


@pytest.fixture(scope="module")
def cleaned(snapshot: MarketSnapshot) -> pd.DataFrame:
    """The fixture chain after the standard cleaning pipeline."""
    return clean_option_chain(snapshot)


@pytest.fixture(scope="module")
def with_iv(snapshot: MarketSnapshot, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Cleaned chain with an implied volatility column attached."""
    implied = implied_volatility_array(
        cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
        cleaned["time_to_expiry"].values, snapshot.rate,
        cleaned["option_type"].values, snapshot.dividend_yield,
    )
    return cleaned.assign(iv=implied).dropna(subset=["iv"])


class TestSnapshotLoading:
    """Round-tripping snapshots through disk."""

    def test_fixture_loads_with_sane_market_inputs(self, snapshot: MarketSnapshot):
        """The committed fixture must have plausible spot, rate and yield."""
        assert snapshot.ticker == "SPY"
        assert 50.0 < snapshot.spot < 10_000.0
        assert -0.01 < snapshot.rate < 0.15
        assert 0.0 <= snapshot.dividend_yield < 0.10
        assert len(snapshot.chain) > 100

    def test_save_and_load_round_trip(self, snapshot: MarketSnapshot, tmp_path: Path):
        """Persisting and reloading must preserve every pricing input.

        The metadata sidecar matters as much as the chain: a chain divorced from
        the spot it was observed against is not analysable.
        """
        path = save_snapshot(snapshot, tmp_path)
        reloaded = load_snapshot(path)
        assert reloaded.ticker == snapshot.ticker
        assert reloaded.spot == pytest.approx(snapshot.spot)
        assert reloaded.rate == pytest.approx(snapshot.rate)
        assert reloaded.dividend_yield == pytest.approx(snapshot.dividend_yield)
        assert reloaded.as_of == snapshot.as_of
        assert len(reloaded.chain) == len(snapshot.chain)

    def test_missing_metadata_is_reported_clearly(self, tmp_path: Path):
        """A chain without its sidecar is unusable and must say so."""
        orphan = tmp_path / "SPY_orphan_chain.csv"
        orphan.write_text("strike,bid,ask\n100,1,2\n")
        with pytest.raises(FileNotFoundError, match="metadata sidecar not found"):
            load_snapshot(orphan)

    def test_missing_chain_is_reported_clearly(self, tmp_path: Path):
        """A path that does not exist should fail before anything else."""
        with pytest.raises(FileNotFoundError, match="chain file not found"):
            load_snapshot(tmp_path / "nope_chain.csv")


class TestCleaning:
    """Each filter, and the specific failure it prevents."""

    def test_cleaning_removes_a_substantial_fraction(self, snapshot, cleaned):
        """Real chains are mostly unusable; keeping everything would be the bug.

        Roughly half of a raw SPY chain fails at least one quality filter. If this
        ratio ever approached 100% the filters would have silently stopped working.
        """
        retained = len(cleaned) / len(snapshot.chain)
        assert 0.1 < retained < 0.9, f"retained {retained:.1%}, filters look wrong"

    def test_no_zero_or_crossed_quotes_survive(self, cleaned):
        """A zero bid means nobody will buy at any price; the mid is then fiction."""
        assert (cleaned["bid"] > 0).all()
        assert (cleaned["ask"] >= cleaned["bid"]).all()

    def test_mid_is_used_rather_than_last_price(self, cleaned):
        """The output must carry a bid/ask mid, and no lastPrice column at all.

        ``lastPrice`` is the single most dangerous field in an option chain: it can
        be hours stale, from a different spot regime. In the snapshot this fixture
        came from, a 630-strike call showed lastPrice 107.38 against a 124.62/127.43
        market — stale by seventeen dollars, which would imply a nonsensical vol.
        The clean frame excludes it entirely so it cannot be used by accident.
        """
        assert "mid" in cleaned.columns
        assert "lastPrice" not in cleaned.columns
        expected = 0.5 * (cleaned["bid"] + cleaned["ask"])
        assert np.allclose(cleaned["mid"], expected)

    def test_spread_filter_is_applied(self, cleaned):
        """Quotes whose mid is essentially a guess must be gone."""
        assert (cleaned["relative_spread"] <= 0.25 + 1e-12).all()

    def test_liquidity_and_price_floors_are_applied(self, cleaned):
        """Sub-nickel and untraded contracts carry no usable information."""
        assert (cleaned["mid"] >= 0.05 - 1e-12).all()
        assert (cleaned["open_interest"] >= 10).all()

    def test_only_out_of_the_money_options_are_kept(self, cleaned):
        """Deep ITM options are almost all intrinsic value, so their vol is noise.

        Put-call parity means nothing is lost by discarding them: the OTM option at
        the same strike carries the same information with far better conditioning.
        """
        calls = cleaned[cleaned["option_type"] == "call"]
        puts = cleaned[cleaned["option_type"] == "put"]
        assert (calls["strike"] >= calls["forward"]).all()
        assert (puts["strike"] < puts["forward"]).all()

    def test_moneyness_is_measured_against_the_forward(self, snapshot, cleaned):
        """An option is at the money when K = F, not when K = S.

        Using spot would tilt the entire smile by the cost of carry — for SPY at a
        3.7% rate and 1.2% yield that is ~2.5% per year of maturity, comparable to
        the whole bid-ask spread.
        """
        expected_forward = snapshot.spot * np.exp(
            (snapshot.rate - snapshot.dividend_yield) * cleaned["time_to_expiry"]
        )
        assert np.allclose(cleaned["forward"], expected_forward)
        assert np.allclose(cleaned["moneyness"], cleaned["strike"] / cleaned["forward"])
        assert np.allclose(cleaned["log_moneyness"], np.log(cleaned["moneyness"]))

    def test_time_to_expiry_matches_day_count(self, cleaned):
        """Years must be days / 365, consistent with the theta convention."""
        assert np.allclose(cleaned["time_to_expiry"], cleaned["days_to_expiry"] / DAYS_PER_YEAR)
        assert (cleaned["time_to_expiry"] > 0).all()

    def test_disabling_otm_filter_keeps_both_wings(self, snapshot):
        """The OTM restriction is a default, not a hard-coded assumption."""
        everything = clean_option_chain(snapshot, otm_only=False)
        otm_only = clean_option_chain(snapshot, otm_only=True)
        assert len(everything) > len(otm_only)

    def test_stricter_filters_keep_fewer_rows(self, snapshot):
        """Tightening any filter must be monotone in what survives."""
        loose = clean_option_chain(snapshot, max_relative_spread=0.5, min_open_interest=0)
        tight = clean_option_chain(snapshot, max_relative_spread=0.05, min_open_interest=500)
        assert len(tight) < len(loose)

    def test_output_columns_are_stable(self, cleaned):
        """Downstream code and figures depend on this schema."""
        required = {
            "option_type", "strike", "expiry", "days_to_expiry", "time_to_expiry",
            "bid", "ask", "mid", "spread", "relative_spread", "volume",
            "open_interest", "forward", "moneyness", "log_moneyness",
        }
        assert required.issubset(set(cleaned.columns))

    def test_rows_are_sorted_for_plotting(self, cleaned):
        """Sorted by expiry then strike, so smile curves plot without reordering."""
        assert cleaned.sort_values(["time_to_expiry", "strike"]).index.equals(cleaned.index)


class TestCleanedDataIsPriceable:
    """The point of cleaning: what survives must actually yield implied vols."""

    def test_every_cleaned_quote_produces_an_implied_vol(self, snapshot, cleaned):
        """Zero NaNs is the real test that the filters did their job.

        Each NaN would be a quote that survived cleaning but violates the
        no-arbitrage bounds — that is, a filter that failed to catch bad data.
        """
        implied = implied_volatility_array(
            cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
            cleaned["time_to_expiry"].values, snapshot.rate,
            cleaned["option_type"].values, snapshot.dividend_yield,
        )
        nan_rate = float(np.isnan(implied).mean())
        assert nan_rate == 0.0, f"{nan_rate:.1%} of cleaned quotes failed to invert"

    def test_implied_vols_are_in_a_plausible_range(self, snapshot, cleaned):
        """SPY vol should sit between a few percent and roughly 200%.

        A value outside that band means the rate, dividend yield or day count is
        wrong, not that the market has done something exotic.
        """
        implied = implied_volatility_array(
            cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
            cleaned["time_to_expiry"].values, snapshot.rate,
            cleaned["option_type"].values, snapshot.dividend_yield,
        )
        assert implied.min() > 0.02
        assert implied.max() < 2.0

    def test_implied_vols_reprice_the_market_mid(self, snapshot, cleaned):
        """Every solved vol must return the quote it came from.

        End-to-end confirmation that the data pipeline and the solver agree on
        units and conventions — a day-count or rate mismatch would show up here.
        """
        from options_engine.pricing.black_scholes import black_scholes_price

        implied = implied_volatility_array(
            cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
            cleaned["time_to_expiry"].values, snapshot.rate,
            cleaned["option_type"].values, snapshot.dividend_yield,
        )
        repriced = np.array([
            float(black_scholes_price(
                snapshot.spot, row.strike, row.time_to_expiry, snapshot.rate,
                sigma, row.option_type, snapshot.dividend_yield,
            ))
            for sigma, row in zip(implied, cleaned.itertuples())
        ])
        assert np.max(np.abs(repriced - cleaned["mid"].values)) < 1e-6

    def test_our_implied_vols_broadly_track_the_vendor_figure(self, snapshot, cleaned):
        """A sanity check against yfinance's own implied volatility column.

        Agreement is close but not exact, and the difference is *explained*: the
        vendor computes implied vol with zero rates and zero dividends. Recomputing
        under their assumptions collapses the median gap from ~0.8 vol points to
        ~0.14, which both validates this solver and shows why the vendor's number
        is the less correct of the two.
        """
        if "vendor_iv" not in cleaned.columns:
            pytest.skip("fixture has no vendor implied volatility column")

        ours = implied_volatility_array(
            cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
            cleaned["time_to_expiry"].values, snapshot.rate,
            cleaned["option_type"].values, snapshot.dividend_yield,
        )
        theirs = cleaned["vendor_iv"].values
        assert float(np.nanmedian(np.abs(ours - theirs))) < 0.05

        # Under the vendor's own (zero rate, zero dividend) assumptions we agree far
        # more closely — which localises the discrepancy to the inputs, not the math.
        matched = implied_volatility_array(
            cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
            cleaned["time_to_expiry"].values, 0.0,
            cleaned["option_type"].values, 0.0,
        )
        assert float(np.nanmedian(np.abs(matched - theirs))) < float(
            np.nanmedian(np.abs(ours - theirs))
        )


class TestSmileShape:
    """Structural facts about the surface that any correct pipeline must reproduce."""

    def test_equity_index_skew_is_downward_sloping(self, with_iv):
        """Low strikes imply higher volatility than high strikes.

        This is the defining feature of the post-1987 equity index smile and the
        single clearest refutation of constant volatility. If this test ever fails,
        either the data is broken or something extraordinary has happened.
        """
        for _, group in with_iv.groupby("days_to_expiry"):
            if len(group) < 8:
                continue
            low = group[group["moneyness"] < 0.95]["iv"]
            high = group[group["moneyness"] > 1.02]["iv"]
            if len(low) < 2 or len(high) < 2:
                continue
            assert low.mean() > high.mean(), "skew is not downward sloping"

    def test_skew_flattens_with_maturity(self, with_iv):
        """Short-dated smiles are steep; long-dated ones are much flatter.

        A square-root-of-time effect: cumulative returns aggregate towards
        normality, so the risk-neutral distribution's excess skewness decays roughly
        as 1/sqrt(T). Measured on this snapshot: about +15 vol points of skew at 11
        days, falling to under +2 points at 331 days.
        """
        slopes = {}
        for days, group in with_iv.groupby("days_to_expiry"):
            group = group[(group["moneyness"] > 0.85) & (group["moneyness"] < 1.05)]
            if len(group) < 8:
                continue
            slope = np.polyfit(group["log_moneyness"], group["iv"], 1)[0]
            slopes[days] = slope

        assert len(slopes) >= 2, "need at least two expiries to compare skew"
        ordered = [slopes[d] for d in sorted(slopes)]
        # Slopes are negative (downward skew); flattening means moving towards zero.
        assert abs(ordered[-1]) < abs(ordered[0])

    def test_implied_vol_is_not_constant_across_strikes(self, with_iv):
        """The headline Phase 4 result, stated as an assertion.

        Black-Scholes assumes one volatility for the whole underlying. If that held,
        every strike would imply the same number and this test would fail. The
        spread is tens of vol points.
        """
        for _, group in with_iv.groupby("days_to_expiry"):
            if len(group) < 8:
                continue
            assert group["iv"].max() - group["iv"].min() > 0.02


@pytest.mark.network
class TestLiveFetch:
    """Exercises the live yfinance path. Skipped unless ``-m network`` is passed."""

    def test_fetch_returns_a_usable_snapshot(self):
        """A live fetch must produce data that survives cleaning and inverts."""
        from options_engine.data.market_data import fetch_option_chain

        snapshot = fetch_option_chain("SPY", cache_dir=None)
        assert snapshot.spot > 0
        assert len(snapshot.chain) > 100

        cleaned = clean_option_chain(snapshot)
        assert len(cleaned) > 50

        implied = implied_volatility_array(
            cleaned["mid"].values, snapshot.spot, cleaned["strike"].values,
            cleaned["time_to_expiry"].values, snapshot.rate,
            cleaned["option_type"].values, snapshot.dividend_yield,
        )
        assert float(np.isnan(implied).mean()) < 0.02

    def test_expiries_span_the_term_structure(self):
        """Expiry selection must sample the curve, not just the front month.

        SPY lists daily expiries, so naively taking the first N dates returns a
        month of data and no term structure at all.
        """
        from options_engine.data.market_data import fetch_option_chain

        snapshot = fetch_option_chain("SPY", cache_dir=None)
        days = sorted(snapshot.chain["days_to_expiry"].unique())
        assert max(days) > 5 * min(days), f"expiries are clustered: {days}"
