#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from scan_episodic_pivots import BREAKOUT_ROOT, EpisodicPivotConfig
from scan_r3000_2025_episodic_pivots import (
    DATA_VENDOR,
    FEATURE_START,
    LANE_NAMES,
    OUT_DIR as EP_OUT_DIR,
    SIGNAL_END,
    SIGNAL_START,
    add_numba_features,
    load_daily,
    load_universe,
)


OUT_DIR = BREAKOUT_ROOT / "analysis_outputs" / "episodic_pivot_entry_rules_2025"


@dataclass(frozen=True)
class EntryRuleConfig:
    signal_start: str = SIGNAL_START
    signal_end: str = SIGNAL_END
    feature_start: str = FEATURE_START
    watch_days: int = 20
    max_hold_days: int = 60
    pullback_ma_buffer: float = 0.03
    min_price: float = 5.0
    min_avg_dollar_volume_20: float = 5_000_000.0
    min_market_cap: float = 300_000_000.0


@njit
def _compute_entry_features_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    dma10 = np.empty(n, dtype=np.float64)
    dma20 = np.empty(n, dtype=np.float64)
    avgdv20_prior = np.empty(n, dtype=np.float64)
    typical_price = np.empty(n, dtype=np.float64)
    dollar_volume = np.empty(n, dtype=np.float64)
    cum_tpv = np.empty(n, dtype=np.float64)
    cum_volume = np.empty(n, dtype=np.float64)
    for i in range(n):
        dma10[i] = np.nan
        dma20[i] = np.nan
        avgdv20_prior[i] = np.nan
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0
        dollar_volume[i] = close[i] * volume[i]

    start = 0
    while start < n:
        end = start + 1
        code = codes[start]
        while end < n and codes[end] == code:
            end += 1

        sum10 = 0.0
        sum20 = 0.0
        sumdv20 = 0.0
        running_tpv = 0.0
        running_volume = 0.0
        for i in range(start, end):
            local_idx = i - start
            sum10 += close[i]
            sum20 += close[i]
            if local_idx >= 10:
                sum10 -= close[i - 10]
            if local_idx >= 20:
                sum20 -= close[i - 20]

            if local_idx >= 9:
                dma10[i] = sum10 / 10.0
            if local_idx >= 19:
                dma20[i] = sum20 / 20.0

            if local_idx == 20:
                sumdv20 = 0.0
                for j in range(i - 20, i):
                    sumdv20 += dollar_volume[j]
                avgdv20_prior[i] = sumdv20 / 20.0
            elif local_idx > 20:
                sumdv20 += dollar_volume[i - 1]
                sumdv20 -= dollar_volume[i - 21]
                avgdv20_prior[i] = sumdv20 / 20.0

            running_tpv += typical_price[i] * volume[i]
            running_volume += volume[i]
            cum_tpv[i] = running_tpv
            cum_volume[i] = running_volume

        start = end
    return dma10, dma20, avgdv20_prior, typical_price, dollar_volume, cum_tpv, cum_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate post-EP watchlist entry, stop, and filter rules for 2025.")
    parser.add_argument("--signal-start", default=SIGNAL_START)
    parser.add_argument("--signal-end", default=SIGNAL_END)
    parser.add_argument("--feature-start", default=FEATURE_START)
    parser.add_argument("--watch-days", type=int, default=20)
    parser.add_argument("--max-hold-days", type=int, default=60)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def pct(value: object) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    return f"{100.0 * value:.1f}%"


def num(value: object, digits: int = 2) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def add_entry_features(features: pd.DataFrame, codes: np.ndarray) -> pd.DataFrame:
    dma10, dma20, avgdv20_prior, typical_price, dollar_volume, cum_tpv, cum_volume = _compute_entry_features_numba(
        features["high"].to_numpy(dtype=np.float64),
        features["low"].to_numpy(dtype=np.float64),
        features["close"].to_numpy(dtype=np.float64),
        features["volume"].to_numpy(dtype=np.float64),
        codes.astype(np.int64),
    )
    out = features.copy()
    out["dma10"] = dma10
    out["dma20"] = dma20
    out["avg_dollar_volume_20"] = avgdv20_prior
    out["typical_price"] = typical_price
    out["dollar_volume"] = dollar_volume
    out["cum_tpv"] = cum_tpv
    out["cum_volume"] = cum_volume
    return out


def load_features_and_events(cfg: EntryRuleConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    ep_cfg = EpisodicPivotConfig(start=cfg.feature_start)
    daily = load_daily(cfg.feature_start)
    features, codes, group_end_by_row = add_numba_features(daily, ep_cfg)
    features = add_entry_features(features, codes)
    signal_mask = (
        (features["lane_code"].to_numpy() > 0)
        & (features["date"].to_numpy() >= np.datetime64(cfg.signal_start))
        & (features["date"].to_numpy() < np.datetime64(cfg.signal_end))
    )
    event_indices = np.flatnonzero(signal_mask).astype(np.int64)
    events = features.iloc[event_indices].copy()
    events["ep_lane"] = [LANE_NAMES[int(code)] for code in events["lane_code"]]
    events["event_index"] = event_indices
    universe = load_universe()
    if not universe.empty:
        meta_cols = [
            col
            for col in ["symbol", "company_name", "exchange", "market_cap", "sector", "industry", "rank_market_cap"]
            if col in universe.columns
        ]
        events = events.merge(universe[meta_cols], on="symbol", how="left")
    return features, events, event_indices, group_end_by_row


def anchor_vwap_at(features: pd.DataFrame, start_idx: int, idx: int) -> float:
    tpv_sum = float(features["cum_tpv"].iat[idx])
    vol_sum = float(features["cum_volume"].iat[idx])
    if start_idx > 0 and features["symbol"].iat[start_idx - 1] == features["symbol"].iat[start_idx]:
        tpv_sum -= float(features["cum_tpv"].iat[start_idx - 1])
        vol_sum -= float(features["cum_volume"].iat[start_idx - 1])
    if vol_sum <= 0:
        return np.nan
    return tpv_sum / vol_sum


def find_entry(features: pd.DataFrame, event: pd.Series, group_end_by_row: np.ndarray, entry_mode: str, cfg: EntryRuleConfig) -> dict[str, Any] | None:
    ep_idx = int(event["event_index"])
    end = int(group_end_by_row[ep_idx])
    stop = min(end, ep_idx + 1 + cfg.watch_days)
    ep_high = float(event["high"])
    ep_low = float(event["low"])
    ep_close = float(event["close"])

    candidates: list[dict[str, Any]] = []
    for idx in range(ep_idx + 1, stop):
        row = features.iloc[idx]
        close = float(row["close"])
        open_ = float(row["open"])
        low = float(row["low"])
        dma10 = float(row["dma10"]) if pd.notna(row["dma10"]) else np.nan
        dma20 = float(row["dma20"]) if pd.notna(row["dma20"]) else np.nan

        breakout = close > ep_high
        pullback10 = (
            np.isfinite(dma10)
            and low <= dma10 * (1.0 + cfg.pullback_ma_buffer)
            and close >= dma10
            and close > open_
            and close > ep_low
            and close >= ep_close * 0.80
        )
        pullback20 = (
            np.isfinite(dma20)
            and low <= dma20 * (1.0 + cfg.pullback_ma_buffer)
            and close >= dma20
            and close > open_
            and close > ep_low
            and close >= ep_close * 0.80
        )

        if entry_mode == "ep_high_close_breakout" and breakout:
            return {"entry_index": idx, "entry_trigger": "ep_high_close_breakout"}
        if entry_mode == "pullback_10ma_hold" and pullback10:
            return {"entry_index": idx, "entry_trigger": "pullback_10ma_hold"}
        if entry_mode == "pullback_20ma_hold" and pullback20:
            return {"entry_index": idx, "entry_trigger": "pullback_20ma_hold"}
        if entry_mode == "first_valid_entry":
            if breakout:
                candidates.append({"entry_index": idx, "entry_trigger": "ep_high_close_breakout"})
            if pullback10:
                candidates.append({"entry_index": idx, "entry_trigger": "pullback_10ma_hold"})
            if pullback20:
                candidates.append({"entry_index": idx, "entry_trigger": "pullback_20ma_hold"})
            if candidates:
                priority = {
                    "ep_high_close_breakout": 0,
                    "pullback_10ma_hold": 1,
                    "pullback_20ma_hold": 2,
                }
                return sorted(candidates, key=lambda x: priority[x["entry_trigger"]])[0]
    return None


def stop_exit_price(open_: float, stop_level: float) -> float:
    return open_ if open_ < stop_level else stop_level


def simulate_trade(
    features: pd.DataFrame,
    event: pd.Series,
    group_end_by_row: np.ndarray,
    entry_mode: str,
    stop_mode: str,
    cfg: EntryRuleConfig,
) -> dict[str, Any] | None:
    entry = find_entry(features, event, group_end_by_row, entry_mode, cfg)
    if entry is None:
        return None

    ep_idx = int(event["event_index"])
    entry_idx = int(entry["entry_index"])
    end = int(group_end_by_row[ep_idx])
    hold_end = min(end - 1, entry_idx + cfg.max_hold_days)
    entry_row = features.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    ep_low = float(event["low"])
    max_high = float(entry_row["high"])
    min_low = float(entry_row["low"])
    exit_idx = hold_end
    exit_price = float(features.iloc[hold_end]["close"])
    exit_reason = "max_hold"

    for idx in range(entry_idx + 1, hold_end + 1):
        row = features.iloc[idx]
        open_ = float(row["open"])
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        max_high = max(max_high, high)
        min_low = min(min_low, low)

        prior_low = float(features.iloc[idx - 1]["low"])
        avwap = anchor_vwap_at(features, ep_idx, idx)

        prior_low_modes = {"prior_low", "ep_low_or_prior_low", "prior_low_or_vwap", "combined"}
        ep_low_modes = {"ep_low", "ep_low_or_vwap", "ep_low_or_prior_low", "combined"}
        vwap_modes = {"anchored_vwap_close", "ep_low_or_vwap", "prior_low_or_vwap", "combined"}

        if stop_mode in prior_low_modes and np.isfinite(prior_low) and low < prior_low:
            exit_idx = idx
            exit_price = stop_exit_price(open_, prior_low)
            exit_reason = "prior_low_stop"
            break
        if stop_mode in ep_low_modes and np.isfinite(ep_low) and low < ep_low:
            exit_idx = idx
            exit_price = stop_exit_price(open_, ep_low)
            exit_reason = "ep_low_stop"
            break
        if stop_mode in vwap_modes and np.isfinite(avwap) and close < avwap:
            exit_idx = idx
            exit_price = close
            exit_reason = "anchored_vwap_close_stop"
            break

    exit_row = features.iloc[exit_idx]
    return_pct = exit_price / entry_price - 1.0
    mfe_pct = max_high / entry_price - 1.0
    mae_pct = min_low / entry_price - 1.0
    return {
        "symbol": event["symbol"],
        "company_name": event.get("company_name", ""),
        "exchange": event.get("exchange", ""),
        "sector": event.get("sector", ""),
        "industry": event.get("industry", ""),
        "market_cap": event.get("market_cap", np.nan),
        "rank_market_cap": event.get("rank_market_cap", np.nan),
        "ep_date": event["date"],
        "ep_lane": event["ep_lane"],
        "ep_score": event["ep_score"],
        "ep_high": event["high"],
        "ep_low": event["low"],
        "ep_close": event["close"],
        "ep_gap_pct": event["gap_pct"],
        "ep_day_change_pct": event["day_change_pct"],
        "ep_volume_ratio_20": event["volume_ratio_20"],
        "ep_avg_dollar_volume_20": event["avg_dollar_volume_20"],
        "entry_mode": entry_mode,
        "entry_trigger": entry["entry_trigger"],
        "stop_mode": stop_mode,
        "entry_date": entry_row["date"],
        "entry_price": entry_price,
        "entry_close": entry_row["close"],
        "entry_dma10": entry_row["dma10"],
        "entry_dma20": entry_row["dma20"],
        "entry_days_from_ep": entry_idx - ep_idx,
        "exit_date": exit_row["date"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "hold_days": exit_idx - entry_idx,
        "return_pct": return_pct,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    }


def summarize_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for keys, sub in trades.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        returns = pd.to_numeric(sub["return_pct"], errors="coerce")
        wins = returns > 0
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "trades": int(len(sub)),
                "unique_symbols": int(sub["symbol"].nunique()),
                "win_rate": float(wins.mean()),
                "median_return_pct": float(returns.median()),
                "mean_return_pct": float(returns.mean()),
                "p25_return_pct": float(returns.quantile(0.25)),
                "p75_return_pct": float(returns.quantile(0.75)),
                "profit_factor": float(gains / losses) if losses > 0 else np.nan,
                "median_mfe_pct": float(pd.to_numeric(sub["mfe_pct"], errors="coerce").median()),
                "median_mae_pct": float(pd.to_numeric(sub["mae_pct"], errors="coerce").median()),
                "avg_hold_days": float(pd.to_numeric(sub["hold_days"], errors="coerce").mean()),
                "stop_rate": float(sub["exit_reason"].ne("max_hold").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_filter(trades: pd.DataFrame, name: str, cfg: EntryRuleConfig) -> pd.Series:
    mask = pd.Series(True, index=trades.index)
    if name in {"price", "price_liquidity", "price_liquidity_cap", "final_tradable"}:
        mask &= pd.to_numeric(trades["entry_price"], errors="coerce") >= cfg.min_price
    if name in {"liquidity", "price_liquidity", "price_liquidity_cap", "final_tradable"}:
        mask &= pd.to_numeric(trades["ep_avg_dollar_volume_20"], errors="coerce") >= cfg.min_avg_dollar_volume_20
    if name in {"market_cap", "price_liquidity_cap", "final_tradable"}:
        mask &= pd.to_numeric(trades["market_cap"], errors="coerce") >= cfg.min_market_cap
    if name in {"sector_known", "final_tradable"}:
        sector = trades["sector"].fillna("").astype(str).str.strip()
        mask &= sector.ne("") & ~sector.str.upper().isin({"UNKNOWN", "NAN"})
    if name in {"exclude_biotech", "final_tradable"}:
        industry = trades["industry"].fillna("").astype(str)
        mask &= ~industry.str.contains("Biotechnology", case=False, na=False)
    return mask


def build_trades(features: pd.DataFrame, events: pd.DataFrame, group_end_by_row: np.ndarray, cfg: EntryRuleConfig) -> pd.DataFrame:
    entry_modes = [
        "ep_high_close_breakout",
        "pullback_10ma_hold",
        "pullback_20ma_hold",
        "first_valid_entry",
    ]
    stop_modes = [
        "time_no_stop",
        "ep_low",
        "anchored_vwap_close",
        "ep_low_or_vwap",
        "ep_low_or_prior_low",
        "prior_low_or_vwap",
        "prior_low",
        "combined",
    ]
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        for entry_mode in entry_modes:
            for stop_mode in stop_modes:
                trade = simulate_trade(features, event, group_end_by_row, entry_mode, stop_mode, cfg)
                if trade is not None:
                    rows.append(trade)
    return pd.DataFrame(rows)


def build_filter_summary(trades: pd.DataFrame, cfg: EntryRuleConfig) -> pd.DataFrame:
    base = trades.loc[(trades["entry_mode"] == "first_valid_entry") & (trades["stop_mode"] == "combined")].copy()
    rows: list[pd.DataFrame] = []
    for name in [
        "none",
        "price",
        "liquidity",
        "price_liquidity",
        "market_cap",
        "price_liquidity_cap",
        "sector_known",
        "exclude_biotech",
        "final_tradable",
    ]:
        if name == "none":
            subset = base
        else:
            subset = base.loc[apply_filter(base, name, cfg)]
        summary = summarize_trades(subset.assign(filter_name=name), ["filter_name"])
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(
    out_dir: Path,
    cfg: EntryRuleConfig,
    features: pd.DataFrame,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    entry_summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# EP Entry Rule Validation - Russell 3000 2025")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed_seconds:.2f} seconds")
    lines.append(f"- Data: {DATA_VENDOR}")
    lines.append(f"- Universe rows: {len(features):,}; symbols: {features['symbol'].nunique():,}")
    lines.append(f"- EP watchlist events: {len(events):,}; symbols: {events['symbol'].nunique():,}")
    lines.append(f"- Watch window: {cfg.watch_days} trading days after EP")
    lines.append(f"- Entry price: daily close on trigger day")
    lines.append(f"- Max hold: {cfg.max_hold_days} trading days")
    lines.append("- Anchored VWAP stop uses daily typical-price VWAP from the EP day.")
    lines.append("")
    lines.append("## Step 1 - EP Watchlist")
    lines.append("")
    lines.append("| EP lane | Watchlist events | Symbols |")
    lines.append("|---|---:|---:|")
    for row in events.groupby("ep_lane").agg(events=("symbol", "size"), symbols=("symbol", "nunique")).reset_index().itertuples(index=False):
        lines.append(f"| {row.ep_lane} | {row.events:,} | {row.symbols:,} |")
    lines.append("")
    lines.append("## Step 2/3 - Entry And Stop Variants")
    lines.append("")
    lines.append(
        "| Entry | Stop | Trades | Symbols | Win | Median R | Mean R | PF | Median MFE | Median MAE | Stop rate |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    view = entry_summary.sort_values(["entry_mode", "stop_mode"], kind="mergesort")
    for row in view.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.entry_mode),
                    str(row.stop_mode),
                    f"{int(row.trades):,}",
                    f"{int(row.unique_symbols):,}",
                    pct(row.win_rate),
                    pct(row.median_return_pct),
                    pct(row.mean_return_pct),
                    num(row.profit_factor, 2),
                    pct(row.median_mfe_pct),
                    pct(row.median_mae_pct),
                    pct(row.stop_rate),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Step 4 - Filters On First Valid Entry + Combined Stop")
    lines.append("")
    lines.append(
        "| Filter | Trades | Symbols | Win | Median R | Mean R | PF | Median MFE | Median MAE | Stop rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in filter_summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.filter_name),
                    f"{int(row.trades):,}",
                    f"{int(row.unique_symbols):,}",
                    pct(row.win_rate),
                    pct(row.median_return_pct),
                    pct(row.mean_return_pct),
                    num(row.profit_factor, 2),
                    pct(row.median_mfe_pct),
                    pct(row.median_mae_pct),
                    pct(row.stop_rate),
                ]
            )
            + " |"
        )
    base_final = trades.loc[(trades["entry_mode"] == "first_valid_entry") & (trades["stop_mode"] == "combined")].copy()
    final = base_final.loc[apply_filter(base_final, "final_tradable", cfg)].copy()
    lines.append("")
    lines.append("## Rule Decision After This Pass")
    lines.append("")
    lines.append("1. Detect EP with the existing three-lane price/volume rules.")
    lines.append(f"2. Add each EP to a watchlist for {cfg.watch_days} trading days.")
    lines.append("3. Prefer pullback entries over blind EP-close buying: 10MA hold first, 20MA hold second, EP-high close breakout as momentum confirmation.")
    lines.append("4. Do not use prior-day low as the default daily-system hard stop; it stopped 100% of combined-stop trades and cut many runners.")
    lines.append("5. Use EP low as the disaster stop and EP-anchored VWAP close as the trend-health stop, then tune both on 5-minute data.")
    lines.append(
        f"6. Tradable filter: entry price >= ${cfg.min_price:.2f}, EP prior 20-day dollar volume >= ${cfg.min_avg_dollar_volume_20:,.0f}, "
        f"market cap >= ${cfg.min_market_cap:,.0f}, sector known, and industry is not Biotechnology."
    )
    lines.append("")
    lines.append("Top final-tradable trades by return:")
    lines.append("")
    lines.append("| Symbol | EP date | Entry | Exit | Sector | Industry | R | MFE | MAE | Reason |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|---:|---|")
    for row in final.sort_values("return_pct", ascending=False, kind="mergesort").head(20).itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.symbol),
                    str(pd.Timestamp(row.ep_date).date()),
                    str(pd.Timestamp(row.entry_date).date()),
                    str(pd.Timestamp(row.exit_date).date()),
                    str(row.sector),
                    str(row.industry)[:32],
                    pct(row.return_pct),
                    pct(row.mfe_pct),
                    pct(row.mae_pct),
                    str(row.exit_reason),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(pd.Series(asdict(cfg)).to_json(indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- This is a daily-bar validation. Intraday EP high breaks and VWAP crosses need 5-minute validation before live use.")
    lines.append("- Entry uses close confirmation, which is conservative but can enter later than an intraday trigger.")
    lines.append("- No commission, slippage, position sizing, or portfolio overlap constraints are included yet.")
    lines.append("- The Russell 3000 file is a current-universe local dataset; survivorship bias may exist.")
    (out_dir / "ep_entry_rules_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import time

    started = time.perf_counter()
    args = parse_args()
    cfg = EntryRuleConfig(
        signal_start=args.signal_start,
        signal_end=args.signal_end,
        feature_start=args.feature_start,
        watch_days=args.watch_days,
        max_hold_days=args.max_hold_days,
    )
    out_dir = Path(args.out_dir)
    features, events, _event_indices, group_end_by_row = load_features_and_events(cfg)
    trades = build_trades(features, events, group_end_by_row, cfg)
    entry_summary = summarize_trades(trades, ["entry_mode", "stop_mode"])
    filter_summary = build_filter_summary(trades, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    watch_cols = [
        "symbol",
        "company_name",
        "exchange",
        "sector",
        "industry",
        "market_cap",
        "rank_market_cap",
        "date",
        "ep_lane",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "gap_pct",
        "day_change_pct",
        "volume_ratio_20",
        "avg_dollar_volume_20",
        "ep_score",
        "event_index",
    ]
    events.loc[:, [c for c in watch_cols if c in events.columns]].to_csv(out_dir / "ep_watchlist_2025.csv", index=False)
    trades.to_csv(out_dir / "ep_entry_trades_2025.csv", index=False)
    entry_summary.to_csv(out_dir / "ep_entry_stop_summary_2025.csv", index=False)
    filter_summary.to_csv(out_dir / "ep_filter_summary_2025.csv", index=False)
    elapsed = time.perf_counter() - started
    write_report(out_dir, cfg, features, events, trades, entry_summary, filter_summary, elapsed)

    print(f"Watchlist EP events: {len(events):,} across {events['symbol'].nunique():,} symbols.")
    print(f"Simulated trade variants: {len(trades):,}.")
    print(f"Runtime: {elapsed:.2f}s.")
    print(f"Wrote {out_dir / 'ep_entry_rules_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
