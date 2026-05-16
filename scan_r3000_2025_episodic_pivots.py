#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

from scan_episodic_pivots import BREAKOUT_ROOT, LOCAL_DAILY_PATH, EpisodicPivotConfig


UNIVERSE_PATH = BREAKOUT_ROOT / "analysis_outputs" / "russell3000_full_dataset" / "universe.parquet"
OUT_DIR = BREAKOUT_ROOT / "analysis_outputs" / "episodic_pivot_r3000_2025"
SIGNAL_START = "2025-01-01"
SIGNAL_END = "2026-01-01"
FEATURE_START = "2016-04-25"
DATA_VENDOR = "local Russell 3000 daily_history.parquet"

LANE_NONE = 0
LANE_A_PLUS = 1
LANE_GAP = 2
LANE_DISPLACEMENT = 3
LANE_NAMES = {
    LANE_A_PLUS: "a_plus_gap_ep",
    LANE_GAP: "gap_ep",
    LANE_DISPLACEMENT: "displacement_ep",
}


@njit
def _group_end_by_row(codes: np.ndarray) -> np.ndarray:
    n = len(codes)
    out = np.empty(n, dtype=np.int64)
    start = 0
    while start < n:
        end = start + 1
        code = codes[start]
        while end < n and codes[end] == code:
            end += 1
        for i in range(start, end):
            out[i] = end
        start = end
    return out


@njit
def _compute_features_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    codes: np.ndarray,
    min_history_days: int,
    breakout_window: int,
    stronger_breakout_window: int,
) -> tuple[np.ndarray, ...]:
    n = len(close)
    prev_close = np.empty(n, dtype=np.float64)
    gap_pct = np.empty(n, dtype=np.float64)
    day_change_pct = np.empty(n, dtype=np.float64)
    intraday_return_pct = np.empty(n, dtype=np.float64)
    close_range_pos = np.empty(n, dtype=np.float64)
    avg_volume_20 = np.empty(n, dtype=np.float64)
    volume_ratio_20 = np.empty(n, dtype=np.float64)
    high20_prior = np.empty(n, dtype=np.float64)
    high50_prior = np.empty(n, dtype=np.float64)
    above_high20 = np.zeros(n, dtype=np.bool_)
    above_high50 = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        prev_close[i] = np.nan
        gap_pct[i] = np.nan
        day_change_pct[i] = np.nan
        intraday_return_pct[i] = np.nan
        close_range_pos[i] = np.nan
        avg_volume_20[i] = np.nan
        volume_ratio_20[i] = np.nan
        high20_prior[i] = np.nan
        high50_prior[i] = np.nan

    start = 0
    while start < n:
        end = start + 1
        code = codes[start]
        while end < n and codes[end] == code:
            end += 1

        rolling_volume_sum = 0.0
        for i in range(start, end):
            if i > start:
                prev_close[i] = close[i - 1]
                if close[i - 1] > 0.0:
                    gap_pct[i] = open_[i] / close[i - 1] - 1.0
                    day_change_pct[i] = close[i] / close[i - 1] - 1.0
            if open_[i] > 0.0:
                intraday_return_pct[i] = close[i] / open_[i] - 1.0

            day_range = high[i] - low[i]
            if day_range > 0.0:
                close_range_pos[i] = (close[i] - low[i]) / day_range

            local_idx = i - start
            if local_idx >= min_history_days:
                if local_idx == min_history_days:
                    rolling_volume_sum = 0.0
                    for j in range(i - 20, i):
                        rolling_volume_sum += volume[j]
                else:
                    rolling_volume_sum += volume[i - 1]
                    rolling_volume_sum -= volume[i - 21]
                avg_volume_20[i] = rolling_volume_sum / 20.0
                if avg_volume_20[i] > 0.0:
                    volume_ratio_20[i] = volume[i] / avg_volume_20[i]

            if local_idx >= 20:
                hi = -np.inf
                j0 = i - breakout_window
                if j0 < start:
                    j0 = start
                for j in range(j0, i):
                    if high[j] > hi:
                        hi = high[j]
                high20_prior[i] = hi
                above_high20[i] = close[i] > hi

            if local_idx >= 40:
                hi = -np.inf
                j0 = i - stronger_breakout_window
                if j0 < start:
                    j0 = start
                for j in range(j0, i):
                    if high[j] > hi:
                        hi = high[j]
                high50_prior[i] = hi
                above_high50[i] = close[i] > hi

        start = end

    return (
        prev_close,
        gap_pct,
        day_change_pct,
        intraday_return_pct,
        close_range_pos,
        avg_volume_20,
        volume_ratio_20,
        high20_prior,
        high50_prior,
        above_high20,
        above_high50,
    )


@njit
def _classify_events_numba(
    open_: np.ndarray,
    close: np.ndarray,
    gap_pct: np.ndarray,
    day_change_pct: np.ndarray,
    close_range_pos: np.ndarray,
    volume_ratio_20: np.ndarray,
    above_high20: np.ndarray,
    above_high50: np.ndarray,
    gap_ep_min: float,
    gap_ep_volume_ratio_min: float,
    gap_ep_close_range_min: float,
    a_plus_gap_min: float,
    a_plus_volume_ratio_min: float,
    a_plus_close_range_min: float,
    displacement_day_change_min: float,
    displacement_volume_ratio_min: float,
    displacement_close_range_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    lane = np.zeros(n, dtype=np.int16)
    ep_score = np.empty(n, dtype=np.float64)
    for i in range(n):
        ep_score[i] = np.nan
        common = (
            not np.isnan(gap_pct[i])
            and not np.isnan(volume_ratio_20[i])
            and not np.isnan(close_range_pos[i])
            and close[i] > open_[i]
            and close_range_pos[i] >= 0.50
            and volume_ratio_20[i] > 0.0
        )
        breakout_like = above_high20[i] or day_change_pct[i] >= 0.10
        gap_ep = (
            common
            and gap_pct[i] >= gap_ep_min
            and volume_ratio_20[i] >= gap_ep_volume_ratio_min
            and close_range_pos[i] >= gap_ep_close_range_min
            and breakout_like
        )
        a_plus = (
            gap_ep
            and gap_pct[i] >= a_plus_gap_min
            and volume_ratio_20[i] >= a_plus_volume_ratio_min
            and close_range_pos[i] >= a_plus_close_range_min
            and above_high50[i]
        )
        displacement = (
            common
            and day_change_pct[i] >= displacement_day_change_min
            and volume_ratio_20[i] >= displacement_volume_ratio_min
            and close_range_pos[i] >= displacement_close_range_min
            and above_high20[i]
        )
        if a_plus:
            lane[i] = LANE_A_PLUS
        elif gap_ep:
            lane[i] = LANE_GAP
        elif displacement:
            lane[i] = LANE_DISPLACEMENT
        else:
            continue

        gap_component = gap_pct[i]
        if gap_component < 0.0:
            gap_component = 0.0
        elif gap_component > 0.30:
            gap_component = 0.30

        volume_component = volume_ratio_20[i]
        if volume_component < 0.0:
            volume_component = 0.0
        elif volume_component > 10.0:
            volume_component = 10.0

        day_component = day_change_pct[i]
        if day_component < 0.0:
            day_component = 0.0
        elif day_component > 0.60:
            day_component = 0.60

        close_range_component = close_range_pos[i]
        if close_range_component < 0.0:
            close_range_component = 0.0
        elif close_range_component > 1.0:
            close_range_component = 1.0

        high20_bonus = 10.0 if above_high20[i] else 0.0
        high50_bonus = 10.0 if above_high50[i] else 0.0
        ep_score[i] = (
            100.0 * gap_component
            + 12.0 * volume_component
            + 45.0 * day_component
            + 20.0 * close_range_component
            + high20_bonus
            + high50_bonus
        )
    return lane, ep_score


@njit(parallel=True)
def _forward_returns_for_events_numba(
    event_indices: np.ndarray,
    group_end_by_row: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = len(event_indices)
    k = len(windows)
    future_sessions = np.empty(m, dtype=np.int64)
    returns = np.empty((m, k), dtype=np.float64)
    returns_to_end = np.empty(m, dtype=np.float64)
    for e in prange(m):
        idx = event_indices[e]
        end = group_end_by_row[idx]
        sessions = end - idx - 1
        future_sessions[e] = sessions
        entry_close = close[idx]

        max_to_end = -np.inf
        for j in range(idx + 1, end):
            if high[j] > max_to_end:
                max_to_end = high[j]
        if sessions <= 0 or entry_close <= 0.0 or max_to_end == -np.inf:
            returns_to_end[e] = np.nan
        else:
            returns_to_end[e] = max_to_end / entry_close - 1.0

        for w_i in range(k):
            w = windows[w_i]
            stop = idx + 1 + w
            if stop > end:
                stop = end
            max_high = -np.inf
            for j in range(idx + 1, stop):
                if high[j] > max_high:
                    max_high = high[j]
            if stop <= idx + 1 or entry_close <= 0.0 or max_high == -np.inf:
                returns[e, w_i] = np.nan
            else:
                returns[e, w_i] = max_high / entry_close - 1.0
    return future_sessions, returns, returns_to_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Russell 3000 2025 episodic pivot signals with Numba.")
    parser.add_argument("--signal-start", default=SIGNAL_START)
    parser.add_argument("--signal-end", default=SIGNAL_END)
    parser.add_argument("--feature-start", default=FEATURE_START)
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


def load_daily(feature_start: str) -> pd.DataFrame:
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    daily = pd.read_parquet(LOCAL_DAILY_PATH, columns=cols)
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["symbol", "date", "open", "high", "low", "close"]).copy()
    daily = daily.loc[daily["date"] >= pd.Timestamp(feature_start), cols]
    return daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        return pd.DataFrame(columns=["symbol"])
    universe = pd.read_parquet(UNIVERSE_PATH)
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    keep = [
        col
        for col in [
            "symbol",
            "company_name",
            "exchange",
            "market_cap",
            "sector",
            "industry",
            "rank_market_cap",
        ]
        if col in universe.columns
    ]
    return universe.loc[:, keep].drop_duplicates("symbol")


def add_numba_features(daily: pd.DataFrame, cfg: EpisodicPivotConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    codes = pd.factorize(daily["symbol"], sort=False)[0].astype(np.int64)
    open_ = daily["open"].to_numpy(dtype=np.float64)
    high = daily["high"].to_numpy(dtype=np.float64)
    low = daily["low"].to_numpy(dtype=np.float64)
    close = daily["close"].to_numpy(dtype=np.float64)
    volume = daily["volume"].to_numpy(dtype=np.float64)

    (
        prev_close,
        gap_pct,
        day_change_pct,
        intraday_return_pct,
        close_range_pos,
        avg_volume_20,
        volume_ratio_20,
        high20_prior,
        high50_prior,
        above_high20,
        above_high50,
    ) = _compute_features_numba(
        open_,
        high,
        low,
        close,
        volume,
        codes,
        cfg.min_history_days,
        cfg.breakout_window,
        cfg.stronger_breakout_window,
    )
    lane, ep_score = _classify_events_numba(
        open_,
        close,
        gap_pct,
        day_change_pct,
        close_range_pos,
        volume_ratio_20,
        above_high20,
        above_high50,
        cfg.gap_ep_min,
        cfg.gap_ep_volume_ratio_min,
        cfg.gap_ep_close_range_min,
        cfg.a_plus_gap_min,
        cfg.a_plus_volume_ratio_min,
        cfg.a_plus_close_range_min,
        cfg.displacement_day_change_min,
        cfg.displacement_volume_ratio_min,
        cfg.displacement_close_range_min,
    )

    features = daily.copy()
    features["prev_close"] = prev_close
    features["gap_pct"] = gap_pct
    features["day_change_pct"] = day_change_pct
    features["intraday_return_pct"] = intraday_return_pct
    features["close_range_pos"] = close_range_pos
    features["avg_volume_20"] = avg_volume_20
    features["volume_ratio_20"] = volume_ratio_20
    features["high20_prior"] = high20_prior
    features["high50_prior"] = high50_prior
    features["above_high20"] = above_high20
    features["above_high50"] = above_high50
    features["lane_code"] = lane
    features["ep_score"] = ep_score
    return features, codes, _group_end_by_row(codes)


def attach_forward_returns(
    events: pd.DataFrame,
    event_indices: np.ndarray,
    group_end_by_row: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    window_array = np.asarray(windows, dtype=np.int64)
    future_sessions, returns, returns_to_end = _forward_returns_for_events_numba(
        event_indices.astype(np.int64),
        group_end_by_row,
        high.astype(np.float64),
        close.astype(np.float64),
        window_array,
    )
    out = events.copy()
    out["future_sessions_available"] = future_sessions
    for i, window in enumerate(windows):
        out[f"max_return_{window}d_pct"] = returns[:, i]
    out["max_return_to_end_pct"] = returns_to_end
    return out


def summarize_by_rule(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule, sub in events.groupby("ep_lane", sort=False):
        returns = pd.to_numeric(sub["max_return_to_end_pct"], errors="coerce")
        rows.append(
            {
                "rule": rule,
                "signals": int(len(sub)),
                "unique_symbols": int(sub["symbol"].nunique()),
                "median_max_return_to_end_pct": float(returns.median()),
                "mean_max_return_to_end_pct": float(returns.mean()),
                "p90_max_return_to_end_pct": float(returns.quantile(0.90)),
                "max_return_to_end_pct": float(returns.max()),
                "pct_positive": float((returns > 0).mean()),
                "pct_ge_20": float((returns >= 0.20).mean()),
                "pct_ge_50": float((returns >= 0.50).mean()),
                "pct_ge_100": float((returns >= 1.00).mean()),
                "pct_ge_200": float((returns >= 2.00).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("signals", ascending=False, kind="mergesort")


def build_symbol_rule_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    best = events.sort_values(
        ["ep_lane", "symbol", "max_return_to_end_pct", "ep_score", "date"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).drop_duplicates(["ep_lane", "symbol"], keep="first")

    agg = (
        events.groupby(["ep_lane", "symbol"], sort=False)
        .agg(
            signal_count=("date", "size"),
            first_signal_date=("date", "min"),
            latest_signal_date=("date", "max"),
        )
        .reset_index()
    )
    selected = best.drop(columns=["signal_count", "first_signal_date", "latest_signal_date"], errors="ignore")
    return agg.merge(selected, on=["ep_lane", "symbol"], how="left").sort_values(
        ["ep_lane", "max_return_to_end_pct"], ascending=[True, False], kind="mergesort"
    )


def top_table(frame: pd.DataFrame, rule: str, limit: int = 20) -> pd.DataFrame:
    return (
        frame.loc[frame["ep_lane"].eq(rule)]
        .sort_values(["max_return_to_end_pct", "ep_score"], ascending=[False, False], kind="mergesort")
        .head(limit)
    )


def write_report(
    out_dir: Path,
    cfg: EpisodicPivotConfig,
    signal_start: str,
    signal_end: str,
    features: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    symbol_rule: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_start = features["date"].min().date()
    data_end = features["date"].max().date()
    lines: list[str] = []
    lines.append("# Russell 3000 2025 Episodic Pivot Scan")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed_seconds:.2f} seconds, Numba JIT enabled")
    lines.append(f"- Universe: {features['symbol'].nunique():,} symbols")
    lines.append(f"- Data: {DATA_VENDOR}")
    lines.append(f"- Data period loaded: {data_start} to {data_end}")
    lines.append(f"- Signal period: {signal_start} <= date < {signal_end}")
    lines.append("- Forward return: max future high after the signal close, excluding the signal day.")
    lines.append("- No entry execution, stop, commission, slippage, or liquidity filter is applied.")
    lines.append("")
    lines.append("## Rule Summary")
    lines.append("")
    lines.append(
        "| Rule | Signals | Symbols | Median max | Mean max | P90 max | Best max | Positive | >=20% | >=50% | >=100% | >=200% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.rule),
                    f"{int(row.signals):,}",
                    f"{int(row.unique_symbols):,}",
                    pct(row.median_max_return_to_end_pct),
                    pct(row.mean_max_return_to_end_pct),
                    pct(row.p90_max_return_to_end_pct),
                    pct(row.max_return_to_end_pct),
                    pct(row.pct_positive),
                    pct(row.pct_ge_20),
                    pct(row.pct_ge_50),
                    pct(row.pct_ge_100),
                    pct(row.pct_ge_200),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Top Signals By Rule")
    for rule in ["a_plus_gap_ep", "gap_ep", "displacement_ep"]:
        lines.append("")
        lines.append(f"### {rule}")
        lines.append("")
        lines.append("| Symbol | Company | Date | Gap | Day change | Vol x20 | Max to end | Max 20d | Max 60d | Max 120d | Score |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in top_table(events, rule).itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.symbol),
                        str(getattr(row, "company_name", ""))[:48],
                        str(pd.Timestamp(row.date).date()),
                        pct(row.gap_pct),
                        pct(row.day_change_pct),
                        num(row.volume_ratio_20, 2),
                        pct(row.max_return_to_end_pct),
                        pct(getattr(row, "max_return_20d_pct", np.nan)),
                        pct(getattr(row, "max_return_60d_pct", np.nan)),
                        pct(getattr(row, "max_return_120d_pct", np.nan)),
                        num(row.ep_score, 1),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Extracted Symbol Counts")
    lines.append("")
    lines.append(
        "The file `ep_2025_symbols_by_rule.csv` contains one row per symbol per exclusive rule lane, using the best forward max signal for that symbol/rule."
    )
    lines.append(f"Rows: {len(symbol_rule):,}")
    lines.append("")
    lines.append("## Rule Config")
    lines.append("")
    lines.append("```json")
    lines.append(pd.Series(asdict(cfg)).to_json(indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- A+ Gap EP is a stricter subset of Gap EP, but this report uses exclusive lanes: A+ > Gap > Displacement.")
    lines.append("- The Russell 3000 file is a local current-universe dataset, so survivorship bias may exist.")
    lines.append("- Catalyst/news is not checked; signals are price/volume proxies.")
    lines.append("- Forward maximum is not a tradable return without an exit rule.")
    (out_dir / "ep_2025_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import time

    started = time.perf_counter()
    args = parse_args()
    out_dir = Path(args.out_dir)
    cfg = EpisodicPivotConfig(start=args.feature_start)

    daily = load_daily(args.feature_start)
    features, _codes, group_end_by_row = add_numba_features(daily, cfg)
    signal_mask = (
        (features["lane_code"].to_numpy() > 0)
        & (features["date"].to_numpy() >= np.datetime64(args.signal_start))
        & (features["date"].to_numpy() < np.datetime64(args.signal_end))
    )
    event_indices = np.flatnonzero(signal_mask).astype(np.int64)
    events = features.iloc[event_indices].copy()
    events["ep_lane"] = [LANE_NAMES[int(code)] for code in events["lane_code"]]
    events = attach_forward_returns(
        events,
        event_indices,
        group_end_by_row,
        features["high"].to_numpy(dtype=np.float64),
        features["close"].to_numpy(dtype=np.float64),
        cfg.forward_windows,
    )

    universe = load_universe()
    if not universe.empty and not events.empty:
        events = events.merge(universe, on="symbol", how="left")

    keep_cols = [
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
        "prev_close",
        "gap_pct",
        "day_change_pct",
        "intraday_return_pct",
        "volume_ratio_20",
        "close_range_pos",
        "above_high20",
        "above_high50",
        "ep_score",
        "future_sessions_available",
        "max_return_20d_pct",
        "max_return_60d_pct",
        "max_return_120d_pct",
        "max_return_252d_pct",
        "max_return_to_end_pct",
    ]
    events = events.loc[:, [col for col in keep_cols if col in events.columns]].sort_values(
        ["date", "ep_score", "symbol"], ascending=[True, False, True], kind="mergesort"
    )
    summary = summarize_by_rule(events)
    symbol_rule = build_symbol_rule_table(events)
    top = events.sort_values(["max_return_to_end_pct", "ep_score"], ascending=[False, False], kind="mergesort").head(200)

    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "ep_2025_events.csv", index=False)
    symbol_rule.to_csv(out_dir / "ep_2025_symbols_by_rule.csv", index=False)
    summary.to_csv(out_dir / "ep_2025_summary_by_rule.csv", index=False)
    top.to_csv(out_dir / "ep_2025_top200_signals.csv", index=False)
    elapsed = time.perf_counter() - started
    write_report(out_dir, cfg, args.signal_start, args.signal_end, features, events, summary, symbol_rule, elapsed)

    print(f"Loaded {len(features):,} rows for {features['symbol'].nunique():,} symbols.")
    print(f"Detected {len(events):,} 2025 EP signals across {events['symbol'].nunique():,} symbols.")
    print(f"Runtime: {elapsed:.2f}s with Numba JIT.")
    print(f"Wrote {out_dir / 'ep_2025_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
