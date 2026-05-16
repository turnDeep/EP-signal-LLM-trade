#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BREAKOUT_ROOT = Path(r"C:\Users\plane\BreakOut")
LOCAL_DAILY_PATH = BREAKOUT_ROOT / "analysis_outputs" / "russell3000_daily_10y_dataset" / "daily_history.parquet"
DEFAULT_OUT_DIR = BREAKOUT_ROOT / "analysis_outputs" / "episodic_pivot_mentioned_symbols"
DEFAULT_SYMBOLS = ["BE", "TTMI", "LWLG", "DELL", "MRAM", "BAND", "OSS", "RKLB"]


@dataclass(frozen=True)
class EpisodicPivotConfig:
    start: str = "2023-01-01"
    min_history_days: int = 20
    gap_ep_min: float = 0.08
    gap_ep_volume_ratio_min: float = 2.0
    gap_ep_close_range_min: float = 0.60
    a_plus_gap_min: float = 0.10
    a_plus_volume_ratio_min: float = 3.0
    a_plus_close_range_min: float = 0.70
    displacement_day_change_min: float = 0.12
    displacement_volume_ratio_min: float = 2.5
    displacement_close_range_min: float = 0.65
    breakout_window: int = 20
    stronger_breakout_window: int = 50
    forward_windows: tuple[int, ...] = (20, 60, 120, 252)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan mentioned symbols for episodic pivot events.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to scan.")
    parser.add_argument("--start", default=EpisodicPivotConfig.start, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional end date, YYYY-MM-DD. yfinance end is exclusive.")
    parser.add_argument(
        "--source",
        choices=["auto", "yfinance", "local"],
        default="auto",
        help="Data source. auto tries yfinance first, then local parquet for failures.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    return parser.parse_args()


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper().lstrip("$")
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(col[0]) for col in frame.columns]
    return frame


def fetch_yfinance(symbols: list[str], start: str, end: str | None) -> tuple[pd.DataFrame, list[str]]:
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(), symbols

    pieces: list[pd.DataFrame] = []
    failed: list[str] = []
    for symbol in symbols:
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
            )
        except Exception:
            failed.append(symbol)
            continue
        raw = flatten_yfinance_columns(raw)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if raw.empty or any(col not in raw.columns for col in required):
            failed.append(symbol)
            continue
        one = raw.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        one["symbol"] = symbol
        pieces.append(one[["symbol", "date", "open", "high", "low", "close", "volume"]])
    if not pieces:
        return pd.DataFrame(), failed
    return pd.concat(pieces, ignore_index=True), failed


def load_local(symbols: list[str], start: str, end: str | None) -> pd.DataFrame:
    if not LOCAL_DAILY_PATH.exists():
        return pd.DataFrame()
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(LOCAL_DAILY_PATH, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    mask = df["symbol"].isin(symbols) & (df["date"] >= pd.Timestamp(start))
    if end:
        mask &= df["date"] < pd.Timestamp(end)
    return df.loc[mask, cols].copy()


def load_prices(symbols: list[str], start: str, end: str | None, source: str) -> tuple[pd.DataFrame, str, list[str]]:
    if source == "local":
        local = load_local(symbols, start, end)
        missing = sorted(set(symbols) - set(local["symbol"].dropna().unique()))
        return local, "local_parquet", missing

    yf_df, yf_failed = fetch_yfinance(symbols, start, end)
    if source == "yfinance":
        return yf_df, "yfinance", yf_failed

    have = set(yf_df["symbol"].dropna().unique()) if not yf_df.empty else set()
    missing = sorted(set(symbols) - have)
    if missing:
        local = load_local(missing, start, end)
        if not local.empty:
            yf_df = pd.concat([yf_df, local], ignore_index=True)
            have.update(local["symbol"].dropna().unique())
    still_missing = sorted(set(symbols) - have)
    vendor = "yfinance_with_local_fallback" if missing else "yfinance"
    return yf_df, vendor, still_missing


def prepare_features(prices: pd.DataFrame, cfg: EpisodicPivotConfig) -> pd.DataFrame:
    df = prices.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df.dropna(subset=["symbol", "date", "open", "high", "low", "close"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"], kind="mergesort")
        .reset_index(drop=True)
    )

    g = df.groupby("symbol", sort=False)
    df["prev_close"] = g["close"].shift(1)
    df["gap_pct"] = df["open"] / df["prev_close"].replace(0, np.nan) - 1.0
    df["day_change_pct"] = df["close"] / df["prev_close"].replace(0, np.nan) - 1.0
    df["intraday_return_pct"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
    day_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_range_pos"] = (df["close"] - df["low"]) / day_range
    df["dollar_volume"] = df["close"] * df["volume"]

    df["avg_volume_20"] = g["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=cfg.min_history_days).mean())
    df["avg_dollar_volume_20"] = g["dollar_volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=cfg.min_history_days).mean()
    )
    df["volume_ratio_20"] = df["volume"] / df["avg_volume_20"].replace(0, np.nan)
    df["high20_prior"] = g["high"].transform(lambda s: s.shift(1).rolling(cfg.breakout_window, min_periods=20).max())
    df["high50_prior"] = g["high"].transform(lambda s: s.shift(1).rolling(cfg.stronger_breakout_window, min_periods=40).max())
    df["low63_prior"] = g["low"].transform(lambda s: s.shift(1).rolling(63, min_periods=40).min())
    df["prior_63d_return"] = df["prev_close"] / g["close"].shift(63).replace(0, np.nan) - 1.0
    df["above_high20"] = df["close"] > df["high20_prior"]
    df["above_high50"] = df["close"] > df["high50_prior"]
    return df


def classify_events(df: pd.DataFrame, cfg: EpisodicPivotConfig) -> pd.DataFrame:
    common = (
        df["prev_close"].notna()
        & df["avg_volume_20"].notna()
        & (df["close"] > df["open"])
        & (df["close_range_pos"] >= 0.50)
        & (df["volume_ratio_20"] > 0)
    )
    breakout_like = df["above_high20"] | (df["day_change_pct"] >= 0.10)
    gap_ep = (
        common
        & (df["gap_pct"] >= cfg.gap_ep_min)
        & (df["volume_ratio_20"] >= cfg.gap_ep_volume_ratio_min)
        & (df["close_range_pos"] >= cfg.gap_ep_close_range_min)
        & breakout_like
    )
    a_plus = (
        gap_ep
        & (df["gap_pct"] >= cfg.a_plus_gap_min)
        & (df["volume_ratio_20"] >= cfg.a_plus_volume_ratio_min)
        & (df["close_range_pos"] >= cfg.a_plus_close_range_min)
        & df["above_high50"]
    )
    displacement = (
        common
        & (df["day_change_pct"] >= cfg.displacement_day_change_min)
        & (df["volume_ratio_20"] >= cfg.displacement_volume_ratio_min)
        & (df["close_range_pos"] >= cfg.displacement_close_range_min)
        & df["above_high20"]
    )
    mask = gap_ep | displacement
    events = df.loc[mask].copy()
    if events.empty:
        return events

    events["ep_lane"] = np.select(
        [a_plus.loc[mask], gap_ep.loc[mask], displacement.loc[mask]],
        ["a_plus_gap_ep", "gap_ep", "displacement_ep"],
        default="ep_candidate",
    )
    events["ep_score"] = (
        100.0 * events["gap_pct"].clip(lower=0, upper=0.30)
        + 12.0 * events["volume_ratio_20"].clip(lower=0, upper=10)
        + 45.0 * events["day_change_pct"].clip(lower=0, upper=0.60)
        + 20.0 * events["close_range_pos"].clip(lower=0, upper=1.0)
        + 10.0 * events["above_high20"].astype(float)
        + 10.0 * events["above_high50"].astype(float)
    )
    return events.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def forward_stats(features: pd.DataFrame, events: pd.DataFrame, cfg: EpisodicPivotConfig) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    by_symbol = {
        symbol: sub.sort_values("date", kind="mergesort").reset_index(drop=True)
        for symbol, sub in features.groupby("symbol", sort=False)
    }
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for event in events.itertuples(index=False):
        symbol = str(event.symbol)
        daily = by_symbol[symbol]
        idx = int(daily["date"].searchsorted(pd.Timestamp(event.date), side="right"))
        event_close = float(event.close)
        out = event._asdict()
        last_close = float(daily["close"].iloc[-1])
        out["last_date"] = daily["date"].iloc[-1]
        out["current_return_pct"] = last_close / event_close - 1.0 if event_close > 0 else np.nan
        for window in cfg.forward_windows:
            future = daily.iloc[idx : idx + window]
            if future.empty or event_close <= 0:
                out[f"max_return_{window}d_pct"] = np.nan
                out[f"close_return_{window}d_pct"] = np.nan
                out[f"max_drawdown_{window}d_pct"] = np.nan
                out[f"days_to_max_{window}d"] = np.nan
                continue
            max_high = float(future["high"].max())
            min_low = float(future["low"].min())
            max_idx = int(future["high"].to_numpy().argmax())
            out[f"max_return_{window}d_pct"] = max_high / event_close - 1.0
            out[f"close_return_{window}d_pct"] = float(future["close"].iloc[-1]) / event_close - 1.0
            out[f"max_drawdown_{window}d_pct"] = min_low / event_close - 1.0
            out[f"days_to_max_{window}d"] = int(max_idx + 1)
        rows.append(out)
    return pd.DataFrame(rows)


def best_by_symbol(events: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame({"symbol": symbols, "status": "no_ep_detected"})
    ranked = events.sort_values(
        ["symbol", "ep_score", "max_return_120d_pct", "date"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    best = ranked.drop_duplicates("symbol", keep="first").copy()
    missing = sorted(set(symbols) - set(best["symbol"].dropna().unique()))
    if missing:
        best = pd.concat([best, pd.DataFrame({"symbol": missing, "status": "no_ep_detected"})], ignore_index=True)
    best["status"] = best.get("status", pd.Series(index=best.index, dtype=object)).fillna("detected")
    return best.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def latest_by_symbol(events: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame({"symbol": symbols, "status": "no_ep_detected"})
    ranked = events.sort_values(["symbol", "date", "ep_score"], ascending=[True, False, False], kind="mergesort")
    latest = ranked.drop_duplicates("symbol", keep="first").copy()
    missing = sorted(set(symbols) - set(latest["symbol"].dropna().unique()))
    if missing:
        latest = pd.concat([latest, pd.DataFrame({"symbol": missing, "status": "no_ep_detected"})], ignore_index=True)
    latest["status"] = latest.get("status", pd.Series(index=latest.index, dtype=object)).fillna("detected")
    return latest.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def pct(value: object) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value * 100:.1f}%"


def num(value: object, digits: int = 2) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def write_report(
    out_dir: Path,
    symbols: list[str],
    vendor: str,
    missing: list[str],
    cfg: EpisodicPivotConfig,
    features: pd.DataFrame,
    events: pd.DataFrame,
    best: pd.DataFrame,
    latest: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = ""
    if not features.empty:
        as_of = str(features["date"].max().date())

    lines: list[str] = []
    lines.append("# Episodic Pivot Scan - Mentioned Symbols")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Symbols: {', '.join(symbols)}")
    lines.append(f"- Data vendor: {vendor}")
    lines.append(f"- Data as of: {as_of}")
    lines.append(f"- Start date: {cfg.start}")
    if missing:
        lines.append(f"- Missing symbols: {', '.join(missing)}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append(
        "- Gap EP: gap >= 8%, volume >= 2.0x prior 20-day average, close in top 40% of range, close > open, and breakout-like behavior."
    )
    lines.append(
        "- A+ Gap EP: gap >= 10%, volume >= 3.0x, close in top 30% of range, and close above prior 50-day high."
    )
    lines.append(
        "- Displacement EP: day change >= 12%, volume >= 2.5x, close in top 35% of range, close > open, and close above prior 20-day high."
    )
    lines.append("- Catalyst/news is not verified here; this is a price/volume EP proxy.")
    lines.append("")
    lines.append("## Latest EP Per Symbol")
    lines.append("")
    header = (
        "| Symbol | Status | EP date | Lane | Gap | Day change | Vol x20 | Close range | "
        "Max +60d | Current from EP | Score |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in latest.itertuples(index=False):
        date = getattr(row, "date", "")
        if pd.notna(date) and date != "":
            date = pd.Timestamp(date).date()
        else:
            date = ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(getattr(row, "symbol", "")),
                    str(getattr(row, "status", "")),
                    str(date),
                    str(getattr(row, "ep_lane", "")),
                    pct(getattr(row, "gap_pct", np.nan)),
                    pct(getattr(row, "day_change_pct", np.nan)),
                    num(getattr(row, "volume_ratio_20", np.nan), 2),
                    pct(getattr(row, "close_range_pos", np.nan)),
                    pct(getattr(row, "max_return_60d_pct", np.nan)),
                    pct(getattr(row, "current_return_pct", np.nan)),
                    num(getattr(row, "ep_score", np.nan), 1),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Highest-Score EP Per Symbol")
    lines.append("")
    header = (
        "| Symbol | Status | EP date | Lane | Gap | Day change | Vol x20 | Close range | "
        "Max +120d | Current from EP | Score |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in best.itertuples(index=False):
        date = getattr(row, "date", "")
        if pd.notna(date) and date != "":
            date = pd.Timestamp(date).date()
        else:
            date = ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(getattr(row, "symbol", "")),
                    str(getattr(row, "status", "")),
                    str(date),
                    str(getattr(row, "ep_lane", "")),
                    pct(getattr(row, "gap_pct", np.nan)),
                    pct(getattr(row, "day_change_pct", np.nan)),
                    num(getattr(row, "volume_ratio_20", np.nan), 2),
                    pct(getattr(row, "close_range_pos", np.nan)),
                    pct(getattr(row, "max_return_120d_pct", np.nan)),
                    pct(getattr(row, "current_return_pct", np.nan)),
                    num(getattr(row, "ep_score", np.nan), 1),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## All Detected Events")
    lines.append("")
    lines.append("| Symbol | Date | Lane | Gap | Day change | Vol x20 | Max +60d | Max +120d | Score |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    event_view = events.sort_values(["date", "ep_score"], ascending=[False, False], kind="mergesort")
    for row in event_view.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.symbol),
                    str(pd.Timestamp(row.date).date()),
                    str(row.ep_lane),
                    pct(row.gap_pct),
                    pct(row.day_change_pct),
                    num(row.volume_ratio_20, 2),
                    pct(getattr(row, "max_return_60d_pct", np.nan)),
                    pct(getattr(row, "max_return_120d_pct", np.nan)),
                    num(row.ep_score, 1),
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
    (out_dir / "ep_scan_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    symbols = normalize_symbols(args.symbols)
    cfg = EpisodicPivotConfig(start=args.start)
    out_dir = Path(args.out_dir)

    prices, vendor, missing = load_prices(symbols, cfg.start, args.end, args.source)
    if prices.empty:
        raise SystemExit("No price data loaded.")

    features = prepare_features(prices, cfg)
    events = classify_events(features, cfg)
    events = forward_stats(features, events, cfg)
    best = best_by_symbol(events, symbols)
    latest = latest_by_symbol(events, symbols)

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_feature_cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "gap_pct",
        "day_change_pct",
        "volume_ratio_20",
        "close_range_pos",
        "high20_prior",
        "high50_prior",
        "above_high20",
        "above_high50",
    ]
    features.loc[:, [c for c in selected_feature_cols if c in features.columns]].to_csv(
        out_dir / "ep_daily_features.csv", index=False
    )
    events.to_csv(out_dir / "ep_events.csv", index=False)
    best.to_csv(out_dir / "ep_best_by_symbol.csv", index=False)
    latest.to_csv(out_dir / "ep_latest_by_symbol.csv", index=False)
    write_report(out_dir, symbols, vendor, missing, cfg, features, events, best, latest)

    print(f"Loaded {len(features):,} daily rows for {features['symbol'].nunique()} symbols from {vendor}.")
    if missing:
        print(f"Missing symbols: {', '.join(missing)}")
    print(f"Detected {len(events):,} EP events.")
    print(f"Wrote {out_dir / 'ep_scan_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
