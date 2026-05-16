#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scan_episodic_pivots import BREAKOUT_ROOT, LOCAL_DAILY_PATH, EpisodicPivotConfig
from scan_r3000_2025_episodic_pivots import LANE_NAMES, add_numba_features
from validate_ep_entry_rules_2025 import EntryRuleConfig, add_entry_features, apply_filter


UNIVERSE_PATH = BREAKOUT_ROOT / "analysis_outputs" / "russell3000_full_dataset" / "universe.parquet"
DEFAULT_OUT_DIR = BREAKOUT_ROOT / "analysis_outputs" / "episodic_pivot_daily_entry_signals_20260504_20260508"
DEFAULT_SIGNAL_START = "2026-05-04"
DEFAULT_SIGNAL_END = "2026-05-09"
DEFAULT_FEATURE_START = "2025-01-01"
DEFAULT_RECENT_START = "2026-04-25"
DATA_VENDOR = "local Russell 3000 daily_history.parquet + yfinance recent daily bars"


@dataclass(frozen=True)
class DailyEntrySignalConfig:
    signal_start: str = DEFAULT_SIGNAL_START
    signal_end: str = DEFAULT_SIGNAL_END
    feature_start: str = DEFAULT_FEATURE_START
    recent_start: str = DEFAULT_RECENT_START
    watch_days: int = 20
    pullback_ma_buffer: float = 0.03
    min_price: float = 5.0
    min_avg_dollar_volume_20: float = 5_000_000.0
    min_market_cap: float = 300_000_000.0
    chunk_size: int = 100
    max_charts: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan daily EP 10MA-hold entry signals and render charts.")
    parser.add_argument("--signal-start", default=DEFAULT_SIGNAL_START, help="Inclusive entry signal start date.")
    parser.add_argument("--signal-end", default=DEFAULT_SIGNAL_END, help="Exclusive entry signal end date.")
    parser.add_argument("--feature-start", default=DEFAULT_FEATURE_START)
    parser.add_argument("--recent-start", default=DEFAULT_RECENT_START, help="Recent yfinance fetch start date.")
    parser.add_argument("--watch-days", type=int, default=20)
    parser.add_argument("--pullback-ma-buffer", type=float, default=0.03)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-avg-dollar-volume-20", type=float, default=5_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=300_000_000.0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-charts", type=int, default=0, help="0 means render all final-tradable charts.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional symbol subset for faster debugging.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--skip-yfinance", action="store_true", help="Use local data only.")
    return parser.parse_args()


def pct(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{100.0 * number:.1f}%"


def money(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    return f"${number:,.0f}"


def price(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.2f}"


def clean_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_universe(symbols: list[str] | None = None) -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Universe file not found: {UNIVERSE_PATH}")
    universe = pd.read_parquet(UNIVERSE_PATH)
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    if "yahoo_symbol" not in universe.columns:
        universe["yahoo_symbol"] = universe["symbol"]
    universe["yahoo_symbol"] = universe["yahoo_symbol"].astype(str).str.upper()
    if symbols:
        wanted = {s.upper().lstrip("$") for s in symbols}
        universe = universe.loc[universe["symbol"].isin(wanted)].copy()
    keep = [
        "symbol",
        "yahoo_symbol",
        "exchange",
        "company_name",
        "market_cap",
        "sector",
        "industry",
        "country",
        "rank_market_cap",
    ]
    return universe.loc[:, [col for col in keep if col in universe.columns]].drop_duplicates("symbol")


def load_local_daily(feature_start: str, symbols: set[str] | None = None) -> pd.DataFrame:
    if not LOCAL_DAILY_PATH.exists():
        raise FileNotFoundError(f"Daily file not found: {LOCAL_DAILY_PATH}")
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    daily = pd.read_parquet(LOCAL_DAILY_PATH, columns=cols)
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    daily = daily.loc[daily["date"] >= pd.Timestamp(feature_start), cols]
    if symbols:
        daily = daily.loc[daily["symbol"].isin(symbols)]
    return daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def extract_yfinance_symbol_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            one = raw[ticker].copy()
        elif ticker in raw.columns.get_level_values(1):
            one = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        one = raw.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in one.columns for col in required):
        return pd.DataFrame()
    one = one.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return one[["date", "open", "high", "low", "close", "volume"]]


def fetch_recent_yfinance(universe: pd.DataFrame, start: str, end: str, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for recent-date scans.") from exc

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    mapping = dict(zip(universe["yahoo_symbol"].astype(str), universe["symbol"].astype(str)))
    tickers = list(mapping.keys())
    for offset in range(0, len(tickers), chunk_size):
        chunk = tickers[offset : offset + chunk_size]
        label = f"{offset + 1}-{offset + len(chunk)}"
        print(f"Fetching yfinance batch {label} / {len(tickers)}")
        try:
            raw = yf.download(
                " ".join(chunk),
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            for ticker in chunk:
                status_rows.append({"yahoo_symbol": ticker, "symbol": mapping[ticker], "status": "download_error", "message": str(exc)})
            continue
        for ticker in chunk:
            one = extract_yfinance_symbol_frame(raw, ticker)
            if one.empty:
                status_rows.append({"yahoo_symbol": ticker, "symbol": mapping[ticker], "status": "empty", "message": ""})
                continue
            one["symbol"] = mapping[ticker]
            rows.append(one[["symbol", "date", "open", "high", "low", "close", "volume"]])
            status_rows.append({"yahoo_symbol": ticker, "symbol": mapping[ticker], "status": "ok", "message": "", "rows": len(one)})
    if rows:
        daily = pd.concat(rows, ignore_index=True)
        daily["symbol"] = daily["symbol"].astype(str).str.upper()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
        daily = daily.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    else:
        daily = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    status = pd.DataFrame(status_rows)
    return daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True), status


def combine_daily(local_daily: pd.DataFrame, recent_daily: pd.DataFrame) -> pd.DataFrame:
    if recent_daily.empty:
        combined = local_daily.copy()
    else:
        combined = pd.concat([local_daily, recent_daily], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    combined = combined.sort_values(["symbol", "date"], kind="mergesort")
    combined = combined.drop_duplicates(["symbol", "date"], keep="last")
    return combined.reset_index(drop=True)


def event_rows(features: pd.DataFrame, signal_end: str) -> pd.DataFrame:
    mask = (features["lane_code"].to_numpy() > 0) & (features["date"].to_numpy() < np.datetime64(signal_end))
    idx = np.flatnonzero(mask).astype(np.int64)
    events = features.iloc[idx].copy()
    events["event_index"] = idx
    events["ep_lane"] = [LANE_NAMES[int(code)] for code in events["lane_code"]]
    return events


def first_10ma_hold_entry(
    features: pd.DataFrame,
    event: pd.Series,
    group_end_by_row: np.ndarray,
    cfg: DailyEntrySignalConfig,
) -> dict[str, Any] | None:
    ep_idx = int(event["event_index"])
    end = int(group_end_by_row[ep_idx])
    stop = min(end, ep_idx + 1 + cfg.watch_days)
    ep_low = float(event["low"])
    ep_close = float(event["close"])
    for idx in range(ep_idx + 1, stop):
        row = features.iloc[idx]
        dma10 = float(row["dma10"]) if pd.notna(row["dma10"]) else np.nan
        if not np.isfinite(dma10):
            continue
        close = float(row["close"])
        open_ = float(row["open"])
        low = float(row["low"])
        pullback10 = (
            low <= dma10 * (1.0 + cfg.pullback_ma_buffer)
            and close >= dma10
            and close > open_
            and close > ep_low
            and close >= ep_close * 0.80
        )
        if pullback10:
            return {"entry_index": idx, "entry_trigger": "pullback_10ma_hold"}
    return None


def build_entry_signals(
    features: pd.DataFrame,
    events: pd.DataFrame,
    group_end_by_row: np.ndarray,
    universe: pd.DataFrame,
    cfg: DailyEntrySignalConfig,
) -> pd.DataFrame:
    start = np.datetime64(cfg.signal_start)
    end = np.datetime64(cfg.signal_end)
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        entry = first_10ma_hold_entry(features, event, group_end_by_row, cfg)
        if entry is None:
            continue
        entry_idx = int(entry["entry_index"])
        entry_row = features.iloc[entry_idx]
        entry_date = np.datetime64(entry_row["date"])
        if not (start <= entry_date < end):
            continue
        rows.append(
            {
                "symbol": event["symbol"],
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
                "entry_date": entry_row["date"],
                "entry_trigger": entry["entry_trigger"],
                "entry_index": entry_idx,
                "event_index": int(event["event_index"]),
                "entry_days_from_ep": entry_idx - int(event["event_index"]),
                "entry_open": entry_row["open"],
                "entry_high": entry_row["high"],
                "entry_low": entry_row["low"],
                "entry_close": entry_row["close"],
                "entry_dma10": entry_row["dma10"],
                "entry_dma20": entry_row["dma20"],
                "entry_avg_dollar_volume_20": entry_row["avg_dollar_volume_20"],
                "entry_low_vs_dma10_pct": entry_row["low"] / entry_row["dma10"] - 1.0,
                "entry_close_vs_dma10_pct": entry_row["close"] / entry_row["dma10"] - 1.0,
            }
        )
    signals = pd.DataFrame(rows)
    if signals.empty:
        return signals
    meta_cols = [
        col
        for col in [
            "symbol",
            "yahoo_symbol",
            "exchange",
            "company_name",
            "market_cap",
            "sector",
            "industry",
            "country",
            "rank_market_cap",
        ]
        if col in universe.columns
    ]
    signals = signals.merge(universe[meta_cols], on="symbol", how="left")
    signals["entry_mode"] = "pullback_10ma_hold"
    entry_cfg = EntryRuleConfig(
        signal_start=cfg.signal_start,
        signal_end=cfg.signal_end,
        feature_start=cfg.feature_start,
        watch_days=cfg.watch_days,
        pullback_ma_buffer=cfg.pullback_ma_buffer,
        min_price=cfg.min_price,
        min_avg_dollar_volume_20=cfg.min_avg_dollar_volume_20,
        min_market_cap=cfg.min_market_cap,
    )
    signals["entry_price"] = signals["entry_close"]
    signals["tradable_filter_pass"] = apply_filter(signals, "final_tradable", entry_cfg).to_numpy(dtype=bool)
    signals = signals.sort_values(["entry_date", "ep_score", "symbol"], ascending=[True, False, True], kind="mergesort")
    return signals.reset_index(drop=True)


def draw_candlestick_chart(
    features: pd.DataFrame,
    signal: pd.Series,
    chart_path: Path,
    lookback: int = 80,
    forward: int = 0,
) -> None:
    entry_idx = int(signal["entry_index"])
    event_idx = int(signal["event_index"])
    symbol = str(signal["symbol"])
    start = max(0, min(event_idx - 20, entry_idx - lookback))
    end = min(len(features), entry_idx + forward + 1)
    symbol_frame = features.loc[
        (features["symbol"].eq(symbol))
        & (features.index >= start)
        & (features.index < end)
    ].copy()
    if symbol_frame.empty:
        return
    x = mdates.date2num(pd.to_datetime(symbol_frame["date"]).dt.to_pydatetime())
    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    up_color = "#0f9d58"
    down_color = "#d93025"
    width = 0.62
    for xi, row in zip(x, symbol_frame.itertuples(index=False)):
        color = up_color if row.close >= row.open else down_color
        ax.vlines(xi, row.low, row.high, color=color, linewidth=1.1, alpha=0.95)
        lower = min(row.open, row.close)
        height = abs(row.close - row.open)
        if height <= 0:
            height = max(row.high - row.low, row.close * 0.002, 0.01)
            lower = row.close - height / 2.0
        ax.add_patch(plt.Rectangle((xi - width / 2.0, lower), width, height, color=color, alpha=0.85))
    ax.plot(x, symbol_frame["dma10"], color="#1a73e8", linewidth=1.4, label="10MA")
    ax.plot(x, symbol_frame["dma20"], color="#fbbc04", linewidth=1.2, label="20MA")
    ax.axhline(float(signal["ep_high"]), color="#7e57c2", linestyle="--", linewidth=1.0, label="EP high")
    ax.axhline(float(signal["ep_low"]), color="#5f6368", linestyle=":", linewidth=1.0, label="EP low")
    ep_date = mdates.date2num(pd.Timestamp(signal["ep_date"]).to_pydatetime())
    entry_date = mdates.date2num(pd.Timestamp(signal["entry_date"]).to_pydatetime())
    ax.axvline(ep_date, color="#7e57c2", linewidth=1.0, alpha=0.75)
    ax.axvline(entry_date, color="#0b8043", linewidth=1.2, alpha=0.95)
    ax.scatter([entry_date], [float(signal["entry_close"])], marker="^", s=90, color="#0b8043", zorder=5, label="Entry close")
    title = (
        f"{symbol} 10MA hold entry | entry {pd.Timestamp(signal['entry_date']).date()} "
        f"| EP {pd.Timestamp(signal['ep_date']).date()} {signal['ep_lane']}"
    )
    company = str(signal.get("company_name", "") or "")
    if company:
        title += f" | {company[:42]}"
    ax.set_title(title, loc="left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", ncols=5, fontsize=8, frameon=False)
    volume_colors = [up_color if c >= o else down_color for o, c in zip(symbol_frame["open"], symbol_frame["close"])]
    axv.bar(x, symbol_frame["volume"], color=volume_colors, alpha=0.55, width=width)
    axv.grid(True, axis="y", alpha=0.15)
    axv.set_ylabel("Volume")
    axv.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    axv.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=145)
    plt.close(fig)


def render_charts(features: pd.DataFrame, signals: pd.DataFrame, out_dir: Path, max_charts: int) -> pd.DataFrame:
    if signals.empty:
        signals["chart_path"] = []
        return signals
    chart_dir = out_dir / "charts"
    render_rows = signals.copy()
    if max_charts > 0:
        render_rows = render_rows.head(max_charts).copy()
    chart_paths: dict[int, str] = {}
    for idx, signal in render_rows.iterrows():
        filename = clean_filename(
            f"{signal['symbol']}_{pd.Timestamp(signal['ep_date']).date()}_{pd.Timestamp(signal['entry_date']).date()}.png"
        )
        path = chart_dir / filename
        draw_candlestick_chart(features, signal, path)
        chart_paths[idx] = str(path)
    out = signals.copy()
    out["chart_path"] = [chart_paths.get(idx, "") for idx in out.index]
    return out


def write_report(out_dir: Path, cfg: DailyEntrySignalConfig, signals: pd.DataFrame, elapsed: float, yfinance_status: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = signals.loc[signals["tradable_filter_pass"]].copy() if not signals.empty else pd.DataFrame()
    lines: list[str] = []
    lines.append("# EP Daily 10MA Entry Signals")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed:.2f} seconds")
    lines.append(f"- Data: {DATA_VENDOR}")
    lines.append(f"- Entry signal window: {cfg.signal_start} inclusive to {cfg.signal_end} exclusive")
    lines.append(f"- Watch window: {cfg.watch_days} trading days after the EP day")
    lines.append("- Entry rule: daily close entry on the first post-EP `pullback_10ma_hold` trigger.")
    lines.append(
        f"- Tradable filter: price >= ${cfg.min_price:.2f}, EP prior 20-day dollar volume >= ${cfg.min_avg_dollar_volume_20:,.0f}, "
        f"market cap >= ${cfg.min_market_cap:,.0f}, sector known, exclude Biotechnology."
    )
    if not yfinance_status.empty:
        ok = int(yfinance_status["status"].eq("ok").sum())
        lines.append(f"- yfinance recent fetch: {ok:,}/{len(yfinance_status):,} tickers returned rows")
    lines.append("")
    lines.append("## Systemized Entry Logic")
    lines.append("")
    lines.append("1. Detect EP with the three-lane EP classifier (`a_plus_gap_ep`, `gap_ep`, `displacement_ep`).")
    lines.append(f"2. Put the stock on the watchlist for {cfg.watch_days} trading days after the EP day.")
    lines.append("3. Search only after the EP day. The first valid 10MA-hold day is the only signal for that EP event.")
    lines.append(f"4. Trigger when low <= 10MA x {1.0 + cfg.pullback_ma_buffer:.2f}, close >= 10MA, close > open, close > EP low, and close >= 80% of EP close.")
    lines.append("5. Enter at the trigger day's close; manage with EP low as disaster stop and EP-anchored VWAP close as trend-health stop in the next validation layer.")
    lines.append("")
    lines.append("## Signal Counts")
    lines.append("")
    lines.append(f"- Raw entry signals: {len(signals):,}")
    lines.append(f"- Final tradable signals: {len(final):,}")
    lines.append(f"- Final tradable symbols: {final['symbol'].nunique() if not final.empty else 0:,}")
    lines.append("")
    lines.append("## Final Tradable Signals")
    lines.append("")
    if final.empty:
        lines.append("No final-tradable signals were found in this window.")
    else:
        lines.append("| Symbol | Entry | EP date | Lane | Close | 10MA | EP gap | EP RVOL | MCap | Sector | Industry | Chart |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|")
        for row in final.itertuples(index=False):
            chart = Path(row.chart_path).name if getattr(row, "chart_path", "") else ""
            chart_link = f"[chart](charts/{chart})" if chart else ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.symbol),
                        str(pd.Timestamp(row.entry_date).date()),
                        str(pd.Timestamp(row.ep_date).date()),
                        str(row.ep_lane),
                        price(row.entry_close),
                        price(row.entry_dma10),
                        pct(row.ep_gap_pct),
                        price(row.ep_volume_ratio_20),
                        money(row.market_cap),
                        str(row.sector),
                        str(row.industry)[:34],
                        chart_link,
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## All Raw Signals")
    lines.append("")
    if signals.empty:
        lines.append("No raw signals were found.")
    else:
        lines.append("| Symbol | Entry | EP date | Lane | Close | 10MA | Tradable | Sector | Industry |")
        lines.append("|---|---:|---:|---|---:|---:|---|---|---|")
        for row in signals.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.symbol),
                        str(pd.Timestamp(row.entry_date).date()),
                        str(pd.Timestamp(row.ep_date).date()),
                        str(row.ep_lane),
                        price(row.entry_close),
                        price(row.entry_dma10),
                        "yes" if bool(row.tradable_filter_pass) else "no",
                        str(row.sector),
                        str(row.industry)[:34],
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
    (out_dir / "entry_signal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    cfg = DailyEntrySignalConfig(
        signal_start=args.signal_start,
        signal_end=args.signal_end,
        feature_start=args.feature_start,
        recent_start=args.recent_start,
        watch_days=args.watch_days,
        pullback_ma_buffer=args.pullback_ma_buffer,
        min_price=args.min_price,
        min_avg_dollar_volume_20=args.min_avg_dollar_volume_20,
        min_market_cap=args.min_market_cap,
        chunk_size=args.chunk_size,
        max_charts=args.max_charts,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = load_universe(args.symbols)
    if universe.empty:
        raise RuntimeError("No symbols found in universe.")
    symbols = set(universe["symbol"].astype(str).str.upper())
    local_daily = load_local_daily(cfg.feature_start, symbols)
    if args.skip_yfinance:
        recent_daily = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
        yfinance_status = pd.DataFrame()
    else:
        recent_daily, yfinance_status = fetch_recent_yfinance(universe, cfg.recent_start, cfg.signal_end, cfg.chunk_size)
    combined = combine_daily(local_daily, recent_daily)
    recent_daily.to_csv(out_dir / "recent_yfinance_daily.csv", index=False)
    yfinance_status.to_csv(out_dir / "yfinance_fetch_status.csv", index=False)
    combined.to_parquet(out_dir / "combined_daily_features_input.parquet", index=False)

    ep_cfg = EpisodicPivotConfig(start=cfg.feature_start)
    features, codes, group_end_by_row = add_numba_features(combined, ep_cfg)
    features = add_entry_features(features, codes)
    events = event_rows(features, cfg.signal_end)
    signals = build_entry_signals(features, events, group_end_by_row, universe, cfg)
    if "tradable_filter_pass" not in signals.columns:
        signals["tradable_filter_pass"] = pd.Series(dtype=bool)
    chart_base = signals.loc[signals["tradable_filter_pass"]].copy() if not signals.empty else signals
    charted = render_charts(features, chart_base, out_dir, cfg.max_charts)
    if not signals.empty and not charted.empty:
        signals = signals.merge(
            charted[["symbol", "ep_date", "entry_date", "chart_path"]],
            on=["symbol", "ep_date", "entry_date"],
            how="left",
        )
        signals["chart_path"] = signals["chart_path"].fillna("")
    elif not signals.empty:
        signals["chart_path"] = ""
    else:
        signals["chart_path"] = []

    signals.to_csv(out_dir / "entry_signals_all.csv", index=False)
    signals.loc[signals["tradable_filter_pass"].fillna(False)].to_csv(out_dir / "entry_signals_final_tradable.csv", index=False)
    events.to_csv(out_dir / "ep_events_to_signal_end.csv", index=False)

    elapsed = time.perf_counter() - started
    write_report(out_dir, cfg, signals, elapsed, yfinance_status)
    print(f"Raw entry signals: {len(signals):,}")
    print(f"Final tradable signals: {int(signals['tradable_filter_pass'].sum()) if not signals.empty else 0:,}")
    print(f"Wrote: {out_dir / 'entry_signal_report.md'}")
    print(f"Runtime: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
