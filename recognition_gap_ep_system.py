#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BREAKOUT_ROOT = Path(__file__).parent
DEFAULT_OUT_DIR = BREAKOUT_ROOT / "analysis_outputs" / "recognition_gap_ep_system_20260515"
DEFAULT_POLICY_RANKING = BREAKOUT_ROOT / "analysis_outputs" / "policy_theme_ep_rank_20260510" / "policy_theme_ep_ranking.csv"
DEFAULT_LITE_CANDIDATES = (
    BREAKOUT_ROOT
    / "analysis_outputs"
    / "lite_before_surge_candidates_20260510"
    / "lite_before_surge_candidates.csv"
)
DEFAULT_EP_SIGNALS = (
    BREAKOUT_ROOT
    / "analysis_outputs"
    / "episodic_pivot_daily_entry_signals_20260504_20260508"
    / "entry_signals_final_tradable.csv"
)
DEFAULT_DAILY_FEATURES = (
    BREAKOUT_ROOT
    / "analysis_outputs"
    / "episodic_pivot_daily_entry_signals_20260504_20260508"
    / "combined_daily_features_input.parquet"
)
DEFAULT_RESEARCH_NOTES = BREAKOUT_ROOT / "configs" / "recognition_gap_research_notes.csv"
DEFAULT_EARNINGS_CALENDAR = DEFAULT_OUT_DIR / "fmp_earnings_calendar.csv"
FMP_BASE = "https://financialmodelingprep.com/api/v3"


@dataclass(frozen=True)
class RecognitionGapConfig:
    max_candidates: int = 180
    min_loop_iters: int = 2
    max_loop_iters: int = 6
    halt_delta: float = 1.25
    halt_uncertainty: float = 0.28
    top_n: int = 50


THEME_WEIGHTS = {
    "semiconductor_ai": 18.0,
    "data_center_electrification": 17.0,
    "nuclear_power_grid": 16.0,
    "space_defense": 15.0,
    "critical_minerals": 15.0,
    "clean_energy_ev": 10.0,
    "energy_dominance": 9.0,
}

NEW_REALITY_KEYWORDS = {
    "ai infrastructure": 9,
    "data center": 8,
    "datacenter": 8,
    "hyperscal": 8,
    "800g": 8,
    "1.6t": 9,
    "ethernet": 5,
    "optical": 8,
    "photonic": 8,
    "transceiver": 8,
    "silicon photon": 9,
    "interconnect": 7,
    "advanced packag": 8,
    "chiplet": 8,
    "substrate": 7,
    "printed circuit": 8,
    "pcb": 8,
    "high density": 8,
    "backplane": 7,
    "power management": 7,
    "rack power": 8,
    "liquid cooling": 7,
    "thermal": 6,
    "montitan": 9,
    "enterprise ssd": 8,
    "boot drive": 7,
    "bluefield": 9,
    "dpu": 7,
    "nvidia": 8,
    "book-to-bill": 9,
    "book to bill": 9,
    "backlog": 8,
    "booking": 7,
    "order": 5,
    "design win": 8,
    "visibility": 7,
    "record revenue": 6,
    "capacity expansion": 6,
    "lead time": 6,
}

OLD_PERCEPTION_HINTS = {
    "Hardware, Equipment & Parts": "legacy hardware/equipment supplier",
    "Communication Equipment": "cyclical communication equipment vendor",
    "Electrical Equipment & Parts": "industrial electrical component supplier",
    "Semiconductors": "cyclical semiconductor vendor",
    "Semiconductor Equipment & Materials": "semicap cycle supplier",
    "Industrial Materials": "commodity/materials supplier",
    "Computer Hardware": "hardware cycle vendor",
    "Software - Infrastructure": "software infrastructure vendor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognition Gap EP System with adaptive loop reasoning.")
    parser.add_argument("--policy-ranking", default=str(DEFAULT_POLICY_RANKING))
    parser.add_argument("--lite-candidates", default=str(DEFAULT_LITE_CANDIDATES))
    parser.add_argument("--ep-signals", default=str(DEFAULT_EP_SIGNALS))
    parser.add_argument("--daily-features", default=str(DEFAULT_DAILY_FEATURES))
    parser.add_argument("--earnings-calendar", default=str(DEFAULT_EARNINGS_CALENDAR))
    parser.add_argument("--skip-fmp-earnings", action="store_true")
    parser.add_argument("--require-ep-entry", action="store_true")
    parser.add_argument("--allow-no-ep", action="store_true")
    parser.add_argument("--allow-missing-earnings-anchor", action="store_true")
    parser.add_argument("--research-notes", default=str(DEFAULT_RESEARCH_NOTES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-candidates", type=int, default=180)
    parser.add_argument("--min-loop-iters", type=int, default=2)
    parser.add_argument("--max-loop-iters", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def clamp(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return low
    return max(low, min(high, value))


def pct(value: Any) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return ""
    return f"{100.0 * number:.1f}%"


def price(value: Any) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.2f}"


def money(value: Any) -> str:
    number = safe_float(value)
    if not np.isfinite(number):
        return ""
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    return f"${number:,.0f}"


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def normalize_text(value: Any) -> str:
    return clean_str(value).lower()


def parse_titles(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_str(item) for item in value if clean_str(item)]
    if not clean_str(value):
        return []
    text = clean_str(value)
    if not text:
        return []
    if "||" in text:
        return [part.strip() for part in text.split("||") if part.strip()]
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            return [text]
    return [text]


def parse_date(value: Any) -> pd.Timestamp | None:
    text = clean_str(value)
    if not text:
        return None
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def read_secret_from_env_file(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    env_path = BREAKOUT_ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, raw_value = text.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return ""


def keyword_score(text: str, keywords: dict[str, int]) -> tuple[float, list[str]]:
    total = 0.0
    hits: list[str] = []
    for key, points in keywords.items():
        if key in text:
            total += points
            hits.append(key)
    return total, hits


def load_policy(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    rename = {
        "total_score": "policy_total_score",
        "action_bucket": "policy_action",
        "news_titles": "policy_news_titles",
    }
    df = df.rename(columns=rename)
    keep = [
        "symbol",
        "latest_date",
        "close",
        "dma10",
        "dma20",
        "return_20d_pct",
        "close_vs_10ma_pct",
        "close_vs_20ma_pct",
        "recent_ep_date",
        "recent_ep_lane",
        "recent_ep_score",
        "company_name",
        "exchange",
        "market_cap",
        "sector",
        "industry",
        "policy_themes",
        "policy_total_score",
        "policy_action",
        "policy_news_hits",
        "policy_news_titles",
        "earnings_score",
        "latest_earnings_date",
        "eps_surprise_pct",
        "revenue_yoy_pct",
    ]
    return df[[col for col in keep if col in df.columns]].drop_duplicates("symbol")


def load_lite(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    rename = {
        "lite_score": "lite_score",
        "action": "lite_action",
        "rank": "lite_rank",
        "news_titles": "lite_news_titles",
        "news_hits": "lite_news_hits",
    }
    df = df.rename(columns=rename)
    keep = [
        "symbol",
        "company_name",
        "exchange",
        "market_cap",
        "sector",
        "industry",
        "latest_date",
        "close",
        "dma10",
        "dma20",
        "return_20d_pct",
        "close_vs_10ma_pct",
        "rev_yoy",
        "rev_qoq",
        "rev_accelerating",
        "ttm_revenue",
        "latest_gross_margin",
        "latest_net_income",
        "ps_ratio_ttm",
        "pe_ratio_ttm",
        "eps_surprise_pct",
        "earnings_date",
        "lite_news_titles",
        "lite_news_hits",
        "ai_infra_pts",
        "ai_kw",
        "lite_score",
        "lite_action",
        "lite_rank",
    ]
    return df[[col for col in keep if col in df.columns]].drop_duplicates("symbol")


def load_ep(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.sort_values(["symbol", "ep_score"], ascending=[True, False], kind="mergesort")
    df = df.drop_duplicates("symbol", keep="first")
    rename = {
        "ep_date": "entry_ep_date",
        "ep_lane": "entry_ep_lane",
        "ep_score": "entry_ep_score",
        "entry_date": "ep_entry_date",
        "entry_close": "ep_entry_close",
        "entry_dma10": "ep_entry_dma10",
        "entry_close_vs_dma10_pct": "ep_entry_close_vs_dma10_pct",
    }
    df = df.rename(columns=rename)
    keep = [
        "symbol",
        "company_name",
        "exchange",
        "market_cap",
        "sector",
        "industry",
        "entry_ep_date",
        "entry_ep_lane",
        "entry_ep_score",
        "ep_entry_date",
        "entry_trigger",
        "ep_entry_close",
        "ep_entry_dma10",
        "ep_entry_close_vs_dma10_pct",
    ]
    return df[[col for col in keep if col in df.columns]]


def load_daily_features(path: Path, symbols: set[str]) -> pd.DataFrame:
    if not path.exists() or not symbols:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    daily = pd.read_parquet(path, columns=cols)
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily = daily.loc[daily["symbol"].isin(symbols)].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    return daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def fetch_fmp_earnings_calendar(api_key: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["symbol", "date"])
    url = f"{FMP_BASE}/earning_calendar"
    params = {
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "apikey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame(columns=["symbol", "date"])
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["symbol", "date"])
    df = pd.DataFrame(payload)
    if "symbol" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["symbol", "date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["symbol", "date"])
    return df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def fetch_fmp_earnings_surprises(api_key: str, symbols: set[str], limit: int = 8) -> pd.DataFrame:
    if not api_key or not symbols:
        return pd.DataFrame(columns=["symbol", "date"])
    rows: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        url = f"{FMP_BASE}/earnings-surprises/{symbol}"
        try:
            response = requests.get(url, params={"apikey": api_key, "limit": limit}, timeout=12)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            date_value = parse_date(item.get("date") if isinstance(item, dict) else None)
            if date_value is not None:
                rows.append({"symbol": symbol, "date": date_value})
    if not rows:
        return pd.DataFrame(columns=["symbol", "date"])
    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["symbol", "date"])
    return df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"], kind="mergesort")


def load_or_fetch_earnings_calendar(
    path: Path,
    candidates: pd.DataFrame,
    skip_fmp: bool,
) -> pd.DataFrame:
    anchors = [choose_anchor_date(row)[0] for _, row in candidates.iterrows()]
    anchors = [anchor for anchor in anchors if anchor is not None]
    if not anchors:
        return pd.DataFrame(columns=["symbol", "date"])
    symbols = set(candidates["symbol"].astype(str).str.upper())
    start_date = min(anchors) - pd.Timedelta(days=240)
    end_date = max(anchors) + pd.Timedelta(days=10)

    cached = pd.DataFrame(columns=["symbol", "date"])
    if path.exists():
        cached = pd.read_csv(path)
        if not cached.empty and {"symbol", "date"}.issubset(cached.columns):
            cached["symbol"] = cached["symbol"].astype(str).str.upper()
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
            cached = cached.dropna(subset=["symbol", "date"])
            cache_covers_range = cached["date"].min() <= start_date and cached["date"].max() >= end_date
            cache_has_symbols = bool(symbols.intersection(set(cached["symbol"])))
            if cache_covers_range and cache_has_symbols:
                return cached.loc[cached["symbol"].isin(symbols)].sort_values(["symbol", "date"], kind="mergesort")

    if skip_fmp:
        return cached.loc[cached["symbol"].isin(symbols)] if not cached.empty else cached

    api_key = read_secret_from_env_file("FMP_API_KEY")
    fetched = fetch_fmp_earnings_calendar(api_key, start_date, end_date)
    surprise_dates = fetch_fmp_earnings_surprises(api_key, symbols)
    if not surprise_dates.empty:
        fetched = pd.concat([fetched, surprise_dates], ignore_index=True)
        fetched = fetched.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"], kind="mergesort")
    if fetched.empty:
        return cached.loc[cached["symbol"].isin(symbols)] if not cached.empty else fetched
    path.parent.mkdir(parents=True, exist_ok=True)
    fetched.to_csv(path, index=False)
    return fetched.loc[fetched["symbol"].isin(symbols)].sort_values(["symbol", "date"], kind="mergesort")


def load_research_notes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    keep = [
        "symbol",
        "old_perception",
        "new_reality_hypothesis",
        "theme_tags",
        "evidence_keywords",
        "source_status",
        "analyst_notes",
        "exclude_reason",
    ]
    return df[[col for col in keep if col in df.columns]].drop_duplicates("symbol")


def combine_candidates(policy: pd.DataFrame, lite: pd.DataFrame, ep: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df in [policy, lite, ep]:
        if not df.empty:
            frames.append(df.copy())
    if not frames:
        return pd.DataFrame(columns=["symbol"])
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="symbol", how="outer", suffixes=("", "_dup"))
        for col in list(out.columns):
            if not col.endswith("_dup"):
                continue
            base = col[:-4]
            if base in out.columns:
                out[base] = out[base].combine_first(out[col])
                out = out.drop(columns=[col])
            else:
                out = out.rename(columns={col: base})
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["news_titles_combined"] = out.apply(
        lambda r: parse_titles(r.get("policy_news_titles")) + parse_titles(r.get("lite_news_titles")),
        axis=1,
    )
    if not notes.empty:
        out = out.merge(notes, on="symbol", how="left")
    else:
        for col in [
            "old_perception",
            "new_reality_hypothesis",
            "theme_tags",
            "evidence_keywords",
            "source_status",
            "analyst_notes",
            "exclude_reason",
        ]:
            out[col] = ""
    score_cols = ["policy_total_score", "lite_score", "entry_ep_score"]
    for col in score_cols:
        if col not in out.columns:
            out[col] = np.nan
    out["seed_score"] = (
        pd.to_numeric(out["policy_total_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["lite_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["entry_ep_score"], errors="coerce").fillna(0.0).clip(0, 160) / 2.0
    )
    out = out.sort_values("seed_score", ascending=False, kind="mergesort")
    return out.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def require_ep_entry_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    required = ["entry_ep_date", "ep_entry_date", "entry_ep_lane"]
    for col in required:
        if col not in candidates.columns:
            candidates[col] = ""
    mask = (
        candidates["entry_ep_date"].map(clean_str).ne("")
        & candidates["ep_entry_date"].map(clean_str).ne("")
        & candidates["entry_ep_lane"].map(clean_str).ne("")
    )
    return candidates.loc[mask].reset_index(drop=True)


def require_ep_event_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    for col in ["entry_ep_date", "recent_ep_date", "entry_ep_lane", "recent_ep_lane"]:
        if col not in candidates.columns:
            candidates[col] = ""
    mask = (
        (candidates["entry_ep_date"].map(clean_str).ne("") | candidates["recent_ep_date"].map(clean_str).ne(""))
        & (candidates["entry_ep_lane"].map(clean_str).ne("") | candidates["recent_ep_lane"].map(clean_str).ne(""))
    )
    return candidates.loc[mask].reset_index(drop=True)


def entry_gate(row: pd.Series) -> tuple[str, list[str]]:
    has_entry = (
        bool(clean_str(row.get("entry_ep_date")))
        and bool(clean_str(row.get("ep_entry_date")))
        and bool(clean_str(row.get("entry_ep_lane")))
    )
    v10 = safe_float(row.get("close_vs_10ma_pct"), np.nan)
    details: list[str] = []
    if has_entry:
        trigger = clean_str(row.get("entry_trigger")) or "post-EP pullback entry"
        details.append(f"entry_trigger={trigger}")
        return "entry_ready", details
    if np.isfinite(v10):
        details.append(f"vs10MA={pct(v10)}")
        if v10 > 0.12:
            return "wait_for_pullback", details
        if v10 < -0.03:
            return "wait_for_10ma_reclaim", details
        return "near_10ma_no_confirmed_entry", details
    return "wait_for_entry_signal", details


def choose_anchor_date(row: pd.Series) -> tuple[pd.Timestamp | None, str]:
    candidates = [
        ("entry_ep_date", row.get("entry_ep_date")),
        ("recent_ep_date", row.get("recent_ep_date")),
        ("latest_earnings_date", row.get("latest_earnings_date")),
        ("earnings_date", row.get("earnings_date")),
    ]
    for source, value in candidates:
        date_value = parse_date(value)
        if date_value is not None:
            return date_value, source
    return None, ""


def locate_previous_earnings_date(
    symbol: str,
    anchor_date: pd.Timestamp,
    earnings_calendar: pd.DataFrame,
) -> tuple[pd.Timestamp | None, str]:
    if earnings_calendar.empty:
        return None, ""
    rows = earnings_calendar.loc[earnings_calendar["symbol"] == symbol, ["date"]].dropna()
    if rows.empty:
        return None, ""
    dates = sorted(pd.Timestamp(value).normalize() for value in rows["date"].tolist())
    near_current = [
        date_value
        for date_value in dates
        if anchor_date - pd.Timedelta(days=10) <= date_value <= anchor_date + pd.Timedelta(days=3)
    ]
    current_event_date = max(near_current) if near_current else anchor_date
    previous = [date_value for date_value in dates if date_value < current_event_date - pd.Timedelta(days=21)]
    if not previous:
        return None, ""
    source = "actual_prev_earnings_after_current_earnings" if near_current else "actual_prev_earnings_before_ep"
    return max(previous), source


def add_anchor_recognition_features(
    candidates: pd.DataFrame,
    daily: pd.DataFrame,
    earnings_calendar: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.copy()
    defaults: dict[str, Any] = {
        "anchor_date": "",
        "anchor_source": "",
        "previous_earnings_date": "",
        "previous_earnings_source": "",
        "previous_earnings_snapshot_date": "",
        "current_snapshot_date": "",
        "previous_earnings_close": np.nan,
        "anchor_close": np.nan,
        "current_close_anchor": np.nan,
        "previous_earnings_to_pre_ep_return_pct": np.nan,
        "anchor_to_current_return_pct": np.nan,
        "previous_earnings_to_current_return_pct": np.nan,
        "previous_earnings_20d_return_pct": np.nan,
        "dollar_volume_expansion_prev_earnings_to_current": np.nan,
        "anchor_dollar_volume_expansion": np.nan,
        "current_vs_prev_earnings_to_anchor_high_pct": np.nan,
        "anchor_days_to_current": np.nan,
        "anchor_feature_available": False,
    }
    for col, value in defaults.items():
        out[col] = value
    if daily.empty or out.empty:
        return out

    grouped = {symbol: frame.reset_index(drop=True) for symbol, frame in daily.groupby("symbol", sort=False)}
    for idx, row in out.iterrows():
        symbol = clean_str(row.get("symbol")).upper()
        frame = grouped.get(symbol)
        if frame is None or frame.empty:
            continue
        anchor_date, source = choose_anchor_date(row)
        if anchor_date is None:
            continue
        previous_earnings_date, previous_source = locate_previous_earnings_date(symbol, anchor_date, earnings_calendar)
        if previous_earnings_date is None:
            continue
        dates = frame["date"].to_numpy(dtype="datetime64[ns]")
        anchor_idx = int(np.searchsorted(dates, np.datetime64(anchor_date), side="left"))
        if anchor_idx >= len(frame):
            continue
        previous_idx = int(np.searchsorted(dates, np.datetime64(previous_earnings_date), side="right"))
        if previous_idx >= len(frame) or previous_idx >= anchor_idx:
            continue
        latest_date = parse_date(row.get("latest_date"))
        if latest_date is not None:
            current_idx = int(np.searchsorted(dates, np.datetime64(latest_date), side="right") - 1)
            current_idx = min(max(current_idx, 0), len(frame) - 1)
        else:
            current_idx = len(frame) - 1
        if current_idx <= anchor_idx:
            current_idx = len(frame) - 1
        close = frame["close"].to_numpy(dtype=np.float64)
        high = frame["high"].to_numpy(dtype=np.float64)
        volume = frame["volume"].to_numpy(dtype=np.float64)
        dollar_volume = close * volume

        previous_close = close[previous_idx]
        anchor_close = close[anchor_idx]
        current_close = close[current_idx]
        current_dma10 = np.nanmean(close[max(0, current_idx - 9) : current_idx + 1])
        current_dma20 = np.nanmean(close[max(0, current_idx - 19) : current_idx + 1])
        pre_ep_idx = max(previous_idx, anchor_idx - 1)
        pre_ep_close = close[pre_ep_idx]
        return20_idx = current_idx - 20
        previous_post_end = min(anchor_idx - 1, previous_idx + 19)
        previous_lookback_idx = previous_idx - 20
        anchor_pre20_start = max(0, anchor_idx - 20)
        current20_start = max(0, current_idx - 19)
        old_high = np.nanmax(high[previous_idx:anchor_idx]) if anchor_idx > previous_idx else np.nan
        previous_post_dollar_volume = np.nanmean(dollar_volume[previous_idx : previous_post_end + 1])
        pre_ep_dollar_volume = np.nanmean(dollar_volume[anchor_pre20_start:anchor_idx])
        current20_dollar_volume = np.nanmean(dollar_volume[current20_start : current_idx + 1])
        anchor_dollar_volume = dollar_volume[anchor_idx]

        out.at[idx, "anchor_date"] = frame.at[anchor_idx, "date"].strftime("%Y-%m-%d")
        out.at[idx, "anchor_source"] = source
        out.at[idx, "previous_earnings_date"] = previous_earnings_date.strftime("%Y-%m-%d")
        out.at[idx, "previous_earnings_source"] = previous_source
        out.at[idx, "previous_earnings_snapshot_date"] = frame.at[previous_idx, "date"].strftime("%Y-%m-%d")
        out.at[idx, "current_snapshot_date"] = frame.at[current_idx, "date"].strftime("%Y-%m-%d")
        out.at[idx, "latest_date"] = frame.at[current_idx, "date"].strftime("%Y-%m-%d")
        out.at[idx, "close"] = current_close
        out.at[idx, "dma10"] = current_dma10
        out.at[idx, "dma20"] = current_dma20
        out.at[idx, "close_vs_10ma_pct"] = current_close / current_dma10 - 1.0 if current_dma10 > 0 else np.nan
        out.at[idx, "close_vs_20ma_pct"] = current_close / current_dma20 - 1.0 if current_dma20 > 0 else np.nan
        out.at[idx, "return_20d_pct"] = (
            current_close / close[return20_idx] - 1.0
            if return20_idx >= 0 and close[return20_idx] > 0
            else np.nan
        )
        out.at[idx, "previous_earnings_close"] = previous_close
        out.at[idx, "anchor_close"] = anchor_close
        out.at[idx, "current_close_anchor"] = current_close
        out.at[idx, "previous_earnings_to_pre_ep_return_pct"] = (
            pre_ep_close / previous_close - 1.0 if previous_close > 0 else np.nan
        )
        out.at[idx, "anchor_to_current_return_pct"] = current_close / anchor_close - 1.0 if anchor_close > 0 else np.nan
        out.at[idx, "previous_earnings_to_current_return_pct"] = (
            current_close / previous_close - 1.0 if previous_close > 0 else np.nan
        )
        out.at[idx, "previous_earnings_20d_return_pct"] = (
            previous_close / close[previous_lookback_idx] - 1.0
            if previous_lookback_idx >= 0 and close[previous_lookback_idx] > 0
            else np.nan
        )
        out.at[idx, "dollar_volume_expansion_prev_earnings_to_current"] = (
            current20_dollar_volume / previous_post_dollar_volume if previous_post_dollar_volume > 0 else np.nan
        )
        out.at[idx, "anchor_dollar_volume_expansion"] = (
            anchor_dollar_volume / pre_ep_dollar_volume if pre_ep_dollar_volume > 0 else np.nan
        )
        out.at[idx, "current_vs_prev_earnings_to_anchor_high_pct"] = (
            current_close / old_high - 1.0 if np.isfinite(old_high) and old_high > 0 else np.nan
        )
        out.at[idx, "anchor_days_to_current"] = float(current_idx - anchor_idx)
        out.at[idx, "anchor_feature_available"] = True
    return out


def theme_expert(row: pd.Series) -> tuple[float, list[str]]:
    themes = [part.strip() for part in clean_str(row.get("policy_themes")).split(",") if part.strip()]
    themes.extend([part.strip() for part in clean_str(row.get("theme_tags")).split(";") if part.strip()])
    score = 0.0
    evidence: list[str] = []
    for theme in themes:
        weight = THEME_WEIGHTS.get(theme, 0.0)
        if weight:
            score = max(score, weight)
            evidence.append(f"policy_theme={theme}")
    text = " ".join(
        [
            normalize_text(row.get("company_name")),
            normalize_text(row.get("sector")),
            normalize_text(row.get("industry")),
            normalize_text(row.get("new_reality_hypothesis")),
            normalize_text(row.get("evidence_keywords")),
            " ".join(normalize_text(t) for t in row.get("news_titles_combined", [])),
        ]
    )
    inferred = [
        ("data_center_electrification", ["data center", "hyperscal", "rack power", "cooling"]),
        ("semiconductor_ai", ["semiconductor", "advanced packag", "chiplet", "nvidia", "ai infrastructure"]),
        ("space_defense", ["satellite", "space", "defense", "geospatial"]),
        ("critical_minerals", ["rare earth", "lithium", "uranium", "copper", "mining"]),
        ("nuclear_power_grid", ["nuclear", "grid", "power demand", "electrical"]),
    ]
    for theme, words in inferred:
        if any(word in text for word in words):
            weight = THEME_WEIGHTS.get(theme, 0.0) - 2.0
            if weight > score:
                score = weight
            evidence.append(f"inferred_theme={theme}")
    if len(set(evidence)) >= 2:
        score += 3.0
    return clamp(score, 0.0, 22.0), list(dict.fromkeys(evidence))[:6]


def earnings_backlog_expert(row: pd.Series) -> tuple[float, list[str], float]:
    rev = safe_float(row.get("revenue_yoy_pct"), np.nan)
    if not np.isfinite(rev):
        rev = safe_float(row.get("rev_yoy"), np.nan)
    qoq = safe_float(row.get("rev_qoq"), np.nan)
    eps = safe_float(row.get("eps_surprise_pct"), np.nan)
    gm = safe_float(row.get("latest_gross_margin"), np.nan)
    net_income = safe_float(row.get("latest_net_income"), np.nan)
    text = " ".join(
        [
            " ".join(normalize_text(t) for t in row.get("news_titles_combined", [])),
            normalize_text(row.get("evidence_keywords")),
            normalize_text(row.get("new_reality_hypothesis")),
        ]
    )
    kw, hits = keyword_score(text, NEW_REALITY_KEYWORDS)
    backlog_hits = [h for h in hits if h in {"book-to-bill", "book to bill", "backlog", "booking", "order", "design win", "visibility", "capacity expansion", "lead time"}]

    score = 0.0
    evidence: list[str] = []
    uncertainty = 0.45
    if np.isfinite(rev):
        score += clamp((rev + 0.05) / 0.55 * 9.0, 0.0, 11.0)
        evidence.append(f"rev_yoy={pct(rev)}")
        uncertainty -= 0.08
    if np.isfinite(qoq) and qoq > 0.03:
        score += clamp(qoq / 0.18 * 4.0, 0.0, 4.0)
        evidence.append(f"rev_qoq={pct(qoq)}")
        uncertainty -= 0.04
    if bool(row.get("rev_accelerating")):
        score += 3.0
        evidence.append("three-quarter sequential revenue acceleration")
        uncertainty -= 0.04
    if np.isfinite(eps):
        score += clamp((eps + 0.05) / 0.35 * 4.0, 0.0, 5.0)
        evidence.append(f"eps_surprise={pct(eps)}")
        uncertainty -= 0.03
    if np.isfinite(gm) and gm >= 0.35:
        score += 1.5
        evidence.append(f"gross_margin={pct(gm)}")
    if np.isfinite(net_income) and net_income > 0:
        score += 1.5
        evidence.append("profitable latest quarter")
    if backlog_hits:
        score += min(6.0, 2.0 * len(set(backlog_hits)))
        evidence.append("backlog/bookings language: " + ", ".join(sorted(set(backlog_hits))[:4]))
        uncertainty -= 0.08
    elif kw > 0:
        score += min(3.0, kw / 10.0)
    return clamp(score, 0.0, 26.0), evidence[:8], clamp(uncertainty, 0.15, 0.60)


def recognition_gap_expert(row: pd.Series) -> tuple[float, list[str], list[str], float]:
    industry = clean_str(row.get("industry"))
    note_old = clean_str(row.get("old_perception"))
    old = note_old or OLD_PERCEPTION_HINTS.get(industry, industry or "unclear legacy classification")
    hypothesis = clean_str(row.get("new_reality_hypothesis"))
    source_status = clean_str(row.get("source_status")).lower()
    titles = row.get("news_titles_combined", [])
    text = " ".join(
        [
            normalize_text(row.get("company_name")),
            normalize_text(industry),
            normalize_text(row.get("ai_kw")),
            normalize_text(hypothesis),
            normalize_text(row.get("evidence_keywords")),
            normalize_text(row.get("analyst_notes")),
            " ".join(normalize_text(t) for t in titles),
        ]
    )
    kw, hits = keyword_score(text, NEW_REALITY_KEYWORDS)
    ps = safe_float(row.get("ps_ratio_ttm"), np.nan)
    market_cap = safe_float(row.get("market_cap"), np.nan)
    ret20 = safe_float(row.get("return_20d_pct"), np.nan)
    rev = safe_float(row.get("revenue_yoy_pct"), np.nan)
    if not np.isfinite(rev):
        rev = safe_float(row.get("rev_yoy"), np.nan)

    evidence: list[str] = []
    contradictions: list[str] = []
    if hits:
        evidence.append("new_reality_keywords=" + ", ".join(hits[:7]))
    if hypothesis:
        evidence.append("hypothesis=" + hypothesis)
    evidence.append(f"old_perception={old}")

    recognition_discount = 0.0
    if np.isfinite(ps):
        if ps <= 4.0:
            recognition_discount += 7.0
            evidence.append(f"still low P/S={ps:.1f}x")
        elif ps <= 8.0 and np.isfinite(rev) and rev >= 0.25:
            recognition_discount += 5.0
            evidence.append(f"growth-adjusted P/S still acceptable={ps:.1f}x")
        elif ps > 12.0:
            contradictions.append(f"market may already recognize story: P/S={ps:.1f}x")
    if np.isfinite(market_cap):
        if market_cap <= 5_000_000_000:
            recognition_discount += 4.0
            evidence.append("small enough for institutional rerating")
        elif market_cap >= 50_000_000_000:
            contradictions.append("large-cap limits multibagger asymmetry")
    if np.isfinite(ret20):
        if ret20 > 1.0:
            contradictions.append(f"very large 20d move={pct(ret20)}")
        elif 0.20 <= ret20 <= 0.75:
            evidence.append("price has begun to reprice, but not parabolic")

    score = min(18.0, kw / 3.0) + recognition_discount
    if not hits:
        score *= 0.55
        contradictions.append("weak explicit new-reality language")
    uncertainty = 0.50 - min(0.20, len(hits) * 0.025) + min(0.18, len(contradictions) * 0.045)
    if source_status == "partially_verified":
        score += 3.0
        evidence.append("research_note_status=partially_verified")
        uncertainty -= 0.05
    elif source_status == "needs_primary_source":
        score += 2.0
        evidence.append("research_note_status=needs_primary_source")
        contradictions.append("primary-source verification required")
        uncertainty += 0.08
    elif source_status == "rejected":
        score -= 12.0
        contradictions.append("research note rejects theme fit")
        uncertainty += 0.12
    return clamp(score, 0.0, 30.0), evidence[:9], contradictions[:6], clamp(uncertainty, 0.18, 0.70)


def anchor_delta_expert(row: pd.Series) -> tuple[float, list[str], list[str], float]:
    if not bool(row.get("anchor_feature_available")):
        return 0.0, [], ["previous-earnings anchor comparison unavailable"], 0.55

    anchor_source = clean_str(row.get("anchor_source"))
    anchor_date = clean_str(row.get("anchor_date"))
    previous_earnings_date = clean_str(row.get("previous_earnings_date"))
    previous_earnings_source = clean_str(row.get("previous_earnings_source"))
    previous_snapshot_date = clean_str(row.get("previous_earnings_snapshot_date"))
    current_date = clean_str(row.get("current_snapshot_date"))
    pre_run = safe_float(row.get("previous_earnings_to_pre_ep_return_pct"), np.nan)
    post_run = safe_float(row.get("anchor_to_current_return_pct"), np.nan)
    full_run = safe_float(row.get("previous_earnings_to_current_return_pct"), np.nan)
    pre20 = safe_float(row.get("previous_earnings_20d_return_pct"), np.nan)
    dollar_expansion = safe_float(row.get("dollar_volume_expansion_prev_earnings_to_current"), np.nan)
    anchor_volume_expansion = safe_float(row.get("anchor_dollar_volume_expansion"), np.nan)
    old_high_break = safe_float(row.get("current_vs_prev_earnings_to_anchor_high_pct"), np.nan)
    days_to_current = safe_float(row.get("anchor_days_to_current"), np.nan)
    v10 = safe_float(row.get("close_vs_10ma_pct"), np.nan)

    evidence: list[str] = [
        f"previous_earnings={previous_earnings_date}",
        f"previous_earnings_snapshot={previous_snapshot_date}",
        f"anchor={anchor_date} via {anchor_source}",
        f"current_snapshot={current_date}",
    ]
    if previous_earnings_source:
        evidence.append(f"previous_earnings_source={previous_earnings_source}")
    contradictions: list[str] = []

    pre_awareness = 0.0
    if np.isfinite(pre_run):
        evidence.append(f"previous_earnings_to_pre_EP={pct(pre_run)}")
        if pre_run >= 0.75:
            pre_awareness += 9.0
            contradictions.append("large pre-EP run suggests story was already recognized")
        elif pre_run >= 0.35:
            pre_awareness += 5.0
            contradictions.append("meaningful pre-EP run before the catalyst")
        elif pre_run <= 0.15:
            evidence.append("muted pre-EP drift before catalyst")
    if np.isfinite(pre20) and pre20 >= 0.25:
        pre_awareness += 2.0
        contradictions.append("previous-quarter snapshot was already in momentum")

    post_recognition = 0.0
    if np.isfinite(post_run):
        evidence.append(f"anchor_to_current={pct(post_run)}")
        if 0.08 <= post_run <= 0.75:
            post_recognition += 4.0
            evidence.append("post-anchor repricing is strong but not exhausted")
        elif post_run > 1.00:
            post_recognition += 3.0
            contradictions.append("post-anchor move may already be crowded")
        elif post_run < -0.05:
            contradictions.append("post-anchor price failed to confirm")
    if np.isfinite(full_run):
        evidence.append(f"previous_earnings_to_current={pct(full_run)}")
        if full_run >= 0.50:
            post_recognition += 4.0
        elif full_run >= 0.25:
            post_recognition += 2.0
    if np.isfinite(dollar_expansion):
        evidence.append(f"dollar_volume_expansion={dollar_expansion:.1f}x")
        if dollar_expansion >= 2.0:
            post_recognition += 4.0
        elif dollar_expansion >= 1.25:
            post_recognition += 2.0
        elif dollar_expansion < 0.80:
            contradictions.append("no sustained dollar-volume expansion vs previous earnings snapshot")
    if np.isfinite(anchor_volume_expansion) and anchor_volume_expansion >= 2.0:
        post_recognition += 2.0
        evidence.append(f"anchor_volume_shock={anchor_volume_expansion:.1f}x")
    if np.isfinite(old_high_break):
        evidence.append(f"current_vs_prev_earnings_to_EP_high={pct(old_high_break)}")
        if old_high_break >= 0.10:
            post_recognition += 2.0
        elif old_high_break < -0.05:
            contradictions.append("current price has not cleared the previous-earnings-to-EP high")
    if np.isfinite(v10) and -0.02 <= v10 <= 0.08:
        post_recognition += 2.0
        evidence.append("current price still respects 10MA zone")

    delta = post_recognition - pre_awareness
    score = clamp(4.0 + delta, 0.0, 10.0)
    uncertainty = 0.38
    if previous_earnings_source.startswith("actual_prev_earnings"):
        uncertainty -= 0.06
    if anchor_source in {"recent_ep_date", "entry_ep_date"}:
        uncertainty -= 0.06
    elif anchor_source in {"latest_earnings_date", "earnings_date"}:
        evidence.append("earnings date used as EP proxy")
        uncertainty += 0.02
    if np.isfinite(days_to_current) and days_to_current < 3:
        uncertainty += 0.08
        contradictions.append("post-anchor window is still short")
    if contradictions:
        uncertainty += min(0.12, 0.03 * len(contradictions))
    return score, evidence[:10], contradictions[:6], clamp(uncertainty, 0.18, 0.62)


def technical_expert(row: pd.Series) -> tuple[float, list[str], list[str]]:
    score = 0.0
    evidence: list[str] = []
    contradictions: list[str] = []
    ep_lane = clean_str(row.get("recent_ep_lane")) or clean_str(row.get("entry_ep_lane"))
    ep_score = safe_float(row.get("recent_ep_score"), np.nan)
    if not np.isfinite(ep_score):
        ep_score = safe_float(row.get("entry_ep_score"), np.nan)

    if ep_lane:
        lane_bonus = {"a_plus_gap_ep": 8.0, "gap_ep": 6.5, "displacement_ep": 6.0}.get(ep_lane, 4.0)
        score += lane_bonus
        evidence.append(f"EP={ep_lane}")
    if np.isfinite(ep_score) and ep_score > 90:
        score += 2.0
        evidence.append(f"strong_ep_score={ep_score:.1f}")
    return clamp(score, 0.0, 17.0), evidence[:7], contradictions[:5]


def risk_expert(row: pd.Series) -> tuple[float, list[str]]:
    penalty = 0.0
    risks: list[str] = []
    ps = safe_float(row.get("ps_ratio_ttm"), np.nan)
    rev = safe_float(row.get("revenue_yoy_pct"), np.nan)
    if not np.isfinite(rev):
        rev = safe_float(row.get("rev_yoy"), np.nan)
    net_income = safe_float(row.get("latest_net_income"), np.nan)
    market_cap = safe_float(row.get("market_cap"), np.nan)
    if np.isfinite(ps) and ps > 10.0 and (not np.isfinite(rev) or rev < 0.30):
        penalty += 4.0
        risks.append("high P/S without enough growth")
    if np.isfinite(net_income) and net_income < 0:
        penalty += 2.5
        risks.append("latest quarter unprofitable")
    if np.isfinite(market_cap) and market_cap > 80_000_000_000:
        penalty += 3.0
        risks.append("large cap reduces multibagger torque")
    exclude_reason = clean_str(row.get("exclude_reason"))
    if exclude_reason:
        penalty += 10.0
        risks.append(exclude_reason[:80])
    return clamp(penalty, 0.0, 15.0), risks[:6]


def source_quality_expert(row: pd.Series) -> tuple[float, list[str], float]:
    titles = row.get("news_titles_combined", [])
    title_text = " ".join(normalize_text(t) for t in titles)
    evidence: list[str] = []
    score = 0.0
    uncertainty = 0.35
    if len(titles) >= 3:
        score += 3.0
        evidence.append(f"news_items={len(titles)}")
        uncertainty -= 0.06
    if any(word in title_text for word in ["earnings call transcript", "reports", "results", "financial results"]):
        score += 3.0
        evidence.append("earnings-source present")
        uncertainty -= 0.06
    if any(word in title_text for word in ["analyst", "momentum stock", "could surge", "overvalued"]):
        score += 1.0
        uncertainty += 0.03
    if not titles:
        uncertainty += 0.15
    status = clean_str(row.get("source_status")).lower()
    if status == "partially_verified":
        score += 1.5
        evidence.append("curated research note present")
    elif status == "needs_primary_source":
        uncertainty += 0.08
        evidence.append("curated hypothesis pending verification")
    return clamp(score, 0.0, 7.0), evidence[:5], clamp(uncertainty, 0.18, 0.62)


def deep_loop_target(row: pd.Series, pre_score: float, contradictions: list[str]) -> int:
    target = 2
    if pre_score >= 60:
        target += 1
    if pre_score >= 75:
        target += 1
    if contradictions:
        target += 1
    text = " ".join(
        [
            " ".join(normalize_text(t) for t in row.get("news_titles_combined", [])),
            normalize_text(row.get("evidence_keywords")),
            normalize_text(row.get("new_reality_hypothesis")),
        ]
    )
    if any(word in text for word in ["backlog", "book-to-bill", "nvidia", "bluefield", "data center", "800g", "1.6t"]):
        target += 1
    if safe_float(row.get("previous_earnings_to_current_return_pct"), np.nan) >= 0.50:
        target += 1
    if clean_str(row.get("source_status")).lower() in {"needs_primary_source", "partially_verified"}:
        target += 1
    return int(clamp(target, 2, 6))


def reason_one(row: pd.Series, cfg: RecognitionGapConfig) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    stable_score = 0.0
    uncertainty = 0.55
    all_evidence: list[str] = []
    all_contradictions: list[str] = []
    components: dict[str, float] = {}

    for loop_idx in range(1, cfg.max_loop_iters + 1):
        theme_score, theme_evidence = theme_expert(row)
        earnings_score, earnings_evidence, earnings_uncertainty = earnings_backlog_expert(row)
        gap_score, gap_evidence, gap_contra, gap_uncertainty = recognition_gap_expert(row)
        anchor_score, anchor_evidence, anchor_contra, anchor_uncertainty = anchor_delta_expert(row)
        tech_score, tech_evidence, tech_contra = technical_expert(row)
        risk_penalty, risk_evidence = risk_expert(row)
        source_score, source_evidence, source_uncertainty = source_quality_expert(row)

        loop_components = {
            "theme_score": theme_score,
            "recognition_gap_score": gap_score,
            "anchor_delta_score": anchor_score,
            "earnings_backlog_score": earnings_score,
            "technical_score": tech_score,
            "source_quality_score": source_score,
            "risk_penalty": risk_penalty,
        }
        raw_score = (
            theme_score
            + gap_score
            + anchor_score
            + earnings_score
            + tech_score
            + source_score
            - risk_penalty
        )
        loop_uncertainty = np.mean([earnings_uncertainty, gap_uncertainty, anchor_uncertainty, source_uncertainty])

        if loop_idx == 1:
            stable_score = raw_score
        else:
            # Recurrent-depth style stable update: later loops refine, not overwrite.
            alpha = 0.62
            stable_score = alpha * stable_score + (1.0 - alpha) * raw_score
        uncertainty = 0.70 * uncertainty + 0.30 * loop_uncertainty

        evidence = theme_evidence + gap_evidence + anchor_evidence + earnings_evidence + tech_evidence + source_evidence
        contradictions = gap_contra + anchor_contra + tech_contra + risk_evidence
        all_evidence.extend(evidence)
        all_contradictions.extend(contradictions)
        components = loop_components

        delta = abs(raw_score - stable_score)
        target = deep_loop_target(row, stable_score, all_contradictions)
        trace.append(
            {
                "loop": loop_idx,
                "target_loops": target,
                "raw_score": round(raw_score, 2),
                "stable_score": round(stable_score, 2),
                "uncertainty": round(float(uncertainty), 3),
                "top_evidence": list(dict.fromkeys(evidence))[:8],
                "top_contradictions": list(dict.fromkeys(contradictions))[:6],
            }
        )
        if loop_idx >= cfg.min_loop_iters and loop_idx >= target:
            if delta <= cfg.halt_delta or uncertainty <= cfg.halt_uncertainty:
                break

    final_score = clamp(stable_score, 0.0, 100.0)
    unique_evidence = list(dict.fromkeys([clean_str(item) for item in all_evidence if clean_str(item)]))[:12]
    unique_contra = list(dict.fromkeys([clean_str(item) for item in all_contradictions if clean_str(item)]))[:10]
    gate_status, gate_evidence = entry_gate(row)
    if final_score >= 72 and gate_status == "entry_ready":
        action = "buy_zone_research_validated"
    elif final_score >= 72:
        action = "high_conviction_wait_for_pullback"
    elif final_score >= 62 and gate_status == "entry_ready":
        action = "watchlist_entry_ready"
    elif final_score >= 62:
        action = "watchlist_wait_for_pullback"
    else:
        action = "research_only"

    return {
        "symbol": row.get("symbol"),
        "company_name": row.get("company_name"),
        "sector": row.get("sector"),
        "industry": row.get("industry"),
        "market_cap": row.get("market_cap"),
        "close": row.get("close"),
        "close_vs_10ma_pct": row.get("close_vs_10ma_pct"),
        "return_20d_pct": row.get("return_20d_pct"),
        "policy_themes": clean_str(row.get("policy_themes")),
        "new_reality_hypothesis": clean_str(row.get("new_reality_hypothesis")),
        "source_status": clean_str(row.get("source_status")),
        "recent_ep_lane": clean_str(row.get("recent_ep_lane")) or clean_str(row.get("entry_ep_lane")),
        "recent_ep_date": clean_str(row.get("recent_ep_date")) or clean_str(row.get("entry_ep_date")),
        "entry_gate": gate_status,
        "entry_gate_evidence": " | ".join(gate_evidence),
        "ep_entry_date": clean_str(row.get("ep_entry_date")),
        "entry_trigger": clean_str(row.get("entry_trigger")),
        "anchor_date": clean_str(row.get("anchor_date")),
        "anchor_source": clean_str(row.get("anchor_source")),
        "previous_earnings_date": clean_str(row.get("previous_earnings_date")),
        "previous_earnings_source": clean_str(row.get("previous_earnings_source")),
        "previous_earnings_snapshot_date": clean_str(row.get("previous_earnings_snapshot_date")),
        "current_snapshot_date": clean_str(row.get("current_snapshot_date")),
        "previous_earnings_to_pre_ep_return_pct": row.get("previous_earnings_to_pre_ep_return_pct"),
        "anchor_to_current_return_pct": row.get("anchor_to_current_return_pct"),
        "previous_earnings_to_current_return_pct": row.get("previous_earnings_to_current_return_pct"),
        "dollar_volume_expansion_prev_earnings_to_current": row.get("dollar_volume_expansion_prev_earnings_to_current"),
        "anchor_dollar_volume_expansion": row.get("anchor_dollar_volume_expansion"),
        "current_vs_prev_earnings_to_anchor_high_pct": row.get("current_vs_prev_earnings_to_anchor_high_pct"),
        "ps_ratio_ttm": row.get("ps_ratio_ttm"),
        "revenue_yoy_pct": row.get("revenue_yoy_pct")
        if pd.notna(row.get("revenue_yoy_pct", np.nan))
        else row.get("rev_yoy"),
        "eps_surprise_pct": row.get("eps_surprise_pct"),
        "lite_score": row.get("lite_score"),
        "policy_total_score": row.get("policy_total_score"),
        "recognition_gap_total_score": round(final_score, 2),
        "uncertainty": round(float(uncertainty), 3),
        "loops_used": len(trace),
        "action": action,
        **{key: round(value, 2) for key, value in components.items()},
        "evidence": " | ".join(unique_evidence),
        "contradictions": " | ".join(unique_contra),
        "reasoning_trace": trace,
    }


def build_report(out_dir: Path, cfg: RecognitionGapConfig, ranked: pd.DataFrame, elapsed: float) -> None:
    lines: list[str] = []
    lines.append("# Recognition Gap EP System")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed:.2f} seconds")
    lines.append(f"- Candidates reasoned: {len(ranked):,}")
    lines.append(f"- Loop range: {cfg.min_loop_iters}-{cfg.max_loop_iters}")
    lines.append("- Architecture: prelude candidate merge -> recurrent expert loops -> coda ranking/action.")
    lines.append("- Candidate universe: EP event is required unless `--allow-no-ep` is used.")
    lines.append("- Pullback rule: post-EP pullback/10MA entry is required for buy actions, but it is not part of the recognition-gap score.")
    lines.append("- Anchor rule: actual previous-earnings aftermath is required unless `--allow-missing-earnings-anchor` is used.")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Symbol | Score | Entry Gate | Action | Loops | Anchor | RG Delta | Theme | Close | 20d R | vs10MA | EP | P/S | Rev YoY | Key Evidence | Main Risk |")
    lines.append("|---:|---|---:|---|---|---:|---|---:|---|---:|---:|---:|---|---:|---:|---|---|")
    top = ranked.head(cfg.top_n)
    for rank, row in enumerate(top.itertuples(index=False), 1):
        evidence = clean_str(row.evidence).split(" | ")[0:3]
        risks = clean_str(row.contradictions).split(" | ")[0:2]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    clean_str(row.symbol),
                    f"{float(row.recognition_gap_total_score):.1f}",
                    clean_str(row.entry_gate),
                    clean_str(row.action),
                    str(int(row.loops_used)),
                    clean_str(row.anchor_date)[:10],
                    f"{safe_float(row.anchor_delta_score):.1f}" if np.isfinite(safe_float(row.anchor_delta_score)) else "",
                    clean_str(row.policy_themes)[:38],
                    price(row.close),
                    pct(row.return_20d_pct),
                    pct(row.close_vs_10ma_pct),
                    clean_str(row.recent_ep_lane)[:18],
                    f"{safe_float(row.ps_ratio_ttm):.1f}x" if np.isfinite(safe_float(row.ps_ratio_ttm)) else "",
                    pct(row.revenue_yoy_pct),
                    "; ".join(evidence)[:80],
                    "; ".join(risks)[:70],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## System Definition")
    lines.append("")
    lines.append("1. EP proves that real money is already testing the story.")
    lines.append("2. Policy and industry themes test whether the story can persist beyond one quarter.")
    lines.append("3. Earnings, bookings, backlog, design-win, and visibility language test whether the business has structurally changed.")
    lines.append("4. Anchor delta compares actual previous-earnings aftermath with the current EP aftermath.")
    lines.append("5. Recognition gap is the distance between the old market label and the new economic reality.")
    lines.append("6. Pullback/10MA is an entry gate, not an alpha score component.")
    lines.append("7. Variable loops spend more compute on high-potential, contradictory, or source-weak candidates.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(pd.Series(asdict(cfg)).to_json(indent=2))
    lines.append("```")
    lines.append("")
    (out_dir / "recognition_gap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    cfg = RecognitionGapConfig(
        max_candidates=args.max_candidates,
        min_loop_iters=args.min_loop_iters,
        max_loop_iters=args.max_loop_iters,
        top_n=args.top_n,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(Path(args.policy_ranking))
    lite = load_lite(Path(args.lite_candidates))
    ep = load_ep(Path(args.ep_signals))
    notes = load_research_notes(Path(args.research_notes))
    candidates = combine_candidates(policy, lite, ep, notes)
    if args.require_ep_entry:
        candidates = require_ep_entry_candidates(candidates)
    elif not args.allow_no_ep:
        candidates = require_ep_event_candidates(candidates)
    candidates = candidates.head(cfg.max_candidates).copy()
    if candidates.empty:
        raise RuntimeError("No candidates loaded after EP requirement.")
    daily = load_daily_features(Path(args.daily_features), set(candidates["symbol"].astype(str).str.upper()))
    earnings_calendar = load_or_fetch_earnings_calendar(
        Path(args.earnings_calendar),
        candidates,
        args.skip_fmp_earnings,
    )
    candidates = add_anchor_recognition_features(candidates, daily, earnings_calendar)
    if not args.allow_missing_earnings_anchor:
        candidates = candidates.loc[candidates["anchor_feature_available"].astype(bool)].reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("No candidates loaded after previous-earnings anchor requirement.")

    results = [reason_one(row, cfg) for _, row in candidates.iterrows()]
    ranked = pd.DataFrame(results).sort_values(
        ["recognition_gap_total_score", "uncertainty"],
        ascending=[False, True],
        kind="mergesort",
    )
    traces = {
        row["symbol"]: row["reasoning_trace"]
        for row in results
        if row.get("symbol")
    }
    ranked_no_trace = ranked.drop(columns=["reasoning_trace"])
    elapsed = time.perf_counter() - started

    ranked_no_trace.to_csv(out_dir / "recognition_gap_ranking.csv", index=False)
    ranked_no_trace.head(cfg.top_n).to_csv(out_dir / "recognition_gap_ranking_top.csv", index=False)
    (out_dir / "recognition_gap_traces.json").write_text(json.dumps(traces, indent=2, default=str), encoding="utf-8")
    build_report(out_dir, cfg, ranked_no_trace, elapsed)
    print(f"Candidates reasoned: {len(ranked_no_trace):,}")
    print(f"Top score: {ranked_no_trace['recognition_gap_total_score'].max():.1f}")
    print(f"Wrote: {out_dir / 'recognition_gap_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
