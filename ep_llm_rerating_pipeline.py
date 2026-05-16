#!/usr/bin/env python3
"""
ep_llm_rerating_pipeline.py
─────────────────────────────────────────────────────────────────────────────
EP-mandatory, score-free, LLM-based recognition-gap ranking pipeline.

Purpose
-------
Find EP-confirmed stocks that may have started to rerate after an Episodic
Pivot but have not yet been fully reclassified by the market as growth stocks.

Major design choices
--------------------
1. EP is mandatory. Non-EP candidates are not sent to the LLM.
2. Fixed theme dictionaries, keyword scores, P/S caps, market-cap caps, and
   total numeric scores are removed from the ranking decision.
3. LLM prompts discover themes from the input data and produce narrative JSON.
4. Every LLM stage supports loop reasoning through stable JSON fields:
   - judgment_stability
   - needs_additional_review
   - changed_from_previous_loop
   - next_loop_focus
   - stop_recommendation
5. Fundamental recognition-gap ranking and entry timing are separated.

Expected inputs
---------------
Required:
  --ep-signals     CSV with at least symbol and EP-related columns.

Optional but recommended:
  --daily-features Parquet with symbol,date,open,high,low,close,volume.
  --lite-candidates CSV from any previous broad scanner.
  --policy-ranking CSV from any policy/theme pre-analysis.
  --research-notes CSV with manual notes.
  FMP_API_KEY      Optional for enrichment.
  OPENAI_API_KEY   Required unless --dry-run is used.

Outputs
-------
  ep_llm_candidates.csv
  ep_llm_company_memos.json
  ep_llm_pre_ranking_review.json
  ep_llm_final_ranking.json
  ep_llm_entry_timing.json
  ep_llm_report.md
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency until runtime
    OpenAI = None  # type: ignore


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "analysis_outputs" / "ep_llm_rerating_pipeline"
DEFAULT_EP_SIGNALS = (
    ROOT
    / "analysis_outputs"
    / "episodic_pivot_daily_entry_signals_20260504_20260508"
    / "entry_signals_final_tradable.csv"
)
DEFAULT_DAILY_FEATURES = (
    ROOT
    / "analysis_outputs"
    / "episodic_pivot_daily_entry_signals_20260504_20260508"
    / "combined_daily_features_input.parquet"
)
DEFAULT_POLICY_RANKING = (
    ROOT
    / "analysis_outputs"
    / "policy_theme_ep_rank_20260510"
    / "policy_theme_ep_ranking.csv"
)
DEFAULT_LITE_CANDIDATES = (
    ROOT
    / "analysis_outputs"
    / "lite_before_surge_candidates_20260510"
    / "lite_before_surge_candidates.csv"
)
DEFAULT_RESEARCH_NOTES = ROOT / "configs" / "recognition_gap_research_notes.csv"
FMP_BASE = "https://financialmodelingprep.com/api/v3"


@dataclass(frozen=True)
class LoopConfig:
    individual_min: int = 2
    individual_max: int = 6
    bear_min: int = 2
    bear_max: int = 4
    memo_min: int = 2
    memo_max: int = 4
    pre_ranking_min: int = 2
    pre_ranking_max: int = 4
    ranking_min: int = 2
    ranking_max: int = 6
    entry_min: int = 1
    entry_max: int = 4


@dataclass(frozen=True)
class PipelineConfig:
    model: str
    max_candidates: int
    top_n: int
    min_price: float
    min_dollar_volume_20: float
    require_ep_entry: bool
    asof_date: str
    ep_start_date: str
    enrich_fmp: bool
    dry_run: bool
    temperature: float
    request_sleep: float
    loops: LoopConfig


# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EP-mandatory, score-free LLM recognition-gap ranking pipeline."
    )
    parser.add_argument("--ep-signals", default=str(DEFAULT_EP_SIGNALS))
    parser.add_argument("--daily-features", default=str(DEFAULT_DAILY_FEATURES))
    parser.add_argument("--policy-ranking", default=str(DEFAULT_POLICY_RANKING))
    parser.add_argument("--lite-candidates", default=str(DEFAULT_LITE_CANDIDATES))
    parser.add_argument("--research-notes", default=str(DEFAULT_RESEARCH_NOTES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume-20", type=float, default=5_000_000.0)
    parser.add_argument("--require-ep-entry", action="store_true")
    parser.add_argument(
        "--asof-date",
        default="",
        help="Optional point-in-time ceiling date. Truncates prices, events, news, and FMP fundamentals to this date.",
    )
    parser.add_argument(
        "--ep-start-date",
        default="",
        help="Optional lower bound for EP event dates. Use this for post-EP discovery windows.",
    )
    parser.add_argument("--no-fmp-enrich", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM; emit prompt-shaped placeholders.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-sleep", type=float, default=0.15)

    parser.add_argument("--individual-min-loop-iters", type=int, default=2)
    parser.add_argument("--individual-max-loop-iters", type=int, default=6)
    parser.add_argument("--bear-min-loop-iters", type=int, default=2)
    parser.add_argument("--bear-max-loop-iters", type=int, default=4)
    parser.add_argument("--memo-min-loop-iters", type=int, default=2)
    parser.add_argument("--memo-max-loop-iters", type=int, default=4)
    parser.add_argument("--pre-ranking-min-loop-iters", type=int, default=2)
    parser.add_argument("--pre-ranking-max-loop-iters", type=int, default=4)
    parser.add_argument("--ranking-min-loop-iters", type=int, default=2)
    parser.add_argument("--ranking-max-loop-iters", type=int, default=6)
    parser.add_argument("--entry-min-loop-iters", type=int, default=1)
    parser.add_argument("--entry-max-loop-iters", type=int, default=4)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    loops = LoopConfig(
        individual_min=args.individual_min_loop_iters,
        individual_max=args.individual_max_loop_iters,
        bear_min=args.bear_min_loop_iters,
        bear_max=args.bear_max_loop_iters,
        memo_min=args.memo_min_loop_iters,
        memo_max=args.memo_max_loop_iters,
        pre_ranking_min=args.pre_ranking_min_loop_iters,
        pre_ranking_max=args.pre_ranking_max_loop_iters,
        ranking_min=args.ranking_min_loop_iters,
        ranking_max=args.ranking_max_loop_iters,
        entry_min=args.entry_min_loop_iters,
        entry_max=args.entry_max_loop_iters,
    )
    return PipelineConfig(
        model=args.model,
        max_candidates=args.max_candidates,
        top_n=args.top_n,
        min_price=args.min_price,
        min_dollar_volume_20=args.min_dollar_volume_20,
        require_ep_entry=args.require_ep_entry,
        asof_date=args.asof_date,
        ep_start_date=args.ep_start_date,
        enrich_fmp=not args.no_fmp_enrich,
        dry_run=args.dry_run,
        temperature=args.temperature,
        request_sleep=args.request_sleep,
        loops=loops,
    )


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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def parse_date(value: Any) -> pd.Timestamp | None:
    text = clean_str(value)
    if not text:
        return None
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def parse_titles(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean_str(v) for v in value if clean_str(v)]
    text = clean_str(value)
    if not text:
        return []
    if "||" in text:
        return [p.strip() for p in text.split("||") if p.strip()]
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [clean_str(v) for v in parsed if clean_str(v)]
        except Exception:
            pass
    return [text]


NEWS_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "positive_business_change": [
        r"\bbeat\b",
        r"\braises?\b",
        r"\braised\b",
        r"\bguidance\b",
        r"\brecord\b",
        r"\baccelerat",
        r"\bprofit",
        r"\bmargin",
        r"\bbacklog\b",
    ],
    "earnings_or_guidance": [r"\bearnings?\b", r"\bresults?\b", r"\bq[1-4]\b", r"\bguidance\b", r"\boutlook\b"],
    "order_or_backlog": [r"\border\b", r"\borders\b", r"\bbacklog\b", r"\bbookings?\b", r"\baward(?:ed)?\b"],
    "customer_or_contract": [r"\bcustomer\b", r"\bcontract\b", r"\bpartnership\b", r"\bpartner\b", r"\bdeal\b", r"\bsupplier\b"],
    "product_launch": [r"\blaunch(?:es|ed)?\b", r"\bplatform\b", r"\bproduct\b", r"\bintroduces?\b", r"\bshowcases?\b"],
    "policy_theme": [
        r"\bAI\b",
        r"\bartificial intelligence\b",
        r"\bdata center\b",
        r"\bdatacenter\b",
        r"\bsemiconductor\b",
        r"\bspace\b",
        r"\bdefen[cs]e\b",
        r"\bnuclear\b",
        r"\brare earth\b",
        r"\bcritical minerals?\b",
        r"\bgrid\b",
    ],
    "analyst_upgrade": [r"\bupgrade", r"\bprice target\b", r"\bPT\b", r"\banalyst"],
    "lawsuit": [r"\blawsuit\b", r"\bclass action\b", r"\bshareholder alert\b", r"\binvestigation\b", r"\bsecurities fraud\b"],
    "short_report": [r"\bshort report\b", r"\bshort-seller\b", r"\bshort seller\b", r"\bshort\b"],
    "offering_or_dilution": [r"\boffering\b", r"\bconvertible\b", r"\bwarrants?\b", r"\bdilution\b", r"\bATM\b", r"\bprivate placement\b"],
    "mna_buyout": [r"\bacquisition\b", r"\bbuyout\b", r"\btakeover\b", r"\bmerger\b", r"\bto acquire\b", r"\bacquires\b"],
    "commodity_price_only": [r"\blithium\b", r"\bcopper\b", r"\bzinc\b", r"\bgold\b", r"\bsilver\b", r"\boil\b", r"\bnatural gas\b", r"\buranium\b"],
    "biotech_binary": [r"\bFDA\b", r"\bphase\b", r"\btrial\b", r"\bclinical\b", r"\bapproval\b", r"\bdrug\b"],
    "crypto_noise": [r"\bbitcoin\b", r"\bcrypto\b", r"\bmining\b", r"\bminer\b", r"\bhashrate\b"],
    "generic_momentum_article": [r"\bmomentum stock\b", r"\bstock soars\b", r"\bstock jumps\b", r"\bbest momentum\b", r"\btop-ranked\b"],
}


THEME_TERM_PATTERNS: dict[str, list[str]] = {
    "ai": [r"\bAI\b", r"\bartificial intelligence\b", r"\binference\b", r"\bagentic\b"],
    "data_center": [r"\bdata center\b", r"\bdatacenter\b"],
    "semiconductor": [r"\bsemiconductor\b", r"\bchip\b", r"\bsilicon\b", r"\bmemory\b", r"\bNAND\b", r"\bSSD\b"],
    "power_grid": [r"\bpower\b", r"\bgrid\b", r"\belectrification\b", r"\butility\b"],
    "space_defense": [r"\bspace\b", r"\bdefen[cs]e\b", r"\bsatellite\b", r"\blaunch\b"],
    "critical_minerals": [r"\brare earth\b", r"\bcritical minerals?\b", r"\blithium\b", r"\buranium\b", r"\bmining\b"],
    "voice_ai": [r"\bvoice AI\b", r"\bcontact center\b", r"\bagentforce\b"],
    "backlog_orders": [r"\bbacklog\b", r"\borders?\b", r"\bbookings?\b"],
}


NEGATIVE_NEWS_CATEGORIES = {
    "lawsuit",
    "short_report",
    "offering_or_dilution",
    "mna_buyout",
    "commodity_price_only",
    "biotech_binary",
    "crypto_noise",
    "generic_momentum_article",
}


def pattern_matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def categorize_news_titles(titles: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    matched_titles: dict[str, list[str]] = {}
    for title in titles:
        text = clean_str(title)
        if not text:
            continue
        for category, patterns in NEWS_CATEGORY_PATTERNS.items():
            if any(pattern_matches(text, pattern) for pattern in patterns):
                counts[category] = counts.get(category, 0) + 1
                matched_titles.setdefault(category, []).append(text)
    positive = [
        cat
        for cat in [
            "positive_business_change",
            "earnings_or_guidance",
            "order_or_backlog",
            "customer_or_contract",
            "product_launch",
            "policy_theme",
            "analyst_upgrade",
        ]
        if counts.get(cat, 0) > 0
    ]
    negative = [cat for cat in NEGATIVE_NEWS_CATEGORIES if counts.get(cat, 0) > 0]
    return {
        "news_category_counts": counts,
        "positive_news_flags": positive,
        "negative_news_flags": sorted(negative),
        "order_backlog_evidence": matched_titles.get("order_or_backlog", [])[:5],
        "guidance_evidence": matched_titles.get("earnings_or_guidance", [])[:5],
        "customer_contract_evidence": matched_titles.get("customer_or_contract", [])[:5],
        "product_launch_evidence": matched_titles.get("product_launch", [])[:5],
    }


def extract_theme_terms(titles: list[str]) -> list[str]:
    text = " || ".join(clean_str(title) for title in titles)
    terms: list[str] = []
    for term, patterns in THEME_TERM_PATTERNS.items():
        if any(pattern_matches(text, pattern) for pattern in patterns):
            terms.append(term)
    return terms


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def to_json_text(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, default=json_default)


def strip_history(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_history(v) for k, v in obj.items() if k != "_history"}
    if isinstance(obj, list):
        return [strip_history(v) for v in obj]
    return obj


def read_secret_from_env_file(name: str, root: Path = ROOT) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    env_path = root / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, raw = text.split("=", 1)
        if key.strip() == name:
            return raw.strip().strip('"').strip("'")
    return ""


def optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df.drop_duplicates("symbol")


ASOF_DATE_COLUMNS = [
    "entry_ep_date",
    "recent_ep_date",
    "ep_entry_date",
    "latest_date",
    "latest_earnings_date",
    "earnings_date",
    "latest_reported_date",
    "income_report_date",
]


def parse_asof(value: str) -> pd.Timestamp | None:
    text = clean_str(value)
    if not text:
        return None
    date = pd.to_datetime(text, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"Invalid --asof-date: {value}")
    return pd.Timestamp(date).normalize()


def parse_optional_date_arg(value: str, arg_name: str) -> pd.Timestamp | None:
    text = clean_str(value)
    if not text:
        return None
    date = pd.to_datetime(text, errors="coerce")
    if pd.isna(date):
        raise ValueError(f"Invalid {arg_name}: {value}")
    return pd.Timestamp(date).normalize()


def filter_frame_to_asof(df: pd.DataFrame, asof: pd.Timestamp | None) -> pd.DataFrame:
    if asof is None or df.empty:
        return df
    out = df.copy()
    for col in ASOF_DATE_COLUMNS:
        if col not in out.columns:
            continue
        dates = pd.to_datetime(out[col], errors="coerce").dt.normalize()
        out = out.loc[dates.isna() | (dates <= asof)].copy()
    return out.reset_index(drop=True)


def combine_first_columns(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    out = left.merge(right, on="symbol", how="left", suffixes=("", "_dup"))
    for col in list(out.columns):
        if not col.endswith("_dup"):
            continue
        base = col[:-4]
        if base in out.columns:
            out[base] = out[base].combine_first(out[col])
            out = out.drop(columns=[col])
        else:
            out = out.rename(columns={col: base})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load and filter candidates
# ─────────────────────────────────────────────────────────────────────────────


def filter_ep_event_window(
    df: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    ep_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df.empty or (asof is None and ep_start is None):
        return df
    ep_date_col = None
    for col in ["entry_ep_date", "recent_ep_date"]:
        if col in df.columns:
            ep_date_col = col
            break
    if ep_date_col is None:
        return df
    dates = pd.to_datetime(df[ep_date_col], errors="coerce").dt.normalize()
    mask = dates.notna()
    if asof is not None:
        mask &= dates <= asof
    if ep_start is not None:
        mask &= dates >= ep_start
    return df.loc[mask].copy()


def load_ep_signals(
    path: Path,
    asof: pd.Timestamp | None = None,
    ep_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"EP signals file not found: {path}")
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError("EP signals file must contain a 'symbol' column.")
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # EP-event files use `date/high/low/close` for the event session. Preserve those
    # as EP fields before latest price features overwrite `close`.
    if "entry_ep_date" not in df.columns and "ep_date" not in df.columns and "date" in df.columns and "ep_lane" in df.columns:
        event_rename = {
            "date": "entry_ep_date",
            "open": "ep_open",
            "high": "ep_high",
            "low": "ep_low",
            "close": "ep_close",
            "volume": "ep_volume",
            "gap_pct": "ep_gap_pct",
            "day_change_pct": "ep_day_change_pct",
            "volume_ratio_20": "ep_volume_ratio_20",
            "avg_dollar_volume_20": "ep_avg_dollar_volume_20",
        }
        df = df.rename(columns={k: v for k, v in event_rename.items() if k in df.columns})

    rename = {
        "ep_date": "entry_ep_date",
        "ep_lane": "entry_ep_lane",
        "ep_score": "entry_ep_score",
        "entry_date": "ep_entry_date",
        "entry_close": "ep_entry_close",
        "entry_dma10": "ep_entry_dma10",
        "entry_close_vs_dma10_pct": "ep_entry_close_vs_dma10_pct",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = filter_ep_event_window(df, asof=asof, ep_start=ep_start)

    # Keep strongest EP row per symbol when a numeric EP score exists.
    if "entry_ep_score" in df.columns:
        df["_ep_sort_score"] = pd.to_numeric(df["entry_ep_score"], errors="coerce").fillna(-1)
        df["_ep_sort_date"] = pd.to_datetime(df.get("entry_ep_date"), errors="coerce")
        df = df.sort_values(["symbol", "_ep_sort_score", "_ep_sort_date"], ascending=[True, False, False])
        df = df.drop_duplicates("symbol", keep="first")
        df = df.sort_values(["_ep_sort_score", "_ep_sort_date", "symbol"], ascending=[False, False, True])
        df = df.drop(columns=["_ep_sort_score"])
        if "_ep_sort_date" in df.columns:
            df = df.drop(columns=["_ep_sort_date"])
        return df.reset_index(drop=True)
    if "entry_ep_date" in df.columns:
        df["_ep_sort_date"] = pd.to_datetime(df["entry_ep_date"], errors="coerce")
        df = df.sort_values(["_ep_sort_date", "symbol"], ascending=[False, True])
        df = df.drop_duplicates("symbol", keep="first").drop(columns=["_ep_sort_date"])
        return df.reset_index(drop=True)
    return df.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def has_ep_event(row: pd.Series) -> bool:
    ep_date = clean_str(row.get("entry_ep_date")) or clean_str(row.get("recent_ep_date"))
    ep_lane = clean_str(row.get("entry_ep_lane")) or clean_str(row.get("recent_ep_lane"))
    return bool(ep_date and ep_lane)


def has_ep_entry(row: pd.Series) -> bool:
    return bool(
        clean_str(row.get("entry_ep_date"))
        and clean_str(row.get("ep_entry_date"))
        and clean_str(row.get("entry_ep_lane"))
    )


def load_and_merge_candidates(
    ep_path: Path,
    policy_path: Path,
    lite_path: Path,
    notes_path: Path,
    require_ep_entry: bool,
    max_candidates: int,
    asof: pd.Timestamp | None = None,
    ep_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ep = filter_frame_to_asof(load_ep_signals(ep_path, asof=asof, ep_start=ep_start), asof)
    policy = filter_frame_to_asof(optional_csv(policy_path), asof)
    lite = filter_frame_to_asof(optional_csv(lite_path), asof)
    notes = filter_frame_to_asof(optional_csv(notes_path), asof)

    candidates = ep.copy()
    for extra in [policy, lite, notes]:
        if not extra.empty:
            candidates = combine_first_columns(candidates, extra)

    # Build combined news titles without using fixed keyword scoring.
    candidates["news_titles_combined"] = candidates.apply(
        lambda r: parse_titles(r.get("policy_news_titles"))
        + parse_titles(r.get("lite_news_titles"))
        + parse_titles(r.get("news_titles")),
        axis=1,
    )

    if require_ep_entry:
        candidates = candidates.loc[candidates.apply(has_ep_entry, axis=1)].copy()
    else:
        candidates = candidates.loc[candidates.apply(has_ep_event, axis=1)].copy()

    if candidates.empty:
        requirement = "EP entry" if require_ep_entry else "EP event"
        raise RuntimeError(f"No candidates remain after mandatory {requirement} filter.")

    if "entry_ep_score" in candidates.columns:
        candidates["_sort_ep_score"] = pd.to_numeric(candidates["entry_ep_score"], errors="coerce").fillna(-1)
        candidates["_sort_ep_date"] = pd.to_datetime(candidates.get("entry_ep_date"), errors="coerce")
        candidates = candidates.sort_values(
            ["_sort_ep_score", "_sort_ep_date", "symbol"],
            ascending=[False, False, True],
            kind="mergesort",
        ).drop(columns=["_sort_ep_score", "_sort_ep_date"])

    return candidates.head(max_candidates).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Daily features and minimal filters
# ─────────────────────────────────────────────────────────────────────────────


def load_daily_features(path: Path, symbols: set[str], asof: pd.Timestamp | None = None) -> pd.DataFrame:
    if not path.exists() or not symbols:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    daily = pd.read_parquet(path, columns=cols)
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily = daily.loc[daily["symbol"].isin(symbols)].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    if asof is not None:
        daily = daily.loc[daily["date"] <= asof].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    return daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def choose_ep_date(row: pd.Series) -> pd.Timestamp | None:
    for col in ["entry_ep_date", "recent_ep_date", "latest_earnings_date", "earnings_date"]:
        d = parse_date(row.get(col))
        if d is not None:
            return d
    return None


def add_price_volume_features(candidates: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    feature_defaults: dict[str, Any] = {
        "latest_date": "",
        "close": np.nan,
        "dma10": np.nan,
        "dma20": np.nan,
        "avg_dollar_vol_20": np.nan,
        "return_20d_pct": np.nan,
        "close_vs_10ma_pct": np.nan,
        "close_vs_20ma_pct": np.nan,
        "ep_session_close": np.nan,
        "pre_ep_return_20d": np.nan,
        "pre_ep_return_60d": np.nan,
        "pre_ep_avg_dollar_vol_20": np.nan,
        "pre_ep_avg_dollar_vol_60": np.nan,
        "ep_to_current_return_pct": np.nan,
        "post_ep_return_pct": np.nan,
        "ep_day_dollar_volume": np.nan,
        "ep_dollar_volume_expansion_vs_20d_pre": np.nan,
        "post_ep_avg_dollar_vol": np.nan,
        "post_ep_volume_persistence": np.nan,
        "post_ep_high_pct": np.nan,
        "post_ep_high_return_pct": np.nan,
        "days_since_ep": np.nan,
        "price_volume_feature_available": False,
    }
    for col, val in feature_defaults.items():
        if col not in out.columns:
            out[col] = val

    if daily.empty or out.empty:
        return out

    grouped = {sym: frame.reset_index(drop=True) for sym, frame in daily.groupby("symbol", sort=False)}
    for idx, row in out.iterrows():
        symbol = clean_str(row.get("symbol")).upper()
        g = grouped.get(symbol)
        if g is None or len(g) < 20:
            continue
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        volume = g["volume"].to_numpy(dtype=float)
        dollar_volume = close * volume
        dates = g["date"].to_numpy(dtype="datetime64[ns]")
        cur_idx = len(g) - 1
        dma10 = np.nanmean(close[max(0, cur_idx - 9) : cur_idx + 1])
        dma20 = np.nanmean(close[max(0, cur_idx - 19) : cur_idx + 1])
        dv20 = np.nanmean(dollar_volume[max(0, cur_idx - 19) : cur_idx + 1])
        ret20 = close[cur_idx] / close[cur_idx - 20] - 1.0 if cur_idx >= 20 and close[cur_idx - 20] > 0 else np.nan

        out.at[idx, "latest_date"] = pd.Timestamp(g.at[cur_idx, "date"]).strftime("%Y-%m-%d")
        out.at[idx, "close"] = close[cur_idx]
        out.at[idx, "dma10"] = dma10
        out.at[idx, "dma20"] = dma20
        out.at[idx, "avg_dollar_vol_20"] = dv20
        out.at[idx, "return_20d_pct"] = ret20
        out.at[idx, "close_vs_10ma_pct"] = close[cur_idx] / dma10 - 1.0 if dma10 > 0 else np.nan
        out.at[idx, "close_vs_20ma_pct"] = close[cur_idx] / dma20 - 1.0 if dma20 > 0 else np.nan
        out.at[idx, "price_volume_feature_available"] = True

        ep_date = choose_ep_date(row)
        if ep_date is None:
            continue
        ep_idx = int(np.searchsorted(dates, np.datetime64(ep_date), side="left"))
        if ep_idx < 0 or ep_idx >= len(g):
            continue
        pre_start = max(0, ep_idx - 20)
        pre_start_60 = max(0, ep_idx - 60)
        pre_dv = np.nanmean(dollar_volume[pre_start:ep_idx]) if ep_idx > pre_start else np.nan
        pre_dv_60 = np.nanmean(dollar_volume[pre_start_60:ep_idx]) if ep_idx > pre_start_60 else np.nan
        ep_close = close[ep_idx]
        post_high = np.nanmax(high[ep_idx : cur_idx + 1]) if cur_idx >= ep_idx else np.nan
        post_dv = np.nanmean(dollar_volume[ep_idx + 1 : cur_idx + 1]) if cur_idx > ep_idx else np.nan
        out.at[idx, "ep_session_close"] = ep_close
        out.at[idx, "pre_ep_return_20d"] = ep_close / close[ep_idx - 20] - 1.0 if ep_idx >= 20 and close[ep_idx - 20] > 0 else np.nan
        out.at[idx, "pre_ep_return_60d"] = ep_close / close[ep_idx - 60] - 1.0 if ep_idx >= 60 and close[ep_idx - 60] > 0 else np.nan
        out.at[idx, "pre_ep_avg_dollar_vol_20"] = pre_dv
        out.at[idx, "pre_ep_avg_dollar_vol_60"] = pre_dv_60
        post_return = close[cur_idx] / ep_close - 1.0 if ep_close > 0 else np.nan
        out.at[idx, "ep_to_current_return_pct"] = post_return
        out.at[idx, "post_ep_return_pct"] = post_return
        out.at[idx, "ep_day_dollar_volume"] = dollar_volume[ep_idx]
        out.at[idx, "ep_dollar_volume_expansion_vs_20d_pre"] = dollar_volume[ep_idx] / pre_dv if pre_dv > 0 else np.nan
        out.at[idx, "post_ep_avg_dollar_vol"] = post_dv
        out.at[idx, "post_ep_volume_persistence"] = post_dv / pre_dv if pre_dv > 0 and np.isfinite(post_dv) else np.nan
        post_high_return = post_high / ep_close - 1.0 if ep_close > 0 and np.isfinite(post_high) else np.nan
        out.at[idx, "post_ep_high_pct"] = post_high_return
        out.at[idx, "post_ep_high_return_pct"] = post_high_return
        out.at[idx, "days_since_ep"] = cur_idx - ep_idx
    return out


def apply_minimal_tradeability_filters(df: pd.DataFrame, min_price: float, min_dollar_volume_20: float) -> pd.DataFrame:
    out = df.copy()
    if "close" in out.columns:
        close = pd.to_numeric(out["close"], errors="coerce")
        out = out.loc[close.isna() | (close >= min_price)].copy()
    if "avg_dollar_vol_20" in out.columns:
        dv = pd.to_numeric(out["avg_dollar_vol_20"], errors="coerce")
        out = out.loc[dv.isna() | (dv >= min_dollar_volume_20)].copy()
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FMP enrichment; no filters are applied to these values
# ─────────────────────────────────────────────────────────────────────────────


def fmp_get(endpoint: str, api_key: str, params: dict[str, Any] | None = None, timeout: int = 15) -> Any:
    if not api_key:
        return None
    url = f"{FMP_BASE}/{endpoint}"
    query = {"apikey": api_key}
    if params:
        query.update(params)
    try:
        r = requests.get(url, params=query, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logging.debug("FMP %s failed: %s", endpoint, exc)
        return None


def fmp_row_date(row: dict[str, Any]) -> pd.Timestamp | None:
    for key in ["acceptedDate", "fillingDate", "reportedDate", "date"]:
        date = parse_date(row.get(key))
        if date is not None:
            return date
    return None


def fmp_income(symbol: str, api_key: str, asof: pd.Timestamp | None = None) -> dict[str, Any]:
    data = fmp_get(f"income-statement/{symbol}", api_key, {"limit": 8, "period": "quarter"})
    if not isinstance(data, list):
        return {"symbol": symbol}
    if asof is not None:
        bounded = []
        for row in data:
            if not isinstance(row, dict):
                continue
            row_date = fmp_row_date(row)
            if row_date is not None and row_date <= asof:
                bounded.append(row)
        data = bounded
    if not data:
        return {"symbol": symbol}
    q0 = data[0]
    q1 = data[1] if len(data) > 1 else {}
    q2 = data[2] if len(data) > 2 else {}
    q4 = data[4] if len(data) > 4 else {}
    r0 = q0.get("revenue") or 0
    r1 = q1.get("revenue") or 0
    r2 = q2.get("revenue") or 0
    r4 = q4.get("revenue") or 0
    ttm = sum((q.get("revenue") or 0) for q in data[:4])
    return {
        "symbol": symbol,
        "rev_yoy": (r0 / r4 - 1.0) if r4 else np.nan,
        "rev_qoq": (r0 / r1 - 1.0) if r1 else np.nan,
        "rev_accelerating": bool(r0 > r1 > r2 > 0),
        "ttm_revenue": ttm if ttm > 0 else np.nan,
        "latest_gross_margin": (q0.get("grossProfit", 0) / r0) if r0 else np.nan,
        "latest_net_income": q0.get("netIncome", np.nan),
        "latest_revenue": r0 if r0 else np.nan,
        "latest_reported_date": q0.get("date", ""),
        "income_report_date": clean_str(q0.get("acceptedDate") or q0.get("fillingDate") or q0.get("date")),
    }


def fmp_key_metrics(symbol: str, api_key: str, asof: pd.Timestamp | None = None) -> dict[str, Any]:
    if asof is not None:
        # FMP key-metrics-ttm is not point-in-time in this endpoint.
        return {"symbol": symbol}
    data = fmp_get(f"key-metrics-ttm/{symbol}", api_key)
    if not isinstance(data, list) or not data:
        return {"symbol": symbol}
    m = data[0]
    return {
        "symbol": symbol,
        "ps_ratio_ttm": m.get("priceToSalesRatioTTM", np.nan),
        "pe_ratio_ttm": m.get("peRatioTTM", np.nan),
        "enterprise_value_ttm": m.get("enterpriseValueTTM", np.nan),
    }


def fmp_eps_surprise(symbol: str, api_key: str, asof: pd.Timestamp | None = None) -> dict[str, Any]:
    data = fmp_get(f"earnings-surprises/{symbol}", api_key)
    if not isinstance(data, list) or not data:
        return {"symbol": symbol}
    if asof is not None:
        eligible = []
        for row in data:
            if not isinstance(row, dict):
                continue
            row_date = parse_date(row.get("date"))
            if row_date is not None and row_date <= asof:
                eligible.append(row)
        data = eligible
    if not data:
        return {"symbol": symbol}
    d = data[0]
    act = d.get("actualEarningResult", np.nan)
    est = d.get("estimatedEarning", np.nan)
    surprise = (act - est) / abs(est) if est and pd.notna(act) else np.nan
    return {"symbol": symbol, "eps_surprise_pct": surprise, "earnings_date": d.get("date", "")}


def fmp_news(
    symbol: str,
    api_key: str,
    limit: int = 24,
    asof: pd.Timestamp | None = None,
    ep_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"tickers": symbol, "limit": limit}
    if asof is not None:
        start = (ep_date - pd.Timedelta(days=75)) if ep_date is not None else (asof - pd.Timedelta(days=75))
        params["from"] = start.strftime("%Y-%m-%d")
        params["to"] = asof.strftime("%Y-%m-%d")
    data = fmp_get("stock_news", api_key, params)
    titles = []
    dates = []
    pre_ep_titles: list[str] = []
    post_ep_titles: list[str] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            item_date = parse_date(item.get("publishedDate") or item.get("date"))
            if asof is not None and item_date is not None and item_date > asof:
                continue
            title = clean_str(item.get("title"))
            if title:
                titles.append(title)
                if item_date is not None:
                    dates.append(item_date.strftime("%Y-%m-%d"))
                if ep_date is not None and item_date is not None and item_date < ep_date:
                    pre_ep_titles.append(title)
                elif ep_date is not None and item_date is not None:
                    post_ep_titles.append(title)
    post_categories = categorize_news_titles(post_ep_titles if ep_date is not None else titles)
    pre_terms = extract_theme_terms(pre_ep_titles)
    post_terms = extract_theme_terms(post_ep_titles if ep_date is not None else titles)
    new_terms = [term for term in post_terms if term not in set(pre_terms)]
    return {
        "symbol": symbol,
        "fmp_news_titles": titles,
        "fmp_news_dates": dates,
        "fmp_news_hits": len(titles),
        "pre_ep_news_titles": pre_ep_titles[:12],
        "post_ep_news_titles": post_ep_titles[:12] if ep_date is not None else titles[:12],
        "pre_ep_news_hits": len(pre_ep_titles),
        "post_ep_news_hits": len(post_ep_titles) if ep_date is not None else len(titles),
        "pre_ep_theme_terms": pre_terms,
        "post_ep_theme_terms": post_terms,
        "new_terms_after_ep": new_terms,
        **post_categories,
    }


def enrich_with_fmp(df: pd.DataFrame, enabled: bool, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    if not enabled or df.empty:
        return df
    api_key = read_secret_from_env_file("FMP_API_KEY")
    if not api_key:
        logging.warning("FMP_API_KEY is not set; skipping FMP enrichment.")
        return df
    inc_rows: list[dict[str, Any]] = []
    met_rows: list[dict[str, Any]] = []
    sur_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []
    rows_iter = list(df.iterrows())
    for i, (_, source_row) in enumerate(rows_iter, 1):
        sym = clean_str(source_row.get("symbol")).upper()
        if i % 20 == 1:
            print(f"  FMP enrichment {i}/{len(rows_iter)}...")
        ep_date = choose_ep_date(source_row)
        inc_rows.append(fmp_income(sym, api_key, asof))
        met_rows.append(fmp_key_metrics(sym, api_key, asof))
        sur_rows.append(fmp_eps_surprise(sym, api_key, asof))
        news_rows.append(fmp_news(sym, api_key, asof=asof, ep_date=ep_date))
        time.sleep(0.07)
    out = df.copy()
    for rows in [inc_rows, met_rows, sur_rows, news_rows]:
        out = combine_first_columns(out, pd.DataFrame(rows))
    out["news_titles_combined"] = out.apply(
        lambda r: parse_titles(r.get("news_titles_combined"))
        + parse_titles(r.get("fmp_news_titles")),
        axis=1,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Candidate record for LLM
# ─────────────────────────────────────────────────────────────────────────────


def compact_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, list):
        return [compact_value(v) for v in value]
    if isinstance(value, dict):
        return {k: compact_value(v) for k, v in value.items()}
    text = clean_str(value)
    if text == "":
        return None
    return value


def candidate_to_record(row: pd.Series) -> dict[str, Any]:
    news = parse_titles(row.get("news_titles_combined"))
    fields = [
        "symbol", "company_name", "exchange", "sector", "industry", "country", "market_cap",
        "entry_ep_date", "entry_ep_lane", "entry_ep_score", "ep_high", "ep_low", "ep_close",
        "ep_gap_pct", "ep_day_change_pct", "ep_volume_ratio_20", "ep_avg_dollar_volume_20",
        "ep_entry_date", "entry_trigger",
        "recent_ep_date", "recent_ep_lane", "recent_ep_score",
        "latest_date", "close", "dma10", "dma20", "avg_dollar_vol_20", "return_20d_pct",
        "close_vs_10ma_pct", "close_vs_20ma_pct", "ep_session_close", "ep_to_current_return_pct",
        "pre_ep_return_20d", "pre_ep_return_60d", "pre_ep_avg_dollar_vol_20", "pre_ep_avg_dollar_vol_60",
        "post_ep_return_pct", "ep_day_dollar_volume", "ep_dollar_volume_expansion_vs_20d_pre",
        "post_ep_avg_dollar_vol", "post_ep_volume_persistence", "post_ep_high_pct", "post_ep_high_return_pct",
        "days_since_ep", "ps_ratio_ttm", "pe_ratio_ttm", "rev_yoy", "rev_qoq", "rev_accelerating",
        "revenue_yoy_pct", "policy_themes", "policy_news_hits", "fmp_news_hits", "fmp_news_dates",
        "pre_ep_news_titles", "post_ep_news_titles", "pre_ep_news_hits", "post_ep_news_hits",
        "pre_ep_theme_terms", "post_ep_theme_terms", "new_terms_after_ep", "news_category_counts",
        "positive_news_flags", "negative_news_flags", "order_backlog_evidence", "guidance_evidence",
        "customer_contract_evidence", "product_launch_evidence",
        "ttm_revenue", "latest_revenue", "latest_gross_margin", "latest_net_income",
        "eps_surprise_pct", "earnings_date", "latest_earnings_date", "latest_reported_date",
        "income_report_date", "old_perception",
        "new_reality_hypothesis", "theme_tags", "evidence_keywords", "source_status",
        "analyst_notes", "exclude_reason",
    ]
    record = {field: compact_value(row.get(field)) for field in fields if field in row.index}
    record["news_titles"] = news[:20]
    return {k: v for k, v in record.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# LLM client and loop runner
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_MESSAGE = (
    "You are a careful investment research assistant. Return valid JSON only. "
    "Do not reveal hidden chain-of-thought. Provide conclusions, evidence, counterarguments, "
    "stability checks, and unresolved issues."
)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(match.group(0))


class JsonLLM:
    def __init__(self, model: str, temperature: float = 0.1, dry_run: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.client = None
        if not dry_run:
            if OpenAI is None:
                raise RuntimeError("openai package is not installed. Install it or use --dry-run.")
            self.client = OpenAI(api_key=read_secret_from_env_file("OPENAI_API_KEY"))

    def complete_json(self, prompt: str, dry_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.dry_run:
            return dry_payload or {
                "dry_run": True,
                "loop_control": {
                    "loop_index": 1,
                    "judgment_stability": "stable",
                    "needs_additional_review": False,
                    "reason_for_additional_review": "dry_run",
                    "changed_from_previous_loop": False,
                    "what_changed": [],
                    "overstatement_risks": [],
                    "underweighted_counterarguments": [],
                    "next_loop_focus": [],
                    "stop_recommendation": "stop",
                },
            }
        assert self.client is not None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return extract_json_object(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM JSON call failed after retries: {last_error}")


def loop_control_from(result: dict[str, Any]) -> dict[str, Any]:
    ctrl = result.get("loop_control")
    if isinstance(ctrl, dict):
        return ctrl
    # Some nested stages may accidentally place loop_control under a stage key.
    for value in result.values():
        if isinstance(value, dict) and isinstance(value.get("loop_control"), dict):
            return value["loop_control"]
    return {
        "judgment_stability": "unstable",
        "needs_additional_review": True,
        "stop_recommendation": "continue",
        "reason_for_additional_review": "loop_control missing",
    }


def should_stop(result: dict[str, Any], loop_index: int, min_iters: int, max_iters: int) -> bool:
    if loop_index < min_iters:
        return False
    if loop_index >= max_iters:
        return True
    ctrl = loop_control_from(result)
    return (
        ctrl.get("stop_recommendation") == "stop"
        and ctrl.get("judgment_stability") == "stable"
        and ctrl.get("needs_additional_review") is False
    )


def run_looped_stage(
    llm: JsonLLM,
    prompt_builder: Callable[[int, dict[str, Any] | None, dict[str, Any] | None], str],
    min_iters: int,
    max_iters: int,
    dry_payload_builder: Callable[[int], dict[str, Any]] | None = None,
    sleep_seconds: float = 0.15,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    previous: dict[str, Any] | None = None
    previous_ctrl: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    for loop_index in range(1, max_iters + 1):
        prompt = prompt_builder(loop_index, previous, previous_ctrl)
        dry_payload = dry_payload_builder(loop_index) if dry_payload_builder else None
        result = llm.complete_json(prompt, dry_payload=dry_payload)
        history.append(result)
        final = result
        if should_stop(result, loop_index, min_iters, max_iters):
            break
        previous = result
        previous_ctrl = loop_control_from(result)
        time.sleep(sleep_seconds)
    return final, history


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────


def common_loop_rules() -> str:
    return """
【ループ推論・安定化ルール】
あなたは一度の回答で断定せず、現在の入力、前回評価、反証、未解決論点を比較しながら、評価を安定化させてください。
ただし、内部の思考過程は出力しないでください。出力するのは、判断結果、根拠、反証、修正点、安定性評価のみです。

今回が初回ループの場合:
- 入力データから初回評価を作成する
- 不足情報や不確実な点を明示する
- 強い結論を出す場合は、その根拠と反証を両方示す

今回が2回目以降のループの場合:
- 前回評価と今回評価を比較する
- 前回の強気判断が過剰でないか確認する
- 反証が軽視されていないか確認する
- テーマ解釈が単なるキーワード連想になっていないか確認する
- バリュエーションや株価反応から、すでに再評価済みではないか確認する
- 結論を維持する場合も、なぜ維持するのか説明する
- 結論を修正する場合は、何を理由に変更したのか説明する

評価の目的:
EP後に、まだ市場に成長株として十分に再評価されていない銘柄かを判断すること。

禁止事項:
- 点数を付けない
- 固定キーワードだけで判断しない
- テーマ名の派手さだけで判断しない
- エントリータイミングを銘柄魅力度ランキングに混ぜない
- 10日線・20日線との距離で銘柄魅力度を上下させない
- 株価が上がっただけで再評価初動と断定しない
- 株価が上がりすぎているだけで事業魅力を否定しない
""".strip()


def prompt_individual_assessment(
    company_record: dict[str, Any],
    loop_index: int,
    previous: dict[str, Any] | None,
    previous_ctrl: dict[str, Any] | None,
) -> str:
    return f"""
あなたは米国株の決算後EPと市場の認識ギャップを分析する投資リサーチャーです。
以下の銘柄は、すでにスクリプト側でEP条件を満たしています。あなたの役割はEPの有無を判定することではありません。

目的:
EP後に、この銘柄がまだ市場に成長株として十分に再評価されていないかを判断してください。

{common_loop_rules()}

【ループ入力】
loop_index: {loop_index}
previous_assessment: {to_json_text(previous)}
previous_loop_control: {to_json_text(previous_ctrl)}

【入力データ】
company_record:
{to_json_text(company_record)}

【評価観点】
1. EPが何を示したか
2. 旧来の市場認識は何か
3. 新しい事業実態は何か
4. 入力データから自然に抽出される成長テーマは何か
5. そのテーマは売上・受注・利益率・ガイダンスに結びついているか
6. EP後に市場は再評価を始めたか
7. その再評価はまだ不完全か
8. すでに成長株として織り込まれている可能性はないか
9. 反証はどの程度重いか

【出力形式】
JSONのみで返してください。
{{
  "symbol": "{company_record.get('symbol', '')}",
  "company_name": "{company_record.get('company_name', '')}",
  "ep_interpretation": {{
    "ep_date": "",
    "ep_type": "earnings_gap_ep / guidance_ep / demand_repricing_ep / product_customer_ep / unclear",
    "what_the_ep_signaled": "",
    "ep_quality": "high_quality / moderate_quality / low_quality / unclear",
    "ep_quality_reason": ""
  }},
  "discovered_growth_themes": [
    {{
      "theme_name": "",
      "why_this_theme_emerged": "",
      "business_link": "売上・受注・利益率・ガイダンスとの接続",
      "evidence": [],
      "evidence_quality": "strong / moderate / weak"
    }}
  ],
  "old_vs_new_reality": {{
    "old_market_label": "",
    "new_business_reality": "",
    "recognition_gap_summary": "",
    "gap_supported_by": [],
    "gap_weakened_by": []
  }},
  "post_ep_rerating_assessment": {{
    "has_market_started_to_rerate": "yes / partially / no / unclear",
    "rerating_progress": "early_stage / in_progress / mostly_priced_in / failed_or_unclear",
    "evidence_of_rerating": [],
    "evidence_that_rerating_is_not_complete": [],
    "evidence_that_rerating_may_already_be_complete": []
  }},
  "growth_stock_reclassification_judgment": {{
    "judgment": "not_yet_fully_rerated_growth_candidate / partially_rerated_growth_candidate / already_rerated_growth_stock / not_a_growth_rerating_case / insufficient_evidence",
    "plain_english_conclusion": "",
    "why_this_judgment": [],
    "main_counterarguments": [],
    "what_would_confirm_the_case": [],
    "what_would_invalidate_the_case": []
  }},
  "ranking_use_summary": {{
    "include_in_fundamental_recognition_gap_ranking": "yes / no / research_only",
    "ranking_bucket_candidate": "highest_conviction / strong_candidate / watchlist / research_only / avoid_for_now",
    "reason_for_bucket": ""
  }},
  "loop_control": {{
    "loop_index": {loop_index},
    "judgment_stability": "stable / unstable / conflicted",
    "needs_additional_review": true,
    "reason_for_additional_review": "",
    "changed_from_previous_loop": true,
    "what_changed": [],
    "overstatement_risks": [],
    "underweighted_counterarguments": [],
    "next_loop_focus": [],
    "stop_recommendation": "stop / continue"
  }}
}}
""".strip()


def prompt_bear_case(
    company_record: dict[str, Any],
    initial_assessment: dict[str, Any],
    loop_index: int,
    previous: dict[str, Any] | None,
    previous_ctrl: dict[str, Any] | None,
) -> str:
    return f"""
あなたはEP後の強気投資仮説を疑うリスク担当アナリストです。
強気仮説を補強するのではなく、過大評価、テーマ誤認、織り込み済み、決算一過性、バリュエーション過熱、データ不足を探してください。

{common_loop_rules()}

【ループ入力】
loop_index: {loop_index}
previous_bear_case: {to_json_text(previous)}
previous_loop_control: {to_json_text(previous_ctrl)}

【入力】
company_record:
{to_json_text(company_record)}

initial_assessment:
{to_json_text(initial_assessment)}

【出力形式】
JSONのみで返してください。
{{
  "symbol": "{company_record.get('symbol', '')}",
  "bear_case_assessment": {{
    "main_bear_case": "",
    "theme_risk": "",
    "fundamental_risk": "",
    "valuation_risk": "",
    "priced_in_risk": "",
    "technical_reaction_risk": "",
    "data_quality_risk": "",
    "what_would_invalidate_the_bull_case": [],
    "risk_judgment": "severe / material / manageable / minor"
  }},
  "stabilization_review": {{
    "is_initial_assessment_overstated": false,
    "overstated_parts": [],
    "should_bucket_be_downgraded": false,
    "downgrade_reason": null,
    "should_case_remain_valid": true,
    "reason_case_remains_valid": ""
  }},
  "loop_control": {{
    "loop_index": {loop_index},
    "judgment_stability": "stable / unstable / conflicted",
    "needs_additional_review": true,
    "reason_for_additional_review": "",
    "changed_from_previous_loop": true,
    "what_changed": [],
    "overstatement_risks": [],
    "underweighted_counterarguments": [],
    "next_loop_focus": [],
    "stop_recommendation": "stop / continue"
  }}
}}
""".strip()


def prompt_final_memo(
    company_record: dict[str, Any],
    initial_assessment: dict[str, Any],
    bear_case_assessment: dict[str, Any],
    loop_reviews: list[dict[str, Any]],
    loop_index: int,
    previous: dict[str, Any] | None,
    previous_ctrl: dict[str, Any] | None,
) -> str:
    return f"""
あなたはEP後認識ギャップ銘柄の評価結果を統合する投資リサーチャーです。
以下には、個別銘柄の強気評価、反証評価、過去ループのレビューがあります。
これらを統合し、最終ランキングに渡すための安定化済み投資メモを作成してください。

{common_loop_rules()}

点数は使わないでください。エントリータイミングは銘柄魅力度判断に混ぜないでください。反証と不足情報を必ず反映してください。

【ループ入力】
loop_index: {loop_index}
previous_final_memo: {to_json_text(previous)}
previous_loop_control: {to_json_text(previous_ctrl)}

【入力】
company_record:
{to_json_text(company_record)}

initial_assessment:
{to_json_text(initial_assessment)}

bear_case_assessment:
{to_json_text(bear_case_assessment)}

loop_reviews:
{to_json_text(loop_reviews)}

【出力形式】
JSONのみで返してください。
{{
  "symbol": "{company_record.get('symbol', '')}",
  "company_name": "{company_record.get('company_name', '')}",
  "final_company_memo": {{
    "primary_growth_theme": "",
    "one_sentence_thesis": "",
    "ep_meaning": "",
    "old_market_label": "",
    "new_business_reality": "",
    "growth_stock_reclassification_judgment": "not_yet_fully_rerated_growth_candidate / partially_rerated_growth_candidate / already_rerated_growth_stock / not_a_growth_rerating_case / insufficient_evidence",
    "recognition_gap_remaining": "large / moderate / small / unclear",
    "main_evidence": [],
    "main_counterarguments": [],
    "priced_in_risk": "high / moderate / low / unclear",
    "missing_information": [],
    "confidence_label": "high / moderate / low",
    "ranking_bucket_candidate": "highest_conviction / strong_candidate / watchlist / research_only / avoid_for_now",
    "why_this_bucket": "",
    "stability_summary": "",
    "do_not_mix_with_entry_timing": "このメモは銘柄魅力度評価であり、エントリー判断ではない"
  }},
  "loop_control": {{
    "loop_index": {loop_index},
    "judgment_stability": "stable / unstable / conflicted",
    "needs_additional_review": true,
    "reason_for_additional_review": "",
    "changed_from_previous_loop": true,
    "what_changed": [],
    "overstatement_risks": [],
    "underweighted_counterarguments": [],
    "next_loop_focus": [],
    "stop_recommendation": "stop / continue"
  }}
}}
""".strip()


def prompt_pre_ranking_review(
    company_memos: list[dict[str, Any]],
    loop_index: int,
    previous: dict[str, Any] | None,
    previous_ctrl: dict[str, Any] | None,
) -> str:
    return f"""
あなたはEP後・未再評価成長株候補のランキング前レビュー担当です。
以下には、各銘柄の安定化済み投資メモがあります。まだランキングを確定せず、各銘柄の評価が比較可能か、過大評価や見落としがないかを確認してください。

{common_loop_rules()}

【ループ入力】
loop_index: {loop_index}
previous_pre_ranking_review: {to_json_text(previous)}
previous_loop_control: {to_json_text(previous_ctrl)}

【入力】
company_memos:
{to_json_text(company_memos)}

【確認事項】
1. 似たテーマの銘柄同士で評価基準が一貫しているか
2. ある銘柄だけ反証が軽く扱われていないか
3. ある銘柄だけテーマ解釈が甘くなっていないか
4. すでに成長株として織り込まれている銘柄が上位候補に残りすぎていないか
5. データ不足の銘柄が過度に強いバケットに入っていないか
6. エントリータイミングが銘柄魅力度評価に混ざっていないか

【出力形式】
JSONのみで返してください。
{{
  "pre_ranking_review": {{
    "companies_needing_downgrade_review": [],
    "companies_needing_upgrade_review": [],
    "companies_with_unstable_assessment": [],
    "cross_company_consistency_issues": [],
    "recommended_ranking_focus": []
  }},
  "loop_control": {{
    "loop_index": {loop_index},
    "judgment_stability": "stable / unstable / conflicted",
    "needs_additional_review": true,
    "reason_for_additional_review": "",
    "changed_from_previous_loop": true,
    "what_changed": [],
    "overstatement_risks": [],
    "underweighted_counterarguments": [],
    "next_loop_focus": [],
    "stop_recommendation": "stop / continue"
  }}
}}
""".strip()


def prompt_final_ranking(
    company_memos: list[dict[str, Any]],
    pre_ranking_review: dict[str, Any],
    top_n: int,
    loop_index: int,
    previous: dict[str, Any] | None,
    previous_ctrl: dict[str, Any] | None,
) -> str:
    return f"""
あなたは米国株のEP後認識ギャップ銘柄を評価する投資委員会の議長です。
以下には、EP条件を満たした銘柄の安定化済み投資メモと、ランキング前レビューがあります。

役割:
エントリータイミングではなく、「EP後にまだ市場に成長株として十分に再評価されていない可能性が高い順」にランキングしてください。

{common_loop_rules()}

点数は使わないでください。10日線・20日線との距離、押し目待ち、今すぐ買えるかどうかはランキングに反映しないでください。

【ループ入力】
loop_index: {loop_index}
previous_ranking: {to_json_text(previous)}
previous_loop_control: {to_json_text(previous_ctrl)}
pre_ranking_review:
{to_json_text(pre_ranking_review)}

【入力】
company_memos:
{to_json_text(company_memos)}

【ランキングで重視するもの】
1. EPが本当に事業変化を知らせるイベントだったか
2. 抽出された成長テーマが売上・受注・利益に直接つながるか
3. 旧来の市場認識と新しい事業実態の差が大きいか
4. EP後に再評価は始まったが、まだ完全には織り込まれていないか
5. すでに成長株化しすぎていないか
6. 反証が致命的でないか
7. 次の決算や一次情報でさらに認識が変わる余地があるか

【2回目以降のループで必ず確認すること】
- 前回ランキングから順位を変える必要があるか
- 1位から5位までの差が本当に説明できるか
- すでに織り込み済みの銘柄を上位にしすぎていないか
- データ不足銘柄を上位にしすぎていないか
- テーマの派手さだけで順位を上げていないか

【出力形式】
JSONのみで返してください。fundamental_recognition_gap_ranking は最大 {top_n} 件にしてください。
{{
  "ranking_title": "EP後・未再評価成長株候補ランキング",
  "ranking_basis": "エントリータイミングではなく、EP後にまだ成長株として再評価されきっていない可能性に基づくランキング",
  "dominant_themes_discovered": [],
  "fundamental_recognition_gap_ranking": [
    {{
      "rank": 1,
      "symbol": "",
      "company_name": "",
      "ranking_bucket": "highest_conviction / strong_candidate / watchlist / research_only / avoid_for_now",
      "growth_stock_reclassification_judgment": "not_yet_fully_rerated_growth_candidate / partially_rerated_growth_candidate / already_rerated_growth_stock / not_a_growth_rerating_case / insufficient_evidence",
      "one_sentence_thesis": "",
      "why_ranked_here": "",
      "why_not_higher": "",
      "why_not_lower": "",
      "main_growth_theme": "",
      "old_market_label": "",
      "new_business_reality": "",
      "evidence_that_rerating_is_incomplete": [],
      "evidence_that_may_be_priced_in": [],
      "main_counterarguments": [],
      "next_primary_sources_to_check": []
    }}
  ],
  "ranking_stability_review": {{
    "changed_from_previous_ranking": false,
    "rank_changes": [],
    "top_rank_confidence": "high / moderate / low",
    "most_difficult_comparison": {{"symbols": [], "why_difficult": "", "why_final_order_was_chosen": ""}},
    "remaining_ranking_uncertainties": []
  }},
  "excluded_or_deprioritized": [],
  "overall_caveats": [],
  "loop_control": {{
    "loop_index": {loop_index},
    "judgment_stability": "stable / unstable / conflicted",
    "needs_additional_review": true,
    "reason_for_additional_review": "",
    "changed_from_previous_loop": true,
    "what_changed": [],
    "overstatement_risks": [],
    "underweighted_counterarguments": [],
    "next_loop_focus": [],
    "stop_recommendation": "stop / continue"
  }}
}}
""".strip()


def prompt_entry_timing(
    ranking_result: dict[str, Any],
    price_volume_data: list[dict[str, Any]],
    loop_index: int,
    previous: dict[str, Any] | None,
    previous_ctrl: dict[str, Any] | None,
) -> str:
    return f"""
あなたはEP後銘柄のエントリータイミングだけを評価するトレードリサーチャーです。
以下のランキングは、銘柄魅力度に基づくものであり、今すぐ買える順ではありません。
あなたの役割は、ランキングとは独立して、各銘柄の現在のエントリー状態を評価することです。

【ループ推論・安定化ルール】
前回評価がある場合は比較し、エントリー状態が変わるべきか確認してください。内部の思考過程は出力せず、条件・根拠・リスク・安定性だけを出力してください。

【ループ入力】
loop_index: {loop_index}
previous_entry_view: {to_json_text(previous)}
previous_loop_control: {to_json_text(previous_ctrl)}

【入力】
ranking_result:
{to_json_text(ranking_result)}

price_volume_data:
{to_json_text(price_volume_data)}

【評価観点】
- EP後に押し目を作っているか
- 10日線・20日線との位置
- 出来高が維持されているか
- ブレイク後の失敗兆候があるか
- 追いかけ買いになりすぎていないか
- 確認すべき価格帯や条件は何か

【出力形式】
JSONのみで返してください。
{{
  "entry_timing_view": [
    {{
      "symbol": "",
      "entry_state": "actionable_now / wait_for_pullback / wait_for_confirmation / avoid_for_now / insufficient_data",
      "entry_reason": "",
      "important_conditions": [],
      "risk_if_entering_now": [],
      "what_would_improve_entry_quality": []
    }}
  ],
  "loop_control": {{
    "loop_index": {loop_index},
    "judgment_stability": "stable / unstable / conflicted",
    "needs_additional_review": true,
    "reason_for_additional_review": "",
    "changed_from_previous_loop": true,
    "what_changed": [],
    "overstatement_risks": [],
    "underweighted_counterarguments": [],
    "next_loop_focus": [],
    "stop_recommendation": "stop / continue"
  }}
}}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run payloads
# ─────────────────────────────────────────────────────────────────────────────


def dry_loop_control(loop_index: int, stop_after: int = 1) -> dict[str, Any]:
    return {
        "loop_index": loop_index,
        "judgment_stability": "stable",
        "needs_additional_review": False,
        "reason_for_additional_review": "dry_run",
        "changed_from_previous_loop": loop_index > 1,
        "what_changed": [],
        "overstatement_risks": [],
        "underweighted_counterarguments": [],
        "next_loop_focus": [],
        "stop_recommendation": "stop" if loop_index >= stop_after else "continue",
    }


def dry_individual_payload(symbol: str, company_name: str, loop_index: int) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "company_name": company_name,
        "ep_interpretation": {"ep_date": "", "ep_type": "unclear", "what_the_ep_signaled": "dry_run", "ep_quality": "unclear", "ep_quality_reason": "dry_run"},
        "discovered_growth_themes": [],
        "old_vs_new_reality": {"old_market_label": "dry_run", "new_business_reality": "dry_run", "recognition_gap_summary": "dry_run", "gap_supported_by": [], "gap_weakened_by": []},
        "post_ep_rerating_assessment": {"has_market_started_to_rerate": "unclear", "rerating_progress": "failed_or_unclear", "evidence_of_rerating": [], "evidence_that_rerating_is_not_complete": [], "evidence_that_rerating_may_already_be_complete": []},
        "growth_stock_reclassification_judgment": {"judgment": "insufficient_evidence", "plain_english_conclusion": "dry_run", "why_this_judgment": [], "main_counterarguments": [], "what_would_confirm_the_case": [], "what_would_invalidate_the_case": []},
        "ranking_use_summary": {"include_in_fundamental_recognition_gap_ranking": "research_only", "ranking_bucket_candidate": "research_only", "reason_for_bucket": "dry_run"},
        "loop_control": dry_loop_control(loop_index),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_company(llm: JsonLLM, record: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    symbol = str(record.get("symbol", ""))
    company_name = str(record.get("company_name", ""))
    print(f"  LLM company assessment: {symbol}")

    initial, initial_history = run_looped_stage(
        llm,
        lambda i, prev, ctrl: prompt_individual_assessment(record, i, prev, ctrl),
        cfg.loops.individual_min,
        cfg.loops.individual_max,
        dry_payload_builder=lambda i: dry_individual_payload(symbol, company_name, i),
        sleep_seconds=cfg.request_sleep,
    )

    bear, bear_history = run_looped_stage(
        llm,
        lambda i, prev, ctrl: prompt_bear_case(record, initial, i, prev, ctrl),
        cfg.loops.bear_min,
        cfg.loops.bear_max,
        dry_payload_builder=lambda i: {
            "symbol": symbol,
            "bear_case_assessment": {"main_bear_case": "dry_run", "risk_judgment": "material"},
            "stabilization_review": {"is_initial_assessment_overstated": False, "overstated_parts": [], "should_bucket_be_downgraded": False, "downgrade_reason": None, "should_case_remain_valid": True, "reason_case_remains_valid": "dry_run"},
            "loop_control": dry_loop_control(i),
        },
        sleep_seconds=cfg.request_sleep,
    )

    loop_reviews = initial_history + bear_history
    memo, memo_history = run_looped_stage(
        llm,
        lambda i, prev, ctrl: prompt_final_memo(record, initial, bear, loop_reviews, i, prev, ctrl),
        cfg.loops.memo_min,
        cfg.loops.memo_max,
        dry_payload_builder=lambda i: {
            "symbol": symbol,
            "company_name": company_name,
            "final_company_memo": {
                "primary_growth_theme": "dry_run",
                "one_sentence_thesis": "dry_run",
                "growth_stock_reclassification_judgment": "insufficient_evidence",
                "recognition_gap_remaining": "unclear",
                "main_evidence": [],
                "main_counterarguments": [],
                "priced_in_risk": "unclear",
                "missing_information": [],
                "confidence_label": "low",
                "ranking_bucket_candidate": "research_only",
                "why_this_bucket": "dry_run",
                "do_not_mix_with_entry_timing": "このメモは銘柄魅力度評価であり、エントリー判断ではない",
            },
            "loop_control": dry_loop_control(i),
        },
        sleep_seconds=cfg.request_sleep,
    )

    return {
        "symbol": symbol,
        "company_name": company_name,
        "raw_record": record,
        "initial_assessment": initial,
        "bear_case_assessment": bear,
        "final_memo": memo,
        "histories": {
            "initial": initial_history,
            "bear": bear_history,
            "memo": memo_history,
        },
    }


def run_pre_ranking_review(llm: JsonLLM, company_memos: list[dict[str, Any]], cfg: PipelineConfig) -> dict[str, Any]:
    print("  LLM pre-ranking review")
    final, history = run_looped_stage(
        llm,
        lambda i, prev, ctrl: prompt_pre_ranking_review(company_memos, i, prev, ctrl),
        cfg.loops.pre_ranking_min,
        cfg.loops.pre_ranking_max,
        dry_payload_builder=lambda i: {
            "pre_ranking_review": {
                "companies_needing_downgrade_review": [],
                "companies_needing_upgrade_review": [],
                "companies_with_unstable_assessment": [],
                "cross_company_consistency_issues": [],
                "recommended_ranking_focus": [],
            },
            "loop_control": dry_loop_control(i),
        },
        sleep_seconds=cfg.request_sleep,
    )
    final["_history"] = history
    return final


def run_final_ranking(llm: JsonLLM, company_memos: list[dict[str, Any]], pre_review: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    print("  LLM final fundamental ranking")
    pre_review_for_prompt = strip_history(pre_review)
    final, history = run_looped_stage(
        llm,
        lambda i, prev, ctrl: prompt_final_ranking(company_memos, pre_review_for_prompt, cfg.top_n, i, prev, ctrl),
        cfg.loops.ranking_min,
        cfg.loops.ranking_max,
        dry_payload_builder=lambda i: {
            "ranking_title": "EP後・未再評価成長株候補ランキング",
            "ranking_basis": "dry_run",
            "dominant_themes_discovered": [],
            "fundamental_recognition_gap_ranking": [],
            "ranking_stability_review": {"changed_from_previous_ranking": False, "rank_changes": [], "top_rank_confidence": "low", "most_difficult_comparison": {"symbols": [], "why_difficult": "", "why_final_order_was_chosen": ""}, "remaining_ranking_uncertainties": []},
            "excluded_or_deprioritized": [],
            "overall_caveats": [],
            "loop_control": dry_loop_control(i),
        },
        sleep_seconds=cfg.request_sleep,
    )
    final["_history"] = history
    return final


def run_entry_timing(llm: JsonLLM, ranking: dict[str, Any], candidates: pd.DataFrame, cfg: PipelineConfig) -> dict[str, Any]:
    symbols = [item.get("symbol") for item in ranking.get("fundamental_recognition_gap_ranking", []) if isinstance(item, dict)]
    if not symbols:
        symbols = candidates["symbol"].head(cfg.top_n).astype(str).tolist()
    price_cols = [
        "symbol", "latest_date", "close", "dma10", "dma20", "avg_dollar_vol_20",
        "return_20d_pct", "close_vs_10ma_pct", "close_vs_20ma_pct", "ep_to_current_return_pct",
        "ep_dollar_volume_expansion_vs_20d_pre", "days_since_ep", "entry_ep_date", "entry_ep_lane",
    ]
    price_data = []
    for _, row in candidates.loc[candidates["symbol"].isin(symbols)].iterrows():
        price_data.append({col: compact_value(row.get(col)) for col in price_cols if col in row.index})
    print("  LLM separate entry timing view")
    ranking_for_prompt = strip_history(ranking)
    final, history = run_looped_stage(
        llm,
        lambda i, prev, ctrl: prompt_entry_timing(ranking_for_prompt, price_data, i, prev, ctrl),
        cfg.loops.entry_min,
        cfg.loops.entry_max,
        dry_payload_builder=lambda i: {"entry_timing_view": [], "loop_control": dry_loop_control(i)},
        sleep_seconds=cfg.request_sleep,
    )
    final["_history"] = history
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def flatten_memos_for_csv(company_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in company_results:
        memo = item.get("final_memo", {}).get("final_company_memo", {})
        raw = item.get("raw_record", {})
        rows.append(
            {
                "symbol": item.get("symbol"),
                "company_name": item.get("company_name"),
                "entry_ep_date": raw.get("entry_ep_date") or raw.get("recent_ep_date"),
                "entry_ep_lane": raw.get("entry_ep_lane") or raw.get("recent_ep_lane"),
                "latest_date": raw.get("latest_date"),
                "close": raw.get("close"),
                "avg_dollar_vol_20": raw.get("avg_dollar_vol_20"),
                "return_20d_pct": raw.get("return_20d_pct"),
                "close_vs_10ma_pct": raw.get("close_vs_10ma_pct"),
                "ps_ratio_ttm": raw.get("ps_ratio_ttm"),
                "rev_yoy": raw.get("rev_yoy"),
                "rev_qoq": raw.get("rev_qoq"),
                "primary_growth_theme": memo.get("primary_growth_theme"),
                "growth_stock_reclassification_judgment": memo.get("growth_stock_reclassification_judgment"),
                "recognition_gap_remaining": memo.get("recognition_gap_remaining"),
                "priced_in_risk": memo.get("priced_in_risk"),
                "confidence_label": memo.get("confidence_label"),
                "ranking_bucket_candidate": memo.get("ranking_bucket_candidate"),
                "one_sentence_thesis": memo.get("one_sentence_thesis"),
                "main_evidence": " | ".join(memo.get("main_evidence", [])[:5]) if isinstance(memo.get("main_evidence"), list) else "",
                "main_counterarguments": " | ".join(memo.get("main_counterarguments", [])[:5]) if isinstance(memo.get("main_counterarguments"), list) else "",
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(
    out_dir: Path,
    cfg: PipelineConfig,
    candidates: pd.DataFrame,
    company_results: list[dict[str, Any]],
    ranking: dict[str, Any],
    entry: dict[str, Any],
    elapsed: float,
) -> None:
    lines: list[str] = []
    lines.append("# EP LLM Rerating Pipeline")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed:.1f} seconds")
    lines.append(f"- EP candidates evaluated: {len(company_results):,}")
    lines.append(f"- LLM model: `{cfg.model}`")
    lines.append(f"- Dry run: `{cfg.dry_run}`")
    lines.append("")
    lines.append("## Design")
    lines.append("")
    lines.append("- EP is mandatory before LLM evaluation.")
    lines.append("- Fixed theme dictionaries, keyword scores, P/S caps, market-cap caps, and numeric total scores are not used for ranking.")
    lines.append("- Every LLM stage uses loop-control fields for stability, contradiction review, and additional-review decisions.")
    lines.append("- Fundamental recognition-gap ranking and entry timing are separated.")
    lines.append("")
    lines.append("## Final Ranking")
    lines.append("")
    rows = ranking.get("fundamental_recognition_gap_ranking", [])
    if rows:
        lines.append("| Rank | Symbol | Bucket | Judgment | Theme | Thesis | Key risk |")
        lines.append("|---:|---|---|---|---|---|---|")
        for row in rows:
            if not isinstance(row, dict):
                continue
            risk = " ; ".join(row.get("main_counterarguments", [])[:2]) if isinstance(row.get("main_counterarguments"), list) else ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("rank", "")),
                        clean_str(row.get("symbol")),
                        clean_str(row.get("ranking_bucket")),
                        clean_str(row.get("growth_stock_reclassification_judgment")),
                        clean_str(row.get("main_growth_theme"))[:40],
                        clean_str(row.get("one_sentence_thesis"))[:90],
                        risk[:80],
                    ]
                )
                + " |"
            )
    else:
        lines.append("No ranking rows were returned.")
    lines.append("")
    lines.append("## Separate Entry Timing View")
    lines.append("")
    entry_rows = entry.get("entry_timing_view", [])
    if entry_rows:
        lines.append("| Symbol | Entry state | Reason |")
        lines.append("|---|---|---|")
        for row in entry_rows:
            if isinstance(row, dict):
                lines.append(f"| {clean_str(row.get('symbol'))} | {clean_str(row.get('entry_state'))} | {clean_str(row.get('entry_reason'))[:120]} |")
    else:
        lines.append("No entry timing rows were returned.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(to_json_text(asdict(cfg)))
    lines.append("```")
    lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ep_llm_report.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
def write_no_lookahead_audit(
    out_dir: Path,
    candidates: pd.DataFrame,
    asof: pd.Timestamp | None,
    ep_start: pd.Timestamp | None = None,
) -> None:
    if asof is None:
        return
    rows: list[dict[str, Any]] = []
    fields = [
        "entry_ep_date",
        "recent_ep_date",
        "ep_entry_date",
        "latest_date",
        "earnings_date",
        "latest_earnings_date",
        "latest_reported_date",
        "income_report_date",
    ]
    for field in fields:
        if field not in candidates.columns:
            continue
        dates = pd.to_datetime(candidates[field], errors="coerce").dt.normalize()
        rows.append(
            {
                "field": field,
                "max_date": dates.max().strftime("%Y-%m-%d") if pd.notna(dates.max()) else "",
                "violations_after_asof": int((dates > asof).sum()),
                "min_date": dates.min().strftime("%Y-%m-%d") if pd.notna(dates.min()) else "",
                "violations_before_ep_start": int((dates < ep_start).sum()) if field in {"entry_ep_date", "recent_ep_date"} and ep_start is not None else "",
            }
        )
    if "fmp_news_dates" in candidates.columns:
        news_dates: list[pd.Timestamp] = []
        for value in candidates["fmp_news_dates"]:
            for item in parse_titles(value):
                date = pd.to_datetime(item, errors="coerce")
                if pd.notna(date):
                    news_dates.append(pd.Timestamp(date).normalize())
        if news_dates:
            s = pd.Series(news_dates)
            rows.append(
                {
                    "field": "fmp_news_dates",
                    "max_date": s.max().strftime("%Y-%m-%d"),
                    "violations_after_asof": int((s > asof).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "no_lookahead_audit.csv", index=False)


# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    cfg = build_config(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asof = parse_asof(cfg.asof_date)
    ep_start = parse_optional_date_arg(cfg.ep_start_date, "--ep-start-date")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EP candidates...")
    candidates = load_and_merge_candidates(
        ep_path=Path(args.ep_signals),
        policy_path=Path(args.policy_ranking),
        lite_path=Path(args.lite_candidates),
        notes_path=Path(args.research_notes),
        require_ep_entry=cfg.require_ep_entry,
        max_candidates=cfg.max_candidates,
        asof=asof,
        ep_start=ep_start,
    )
    print(f"  EP candidates after mandatory EP filter: {len(candidates):,}")

    print("Adding price/volume features...")
    daily = load_daily_features(Path(args.daily_features), set(candidates["symbol"].astype(str).str.upper()), asof=asof)
    candidates = add_price_volume_features(candidates, daily)
    before_tradeability = len(candidates)
    candidates = apply_minimal_tradeability_filters(candidates, cfg.min_price, cfg.min_dollar_volume_20)
    print(f"  After minimal price/liquidity filters: {len(candidates):,} / {before_tradeability:,}")
    if candidates.empty:
        raise RuntimeError("No candidates remain after minimal price/liquidity filters.")

    print("Optional FMP enrichment...")
    candidates = enrich_with_fmp(candidates, cfg.enrich_fmp, asof=asof)
    candidates.to_csv(out_dir / "ep_llm_candidates.csv", index=False)
    write_no_lookahead_audit(out_dir, candidates, asof, ep_start)

    llm = JsonLLM(model=cfg.model, temperature=cfg.temperature, dry_run=cfg.dry_run)
    records = [candidate_to_record(row) for _, row in candidates.iterrows()]

    company_results: list[dict[str, Any]] = []
    for record in records:
        company_results.append(evaluate_company(llm, record, cfg))
        (out_dir / "ep_llm_company_memos.partial.json").write_text(
            to_json_text(company_results), encoding="utf-8"
        )

    company_memos = [result["final_memo"] for result in company_results]
    pre_review = run_pre_ranking_review(llm, company_memos, cfg)
    ranking = run_final_ranking(llm, company_memos, pre_review, cfg)
    entry = run_entry_timing(llm, ranking, candidates, cfg)

    elapsed = time.perf_counter() - started

    (out_dir / "ep_llm_company_memos.json").write_text(to_json_text(strip_history(company_results)), encoding="utf-8")
    (out_dir / "ep_llm_pre_ranking_review.json").write_text(to_json_text(strip_history(pre_review)), encoding="utf-8")
    (out_dir / "ep_llm_final_ranking.json").write_text(to_json_text(strip_history(ranking)), encoding="utf-8")
    (out_dir / "ep_llm_entry_timing.json").write_text(to_json_text(strip_history(entry)), encoding="utf-8")
    flatten_memos_for_csv(company_results).to_csv(out_dir / "ep_llm_company_memos.csv", index=False)
    build_markdown_report(out_dir, cfg, candidates, company_results, ranking, entry, elapsed)

    print(f"Wrote: {out_dir / 'ep_llm_report.md'}")
    print(f"Wrote: {out_dir / 'ep_llm_final_ranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
