#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from scan_episodic_pivots import BREAKOUT_ROOT, EpisodicPivotConfig
from scan_r3000_2025_episodic_pivots import LANE_NAMES, add_numba_features
from validate_ep_entry_rules_2025 import add_entry_features


UNIVERSE_PATH = BREAKOUT_ROOT / "analysis_outputs" / "russell3000_full_dataset" / "universe.parquet"
DEFAULT_DAILY_INPUT = (
    BREAKOUT_ROOT
    / "analysis_outputs"
    / "episodic_pivot_daily_entry_signals_20260504_20260508"
    / "combined_daily_features_input.parquet"
)
DEFAULT_OUT_DIR = BREAKOUT_ROOT / "analysis_outputs" / "policy_theme_ep_rank_20260510"
FMP_BASE = "https://financialmodelingprep.com/api/v3"


@dataclass(frozen=True)
class RankConfig:
    recent_sessions: int = 20
    feature_start: str = "2025-01-01"
    min_price: float = 5.0
    min_avg_dollar_volume_20: float = 5_000_000.0
    min_market_cap: float = 300_000_000.0
    fmp_candidate_limit: int = 90
    top_n: int = 40


THEME_KEYWORDS: dict[str, list[str]] = {
    "space_defense": [
        "aerospace",
        "defense",
        "space",
        "satellite",
        "missile",
        "drone",
        "unmanned",
        "geospatial",
    ],
    "critical_minerals": [
        "rare earth",
        "lithium",
        "uranium",
        "copper",
        "graphite",
        "nickel",
        "cobalt",
        "mining",
        "mineral",
        "metals",
        "aluminum",
        "steel",
        "coal",
    ],
    "nuclear_power_grid": [
        "nuclear",
        "uranium",
        "electric",
        "utility",
        "utilities",
        "power",
        "grid",
        "transmission",
        "electrical",
        "energy infrastructure",
    ],
    "semiconductor_ai": [
        "semiconductor",
        "semiconductors",
        "chip",
        "silicon",
        "ai",
        "artificial intelligence",
        "computer hardware",
        "electronic components",
        "scientific & technical instruments",
    ],
    "data_center_electrification": [
        "data center",
        "datacenter",
        "thermal",
        "cooling",
        "electrical equipment",
        "building products",
        "power management",
        "infrastructure",
    ],
    "energy_dominance": [
        "oil",
        "gas",
        "lng",
        "natural gas",
        "pipeline",
        "drilling",
        "exploration",
        "refining",
    ],
    "clean_energy_ev": [
        "solar",
        "wind",
        "renewable",
        "battery",
        "electric vehicle",
        "ev",
        "hydrogen",
        "fuel cell",
    ],
}


SYMBOL_THEME_OVERRIDES: dict[str, list[str]] = {
    "RKLB": ["space_defense"],
    "LUNR": ["space_defense"],
    "RDW": ["space_defense"],
    "ASTS": ["space_defense"],
    "IRDM": ["space_defense"],
    "SPIR": ["space_defense"],
    "PL": ["space_defense"],
    "MP": ["critical_minerals"],
    "UAMY": ["critical_minerals"],
    "USAR": ["critical_minerals"],
    "USAU": ["critical_minerals"],
    "LEU": ["critical_minerals", "nuclear_power_grid"],
    "UEC": ["critical_minerals", "nuclear_power_grid"],
    "DNN": ["critical_minerals", "nuclear_power_grid"],
    "NXE": ["critical_minerals", "nuclear_power_grid"],
    "CCJ": ["critical_minerals", "nuclear_power_grid"],
    "LAC": ["critical_minerals", "clean_energy_ev"],
    "ALB": ["critical_minerals", "clean_energy_ev"],
    "SMR": ["nuclear_power_grid"],
    "OKLO": ["nuclear_power_grid"],
    "CEG": ["nuclear_power_grid"],
    "VST": ["nuclear_power_grid"],
    "GEV": ["nuclear_power_grid", "data_center_electrification"],
    "ETN": ["data_center_electrification", "nuclear_power_grid"],
    "PWR": ["data_center_electrification", "nuclear_power_grid"],
    "POWL": ["data_center_electrification", "nuclear_power_grid"],
    "VRT": ["data_center_electrification", "semiconductor_ai"],
    "DLR": ["data_center_electrification"],
    "EQIX": ["data_center_electrification"],
    "NVDA": ["semiconductor_ai"],
    "AMD": ["semiconductor_ai"],
    "AVGO": ["semiconductor_ai"],
    "MRVL": ["semiconductor_ai"],
    "MU": ["semiconductor_ai"],
    "TXN": ["semiconductor_ai"],
    "ACLS": ["semiconductor_ai"],
    "UCTT": ["semiconductor_ai"],
    "LSCC": ["semiconductor_ai"],
}


POLICY_NEWS_KEYWORDS = [
    "white house",
    "executive order",
    "department of energy",
    "doe",
    "department of defense",
    "dod",
    "pentagon",
    "nasa",
    "artemis",
    "space force",
    "faa",
    "critical minerals",
    "rare earth",
    "reshoring",
    "domestic supply",
    "national security",
    "tariff",
    "chips",
    "semiconductor",
    "data center",
    "power grid",
    "nuclear",
    "uranium",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank policy-theme EP/10MA candidates over recent sessions.")
    parser.add_argument("--daily-input", default=str(DEFAULT_DAILY_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--recent-sessions", type=int, default=20)
    parser.add_argument("--fmp-candidate-limit", type=int, default=90)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--skip-fmp", action="store_true")
    return parser.parse_args()


def norm_text(value: object) -> str:
    return str(value or "").lower()


def keyword_matches(haystack: str, keyword: str) -> bool:
    if len(keyword) <= 3 and keyword.isalnum():
        return re.search(rf"\b{re.escape(keyword)}\b", haystack, flags=re.IGNORECASE) is not None
    return keyword in haystack


def pct(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{100.0 * number:.1f}%"


def price(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.2f}"


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Universe not found: {UNIVERSE_PATH}")
    universe = pd.read_parquet(UNIVERSE_PATH)
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    return universe.drop_duplicates("symbol")


def load_features(path: Path, feature_start: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    daily = pd.read_parquet(path)
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    cols = ["open", "high", "low", "close", "volume"]
    for col in cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.dropna(subset=["symbol", "date", *cols])
    daily = daily.loc[daily["date"] >= pd.Timestamp(feature_start), ["symbol", "date", *cols]]
    daily = daily.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    features, codes, group_end = add_numba_features(daily, EpisodicPivotConfig(start=feature_start))
    features = add_entry_features(features, codes)
    features["prev_close2"] = features.groupby("symbol", sort=False)["close"].shift(1)
    features["up_day"] = features["close"] > features["prev_close2"]
    return features, codes, group_end


def classify_themes(row: pd.Series) -> tuple[list[str], int]:
    symbol = str(row["symbol"]).upper()
    haystack = " ".join(
        [
            norm_text(row.get("company_name")),
            norm_text(row.get("sector")),
            norm_text(row.get("industry")),
        ]
    )
    themes: list[str] = []
    for theme, words in THEME_KEYWORDS.items():
        if any(keyword_matches(haystack, word) for word in words):
            themes.append(theme)
    for theme in SYMBOL_THEME_OVERRIDES.get(symbol, []):
        if theme not in themes:
            themes.append(theme)
    if not themes:
        return [], 0
    strong = {"space_defense", "critical_minerals", "nuclear_power_grid", "semiconductor_ai"}
    base = 18 if any(theme in strong for theme in themes) else 13
    if len(themes) >= 2:
        base += 4
    if any(theme in {"space_defense", "critical_minerals"} for theme in themes):
        base += 3
    return themes, min(base, 24)


def build_recent_metrics(features: pd.DataFrame, universe: pd.DataFrame, cfg: RankConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = features.groupby("symbol", sort=False)
    for symbol, sub in grouped:
        if len(sub) < max(60, cfg.recent_sessions + 25):
            continue
        sub = sub.reset_index(drop=False).rename(columns={"index": "feature_index"})
        latest = sub.iloc[-1]
        prior20 = sub.iloc[-1 - cfg.recent_sessions]
        prior5 = sub.iloc[-6] if len(sub) >= 6 else np.nan
        last20 = sub.iloc[-cfg.recent_sessions:]
        prev20 = sub.iloc[-(cfg.recent_sessions * 2) : -cfg.recent_sessions]
        recent_ep = last20.loc[last20["lane_code"] > 0]
        latest_ep = recent_ep.iloc[-1] if not recent_ep.empty else None
        up_dollar = float(last20.loc[last20["up_day"], "dollar_volume"].sum())
        total_dollar = float(last20["dollar_volume"].sum())
        avg_vol_prev20 = float(prev20["volume"].mean()) if not prev20.empty else np.nan
        avg_vol_last20 = float(last20["volume"].mean())
        close = float(latest["close"])
        dma10 = float(latest["dma10"])
        dma20 = float(latest["dma20"])
        low20 = float(last20["low"].min())
        high20 = float(last20["high"].max())
        rows.append(
            {
                "symbol": symbol,
                "latest_date": latest["date"],
                "close": close,
                "dma10": dma10,
                "dma20": dma20,
                "return_5d_pct": close / float(prior5["close"]) - 1.0 if isinstance(prior5, pd.Series) else np.nan,
                "return_20d_pct": close / float(prior20["close"]) - 1.0,
                "range_20d_pct": high20 / low20 - 1.0 if low20 > 0 else np.nan,
                "close_vs_10ma_pct": close / dma10 - 1.0 if dma10 > 0 else np.nan,
                "close_vs_20ma_pct": close / dma20 - 1.0 if dma20 > 0 else np.nan,
                "close_vs_20d_high_pct": close / high20 - 1.0 if high20 > 0 else np.nan,
                "avg_volume20_vs_prev20": avg_vol_last20 / avg_vol_prev20 if avg_vol_prev20 > 0 else np.nan,
                "latest_volume_ratio_20": latest["volume_ratio_20"],
                "avg_dollar_volume_20": latest["avg_dollar_volume_20"],
                "up_dollar_volume_ratio_20": up_dollar / total_dollar if total_dollar > 0 else np.nan,
                "recent_ep_date": latest_ep["date"] if latest_ep is not None else pd.NaT,
                "recent_ep_lane": LANE_NAMES[int(latest_ep["lane_code"])] if latest_ep is not None else "",
                "recent_ep_score": latest_ep["ep_score"] if latest_ep is not None else np.nan,
                "recent_ep_days_ago": int(len(sub) - 1 - int(latest_ep.name)) if latest_ep is not None else np.nan,
                "low_near_10ma_today": bool(float(latest["low"]) <= dma10 * 1.03) if dma10 > 0 else False,
            }
        )
    metrics = pd.DataFrame(rows)
    meta_cols = [
        col
        for col in [
            "symbol",
            "company_name",
            "exchange",
            "market_cap",
            "sector",
            "industry",
            "country",
            "rank_market_cap",
        ]
        if col in universe.columns
    ]
    metrics = metrics.merge(universe[meta_cols], on="symbol", how="left")
    themes = metrics.apply(classify_themes, axis=1)
    metrics["policy_themes"] = [", ".join(item[0]) for item in themes]
    metrics["policy_base_score"] = [item[1] for item in themes]
    return metrics


def preliminary_scores(frame: pd.DataFrame, cfg: RankConfig) -> pd.DataFrame:
    out = frame.copy()
    tradable = (
        (out["close"] >= cfg.min_price)
        & (out["avg_dollar_volume_20"] >= cfg.min_avg_dollar_volume_20)
        & (pd.to_numeric(out["market_cap"], errors="coerce") >= cfg.min_market_cap)
        & ~out["industry"].fillna("").astype(str).str.contains("Biotechnology", case=False, na=False)
        & out["policy_base_score"].gt(0)
    )
    out = out.loc[tradable].copy()
    ret20 = out["return_20d_pct"].clip(lower=-0.20, upper=0.80)
    vol = np.log1p(out["avg_volume20_vs_prev20"].clip(lower=0, upper=8.0)) / np.log1p(8.0)
    rvol = np.log1p(out["latest_volume_ratio_20"].clip(lower=0, upper=10.0)) / np.log1p(10.0)
    updv = out["up_dollar_volume_ratio_20"].fillna(0.5).clip(0, 1)
    near_high = (1.0 + out["close_vs_20d_high_pct"].clip(lower=-0.25, upper=0.0)) / 1.0
    flow_score = (
        12.0 * ((ret20 + 0.20) / 1.00)
        + 6.0 * vol
        + 5.0 * rvol
        + 4.0 * updv
        + 3.0 * near_high.clip(0, 1)
    )
    out["flow_score_pre"] = flow_score.clip(0, 30)
    lane_score = out["recent_ep_lane"].map({"a_plus_gap_ep": 8.0, "gap_ep": 6.0, "displacement_ep": 5.0}).fillna(0.0)
    above_ma = (out["close_vs_10ma_pct"] >= 0).astype(float) * 5.0 + (out["close_vs_20ma_pct"] >= 0).astype(float) * 3.0
    dist = out["close_vs_10ma_pct"]
    buy_zone = np.select(
        [(dist >= 0) & (dist <= 0.08), (dist > 0.08) & (dist <= 0.12), (dist > 0.12) & (dist <= 0.20)],
        [4.0, 2.0, 0.5],
        default=-2.0,
    )
    out["ep_ma_score_pre"] = (lane_score + above_ma + buy_zone).clip(0, 20)
    out["pre_score"] = out["policy_base_score"] + out["flow_score_pre"] + out["ep_ma_score_pre"]
    return out.sort_values("pre_score", ascending=False, kind="mergesort")


def fmp_get_json(path: str, api_key: str, params: dict[str, Any] | None = None) -> Any:
    params = dict(params or {})
    params["apikey"] = api_key
    response = requests.get(f"{FMP_BASE}/{path.lstrip('/')}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def score_news(symbol: str, api_key: str, from_date: str, to_date: str) -> dict[str, Any]:
    try:
        news = fmp_get_json(
            "stock_news",
            api_key,
            {"tickers": symbol, "from": from_date, "to": to_date, "limit": 10},
        )
    except Exception as exc:
        return {"news_score": 0.0, "policy_news_hits": 0, "news_titles": "", "fmp_news_error": str(exc)[:120]}
    if not isinstance(news, list):
        news = []
    hits = 0
    titles: list[str] = []
    for article in news:
        text = " ".join([norm_text(article.get("title")), norm_text(article.get("text"))])
        article_hits = sum(1 for word in POLICY_NEWS_KEYWORDS if word in text)
        if article_hits:
            hits += article_hits
            title = str(article.get("title", "")).strip()
            if title:
                titles.append(title)
    return {
        "news_score": min(10.0, hits * 1.8),
        "policy_news_hits": int(hits),
        "news_titles": " || ".join(titles[:3]),
        "fmp_news_error": "",
    }


def score_earnings(symbol: str, api_key: str, asof: pd.Timestamp) -> dict[str, Any]:
    out: dict[str, Any] = {
        "earnings_score": 0.0,
        "latest_earnings_date": pd.NaT,
        "eps_surprise_pct": np.nan,
        "revenue_yoy_pct": np.nan,
        "fmp_earnings_error": "",
    }
    try:
        surprises = fmp_get_json(f"earnings-surprises/{symbol}", api_key, {"limit": 4})
    except Exception as exc:
        surprises = []
        out["fmp_earnings_error"] = str(exc)[:120]
    try:
        income = fmp_get_json(f"income-statement/{symbol}", api_key, {"period": "quarter", "limit": 6})
    except Exception as exc:
        income = []
        if not out["fmp_earnings_error"]:
            out["fmp_earnings_error"] = str(exc)[:120]

    score = 0.0
    if isinstance(surprises, list) and surprises:
        latest = surprises[0]
        date = pd.to_datetime(latest.get("date"), errors="coerce")
        out["latest_earnings_date"] = date
        actual = pd.to_numeric(latest.get("actualEarningResult"), errors="coerce")
        estimate = pd.to_numeric(latest.get("estimatedEarning"), errors="coerce")
        if pd.notna(date) and (asof - date).days <= 60:
            score += 4.0
        if pd.notna(actual) and pd.notna(estimate) and abs(float(estimate)) > 1e-9:
            surprise = float(actual - estimate) / abs(float(estimate))
            out["eps_surprise_pct"] = surprise
            score += float(np.clip(surprise, -0.20, 0.50) + 0.20) / 0.70 * 7.0

    if isinstance(income, list) and len(income) >= 5:
        latest_rev = pd.to_numeric(income[0].get("revenue"), errors="coerce")
        year_ago_rev = pd.to_numeric(income[4].get("revenue"), errors="coerce")
        if pd.notna(latest_rev) and pd.notna(year_ago_rev) and float(year_ago_rev) > 0:
            growth = float(latest_rev / year_ago_rev - 1.0)
            out["revenue_yoy_pct"] = growth
            score += float(np.clip(growth, -0.20, 0.60) + 0.20) / 0.80 * 9.0

    out["earnings_score"] = min(20.0, max(0.0, score))
    return out


def enrich_with_fmp(frame: pd.DataFrame, cfg: RankConfig, api_key: str | None) -> pd.DataFrame:
    out = frame.copy()
    out["news_score"] = 0.0
    out["policy_news_hits"] = 0
    out["news_titles"] = ""
    out["earnings_score"] = 0.0
    out["latest_earnings_date"] = pd.NaT
    out["eps_surprise_pct"] = np.nan
    out["revenue_yoy_pct"] = np.nan
    out["fmp_news_error"] = ""
    out["fmp_earnings_error"] = ""
    if not api_key:
        out["news_score"] = 0.0
        out["policy_news_hits"] = 0
        out["earnings_score"] = 0.0
        return out
    asof = pd.to_datetime(out["latest_date"].max())
    from_date = (asof - timedelta(days=45)).strftime("%Y-%m-%d")
    to_date = (asof + timedelta(days=2)).strftime("%Y-%m-%d")
    indices = list(out.head(cfg.fmp_candidate_limit).index)
    for n, idx in enumerate(indices, 1):
        symbol = str(out.at[idx, "symbol"])
        if n % 10 == 1:
            print(f"FMP enrichment {n}-{min(n + 9, len(indices))} / {len(indices)}")
        news = score_news(symbol, api_key, from_date, to_date)
        earnings = score_earnings(symbol, api_key, asof)
        for key, value in {**news, **earnings}.items():
            out.at[idx, key] = value
    out["news_score"] = pd.to_numeric(out["news_score"], errors="coerce").fillna(0.0)
    out["policy_news_hits"] = pd.to_numeric(out["policy_news_hits"], errors="coerce").fillna(0).astype(int)
    out["earnings_score"] = pd.to_numeric(out["earnings_score"], errors="coerce").fillna(0.0)
    out["eps_surprise_pct"] = pd.to_numeric(out["eps_surprise_pct"], errors="coerce")
    out["revenue_yoy_pct"] = pd.to_numeric(out["revenue_yoy_pct"], errors="coerce")
    return out


def final_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["policy_score"] = (out["policy_base_score"] + out["news_score"]).clip(0, 30)
    out["flow_score"] = out["flow_score_pre"].clip(0, 30)
    out["ep_ma_score"] = out["ep_ma_score_pre"].clip(0, 20)
    out["earnings_score"] = out["earnings_score"].clip(0, 20)
    penalty = np.zeros(len(out), dtype=np.float64)
    penalty += np.where(out["close_vs_10ma_pct"] > 0.20, 10.0, 0.0)
    penalty += np.where(out["return_20d_pct"] > 0.80, 8.0, 0.0)
    penalty += np.where(out["close_vs_10ma_pct"] < 0.0, 8.0, 0.0)
    penalty += np.where(out["close_vs_20ma_pct"] < 0.0, 5.0, 0.0)
    out["extension_penalty"] = penalty
    out["total_score"] = (
        out["policy_score"] + out["flow_score"] + out["earnings_score"] + out["ep_ma_score"] - out["extension_penalty"]
    ).clip(0, 100)
    dist = out["close_vs_10ma_pct"]
    out["action_bucket"] = np.select(
        [
            (dist >= 0) & (dist <= 0.08) & (out["total_score"] >= 65),
            (dist > 0.08) & (dist <= 0.14) & (out["total_score"] >= 60),
            dist > 0.14,
            dist < 0,
        ],
        ["buy_zone", "wait_for_10ma_pullback", "extended_wait", "failed_10ma"],
        default="watch",
    )
    return out.sort_values("total_score", ascending=False, kind="mergesort")


def write_report(out_dir: Path, cfg: RankConfig, ranked: pd.DataFrame, elapsed: float, used_fmp: bool) -> None:
    top = ranked.head(cfg.top_n).copy()
    lines: list[str] = []
    lines.append("# Policy Theme x Flow x Earnings x EP/10MA Ranking")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Runtime: {elapsed:.2f} seconds")
    lines.append(f"- Latest price date: {pd.Timestamp(ranked['latest_date'].max()).date()}")
    lines.append(f"- Window: last {cfg.recent_sessions} trading sessions")
    lines.append(f"- FMP enrichment: {'used' if used_fmp else 'not used'}")
    lines.append("- Score weights: policy 30, flow 30, earnings 20, EP/10MA 20, with extension/MA-break penalties.")
    lines.append("")
    lines.append("## Top Ranking")
    lines.append("")
    lines.append(
        "| Rank | Symbol | Score | Action | Themes | Close | 20d R | vs 10MA | EP | EPS surprise | Rev YoY | News hits |"
    )
    lines.append("|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|")
    for rank, row in enumerate(top.itertuples(index=False), 1):
        ep = f"{row.recent_ep_lane} {pd.Timestamp(row.recent_ep_date).date()}" if str(row.recent_ep_lane) else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row.symbol),
                    f"{float(row.total_score):.1f}",
                    str(row.action_bucket),
                    str(row.policy_themes),
                    price(row.close),
                    pct(row.return_20d_pct),
                    pct(row.close_vs_10ma_pct),
                    ep,
                    pct(row.eps_surprise_pct),
                    pct(row.revenue_yoy_pct),
                    str(int(row.policy_news_hits)),
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
    lines.append("## Notes")
    lines.append("")
    lines.append("- `buy_zone` means the stock is still above/near 10MA and not too extended by this scoring model.")
    lines.append("- `wait_for_10ma_pullback` and `extended_wait` can still be strong leaders, but the model avoids chasing them.")
    lines.append("- FMP news is used as confirmation only; price/volume and earnings carry the decision.")
    (out_dir / "policy_theme_ep_ranking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    cfg = RankConfig(
        recent_sessions=args.recent_sessions,
        fmp_candidate_limit=args.fmp_candidate_limit,
        top_n=args.top_n,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    features, _codes, _group_end = load_features(Path(args.daily_input), cfg.feature_start)
    metrics = build_recent_metrics(features, universe, cfg)
    prelim = preliminary_scores(metrics, cfg)
    api_key = None if args.skip_fmp else os.environ.get("FMP_API_KEY")
    enriched = enrich_with_fmp(prelim, cfg, api_key)
    ranked = final_scores(enriched)
    elapsed = time.perf_counter() - started

    metrics.to_csv(out_dir / "policy_theme_metrics_all.csv", index=False)
    prelim.to_csv(out_dir / "policy_theme_preliminary_candidates.csv", index=False)
    ranked.to_csv(out_dir / "policy_theme_ep_ranking.csv", index=False)
    ranked.head(cfg.top_n).to_csv(out_dir / "policy_theme_ep_ranking_top.csv", index=False)
    write_report(out_dir, cfg, ranked, elapsed, bool(api_key))
    print(f"Ranked candidates: {len(ranked):,}")
    print(f"Top score: {ranked['total_score'].max():.1f}")
    print(f"Wrote: {out_dir / 'policy_theme_ep_ranking_report.md'}")
    print(f"Runtime: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
