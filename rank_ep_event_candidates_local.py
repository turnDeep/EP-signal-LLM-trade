#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THEME_KEYWORDS: dict[str, list[str]] = {
    "ai_semiconductor": [
        "ai",
        "artificial intelligence",
        "hyperscaler",
        "data center",
        "datacenter",
        "semiconductor",
        "nvidia",
        "gpu",
        "inference",
        "cloud",
    ],
    "space_defense": ["space", "satellite", "nasa", "defense", "hypersonic", "lunar"],
    "critical_minerals": ["critical minerals", "rare earth", "lithium", "uranium", "mining"],
    "nuclear_grid": ["nuclear", "power grid", "utility", "transmission", "electrical"],
    "medical_ai": ["ai diagnostics", "cardiac diagnostics", "diagnostics", "hla", "sequencing", "clinical"],
}


def keyword_found(haystack: str, keyword: str) -> bool:
    term = keyword.lower().strip()
    if not term:
        return False
    if " " in term:
        pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    else:
        pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local EP-event recognition-gap ranking with entry timing separated.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "nat"}:
        return ""
    return text


def num(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def parse_titles(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean(v) for v in value if clean(v)]
    text = clean(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [clean(v) for v in parsed if clean(v)]
    except Exception:
        pass
    if "||" in text:
        return [part.strip() for part in text.split("||") if part.strip()]
    return [text]


def clip_score(value: float, low: float, high: float, points: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((value - low) / (high - low), 0, 1) * points)


def theme_score(row: pd.Series) -> tuple[float, list[str]]:
    haystack = " ".join(
        [
            clean(row.get("company_name")),
            clean(row.get("sector")),
            clean(row.get("industry")),
            clean(row.get("policy_themes")),
            " ".join(parse_titles(row.get("news_titles_combined"))),
            " ".join(parse_titles(row.get("fmp_news_titles"))),
        ]
    ).lower()
    matched: list[str] = []
    score = 0.0
    for theme, words in THEME_KEYWORDS.items():
        if any(keyword_found(haystack, word) for word in words):
            matched.append(theme)
            score += 4.0
    if "record" in haystack and any(word in haystack for word in ["order", "contract", "award"]):
        score += 3.0
        matched.append("record_order")
    if any(word in haystack for word in ["guidance", "raises", "beat", "top momentum"]):
        score += 1.5
    return min(score, 18.0), list(dict.fromkeys(matched))


def ep_score(row: pd.Series) -> float:
    raw = num(row.get("entry_ep_score"), num(row.get("recent_ep_score"), 0.0))
    lane = clean(row.get("entry_ep_lane") or row.get("recent_ep_lane"))
    lane_bonus = {"a_plus_gap_ep": 8.0, "gap_ep": 6.0, "displacement_ep": 4.0}.get(lane, 0.0)
    volume_exp = num(row.get("ep_dollar_volume_expansion_vs_20d_pre"), num(row.get("ep_volume_ratio_20"), np.nan))
    return min(22.0, clip_score(raw, 50, 180, 14.0) + lane_bonus + clip_score(np.log1p(volume_exp), 0, np.log1p(12), 4.0))


def fundamentals_score(row: pd.Series) -> tuple[float, list[str]]:
    notes: list[str] = []
    rev = num(row.get("rev_yoy"), num(row.get("revenue_yoy_pct"), np.nan))
    qoq = num(row.get("rev_qoq"), np.nan)
    gm = num(row.get("latest_gross_margin"), np.nan)
    ni = num(row.get("latest_net_income"), np.nan)
    eps = num(row.get("eps_surprise_pct"), np.nan)
    accel = bool(row.get("rev_accelerating") is True or clean(row.get("rev_accelerating")).lower() == "true")

    score = 0.0
    score += clip_score(rev, -0.1, 1.0, 16.0)
    score += clip_score(qoq, -0.05, 0.25, 8.0)
    if accel:
        score += 5.0
        notes.append("rev_accelerating")
    score += clip_score(gm, 0.1, 0.7, 7.0)
    if np.isfinite(ni) and ni > 0:
        score += 4.0
        notes.append("profitable")
    if np.isfinite(eps):
        if -2.0 <= eps <= 2.0:
            score += clip_score(eps, -0.2, 0.5, 5.0)
        elif abs(eps) > 2.0:
            notes.append("eps_low_base_caution")

    if np.isfinite(rev) and rev < 0:
        score -= 8.0
        notes.append("revenue_declining")
    if np.isfinite(gm) and gm < 0:
        score -= 6.0
        notes.append("negative_gross_margin")
    return max(0.0, min(score, 40.0)), notes


def data_quality_penalty(row: pd.Series) -> float:
    penalty = 0.0
    latest_reported = pd.to_datetime(row.get("latest_reported_date") or row.get("income_report_date"), errors="coerce")
    asof = pd.to_datetime(row.get("asof_date"), errors="coerce")
    if pd.notna(latest_reported) and pd.notna(asof) and (asof - latest_reported).days > 150:
        penalty += 5.0
    if clean(row.get("symbol")).upper() == "NSA":
        titles = " ".join(parse_titles(row.get("fmp_news_titles")) + parse_titles(row.get("news_titles_combined"))).lower()
        if "anthropic" in titles or "national security" in titles:
            penalty += 12.0
    return penalty


def entry_timing(row: pd.Series) -> tuple[str, str]:
    dist10 = num(row.get("close_vs_10ma_pct"), np.nan)
    days = num(row.get("days_since_ep"), np.nan)
    if np.isfinite(days) and days <= 0:
        return "ep_day_watchlist", "EP当日。翌日以降の押し目・高値突破確認待ち。"
    if np.isfinite(dist10):
        if dist10 < -0.03:
            return "wait_for_10ma_reclaim", "10MAを下回っており、回復確認が必要。"
        if -0.03 <= dist10 <= 0.08:
            return "entry_zone_10ma_hold", "10MA近辺。押し目維持の監視対象。"
        if dist10 <= 0.15:
            return "extended_watch_pullback", "やや伸びているため追わずに浅い押し目待ち。"
        return "extended_no_chase", "10MAから大きく乖離。EP候補として監視し、押し目待ち。"
    return "insufficient_timing_data", "10MA距離が不足。"


def classify_thesis(row: pd.Series, themes: list[str]) -> str:
    industry = clean(row.get("industry"))
    if "ai_semiconductor" in themes:
        return f"{industry} -> AI/data-center semiconductor rerating"
    if "medical_ai" in themes:
        return f"{industry} -> AI/advanced diagnostics platform"
    if "space_defense" in themes:
        return f"{industry} -> space/defense infrastructure"
    if "critical_minerals" in themes or "nuclear_grid" in themes:
        return f"{industry} -> policy-backed strategic resource/infrastructure"
    return f"{industry or 'legacy business'} -> post-EP recognition-gap candidate"


def build_ranking(df: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    out = df.copy()
    out["asof_date"] = asof_date
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        e = ep_score(row)
        f, f_notes = fundamentals_score(row)
        t, themes = theme_score(row)
        penalty = data_quality_penalty(row)
        total = e + f + t - penalty
        state, reason = entry_timing(row)
        rows.append(
            {
                "symbol": clean(row.get("symbol")).upper(),
                "company_name": clean(row.get("company_name")),
                "asof_date": asof_date,
                "total_score": round(total, 2),
                "ep_component": round(e, 2),
                "fundamental_component": round(f, 2),
                "theme_component": round(t, 2),
                "data_quality_penalty": round(penalty, 2),
                "entry_timing_state": state,
                "entry_timing_reason": reason,
                "thesis": classify_thesis(row, themes),
                "matched_themes": ", ".join(themes),
                "fundamental_notes": ", ".join(f_notes),
                "entry_ep_date": clean(row.get("entry_ep_date") or row.get("recent_ep_date")),
                "entry_ep_lane": clean(row.get("entry_ep_lane") or row.get("recent_ep_lane")),
                "entry_ep_score": num(row.get("entry_ep_score"), np.nan),
                "close": num(row.get("close"), np.nan),
                "close_vs_10ma_pct": num(row.get("close_vs_10ma_pct"), np.nan),
                "return_20d_pct": num(row.get("return_20d_pct"), np.nan),
                "ep_to_current_return_pct": num(row.get("ep_to_current_return_pct"), np.nan),
                "revenue_yoy_pct": num(row.get("rev_yoy"), num(row.get("revenue_yoy_pct"), np.nan)),
                "rev_qoq_pct": num(row.get("rev_qoq"), np.nan),
                "rev_accelerating": clean(row.get("rev_accelerating")),
                "latest_gross_margin": num(row.get("latest_gross_margin"), np.nan),
                "latest_net_income": num(row.get("latest_net_income"), np.nan),
                "eps_surprise_pct": num(row.get("eps_surprise_pct"), np.nan),
                "news_titles": " || ".join(parse_titles(row.get("news_titles_combined"))[:3] + parse_titles(row.get("fmp_news_titles"))[:3]),
            }
        )
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(["total_score", "ep_component", "fundamental_component"], ascending=False, kind="mergesort")
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def pct(v: Any) -> str:
    n = num(v)
    return "" if not np.isfinite(n) else f"{n * 100:.1f}%"


def write_report(out_dir: Path, ranked: pd.DataFrame, top_n: int) -> None:
    top = ranked.head(top_n)
    lines = [
        "# EP Event Recognition-Gap Ranking",
        "",
        "- Ranking score excludes entry timing.",
        "- Entry timing is appended as a separate state.",
        "",
        "## Top Candidates",
        "",
        "| Rank | Symbol | Score | Thesis | EP | Rev YoY | QoQ | GM | Entry timing |",
        "|---:|---|---:|---|---|---:|---:|---:|---|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['symbol']} | {r['total_score']:.1f} | "
            f"{r['thesis'][:55]} | {r['entry_ep_lane']} {r['entry_ep_score']:.1f} | "
            f"{pct(r['revenue_yoy_pct'])} | {pct(r['rev_qoq_pct'])} | {pct(r['latest_gross_margin'])} | "
            f"{r['entry_timing_state']} |"
        )
    lines += ["", "## Entry Timing Notes", ""]
    for _, r in top.iterrows():
        lines.append(f"- {r['symbol']}: {r['entry_timing_reason']}")
    (out_dir / "ep_event_recognition_gap_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.candidates)
    ranked = build_ranking(df, args.asof_date)
    ranked.to_csv(out_dir / "ep_event_recognition_gap_ranking.csv", index=False)
    ranked.head(args.top_n).to_csv(out_dir / "ep_event_recognition_gap_ranking_top.csv", index=False)
    write_report(out_dir, ranked, args.top_n)
    print(f"Ranked: {len(ranked):,}")
    print(f"Wrote: {out_dir / 'ep_event_recognition_gap_ranking.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
