"""
EXP-030 Pre-Registered In-Cohort Elbow Selector
File: experiments/EXP-030/select_exp030_elbow.py

Applies the locked deterministic BIDSleep N=47 Macro-F1 vs. parameter count
elbow selection rule to choose CNN-BiGRU-Star BEFORE any downstream evaluation
is run, writing experiments/EXP-030/elbow_selection.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

EXP030_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP030_DIR, "..", ".."))
if EXP030_DIR not in sys.path:
    sys.path.insert(0, EXP030_DIR)

from exp030_models import EXP030_GRID, EXP030_GRID_ORDER


def select_elbow_from_scores(scores_pct: Dict[str, float]) -> Dict[str, Any]:
    """
    Pure deterministic implementation of the locked EXP-030 elbow selection rule.

    Args:
        scores_pct: Mapping from grid tag (p1_100k .. p6_1051k) to BIDSleep N=47
                    participant-mean Macro-F1 in percentage points (%).

    Returns:
        Dictionary with star_tag, star_params, m_max, delta_max, threshold_T,
        guard_triggered, and grid_summary.
    """
    missing = [tag for tag in EXP030_GRID_ORDER if tag not in scores_pct]
    if missing:
        raise ValueError(f"Missing Macro-F1 scores for grid tags: {missing}")

    ordered_tags = sorted(EXP030_GRID_ORDER, key=lambda t: EXP030_GRID[t]["expected_params"])
    m_p2 = float(scores_pct["p2_164k"])
    m_max = float(max(scores_pct[t] for t in ordered_tags))
    delta_max = float(m_max - m_p2)

    if delta_max <= 0.50:
        guard_triggered = True
        threshold_T: Optional[float] = None
        star_tag = "p2_164k"
    else:
        guard_triggered = False
        threshold_T = float(max(m_p2 + 0.90 * delta_max, m_max - 0.20))
        eligible = [t for t in ordered_tags if float(scores_pct[t]) >= threshold_T - 1e-12]
        if not eligible:
            # Fallback to argmax if floating rounding edge occurs (should never happen)
            star_tag = max(ordered_tags, key=lambda t: float(scores_pct[t]))
        else:
            star_tag = eligible[0]  # ordered strictly by ascending parameter count

    grid_summary: List[Dict[str, Any]] = []
    for tag in ordered_tags:
        m_val = float(scores_pct[tag])
        grid_summary.append({
            "tag": tag,
            "n_params": int(EXP030_GRID[tag]["expected_params"]),
            "d_enc": int(EXP030_GRID[tag]["d_enc"]),
            "d_temp": int(EXP030_GRID[tag]["d_temp"]),
            "head_hidden": int(EXP030_GRID[tag]["head_hidden"]),
            "macro_f1_pct": m_val,
            "delta_vs_p2_pp": float(m_val - m_p2),
            "meets_threshold": bool(True if guard_triggered and tag == "p2_164k" else (not guard_triggered and m_val >= (threshold_T - 1e-12))),
            "selected": bool(tag == star_tag),
        })

    return {
        "star_tag": star_tag,
        "star_params": int(EXP030_GRID[star_tag]["expected_params"]),
        "m_p2_164k": m_p2,
        "m_max": m_max,
        "delta_max": delta_max,
        "threshold_T": threshold_T,
        "guard_triggered": guard_triggered,
        "grid_summary": grid_summary,
    }


def run_elbow_selection(exp030_dir: str = EXP030_DIR) -> Dict[str, Any]:
    scores_pct: Dict[str, float] = {}
    for tag in EXP030_GRID_ORDER:
        csv_path = os.path.join(exp030_dir, f"{tag}_subject_metrics.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Missing subject metrics CSV for {tag}: {csv_path}. "
                f"Run all 5 new grid points and --eval-p2-manifest before elbow selection."
            )
        df = pd.read_csv(csv_path)
        assert len(df) == 47, f"Expected 47 subjects in {csv_path}, got {len(df)}"
        scores_pct[tag] = float(df["macro_f1"].mean() * 100.0)

    selection = select_elbow_from_scores(scores_pct)

    # Cross-check against frozen DEC-037 CSV if present
    dec037_csv = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL", "final_model_oof_subject_metrics.csv")
    if os.path.exists(dec037_csv):
        df_dec = pd.read_csv(dec037_csv).groupby("subject_id", as_index=False)["macro_f1"].mean()
        selection["dec037_reference_macro_f1_pct"] = float(df_dec["macro_f1"].mean() * 100.0)

    out_path = os.path.join(exp030_dir, "elbow_selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(
        f"[ELBOW SELECTED] star_tag={selection['star_tag']} ({selection['star_params']:,} params) | "
        f"M_max={selection['m_max']:.2f}% | delta_max={selection['delta_max']:+.2f} pp | "
        f"guard_triggered={selection['guard_triggered']} | saved -> {out_path}"
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-030 Pre-Registered Elbow Selector")
    parser.add_argument("--exp030-dir", type=str, default=EXP030_DIR)
    args = parser.parse_args()
    run_elbow_selection(args.exp030_dir)


if __name__ == "__main__":
    main()
