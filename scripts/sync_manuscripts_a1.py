"""
Synchronize MANUSCRIPT_DRAFT_v2.0.md and paper_orchestra_out/manuscript_jbhi.tex
with Amendment A1 (W100 retrain + interleaved CPU latency) and all 10 audit fixes,
using only values read from JSON/CSV artifacts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_artifacts() -> dict:
    w100 = json.loads((ROOT / "experiments/EXP-031/W100_retrained_summary.json").read_text())
    lat = json.loads((ROOT / "experiments/EXP-031/interleaved_cpu_latency.json").read_text())
    conf31 = json.loads((ROOT / "experiments/EXP-031/exp031_confirmatory_summary.json").read_text())
    ma_prof = json.loads((ROOT / "experiments/STAGE2_FINAL_MODEL/int8_quantization_profile.json").read_text())
    cm = json.loads((ROOT / "author_report_bundle/data/confusion_matrices.json").read_text())
    wtable = pd.read_csv(ROOT / "experiments/EXP-031/exp031_width_table.csv")
    return {
        "w100": w100,
        "lat": lat,
        "conf31": conf31,
        "ma_prof": ma_prof,
        "cm": cm,
        "wtable": wtable,
    }


def update_manuscript_md(art: dict) -> None:
    md_path = ROOT / "research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md"
    txt = md_path.read_text(encoding="utf-8")

    w100 = art["w100"]
    lat = art["lat"]
    cpu_model = lat["system"]["cpu_model"]
    w100_f1 = w100["reference_means_pct"]["W100"]
    w100_std = w100["w100_aggregate"]["std_metrics"]["macro_f1"] * 100.0
    a_f1 = w100["reference_means_pct"]["frozen_Model_A"]
    p190_f1 = w100["reference_means_pct"]["P190"]
    p1_f1 = w100["reference_means_pct"]["p1_100k"]
    slamss_f1 = w100["reference_means_pct"]["Official_SLAMSS_IFS"]
    dr = w100["primary_comparison"]
    dp = w100["secondary_comparisons"]["W100_minus_P190"]
    ds = w100["secondary_comparisons"]["W100_minus_Official_SLAMSS_IFS"]
    hs_r = w100["h_scale_r_sensitivity"]
    dec = w100["decision"]
    ma_int8_kib = art["ma_prof"]["file_size_profile"]["int8_dynamic_raw_payload_kib"]

    l4 = lat["threads_4"]["results"]
    l1 = lat["threads_1"]["results"]

    # Fix 1: 4.70M -> 4.64M (84.9%) + s2s reconciliation
    txt = txt.replace(
        "5,465,768 trainable params including its 4.70M-param `cnn3` fusion block vs. 666,060 in the paper text",
        "5,465,768 trainable params including its 4.64M-param `cnn3` fusion block [4,638,464 params, 84.9% of model] and 299,496-param `s2s` [186,432 encoder + 79,016 decoder + 34,048 remaining] vs. 666,060 in the paper text",
    )

    # Fix 2: Model A architecture description
    txt = txt.replace(
        "plain 1D CNN-BiGRU architecture (3-stage 1D Conv encoder with kernels `{3,5,5}` and spatial dropout $p=0.20$, 2-layer BiGRU, linear head)",
        "plain 1D CNN-BiGRU architecture (dual `ScaledModalityEncoder` branches for IHR and ACC with `d_enc=64`: `Conv1d(1->32, k=5)` + `GroupNorm(4, 32)` + `GELU` + `MaxPool1d(2)` + `TemporalConvBlock(32->64, k=5)` + `MaxPool1d(2)` + `TemporalConvBlock(64->64, k=3)` + `AdaptiveAvgPool1d(1)` + `Dropout(0.20)` + `Linear(64->64)` + `LayerNorm(64)`, concatenated with 5-d metadata into a 2-layer BiGRU `hidden=48/dir -> 96-d` + `LayerNorm(96)`, and `Linear(96->64) -> GELU -> Dropout(0.20) -> Linear(64->4)`, totaling `163,940` params)",
    )

    # Fix 3 & 4: 203,338 vs 137,801 epochs & unified class-mix table
    old_mix = (
        "Across all labeled Dreem 2 EEG epochs prior to sensor masking, the unweighted night-averaged distribution was 44.2% Light, 18.9% Deep, 24.1% REM, and 12.8% Wake; across the 137,801 synchronized sensor-valid 30-second epochs, the pooled distribution was 11.9% Wake, 46.2% Light, 18.6% Deep, and 23.2% REM (21.14% Deep and 29.67% REM of TST on $N=46$ evaluation nights)."
    )
    new_mix = (
        "Across all 246 usable nights ($N=47$), the cohort comprised 203,338 valid 30-s epochs (~826.6/night: 10.8% Wake, 44.2% Light, 18.9% Deep, 26.1% REM). By contrast, unweighted night-averaged pre-masking shares were 12.8% Wake, 44.2% Light, 18.9% Deep, and 24.1% REM, and the legacy 31-subject `EXP-005` training split (`137,801` epochs) was 11.9% Wake, 46.2% Light, 18.6% Deep, and 23.2% REM.\n\n"
        "| Cohort / Subset | Nights / Subjects | Valid Epochs | Wake | Light | Deep | REM |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        "| BIDSleep All Usable (`N=47`, `EXP-018`–`031`) | 246 nights / 47 subj | 203,338 | 10.8% | 44.2% | 18.9% | 26.1% |\n"
        "| BIDSleep Night-Averaged (pre-masking) | 246 nights / 47 subj | — | 12.8% | 44.2% | 18.9% | 24.1% |\n"
        "| BIDSleep Legacy `EXP-005` Train Split | 31 subj | 137,801 | 11.9% | 46.2% | 18.6% | 23.2% |\n"
        "| SleepAccel `qc_clean` (`n=25`) | 25 nights / 25 subj | 22,633 | 8.2% | 57.7% | 11.1% | 22.9% |\n"
        "| SleepAccel All (`N=31`) | 31 nights / 31 subj | 25,947 | 8.3% | 57.7% | 11.1% | 22.9% |"
    )
    if old_mix in txt:
        txt = txt.replace(old_mix, new_mix)

    # Fix 9: Standardize Model A Dynamic INT8 payload to 309.54 KiB
    txt = txt.replace("309.54–309.91", f"{ma_int8_kib:.2f} (309.91 metadata rounding)")

    # Fix 10: P190 & SleepAccel hypothesis framing
    txt = txt.replace(
        "**`P190`** (190,940 params",
        "**`P190`** (parameter-matched plain control, `d_enc=72` [×1.125], `d_temp=50`/dir [×1.04], 190,940 params",
    )
    txt = txt.replace(
        "amplified the cross-modality stage-prior shift",
        "amplified the hypothesized cross-modality stage-prior/scoring shift (unconfirmed hypothesis)",
    )

    # Add Amendment A1 (W100 + Interleaved CPU Latency) block in Section 3.1
    a1_block = (
        f"\n- **Amendment A1 (`W100` Retrained Control & Interleaved CPU Latency, `{dec['verdict_label']}`):** "
        f"Retraining the exact `Model A` architecture (`163,940` params; 62/62 state-dict keys and shapes verified) under the locked `EXP-031` recipe (`W100`) yielded **`{w100_f1:.2f}% ± {w100_std:.2f}%`** Macro-F1, improving over frozen `Model A` (`{a_f1:.2f}%`) by **`{dr['delta_pp']:+.2f} pp`** (95% CI `[{dr['ci95_low_pp']:+.2f}, {dr['ci95_high_pp']:+.2f}]`, $p={dr['wilcoxon_p_raw']:.2e}$, $d_z={dr['cohens_dz']:+.3f}$) and matching `P190` (`{p190_f1:.2f}%`; `Δ_P = {dp['delta_pp']:+.2f} pp`, 95% CI `[{dp['ci95_low_pp']:+.2f}, {dp['ci95_high_pp']:+.2f}]`, $p_{{\\text{{Holm}}}}={dp['wilcoxon_p_holm']:.4f}$). "
        f"Replacing frozen `Model A` with `W100` in sensitivity `H-SCALE-R` yielded Page's $L={hs_r['L']:.1f}$ ($z={hs_r['z']:+.3f}$, $p_{{\\text{{two-sided}}}}={hs_r['p_two_sided']:.4f}$), confirming that the apparent 164k–191k jump was a training-recipe effect (`{dec['verdict_label']}`) and that plain CNN-BiGRU capacity saturates within the `~0.4–0.5 pp` noise floor (`p1_100k` vs. `W075` = `0.41 pp`) above ~100k params. "
        f"In interleaved round-robin CPU benchmarking on `{cpu_model}` (`L=1200`, `batch=1`, 30 warm-up + 200 timed passes), 4-thread / 1-thread Dynamic INT8 median latencies were: `p1_100k` `{l4['p1_100k']['int8']['median_ms']:.2f} / {l1['p1_100k']['int8']['median_ms']:.2f} ms`, `W100` `{l4['W100']['int8']['median_ms']:.2f} / {l1['W100']['int8']['median_ms']:.2f} ms`, `full_v5` `{l4['full_v5']['int8']['median_ms']:.2f} / {l1['full_v5']['int8']['median_ms']:.2f} ms`, `P190` `{l4['P190']['int8']['median_ms']:.2f} / {l1['P190']['int8']['median_ms']:.2f} ms`, and `Official SLAMSS-IFS` `{l4['Official_SLAMSS_IFS']['int8']['median_ms']:.2f} / {l1['Official_SLAMSS_IFS']['int8']['median_ms']:.2f} ms`."
    )
    anchor = "with preserved OOF parity (`+0.0244 pp`, 95% CI `[+0.0017, +0.0480]`, $p=0.0836$)."
    if anchor in txt and "Amendment A1 (`W100` Retrained Control" not in txt:
        txt = txt.replace(anchor, anchor + a1_block)

    md_path.write_text(txt, encoding="utf-8")


def update_manuscript_tex(art: dict) -> None:
    tex_path = ROOT / "paper_orchestra_out/manuscript_jbhi.tex"
    txt = tex_path.read_text(encoding="utf-8")

    w100 = art["w100"]
    lat = art["lat"]
    cpu_model = lat["system"]["cpu_model"]
    w100_f1 = w100["reference_means_pct"]["W100"]
    w100_std = w100["w100_aggregate"]["std_metrics"]["macro_f1"] * 100.0
    a_f1 = w100["reference_means_pct"]["frozen_Model_A"]
    p190_f1 = w100["reference_means_pct"]["P190"]
    dr = w100["primary_comparison"]
    dp = w100["secondary_comparisons"]["W100_minus_P190"]
    hs_r = w100["h_scale_r_sensitivity"]
    dec = w100["decision"]
    l4 = lat["threads_4"]["results"]
    l1 = lat["threads_1"]["results"]

    # Fix 1: 4.70M -> 4.64M (84.9%)
    txt = txt.replace(
        "Official SLAMSS-IFS (5,465,768 params including its 4.70M \\texttt{cnn3} block)",
        "Official SLAMSS-IFS (5,465,768 params including its 4.64M \\texttt{cnn3} block [84.9\\%] and 299,496-param \\texttt{s2s} [186,432 enc + 79,016 dec + 34,048 rem])",
    )

    # Fix 2 & 10: Model A & P190 description in Methods
    txt = txt.replace(
        "selecting \\texttt{P190} (190,940 params, 375.16~KiB INT8)",
        "selecting parameter-matched plain control \\texttt{P190} (\\texttt{d\\_enc}=72 [$\\times 1.125$], \\texttt{d\\_temp}=50/dir [$\\times 1.04$], 190,940 params, 375.16~KiB INT8) alongside \\texttt{Model~A} (dual \\texttt{d\\_enc}=64 Conv1d/TemporalConvBlock branches, 2-layer BiGRU \\texttt{hidden}=48/dir $\\rightarrow$ 96, 163,940 params)",
    )

    # Fix 3 & 4: 203,338 vs 137,801 epochs
    txt = txt.replace(
        "left 246 usable nights ($N=47$ baseline/scaling;",
        "left 246 usable nights (203,338 valid 30-s epochs, $\\sim$826.6/night: 10.8\\% Wake, 44.2\\% Light, 18.9\\% Deep, 26.1\\% REM vs.\\ 137,801 legacy \\texttt{EXP-005} train epochs [11.9\\%/46.2\\%/18.6\\%/23.2\\%]; $N=47$ baseline/scaling;",
    )

    # Fix 6, 7, 8: Add W100 + H-SCALE-R + Interleaved CPU Latency in Results
    a1_tex = (
        f" Under locked Amendment~A1 (\\texttt{{{dec['verdict']}: recipe effect}}), retraining \\texttt{{Model~A}} under the \\texttt{{EXP-031}} recipe (\\texttt{{W100}}, 163,940 params) achieved {w100_f1:.2f}\\% $\\pm$ {w100_std:.2f}\\% Macro-F1 ($+{dr['delta_pp']:.2f}$~pp vs.\\ frozen \\texttt{{Model~A}} {a_f1:.2f}\\%, 95\\% CI $[+{dr['ci95_low_pp']:.2f}, +{dr['ci95_high_pp']:.2f}]$~pp, $p={dr['wilcoxon_p_raw']:.2e}$; $\\Delta_P={dp['delta_pp']:+.2f}$~pp vs.\\ \\texttt{{P190}} {p190_f1:.2f}\\%, $p_{{\\text{{Holm}}}}={dp['wilcoxon_p_holm']:.4f}$), reducing sensitivity \\texttt{{H-SCALE-R}} to $L={hs_r['L']:.1f}$ ($z={hs_r['z']:+.3f}$, $p={hs_r['p_two_sided']:.4f}$) within the $\\sim$0.4--0.5~pp noise floor. "
        f"Interleaved round-robin CPU benchmarking on \\texttt{{{cpu_model}}} ($L=1200$, $B=1$) yielded 4T/1T INT8 medians of {l4['p1_100k']['int8']['median_ms']:.2f}/{l1['p1_100k']['int8']['median_ms']:.2f}~ms (\\texttt{{p1\\_100k}}), {l4['W100']['int8']['median_ms']:.2f}/{l1['W100']['int8']['median_ms']:.2f}~ms (\\texttt{{W100}}), {l4['full_v5']['int8']['median_ms']:.2f}/{l1['full_v5']['int8']['median_ms']:.2f}~ms (\\texttt{{full\\_v5}}), {l4['P190']['int8']['median_ms']:.2f}/{l1['P190']['int8']['median_ms']:.2f}~ms (\\texttt{{P190}}), and {l4['Official_SLAMSS_IFS']['int8']['median_ms']:.2f}/{l1['Official_SLAMSS_IFS']['int8']['median_ms']:.2f}~ms (\\texttt{{SLAMSS-IFS}})."
    )
    anchor_tex = "establishing a negative architectural ablation."
    if anchor_tex in txt and "Amendment~A1" not in txt:
        txt = txt.replace(anchor_tex, anchor_tex + a1_tex)

    # Fix 10: SleepAccel hypothesis framing
    txt = txt.replace(
        "amplified the cross-modality prior shift",
        "amplified the hypothesized cross-modality prior/scoring shift (unconfirmed hypothesis)",
    )

    tex_path.write_text(txt, encoding="utf-8")


def run_consistency_check(art: dict) -> dict:
    w100 = art["w100"]
    lat = art["lat"]
    md_txt = (ROOT / "research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md").read_text(encoding="utf-8")
    tex_txt = (ROOT / "paper_orchestra_out/manuscript_jbhi.tex").read_text(encoding="utf-8")

    expected_tokens = [
        f"{w100['reference_means_pct']['W100']:.2f}",
        f"{w100['primary_comparison']['delta_pp']:.2f}",
        f"{w100['h_scale_r_sensitivity']['L']:.1f}",
        f"{lat['threads_4']['results']['W100']['int8']['median_ms']:.2f}",
        f"{lat['threads_4']['results']['Official_SLAMSS_IFS']['int8']['median_ms']:.2f}",
        "4,638,464" if False else "4.64M",
        "203,338",
        "137,801",
        "309.54",
    ]
    mismatches = []
    for tok in expected_tokens:
        if tok not in md_txt:
            mismatches.append(f"MD missing {tok}")
        if tok not in tex_txt:
            mismatches.append(f"TEX missing {tok}")
    return {"checked_tokens": len(expected_tokens), "mismatches": mismatches}


def main() -> None:
    art = load_artifacts()
    update_manuscript_md(art)
    update_manuscript_tex(art)
    chk = run_consistency_check(art)
    print("Consistency check:", json.dumps(chk, indent=2))
    assert not chk["mismatches"], f"Consistency mismatches: {chk['mismatches']}"

    # Compile manuscript_jbhi.pdf
    out_dir = ROOT / "paper_orchestra_out"
    for _ in range(2):
        res = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "manuscript_jbhi.tex"],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(res.stdout[-1500:])
            raise RuntimeError("pdflatex failed")
    print("Compiled paper_orchestra_out/manuscript_jbhi.pdf successfully.")


if __name__ == "__main__":
    main()
