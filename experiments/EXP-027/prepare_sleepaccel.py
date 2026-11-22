#!/usr/bin/env python3
"""
prepare_sleepaccel.py
=====================
Downloads PhysioNet SleepAccel v1.0.0 (DOI: 10.13026/hmhs-py35), verifies SHA-256
integrity, runs exact BIDSleep prep_57738d534b parity preprocessing, and executes
the pre-flight distribution comparison gate before GPU inference.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data", "external", "sleepaccel_v1.0.0")
DEFAULT_BIDSLEEP_CACHE = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
S3_URI = "s3://physionet-open/sleep-accel/1.0.0/"
HTTPS_URL = "https://physionet.org/files/sleep-accel/1.0.0/"


class ExternalValidationGateError(RuntimeError):
    """Raised when pre-flight unit, range, or distribution parity checks fail."""


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_sleepaccel(data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Downloads SleepAccel v1.0.0 via AWS S3 (--no-sign-request) or PhysioNet HTTPS."""
    os.makedirs(data_dir, exist_ok=True)
    motion_ok = len(glob.glob(os.path.join(data_dir, "motion", "*_acceleration.txt"))) >= 31
    hr_ok = len(glob.glob(os.path.join(data_dir, "heart_rate", "*_heartrate.txt"))) >= 31
    lbl_ok = len(glob.glob(os.path.join(data_dir, "labels", "*_labeled_sleep.txt"))) >= 31
    if motion_ok and hr_ok and lbl_ok:
        return

    if shutil.which("aws"):
        cmd = ["aws", "s3", "sync", "--no-sign-request", S3_URI, data_dir]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0:
            return
    if shutil.which("wget"):
        cmd = ["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-P", data_dir, HTTPS_URL]
        subprocess.run(cmd, check=True)
        return
    raise RuntimeError("Neither `aws` CLI nor `wget` succeeded for downloading SleepAccel v1.0.0.")


def verify_sleepaccel_manifest(data_dir: str, out_json: str) -> Dict[str, Any]:
    """Verifies against SHA256SUMS.txt if present and writes sleepaccel_sha256_manifest.json."""
    sums_path = os.path.join(data_dir, "SHA256SUMS.txt")
    expected: Dict[str, str] = {}
    if os.path.exists(sums_path):
        with open(sums_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    expected[parts[1].lstrip("./")] = parts[0]

    files_info: Dict[str, Any] = {}
    mismatches: List[str] = []
    for sub_dir, pat in [
        ("motion", "*_acceleration.txt"),
        ("heart_rate", "*_heartrate.txt"),
        ("labels", "*_labeled_sleep.txt"),
    ]:
        for fp in sorted(glob.glob(os.path.join(data_dir, sub_dir, pat))):
            rel = os.path.relpath(fp, data_dir)
            digest = compute_sha256(fp)
            matched = (expected[rel] == digest) if rel in expected else None
            if matched is False:
                mismatches.append(rel)
            files_info[rel] = {"sha256": digest, "bytes": os.path.getsize(fp), "upstream_match": matched}

    if mismatches:
        raise ExternalValidationGateError(f"SHA-256 mismatch vs SHA256SUMS.txt: {mismatches}")

    manifest = {"data_dir": data_dir, "n_files": len(files_info), "files": files_info}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def robust_scale(v: np.ndarray, floor: float) -> np.ndarray:
    """Exact BIDSleep prep_57738d534b Cell 16 L1-3 robust scaler with [-10, 10] clip."""
    med = np.nanmedian(v)
    iqr = np.nanpercentile(v, 75.0) - np.nanpercentile(v, 25.0)
    return np.clip(np.nan_to_num((v - med) / max(float(iqr), floor)), -10.0, 10.0).astype(np.float32)


def nearest_gap(t_raw: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Exact BIDSleep prep_57738d534b Cell 16 L5-8 nearest sample distance."""
    i = np.searchsorted(t_raw, grid)
    lo = t_raw[np.clip(i - 1, 0, len(t_raw) - 1)]
    hi = t_raw[np.clip(i, 0, len(t_raw) - 1)]
    return np.minimum(np.abs(grid - lo), np.abs(grid - hi))


def map_sleepaccel_labels(raw_stages: np.ndarray) -> np.ndarray:
    """Exact 4-class map: 0->0, {1,2}->1, 3->2, 5->3, all else -> -100."""
    y = np.full(len(raw_stages), -100, dtype=np.int64)
    y[raw_stages == 0] = 0
    y[(raw_stages == 1) | (raw_stages == 2)] = 1
    y[raw_stages == 3] = 2
    y[raw_stages == 5] = 3
    return y


def _load_txt_auto(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    delim = "," if "," in first else None
    arr = np.loadtxt(path, delimiter=delim, dtype=np.float64)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def preprocess_sleepaccel_arrays(
    t_hr_raw: np.ndarray,
    bpm_raw: np.ndarray,
    t_mo_raw: np.ndarray,
    xyz_raw: np.ndarray,
    t_lbl_raw: np.ndarray,
    stage_raw: np.ndarray,
    subject_id: str,
) -> Dict[str, Any]:
    """
    Pure array-level preprocessing matching BIDSleep prep_57738d534b Cell 9 & Cell 16.
    1. Align labels to contiguous 30s grid [0, 30*T).
    2. Restrict pre-PSG t < 0 to boundary anchor windows ([-120, 0) IHR, [-10, 0) ACC).
    3. Trim to [0, 30*T) BEFORE percentiles, std3 HR filtering, robust_scale, and [-10, 10] clip.
    """
    pos_lbl = t_lbl_raw >= 0
    t_lbl = t_lbl_raw[pos_lbl]
    s_lbl = stage_raw[pos_lbl]
    k_idx = np.round(t_lbl / 30.0).astype(np.int64)
    n_ep = int(np.max(k_idx)) + 1
    stage_grid = np.full(n_ep, -100, dtype=np.int64)
    mapped = map_sleepaccel_labels(s_lbl)
    for k, val in zip(k_idx, mapped):
        if 0 <= k < n_ep:
            stage_grid[k] = val

    n = n_ep * 30
    span_s = float(n)
    grid = np.arange(n, dtype=np.float64)

    # Sort raw streams
    hr_ord = np.argsort(t_hr_raw)
    t_hr_all, bpm_all = t_hr_raw[hr_ord], bpm_raw[hr_ord]
    mo_ord = np.argsort(t_mo_raw)
    t_mo_all, xyz_all = t_mo_raw[mo_ord], xyz_raw[mo_ord]

    # Trim-before-scale: [0, 30*T) for statistics/histograms; narrow [-120, 0) / [-10, 0) only for t=0 interp
    in_hr_night = (t_hr_all >= 0.0) & (t_hr_all <= span_s)
    t_hr_night = t_hr_all[in_hr_night]
    h_night = bpm_all[in_hr_night]

    edges = np.linspace(0.0, span_s, n + 1)
    raw_counts = np.histogram(t_hr_night, bins=edges)[0].astype(np.float64)
    freq_feat = np.log1p(raw_counts.reshape(n_ep, 30).sum(axis=1) / 30.0 * 10.0).astype(np.float32)

    keep_night = np.isfinite(h_night) & (h_night >= 30.0) & (h_night <= 200.0)
    if keep_night.sum() > 10 and h_night[keep_night].std() > 0:
        mu_h = h_night[keep_night].mean()
        sd_h = h_night[keep_night].std()
        keep_night &= np.abs((h_night - mu_h) / sd_h) <= 3.0
    if keep_night.sum() < 10:
        raise ExternalValidationGateError(f"{subject_id}: < 10 valid IHR samples after filtering")

    # Boundary-only t < 0 anchor for HR ([-120, 0) s)
    hr_anchor = (t_hr_all >= -120.0) & (t_hr_all < 0.0) & np.isfinite(bpm_all) & (bpm_all >= 30.0) & (bpm_all <= 200.0)
    t_hr_interp = np.concatenate([t_hr_all[hr_anchor], t_hr_night[keep_night]])
    v_hr_interp = np.concatenate([bpm_all[hr_anchor], h_night[keep_night]])
    hr_i = np.interp(grid, t_hr_interp, v_hr_interp)
    ok_hr = nearest_gap(t_hr_interp, grid) <= 120.0

    # ACC: trim to [0, 30*T) for [0.1, 99.9] percentiles, narrow [-10, 0) anchor for t=0 interp
    in_mo_night = (t_mo_all >= 0.0) & (t_mo_all <= span_s)
    t_mo_night = t_mo_all[in_mo_night]
    xyz_night = xyz_all[in_mo_night]
    mo_anchor = (t_mo_all >= -10.0) & (t_mo_all < 0.0)
    t_mo_interp = np.concatenate([t_mo_all[mo_anchor], t_mo_night])
    xyz_interp_src = np.vstack([xyz_all[mo_anchor], xyz_night])

    comps = []
    for ax_i in range(3):
        v_night = xyz_night[:, ax_i]
        lo, hi = np.percentile(v_night, [0.1, 99.9])
        v_clipped = np.clip(xyz_interp_src[:, ax_i], lo, hi)
        comps.append(np.interp(grid, t_mo_interp, v_clipped))
    mag = np.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2)
    ok_mo = nearest_gap(t_mo_interp, grid) <= 10.0

    # Raw physical medians over [0, 30*T) valid samples for pre-flight unit check
    raw_hr_median = float(np.nanmedian(h_night[keep_night]))
    raw_mag_median = float(np.nanmedian(mag[ok_mo])) if np.any(ok_mo) else float(np.nanmedian(mag))

    # Per-night robust scaling on [0, 30*T) + [-10, 10] clipping + gap masking
    ihr_scaled = robust_scale(hr_i, floor=2.0) * ok_hr.astype(np.float32)
    acc_scaled = robust_scale(mag, floor=0.005) * ok_mo.astype(np.float32)

    cov_hr = ok_hr.reshape(n_ep, 30).mean(axis=1)
    cov_mo = ok_mo.reshape(n_ep, 30).mean(axis=1)
    valid_ep = (cov_hr >= 0.5) & (cov_mo >= 0.5)

    ihr_2d = ihr_scaled.reshape(n_ep, 30).copy()
    acc_2d = acc_scaled.reshape(n_ep, 30).copy()
    ihr_2d[~valid_ep] = 0.0
    acc_2d[~valid_ep] = 0.0

    y_final = stage_grid.copy()
    y_final[~valid_ep] = -100

    usable_frac = float(np.mean(y_final >= 0))
    qc_pass = (usable_frac >= 0.60) and (n_ep >= 600)

    return {
        "subject": subject_id,
        "night": "1",
        "ihr": ihr_2d.astype(np.float32),
        "accel": acc_2d.astype(np.float32),
        "freq": freq_feat.reshape(n_ep, 1).astype(np.float32),
        "time": np.zeros((n_ep, 1), dtype=np.float32),
        "y": y_final.astype(np.int64),
        "n_epochs": n_ep,
        "usable_frac": usable_frac,
        "qc_pass": bool(qc_pass),
        "raw_hr_median": raw_hr_median,
        "raw_mag_median": raw_mag_median,
        "cov_hr_mean": float(cov_hr.mean()),
        "cov_mo_mean": float(cov_mo.mean()),
    }


def preprocess_sleepaccel_subject(subject_id: str, data_dir: str) -> Dict[str, Any]:
    hr_arr = _load_txt_auto(os.path.join(data_dir, "heart_rate", f"{subject_id}_heartrate.txt"))
    mo_arr = _load_txt_auto(os.path.join(data_dir, "motion", f"{subject_id}_acceleration.txt"))
    lb_arr = _load_txt_auto(os.path.join(data_dir, "labels", f"{subject_id}_labeled_sleep.txt"))
    return preprocess_sleepaccel_arrays(
        t_hr_raw=hr_arr[:, 0],
        bpm_raw=hr_arr[:, 1],
        t_mo_raw=mo_arr[:, 0],
        xyz_raw=mo_arr[:, 1:4],
        t_lbl_raw=lb_arr[:, 0],
        stage_raw=lb_arr[:, 1],
        subject_id=subject_id,
    )


def run_preflight_distribution_gate(
    records: List[Dict[str, Any]],
    bidsleep_cache_dir: Optional[str] = DEFAULT_BIDSLEEP_CACHE,
) -> Dict[str, Any]:
    """
    Aborts before GPU inference if raw unit mismatch (m/s^2 vs g, RR ms vs bpm),
    non-finite values, |z| > 10.0, or out-of-range meta[:, 0] is detected.
    """
    if not records:
        raise ExternalValidationGateError("No SleepAccel records supplied to pre-flight gate.")

    raw_mag_meds = [float(r["raw_mag_median"]) for r in records]
    raw_hr_meds = [float(r["raw_hr_median"]) for r in records]
    cohort_mag_med = float(np.median(raw_mag_meds))
    cohort_hr_med = float(np.median(raw_hr_meds))

    if not (0.90 <= cohort_mag_med <= 1.10):
        raise ExternalValidationGateError(
            f"ACC unit mismatch: cohort median={cohort_mag_med:.3f} (expected [0.90, 1.10] g; ~9.81 indicates m/s^2)."
        )
    if not (45.0 <= cohort_hr_med <= 95.0):
        raise ExternalValidationGateError(
            f"IHR unit mismatch: cohort median={cohort_hr_med:.2f} (expected [45.0, 95.0] bpm; ~600-1200 indicates RR ms)."
        )

    std_ihr_meds, std_acc_meds, std_ihr_iqrs, std_acc_iqrs = [], [], [], []
    for r in records:
        valid = r["y"] >= 0
        if not np.any(valid):
            continue
        ihr_v = r["ihr"][valid]
        acc_v = r["accel"][valid]
        freq_v = r["freq"]
        if not (np.all(np.isfinite(ihr_v)) and np.all(np.isfinite(acc_v)) and np.all(np.isfinite(freq_v))):
            raise ExternalValidationGateError(f"Non-finite values in {r['subject']}")
        if np.max(np.abs(ihr_v)) > 10.0 or np.max(np.abs(acc_v)) > 10.0:
            raise ExternalValidationGateError(f"Unclipped |z| > 10.0 in {r['subject']}")
        if np.min(freq_v) < 0.0 or np.max(freq_v) > 5.0:
            raise ExternalValidationGateError(f"Out-of-range meta[:,0] freq in {r['subject']}")

        std_ihr_meds.append(float(np.median(ihr_v)))
        std_acc_meds.append(float(np.median(acc_v)))
        std_ihr_iqrs.append(float(np.percentile(ihr_v, 75) - np.percentile(ihr_v, 25)))
        std_acc_iqrs.append(float(np.percentile(acc_v, 75) - np.percentile(acc_v, 25)))

    med_ihr = float(np.median(std_ihr_meds))
    med_acc = float(np.median(std_acc_meds))
    iqr_ihr = float(np.median(std_ihr_iqrs))
    iqr_acc = float(np.median(std_acc_iqrs))

    if abs(med_ihr) >= 0.25 or abs(med_acc) >= 0.25:
        raise ExternalValidationGateError(f"Standardized median drift: IHR={med_ihr:.3f}, ACC={med_acc:.3f}")
    if not (0.5 <= iqr_ihr <= 2.0) or not (0.5 <= iqr_acc <= 2.0):
        raise ExternalValidationGateError(f"Standardized IQR out of [0.5, 2.0]: IHR={iqr_ihr:.3f}, ACC={iqr_acc:.3f}")

    bidsleep_ref = {}
    if bidsleep_cache_dir and os.path.isdir(bidsleep_cache_dir):
        b_files = sorted(glob.glob(os.path.join(bidsleep_cache_dir, "*.npz")))[:20]
        if b_files:
            b_ihr = [float(np.median(np.load(f)["X"][:, 0, :])) for f in b_files]
            b_acc = [float(np.median(np.load(f)["X"][:, 6, :])) for f in b_files]
            bidsleep_ref = {"sample_n": len(b_files), "ihr_med": float(np.median(b_ihr)), "acc_med": float(np.median(b_acc))}

    return {
        "passed": True,
        "n_records": len(records),
        "n_qc_pass": int(sum(1 for r in records if r["qc_pass"])),
        "raw_mag_median_g": cohort_mag_med,
        "raw_hr_median_bpm": cohort_hr_med,
        "std_ihr_median": med_ihr,
        "std_acc_median": med_acc,
        "std_ihr_iqr": iqr_ihr,
        "std_acc_iqr": iqr_acc,
        "bidsleep_ref": bidsleep_ref,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare PhysioNet SleepAccel v1.0.0 for EXP-027/028")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--bidsleep-cache", default=DEFAULT_BIDSLEEP_CACHE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    download_sleepaccel(args.data_dir)
    verify_sleepaccel_manifest(args.data_dir, os.path.join(args.out_dir, "sleepaccel_sha256_manifest.json"))

    lbl_files = sorted(glob.glob(os.path.join(args.data_dir, "labels", "*_labeled_sleep.txt")))
    subjects = [os.path.basename(f).replace("_labeled_sleep.txt", "") for f in lbl_files]

    prep_dir = os.path.join(args.out_dir, "cache", "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)
    records = []
    qc_rows = []
    for sub in subjects:
        rec = preprocess_sleepaccel_subject(sub, args.data_dir)
        records.append(rec)
        np.savez_compressed(
            os.path.join(prep_dir, f"{sub}.npz"),
            ihr=rec["ihr"],
            accel=rec["accel"],
            freq=rec["freq"],
            time=rec["time"],
            y=rec["y"],
            qc_pass=np.array(rec["qc_pass"]),
        )
        qc_rows.append({
            "subject_id": sub,
            "n_epochs": rec["n_epochs"],
            "usable_frac": rec["usable_frac"],
            "qc_pass": rec["qc_pass"],
            "raw_hr_median": rec["raw_hr_median"],
            "raw_mag_median": rec["raw_mag_median"],
            "cov_hr_mean": rec["cov_hr_mean"],
            "cov_mo_mean": rec["cov_mo_mean"],
        })

    pd.DataFrame(qc_rows).to_csv(os.path.join(args.out_dir, "sleepaccel_qc_summary.csv"), index=False)
    gate = run_preflight_distribution_gate(records, args.bidsleep_cache)
    with open(os.path.join(args.out_dir, "preflight_distribution_gate.json"), "w", encoding="utf-8") as f:
        json.dump(gate, f, indent=2)
    print(f"SleepAccel pre-flight PASS: n_records={gate['n_records']}, n_qc_pass={gate['n_qc_pass']}, "
          f"ACC_med={gate['raw_mag_median_g']:.3f}g, HR_med={gate['raw_hr_median_bpm']:.1f}bpm")


if __name__ == "__main__":
    main()
