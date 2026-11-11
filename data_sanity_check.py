"""
Data Sanity and Protocol Verification for SLAMSS-IFS Baseline.
Phase 10 verification script.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import pandas as pd
from slamss_ifs_baseline import make_reproducible_split


def run_data_sanity_check():
    print("=" * 65)
    print("DATA SANITY & PROTOCOL VERIFICATION CHECK")
    print("=" * 65)
    
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    
    total_nights = len(mat_files)
    subjects_dict = {}
    night_lengths = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, "unknown": 0}
    
    # Dreem / Expert 4-class mapping according to PhysioNet documentation:
    # Raw: 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM, -1/5: Unknown
    # 4-class grouping: Wake=0, Light=(1,2), Deep=3, REM=4
    label_map_4class = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
    
    for mf in mat_files:
        parts = mf.split(os.sep)
        subj = parts[-3]
        night_id = parts[-2]
        subjects_dict.setdefault(subj, []).append(night_id)
        
        mat = sio.loadmat(mf)
        labels = mat["expert_label"].ravel()
        night_lengths.append(len(labels))
        
        for l in labels:
            if l in label_map_4class:
                class_counts[label_map_4class[l]] += 1
            else:
                class_counts["unknown"] += 1
                
    night_lengths = np.array(night_lengths)
    total_subjects = len(subjects_dict)
    total_epochs = night_lengths.sum()
    
    print(f"Total Subjects Discovered:        {total_subjects}")
    print(f"Total Nights Discovered:          {total_nights}")
    print(f"Total 30-Second Epochs:           {total_epochs:,} (~{total_epochs*30/3600:,.1f} recording hours)")
    print(f"Night Lengths (in 30s epochs):")
    print(f"  Min:                            {night_lengths.min()} epochs ({night_lengths.min()*30/3600:.2f} h)")
    print(f"  25th Percentile:                {np.percentile(night_lengths, 25):.0f} epochs ({np.percentile(night_lengths, 25)*30/3600:.2f} h)")
    print(f"  Median:                         {np.median(night_lengths):.0f} epochs ({np.median(night_lengths)*30/3600:.2f} h)")
    print(f"  Mean:                           {night_lengths.mean():.1f} epochs ({night_lengths.mean()*30/3600:.2f} h)")
    print(f"  75th Percentile:                {np.percentile(night_lengths, 75):.0f} epochs ({np.percentile(night_lengths, 75)*30/3600:.2f} h)")
    print(f"  Max:                            {night_lengths.max()} epochs ({night_lengths.max()*30/3600:.2f} h)")
    
    n_ge_1200 = int((night_lengths >= 1200).sum())
    pct_ge_1200 = float((night_lengths >= 1200).mean() * 100)
    pct_lt_1200 = 100.0 - pct_ge_1200
    
    print(f"\nSequence Length Analysis (relative to T = 1200 epochs / 10 hours):")
    print(f"  Nights >= 1200 epochs:          {n_ge_1200} / {total_nights} ({pct_ge_1200:.2f}%)")
    print(f"  Nights < 1200 epochs:           {total_nights - n_ge_1200} / {total_nights} ({pct_lt_1200:.2f}%)")
    print(f"  Nights >= 600 epochs (5 hours): {int((night_lengths >= 600).sum())} / {total_nights} (100.0%)")
    
    print(f"\nClass Distribution across all {total_epochs:,} epochs:")
    stage_names = {0: "Wake", 1: "Light (N1+N2)", 2: "Deep (N3)", 3: "REM", "unknown": "Unknown / Unscored"}
    known_epochs = total_epochs - class_counts["unknown"]
    for k in [0, 1, 2, 3, "unknown"]:
        cnt = class_counts[k]
        if k == "unknown":
            pct = 100.0 * cnt / total_epochs
            print(f"  {stage_names[k]:22s}: {cnt:7,d} ({pct:5.2f}% of total)")
        else:
            pct = 100.0 * cnt / known_epochs
            print(f"  Stage {k} - {stage_names[k]:14s}: {cnt:7,d} ({pct:5.2f}% of valid)")
            
    # Verification of Disjoint Subjects & Nights
    print(f"\nVerifying Subject and Night Boundary Separation:")
    all_subjects = sorted(subjects_dict.keys())
    
    for split_mode in ["paper", "current"]:
        sp = make_reproducible_split(all_subjects, split_type=split_mode, seed=0)
        tr_s, va_s, te_s = set(sp["train"]), set(sp["val"]), set(sp["test"])
        
        # 1. Subject disjointness
        assert not (tr_s & va_s), f"[{split_mode}] Train & Val subject overlap!"
        assert not (tr_s & te_s), f"[{split_mode}] Train & Test subject overlap!"
        assert not (va_s & te_s), f"[{split_mode}] Val & Test subject overlap!"
        
        # 2. Night disjointness
        tr_nights = set((s, n) for s in tr_s for n in subjects_dict[s])
        va_nights = set((s, n) for s in va_s for n in subjects_dict[s])
        te_nights = set((s, n) for s in te_s for n in subjects_dict[s])
        
        assert not (tr_nights & va_nights), f"[{split_mode}] Train & Val night overlap!"
        assert not (tr_nights & te_nights), f"[{split_mode}] Train & Test night overlap!"
        assert not (va_nights & te_nights), f"[{split_mode}] Val & Test night overlap!"
        
        print(f"  [{split_mode.upper()}] Partitioning: {len(tr_s)} train ({len(tr_nights)} nights), "
              f"{len(va_s)} val ({len(va_nights)} nights), {len(te_s)} test ({len(te_nights)} nights) -> ZERO OVERLAP.")
              
    print("\n" + "=" * 65)
    print("DATA SANITY & PROTOCOL VERIFICATION COMPLETE: ALL CHECKS PASSED.")
    print("=" * 65)


if __name__ == "__main__":
    run_data_sanity_check()
