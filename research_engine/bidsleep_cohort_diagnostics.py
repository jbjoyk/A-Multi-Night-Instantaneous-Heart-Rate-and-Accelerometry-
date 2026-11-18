#!/usr/bin/env python3
"""
BIDSleep Cohort Data Diagnostics Script (Read-Only)
Candidate 4: Computes IHR sampling quality, calibration-night class coverage,
and subject-level data integrity metrics across the PhysioNet BIDSleep dataset.
"""

import os
import json
import numpy as np
import pandas as pd

def run_diagnostics():
    qc_path = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/qc_table.csv"
    inv_path = "review_package/night_inventory.csv"
    out_dir = "research_knowledge/updates"
    os.makedirs(out_dir, exist_ok=True)
    
    qc = pd.read_csv(qc_path)
    inv = pd.read_csv(inv_path)
    
    # ---------------------------------------------------------
    # 1. IHR Sampling Quality Summary
    # ---------------------------------------------------------
    usable_qc = qc[qc['status'] == 'ok']
    all_qc = qc
    
    metrics = ['raw_hr_rate_hz', 'usable_frac', 'cov_hr', 'cov_motion', 'hr_median']
    rows_quality = []
    
    for scope_name, df_scope in [('all_recorded_nights (n=253)', all_qc), ('usable_nights (n=246)', usable_qc)]:
        for m in metrics:
            vals = df_scope[m].dropna()
            rows_quality.append({
                'cohort_scope': scope_name,
                'metric': m,
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                'p05': float(vals.quantile(0.05)),
                'p25': float(vals.quantile(0.25)),
                'median': float(vals.median()),
                'p75': float(vals.quantile(0.75)),
                'p95': float(vals.quantile(0.95)),
                'max': float(vals.max()),
                'iqr': float(vals.quantile(0.75) - vals.quantile(0.25)),
                'n_nights': len(vals)
            })
            
    df_quality = pd.DataFrame(rows_quality)
    quality_csv_path = os.path.join(out_dir, "bidsleep_ihr_sampling_quality_summary.csv")
    df_quality.to_csv(quality_csv_path, index=False)
    print(f"Saved: {quality_csv_path}")
    
    # ---------------------------------------------------------
    # 2. Calibration-Night Class Coverage (k = 1, 2, 3, 4)
    # ---------------------------------------------------------
    usable_inv = inv[inv['usable'] == True].sort_values(['subject_id', 'night']).copy()
    
    # For each subject, order usable nights chronologically (1-based index)
    usable_inv['usable_night_idx'] = usable_inv.groupby('subject_id').cumcount() + 1
    sub_night_counts = usable_inv.groupby('subject_id')['usable_night_idx'].max()
    
    coverage_rows = []
    for k in [1, 2, 3, 4]:
        # Filter subjects who possess at least k calibration nights AND at least 1 evaluation night (n_usable >= k + 1)
        evaluable_subs = sub_night_counts[sub_night_counts >= k + 1].index
        sub_k = usable_inv[(usable_inv['subject_id'].isin(evaluable_subs)) & (usable_inv['usable_night_idx'] <= k)]
        
        # Aggregate stage epochs across the first k calibration nights for evaluable subjects
        agg = sub_k.groupby('subject_id').agg({
            'n_epochs_labeled': 'sum',
            'wake_frac': lambda w: (w * sub_k.loc[w.index, 'n_epochs_labeled']).sum() / sub_k.loc[w.index, 'n_epochs_labeled'].sum(),
            'light_frac': lambda l: (l * sub_k.loc[l.index, 'n_epochs_labeled']).sum() / sub_k.loc[l.index, 'n_epochs_labeled'].sum(),
            'deep_frac': lambda d: (d * sub_k.loc[d.index, 'n_epochs_labeled']).sum() / sub_k.loc[d.index, 'n_epochs_labeled'].sum(),
            'rem_frac': lambda r: (r * sub_k.loc[r.index, 'n_epochs_labeled']).sum() / sub_k.loc[r.index, 'n_epochs_labeled'].sum()
        }).reset_index()
        
        # Calculate min stage fraction
        agg['min_stage_frac'] = agg[['wake_frac', 'light_frac', 'deep_frac', 'rem_frac']].min(axis=1)
        
        n_evaluable_subs = len(agg)
        zero_deep = int((agg['deep_frac'] == 0).sum())
        zero_rem = int((agg['rem_frac'] == 0).sum())
        zero_wake = int((agg['wake_frac'] == 0).sum())
        
        lt_1pct = int((agg['min_stage_frac'] < 0.01).sum())
        lt_2pct = int((agg['min_stage_frac'] < 0.02).sum())
        lt_3pct = int((agg['min_stage_frac'] < 0.03).sum())
        lt_5pct = int((agg['min_stage_frac'] < 0.05).sum())
        
        coverage_rows.append({
            'calibration_nights_k': k,
            'evaluable_subjects': n_evaluable_subs,
            'eligible_subjects': n_evaluable_subs,
            'zero_deep_subjects': zero_deep,
            'zero_rem_subjects': zero_rem,
            'zero_wake_subjects': zero_wake,
            'min_stage_lt_1pct_subs': lt_1pct,
            'min_stage_lt_2pct_subs': lt_2pct,
            'min_stage_lt_3pct_subs': lt_3pct,
            'min_stage_lt_5pct_subs': lt_5pct,
            'mean_min_stage_pct': float(agg['min_stage_frac'].mean() * 100),
            'median_min_stage_pct': float(agg['min_stage_frac'].median() * 100),
            'mean_deep_pct': float(agg['deep_frac'].mean() * 100),
            'mean_rem_pct': float(agg['rem_frac'].mean() * 100),
            'mean_wake_pct': float(agg['wake_frac'].mean() * 100),
            'mean_light_pct': float(agg['light_frac'].mean() * 100)
        })
        
    df_coverage = pd.DataFrame(coverage_rows)
    coverage_csv_path = os.path.join(out_dir, "bidsleep_calibration_class_coverage_summary.csv")
    df_coverage.to_csv(coverage_csv_path, index=False)
    print(f"Saved: {coverage_csv_path}")
    
    # ---------------------------------------------------------
    # 3. Subject-Level Diagnostic Table
    # ---------------------------------------------------------
    subject_rows = []
    all_subjects = sorted(inv['subject_id'].unique())
    
    for s in all_subjects:
        s_inv = inv[inv['subject_id'] == s]
        s_qc = qc[qc['subject'] == s]
        
        n_rec = len(s_inv)
        n_usable = int(s_inv['usable'].sum())
        n_exc = n_rec - n_usable
        
        mean_ihr_hz = float(s_qc['raw_hr_rate_hz'].mean())
        mean_usable_frac = float(s_qc['usable_frac'].mean())
        
        # Calibration night metrics for k=2
        s_usable_inv = s_inv[s_inv['usable'] == True].sort_values('night')
        if len(s_usable_inv) >= 2:
            cal_2 = s_usable_inv.iloc[:2]
            tot_ep = cal_2['n_epochs_labeled'].sum()
            w_pct = float((cal_2['wake_frac'] * cal_2['n_epochs_labeled']).sum() / tot_ep * 100)
            l_pct = float((cal_2['light_frac'] * cal_2['n_epochs_labeled']).sum() / tot_ep * 100)
            d_pct = float((cal_2['deep_frac'] * cal_2['n_epochs_labeled']).sum() / tot_ep * 100)
            r_pct = float((cal_2['rem_frac'] * cal_2['n_epochs_labeled']).sum() / tot_ep * 100)
            min_pct = min(w_pct, l_pct, d_pct, r_pct)
        else:
            w_pct = l_pct = d_pct = r_pct = min_pct = np.nan
            
        qualifies_exp019 = (n_usable >= 3)
        
        subject_rows.append({
            'subject_id': s,
            'total_nights_recorded': n_rec,
            'usable_nights': n_usable,
            'excluded_nights': n_exc,
            'mean_ihr_hz': round(mean_ihr_hz, 4),
            'mean_usable_frac': round(mean_usable_frac, 4),
            'k2_wake_pct': round(w_pct, 2) if not np.isnan(w_pct) else "NA",
            'k2_light_pct': round(l_pct, 2) if not np.isnan(l_pct) else "NA",
            'k2_deep_pct': round(d_pct, 2) if not np.isnan(d_pct) else "NA",
            'k2_rem_pct': round(r_pct, 2) if not np.isnan(r_pct) else "NA",
            'k2_min_stage_pct': round(min_pct, 2) if not np.isnan(min_pct) else "NA",
            'qualifies_exp019_eval': qualifies_exp019
        })
        
    df_subjects = pd.DataFrame(subject_rows)
    subject_csv_path = os.path.join(out_dir, "bidsleep_subject_diagnostic_table.csv")
    df_subjects.to_csv(subject_csv_path, index=False)
    print(f"Saved: {subject_csv_path}")

if __name__ == "__main__":
    run_diagnostics()
