# Concepts Brief: EXP-019 Multi-Night Within-Person Sleep Staging Adaptation

* **Author:** jbjoyk
* **Status:** Aligned / Frozen
* **Date:** 2026-11-16
* **Governing Documents:** `RESEARCH_DIRECTION_v3_FULL.md`, `RESEARCH_DIRECTION_v3.1_AMENDMENT.md`, `PREREG_EXP-019_H-A.md`

---

## 1. Executive Summary & Problem Context
Sleep staging from consumer wearables (instantaneous heart rate [IHR] and triaxial accelerometry) suffers from substantial between-subject physiological variance. Most existing deep learning pipelines evaluate models either on single pooled nights or cross-sectional splits, completely neglecting that the same individual generates repeated multi-night time-series with person-specific autonomic baselines.

The BIDSleep dataset (Song et al., 2026; 47 healthy adults, 253 recorded nights, Apple Watch IHR/accel + Dreem 2 EEG-derived AASM labels) provides an empirical substrate to investigate longitudinal personalization. This project investigates whether adapting a subject-independent zero-shot model using a person's first $k=2$ labeled nights improves staging accuracy on that person's subsequent nights under a strict, leakage-free nested cross-validation protocol.

---

## 2. Core Confirmatory Hypotheses
* **Primary (H-A):** Fine-tuning the zero-shot (ZS) model on a held-out subject's first $k=2$ labeled calibration nights ($FT\text{-}2$) increases Macro-F1 on that subject's remaining evaluation nights ($E$) relative to $ZS$ by at least $+1.5\text{ pp}$, with the paired subject-level bootstrap 95% CI strictly excluding 0.
* **Secondary Control (S1 - Personalization Specificity):** $FT\text{-}2$ on the subject's own 2 nights outperforms $FT\text{-}other\text{-}2$ (fine-tuning on 2 nights from random training subjects of the same fold under identical hyperparameters), proving that performance gains stem from genuine subject-specific personalization rather than generic optimization or entropy shifts.
* **Secondary Baseline (S2 - Adaptation vs. Scratch):** $FT\text{-}2$ outperforms $PS\text{-}2$ (a model trained from scratch on the 2 calibration nights alone).
* **Secondary Feasibility (S3 - Unsupervised Adaptation):** $UA\text{-}2$ (label-free test-time entropy minimization on the 2 calibration nights' raw inputs) outperforms $ZS$.

---

## 3. Mandatory Governance Constraints
1. **Architecture Freeze:** Backbone is strictly Model A (`ConventionalCNNBiGRU`, 163,940 parameters). No architectural modifications, gating, or structural mutations. Architecture search is closed (DEC-019, DEC-020).
2. **Task Definition:** 4-class staging (0: Wake, 1: Light [N1+N2], 2: Deep [N3], 3: REM). Epoch length is 30 s. Context window is full-night ($L=1200$) with sliding-window inference. Circadian feature is neutralized (`time_feat=0`). Labels: `expert_label` from `labels.mat`. Unknown/out-of-range epochs are masked.
3. **Unit of Analysis:** The subject ($N=46$ eligible subjects with $\ge 3$ usable nights, 152 evaluation nights total; `Bidslab49` excluded from H-A due to having only 2 usable nights). Outer cross-validation uses the locked 5-fold subject manifest (hash `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`).
4. **Leakage-Free Nested Protocol:** Hyperparameter selection is strictly nested and evaluated exclusively on the 5 inner-validation subjects per outer fold. Outer test subjects are evaluated exactly once per seed.
5. **Execution Separation:** Stage D (hyperparameter tuning and paired standard deviation $s_{\Delta}$ verification on inner-validation folds) must complete and confirm the threshold before Stage C (outer test evaluation) is locked and executed.

---

## 4. Alternatives Considered & Dispositions
* **Per-Subject Architecture Search (NAS):** REJECTED. Overfits catastrophically on 2 nights, introduces massive compute overhead, and violates matched-capacity comparability.
* **Synthetic Sleep Generation (Diffusion/GAN):** REJECTED. Hallucinates physiological dynamics and obscures true autonomic sleep mechanisms.
* **Random Night Splitting (Shuffled CV):** REJECTED. Introduces forward-in-time temporal leakage; calibration must be strictly forward chronological ($C \to E$).
* **5-Class Staging Re-expansion:** REJECTED for H-A primary. Kept strictly as a secondary descriptive reporting layer to match baseline literature.
