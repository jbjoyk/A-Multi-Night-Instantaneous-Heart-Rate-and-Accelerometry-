# Stage 5 Initial Zoom Out Review Report

**Target Document**: [`experiments/EXP-018/EXP018_PHASE1_REPORT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/EXP018_PHASE1_REPORT.md)  
**Auditors**: Skeptical Architect (`egm-zoom-out`) & Inventive Problem-Solver (`egm-inventive-catalyst`)  
**Standard**: `zoom-out-standard.md` & TRIZ Contradiction Analysis  
**Review Status**: 🟡 **CONDITIONAL PASS** (0 Blockers, 2 Majors, 2 Nits)  

---

## Consolidated Findings Matrix

| Finding ID | Reviewer | Severity | Scope | Core Issue | Recommended Resolution |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `ZOOM-5-1` | `egm-zoom-out`, `egm-inventive` | 🟡 Major | Sec 3 (lines 33–41) | **Parametric Saturation Paradox**: Single-exponential fit reports asymptote $A=0.6435$ and $L^*=46.3$ epochs (23.1 min), while empirical performance at $L=600$ (0.6529) and $L=1200$ (0.6993) strongly surpasses this ceiling. | Explicitly document Section 3 as a single-scale baseline fit; annotate that sleep staging displays dual-regime physiology (local epoch transition inertia for $L \le 60$ vs. multi-hour ultradian NREM-REM cycles for $L \ge 300$). |
| `ZOOM-5-2` | `egm-zoom-out` | 🟡 Major | Sec 1 (lines 10–18) | **Unqualified Cohort Scope**: Executive summary does not explicitly restate that Phase 1 gating is evaluated on the 5-subject discovery validation set ($N=5$, 28 nights) as a prerequisite to 47-subject Phase 2 CV. | Add an explicit qualification in Section 1 specifying: `(Evaluated on Phase 1 Validation Partition: N=5 subjects, 28 nights; pre-registered gate to authorize 5-fold CV)`. |
| `ZOOM-5-3` | `egm-inventive`, `egm-zoom-out` | 🟢 Nit | Sec 2, 4 (lines 20–31, 43–54) | **Intermediate Horizon Volatility**: Wake F1 drops between $L=60$ ($0.6165$) and $L=120$ ($0.5761$) and TST MAE fluctuates before full-night recovery at $L=1200$. | Add a brief physiological note that intermediate BiGRU context functions as an inter-epoch low-pass filter until global lights-out/awakening anchors take effect at $L=1200$. |
| `ZOOM-5-4` | `egm-inventive` | 🟢 Nit | Sec 3 (line 35) | **Single-Scalar $L^*$ Conceals Stage Heterogeneity**: Reporting a single scalar $L^*$ masks that Deep sleep scales continuously through $L=1200$ ($82.1\%$ recall) whereas REM saturates around $L=300$. | Clarify in text that the saturation threshold represents macro-average and that stage-specific dynamics differ. |

---

## Detailed Reviewer Observations

### 1. Skeptical Architect (`egm-zoom-out`)
- **Scientific Integrity**: The empirical result ($\Delta \text{Macro-F1} = +4.27\text{ pp}$ [95% CI: $+2.53, +6.46\text{ pp}$]) is genuine, verified against the 21 raw JSON artifacts, and satisfies the pre-registered `SUPPORTED` criterion.
- **The Saturation Fit Paradox**: A concave single exponential latches onto the steep initial rise ($L=1 \to 60$) and the intermediate plateau ($L=60 \to 120$), setting an artificial ceiling at $0.6435$. Because $L=600$ ($0.6529$) and $L=1200$ ($0.6993$) break through this bound, presenting $L^* = 46.3$ epochs without qualification contradicts Section 1 and risks misleading citations.

### 2. Inventive Problem-Solver (`egm-inventive-catalyst`)
- **Physiological Reality**: The data reveals that sleep context operates under two distinct biological time constants:
  1. *Local Transition Smoothing ($\tau_1 \approx 10\text{--}30\text{ min}$)*: Resolves momentary artifact vs sleep inertia.
  2. *Ultradian Macro-Architecture ($\tau_2 \approx 90\text{--}120\text{ min}$)*: Captures NREM-REM cyclical progression.
- **Micro-Arousal vs. Macro-Continuity**: Intermediate context ($L=120\text{--}300$) slightly suppresses brief wake arousals in favor of continuous sleep state momentum. At $L=1200$, full-night framing restores wake detection precision.
