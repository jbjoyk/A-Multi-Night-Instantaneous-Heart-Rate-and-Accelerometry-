# Stage 7 Critic Review Report: EXP-018 Phase 1 Final Report

**Target Document**: [`experiments/EXP-018/EXP018_PHASE1_REPORT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/EXP018_PHASE1_REPORT.md) (Lines 1–88)  
**Auditor Panel**: Hostile Design Critic (`egm-design`), Document Architect (`egm-structural`), Technical Copy Editor (`egm-style`)  
**Review Standards**: `feature-spec-standard.md`, Scientific Integrity, Tabular Layout, Plain-English Copy  
**Review Status**: ⚠️ **REVISIONS REQUIRED (1 Blocker, 6 Majors, 6 Nits)**

---

## Consolidated Findings Matrix

| Finding ID | Reviewer | Severity | Scope | Core Issue | Recommended Resolution |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `CRIT-7-1` | `egm-design`, `egm-struct` | 🔴 Blocker | Sec 3 (lines 33–41) | **Mathematically Incoherent Saturation Model**: Single-exponential fit reports ceiling $A=0.6435$ and saturation $L^*=46.3$ epochs (23.1 min), while empirical Macro-F1 at $L=600$ ($0.6529$) and $L=1200$ ($0.6993$) breaks decisively above $A$. | Explicitly qualify Section 3 as a single-scale baseline fit; document the dual-regime physiological scaling (local transition inertia for $L \le 60$ vs. ultradian/macro-sleep architecture for $L \ge 300$). |
| `CRIT-7-2` | `egm-design` | 🟡 Major | Sec 2, 4 (lines 20–54) | **Confounded Inference Mode at $L=1200$**: At $L=1200$, almost all nights have $N \le L$ and bypass sliding-window margin truncation entirely (Case 1), whereas $L \le 600$ undergo slicing/stitching (Case 2). | Document this structural inference difference as an operational factor contributing to the $L=1200$ performance surge, and pre-register a sliding-window diagnostic ablation for Phase 2. |
| `CRIT-7-3` | `egm-design`, `egm-struct` | 🟡 Major | Sec 1 (lines 10–18) | **Statistical Scope & Small Validation Cohort ($N=5$)**: Asserting definitive population claims from a 5-subject bootstrap underestimates variance and obscures that Phase 1 is a preliminary gating discovery tool. | Clarify in Section 1 that Phase 1 evaluates $N=5$ validation subjects (28 nights) as a pre-registered gate authorizing full 47-subject 5-fold CV. Include the 5 individual subject deltas ($+1.51$ to $+8.33\text{ pp}$) to show 100% direction consistency. |
| `CRIT-7-4` | `egm-design` | 🟡 Major | Sec 4 (lines 43–54) | **Clinical Polysomnography Metric Dissociation**: While Macro-F1 improves from $L=60$ to $L=600$, clinical metrics (TST, SOL, WASO, SE MAE) all degrade before recovering at $L=1200$. | Add a physiological explanation in Section 4: intermediate BiGRU horizons act as a low-pass filter, over-smoothing micro-arousals (Wake recall drops from 50.8% to 46.4%), which inflates TST error until full-night boundary anchors take effect at $L=1200$. |
| `CRIT-7-5` | `egm-struct` | 🟡 Major | Sec 5 (lines 58–80) | **Non-Portable Absolute URIs in Git Repository**: Table 5 contains machine-specific absolute file paths (`file:///home/rj2826/...`) that break on GitHub and across collaborator workstations. | Convert all 21 run artifact links in Table 5 to repository-relative markdown paths (`phase1_results/run_LXXXX_seedYY.json`). |
| `CRIT-7-6` | `egm-struct` | 🟡 Major | Sec 2 (lines 20–30) | **Incomplete Dispersion Reporting in Table 2**: Section header promises `(Mean ± SD across 3 Seeds)`, but only Macro-F1 includes standard deviation, while Accuracy, $\kappa$, and MCC omit variance. | Add `± SD` for Accuracy, Cohen's $\kappa$, and MCC using the verified values from `phase1_summary.json`. |
| `CRIT-7-7` | `egm-style`, `egm-struct` | 🟡 Major | Sec 1, 4, 6 | **Undefined Acronyms and Architecture**: "Model A", "CV", "MCC", "TST", "SOL", "WASO" are introduced without expansion or reference. | Expand all acronyms upon first usage and formally define Model A (`ConventionalCNNBiGRU`, 163,940 parameters; `exp013_models.py`). Link `design_draft.md` and state the quantitative gating rule ($\Delta \ge +1.0\text{ pp}$, $\text{CI}_{\text{lower}} > 0$). |
| `CRIT-7-8` | `egm-style`, `egm-struct` | 🟢 Nit | Sec 2, 4, 5 | **Redundant Unit Suffixes in Tables**: Cells repeat `m`, `%`, `s` when units are already defined in column headers (`(min)`, `(%)`, `(s)`). `m` also ambiguously denotes meters in SI notation. | Strip unit characters from data cells; specify units exclusively in table column headers. |
| `CRIT-7-9` | `egm-design` | 🟢 Nit | Sec 2 (line 27) | **Unaddressed $L=120$ Plateau**: Macro-F1 plateaus between $L=60$ ($0.6117$) and $L=120$ ($0.6100$) with a 4 pp drop in Wake F1. | Add a concise note that a 60-minute window sits at an ambiguous temporal scale between local transitions and 90-minute ultradian cycles. |
| `CRIT-7-10` | `egm-design` | 🟢 Nit | Sec 1 (line 13) | **Subject-Paired (+4.27 pp) vs Marginal Difference (+4.12 pp)**: The difference between the paired mean delta and marginal means difference is unexplained. | Add a 1-line note explaining that subject-paired mean delta (+4.27 pp) weights subjects equally, whereas marginal difference (+4.12 pp) weights total pooled epochs. |
| `CRIT-7-11` | `egm-style` | 🟢 Nit | Sec 1 (lines 15–16) | **Redundant Authorization Statement**: Consecutive lines repeat `Phase 2 CV is AUTHORIZED.` | Consolidate into the structured key-value line. |
| `CRIT-7-12` | `egm-style` | 🟢 Nit | Sec 6 (line 87) | **Emotional Rhetoric in Governance**: "strictly LOCKED" reads as defensive. | Calibrate to dry technical language: "11 held-out test subjects unaccessed; zero test evaluations performed." |
| `CRIT-7-13` | `egm-struct` | 🟢 Nit | Sec 2, 4, 5 | **Numeric Column Text Alignment**: Decimal data columns are center-aligned instead of right-aligned. | Right-align numeric columns (`---:`) for vertical decimal alignment. |

---

## Detailed Reviewer Observations

### 1. Hostile Design Critic (`egm-design`)
- **Blocker on Monophasic Fit**: In any mathematical saturation model $F_1(L) = A - B e^{-k L}$ with $B, k > 0$, $F_1(L)$ is strictly bounded above by $A$. Presenting $A = 0.6435$ alongside empirical observations of $0.6529$ ($L=600$) and $0.6993$ ($L=1200$) is an unacceptable mathematical contradiction. The model is severely misspecified because the steep slope at $L=1 \to 20$ biases the least-squares solver.
- **Inference Confound at $L=1200$**: The whole-night bypass rule for $N \le L$ means $L=1200$ experiences 0% boundary stitching loss, while $L=600$ experiences repeated margin discarding and stitching. This architectural asymmetry must be acknowledged before interpreting the $L=1200$ jump.
- **Clinical PSG Metric Dissociation**: Between $L=60$ and $L=600$, classification Macro-F1 rises by $+4.12\text{ pp}$, but Total Sleep Time error increases by $+5.57\text{ min}$ and Wake recall falls by $-4.4\text{ pp}$. The report must discuss this micro-arousal over-smoothing effect.

### 2. Document Architect (`egm-structural`)
- **Link Portability**: Hardcoded absolute machine paths (`file:///home/rj2826/...`) in Table 5 must be converted to portable relative paths for Git/GitHub sharing.
- **Dispersion Reporting**: Standard deviations for Accuracy, $\kappa$, and MCC are recorded in `phase1_summary.json` and must be presented in Table 2 to honor the section title `(Mean ± SD across 3 Seeds)`.
- **Subject-Level Transparency**: Reporting the individual deltas for all 5 subjects (`Bidslab13`: $+4.17$, `Bidslab40`: $+3.39$, `Bidslab55`: $+3.92$, `Bidslab56`: $+8.33$, `Bidslab64`: $+1.51\text{ pp}$) provides the foundational evidence proving 100% directional consistency.

### 3. Technical Copy Editor (`egm-style`)
- **Acronym Definitions**: Expand TST, SOL, WASO, SE, MCC, CV on first reference.
- **Table Cell Formatting**: Remove redundant `m`, `%`, `s` unit suffixes from numeric table cells when units are already defined in table headers.
- **Neutral Tone**: Tone down conversational padding ("100% success") and defensive emphasis ("strictly LOCKED").
