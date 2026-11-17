# EGM Synchronization Review Report: Post-Sync Audit of Research Workspace

**Audit Date:** 2026-11-17  
**Auditor:** EGM Review System (Automated Forensic & Governance Audit)  
**Governing Documents Audited:**
- `RESEARCH_DIRECTION_v4_FULL.md` (Authoritative Direction)
- `PREREG_EXP-019_H-A.md` (Commit `4dff5b7`, Locked Preregistration)
- `RESEARCH_STATE.md`
- `HANDOFF.md`
- `DECISIONS.md`
- `EXPERIMENTS.md`
- `concepts_brief.md`
- `research_memory.json`
- `research_knowledge/` (Knowledge Base)
- `experiments/EXP-019/` (EXP-019 Confirmatory Artifacts)

---

## 1. Overall Verdict

### **PASS WITH WARNINGS**

**Verdict Summary:**
The core research direction, problem framing, contribution hierarchy (C1–C3), and roadmap phasing have been successfully aligned with `RESEARCH_DIRECTION_v4_FULL.md`. All locked EXP-019 artifacts, code backbones, numerical results, and split manifests remain strictly intact and bitwise immutable.

The **WARNINGS** reflect five specific residual consistency gaps where historical "NOT SUPPORTED" phrasing or premature conclusions about unsupervised adaptation linger in secondary documents (`concepts_brief.md`, `novelty_registry.md`, `P007` review note, DEC-028 inline labels, and `EXP019_FINAL_REPORT.md` awaiting Phase A1 erratum). These do not compromise active decision-making, but require documented remediation before initiating `egm-brainstorm`.

---

## 2. Direction Alignment Audit

| Requirement | Finding | Status |
|---|---|---|
| **Active Research Position** | Consistently documented across `RESEARCH_STATE.md`, `HANDOFF.md`, and `DECISIONS.md` (DEC-029) as: *"A pre-registered, leakage-free replication and extension of multi-night supervised personalization using consumer-smartwatch IHR (~0.2 Hz) + accelerometry in healthy adults."* | **PASS** |
| **Withdrawal of "Open Lane" Novelty Claim** | The prior claim ("multi-night personalization is an open/underexplored lane") is formally withdrawn in `RESEARCH_STATE.md`, `HANDOFF.md`, `DECISIONS.md` (DEC-029), and `EXPERIMENTS.md`, following discovery of van der Aar et al. 2025 (*Physiological Measurement*, DOI: [10.1088/1361-6579/ae0119](https://doi.org/10.1088/1361-6579/ae0119)). | **PASS** |
| **Historical v3 Direction Preserved** | `RESEARCH_DIRECTION_v3_FULL.md` remains unmodified on disk as historical record, explicitly superseded by v4 per DEC-029. | **PASS** |

---

## 3. Hypothesis H-A Governance Audit

The locked preregistration text (`git show 4dff5b7:PREREG_EXP-019_H-A.md` Section 6) specifies:
- **SUPPORTED:** $\bar{\Delta} \ge +1.50\text{ pp}$ AND 95% bootstrap CI lower bound $> 0$.
- **NOT SUPPORTED:** 95% bootstrap CI upper bound $< +1.50\text{ pp}$ (ruling out an effect of practical size).
- **INCONCLUSIVE:** Anything else.

Observed EXP-019 confirmatory result:
- $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ (from $64.58\% \pm 7.74\%$ to $65.59\% \pm 8.59\%$)
- 95% bootstrap CI: $\mathbf{[-0.17, +1.99]\text{ pp}}$
- Wilcoxon signed-rank $p = 0.00131$; win rate: 73.9% (34/46)

### Audit Findings:
1. **Rule Evaluation:** Because the upper bound ($+1.99\text{ pp}$) is NOT $< +1.50\text{ pp}$, the preregistered rule evaluates strictly to **`INCONCLUSIVE`**.
2. **Reconciliation Tracking:** `RESEARCH_STATE.md`, `HANDOFF.md`, `DECISIONS.md` (DEC-029), `EXPERIMENTS.md`, `concepts_brief.md`, and `research_knowledge/README.md` have been updated to reflect the reconciled status **`INCONCLUSIVE`** and explicitly document why earlier reports labeled it "NOT SUPPORTED".
3. **Residual Gaps:**
   - `concepts_brief.md:195` still contains the un-reconciled phrase: `H-A (Not Supported) remains permanent`.
   - `research_knowledge/novelty_registry.md:7` (table row `N001`) reads: `Tested in EXP-019; H-A Not Supported`.
   - `DECISIONS.md:458` (DEC-028) retains the original text `Formal Verdict: **NOT SUPPORTED**` without an inline pointer to the DEC-029 erratum.
   - `experiments/EXP-019/EXP019_FINAL_REPORT.md` has not yet had the Phase A1 formal erratum appended to Section 1.

---

## 4. Secondary Hypothesis S3 Audit Status

Observed EXP-019 result:
- $\bar{\Delta} = \mathbf{0.00\text{ pp}}$ [95% CI: $\mathbf{0.00, 0.00\text{ pp}}$] identically across all 46 subjects to the last decimal place.

### Audit Findings:
1. **Status Alignment:** `RESEARCH_STATE.md`, `HANDOFF.md`, `DECISIONS.md` (DEC-029), and `EXPERIMENTS.md` correctly classify S3 as **`UNRESOLVED`**, noting that the identical zero delta indicates an implementation fault or no-op (e.g. inference not loading adapted weights, or zero gradients).
2. **Non-Evidence Rule:** All primary working documents explicitly state that S3 must NOT be cited as evidence against unsupervised adaptation until the Phase A2 audit is complete.
3. **Residual Gaps:**
   - In `research_knowledge/papers/P007_lorenzen2023_wearable_eeg_personalization.md` (lines 40 and 44), notes added during seed ingestion claim that EXP-019 S3 *"confirmed failure of unsupervised adaptation"* and *"aligns with a broader pattern... where unsupervised alignment fails."* This contradicts v4 governance and prematurely treats a suspected no-op as scientific evidence.

---

## 5. Locked-Artifact Integrity Audit

Forensic inspection was executed across all locked artifacts against Git commit `4dff5b7`:

| Locked Artifact | Target Path | Modification Detected? | Audit Status |
|---|---|---|---|
| **Preregistration Document** | `PREREG_EXP-019_H-A.md` | **None** (`git diff 4dff5b7..HEAD` is empty) | **PASS (Bitwise Match)** |
| **Full Results JSON** | `experiments/EXP-019/exp019_stage_c_results.json` | **None** (Clean, unmodified) | **PASS (Intact)** |
| **Per-Subject Metrics CSV** | `experiments/EXP-019/exp019_subject_metrics.csv` | **None** (Clean, unmodified) | **PASS (Intact)** |
| **Summary Metrics CSV** | `experiments/EXP-019/exp019_summary.csv` | **None** (Clean, unmodified) | **PASS (Intact)** |
| **Stage D Configs JSON** | `experiments/EXP-019/exp019_stage_d_selected_configs.json` | **None** (Clean, unmodified) | **PASS (Intact)** |
| **15 Model A Checkpoints** | `experiments/EXP-018/checkpoints/modelA_L1200_fold*.pt` | **None** (Clean, unmodified) | **PASS (Intact)** |
| **Subject Split Manifests** | `review_package/fold_manifests.csv` | **None** (Clean, unmodified) | **PASS (Intact)** |

**Conclusion:** Zero locked artifacts, numbers, or weights were altered during the synchronization.

---

## 6. Prior-Art Alignment Audit

1. **Major Peer Reference:** van der Aar et al. 2025 (*Physiological Measurement*, DOI: [10.1088/1361-6579/ae0119](https://doi.org/10.1088/1361-6579/ae0119)) is established in `RESEARCH_STATE.md`, `RESEARCH_DIRECTION_v4_FULL.md`, and `research_knowledge/` as the primary benchmark and precedent.
2. **Protocol Distinctions Documented:**
   - Population: Sleep-disordered adults (van der Aar) vs healthy adults (BIDSleep).
   - Sensor Modality: Continuous wrist PPG (van der Aar) vs ~0.2 Hz instantaneous heart rate (IHR) + accelerometry (BIDSleep).
   - Calibration Scope: Full partial retraining of encoder layers (van der Aar) vs frozen backbone with head/GRU fine-tuning under strict chronological splits.
   - Evaluation: PSG-tested testing with up to 7 home nights vs Dreem 2 EEG-labeled home sleep.
3. **Claim Boundaries:** System is explicitly barred from claiming "first multi-night personalization", "open lane", or general superiority across mismatched protocols.

---

## 7. Governance Compliance Audit

| Policy / Guardrail | Compliance Evidence | Status |
|---|---|---|
| **EXP-020 Authorization** | Formally marked as **Gated & Unauthorized** in `RESEARCH_STATE.md` (Section 5) and `DECISIONS.md` (DEC-029). Cannot start before Phases A–C and a committed preregistration block. | **PASS** |
| **Preregistration Mandate** | Strict requirement enforced: zero confirmatory experiments without an immutable committed block in `EXPERIMENTS.md`. | **PASS** |
| **NAS Termination** | Neural Architecture Search is permanently closed (DEC-020, DEC-026 Erratum). NAS Iterations 2+ are confirmed as unauthorized. | **PASS** |
| **EXP-012 Test Set Status** | Formally retired as **SPENT** in `RESEARCH_STATE.md` (Section 3), `RESEARCH_DIRECTION_v4_FULL.md` (Section 7), and `DECISIONS.md` (DEC-029). | **PASS** |
| **Anti-Score-Chasing** | Post-hoc threshold changes and experiment cherry-picking are strictly barred. | **PASS** |

---

## 8. Roadmap & Phasing Alignment Audit

The Phase A–E structure from `RESEARCH_DIRECTION_v4_FULL.md` is faithfully mirrored in `RESEARCH_STATE.md` and `HANDOFF.md`:

```
Phase A: Close EXP-019 (A1: Settle label -> A2: S3 no-op audit -> A3: Report missing outputs -> A4: S1 wording -> A5: NAS erratum)
  │
  ▼
Phase B: Evidence Base (B1: KB corrections -> B2: Systematic search -> B3: Full-text reviews -> Closest-ten table)
  │
  ▼
Phase C: Data Search (C1: Systematic search for second multi-night labeled wearable dataset)
  │
  ▼
Phase D: Gated Experiments (D1: EXP-020 regularized adaptation -> D2: EXP-019B heterogeneity -> D3: EXP-027 zero-shot -> D4: EXP-028 clinical agreement -> D5: Second-dataset replication)
  │
  ▼
Phase E: Synthesis & Writing
```

### Missing Pre-Registered Outputs Tracked:
`RESEARCH_STATE.md`, `HANDOFF.md`, and `RESEARCH_DIRECTION_v4_FULL.md` explicitly list all required Phase A3 outputs:
- $k=1–4$ calibration learning curve (with $n$ per $k$).
- Per-stage F1 scores and confusion matrices.
- Clinical sleep-summary agreement (TST, SOL, WASO, SE, REM%, Deep%).
- Holm-adjusted secondary p-values for S1–S3.
- Stage-D empirical variance / spread estimates.

---

## 9. Contradictions & Discrepancies Found

The following five specific textual discrepancies must be remediated:

| # | File | Exact Location | Discrepancy Found | Authoritative Source | Required Correction |
|---|---|---|---|---|---|
| **1** | `concepts_brief.md` | Line 195 | `- **No modification of locked EXP-019 results**: H-A (Not Supported) remains permanent.` | `PREREG_EXP-019_H-A.md` (4dff5b7) & `RESEARCH_DIRECTION_v4_FULL.md` | Change to: `H-A (Reconciled Inconclusive under preregistered rule) remains permanent.` |
| **2** | `research_knowledge/novelty_registry.md` | Line 7 (Table row `N001`) | `| N001 | ... | Low (Tested in EXP-019; H-A Not Supported) |` | `RESEARCH_DIRECTION_v4_FULL.md` Section 5 | Change to: `Low (Tested in EXP-019; H-A Inconclusive / Not Supported in early report)` |
| **3** | `research_knowledge/papers/P007_lorenzen2023_wearable_eeg_personalization.md` | Lines 40 & 44 | Claims EXP-019 S3 *"confirmed failure of unsupervised adaptation"* and *"aligns with a broader pattern... where unsupervised alignment fails."* | `RESEARCH_DIRECTION_v4_FULL.md` Section 1 & `DECISIONS.md` (DEC-029) | Revise to state that EXP-019 S3 is unresolved pending Phase A2 implementation check, and cannot yet be cited as scientific evidence against unsupervised adaptation. |
| **4** | `DECISIONS.md` | Lines 458 & 472 (under DEC-028) | DEC-028 lists `Formal Verdict: **NOT SUPPORTED**` for H-A and S3 without an inline link/erratum marker pointing to DEC-029. | `DECISIONS.md` DEC-029 | Add inline bracketed notes: `[Reconciled to INCONCLUSIVE in DEC-029 erratum per locked preregistration text]` and `[Reclassified to UNRESOLVED in DEC-029 pending no-op audit]`. |
| **5** | `experiments/EXP-019/EXP019_FINAL_REPORT.md` | Section 1 & Section 3.1 | Historical report text retains original "NOT SUPPORTED" label for H-A and S3. | `RESEARCH_DIRECTION_v4_FULL.md` Section 9, Step A1 | Append a formal **Erratum / Governance Reconciliation Note** to the report header and Section 1 referencing DEC-029 and commit `4dff5b7`. |

---

## 10. File-Change Audit (Categorization of Previous Sync)

- **Correct and Necessary:**
  - `RESEARCH_STATE.md`: Full synchronization with v4 problem statement, replication/extension framing, C1–C3 hierarchy, reconciled H-A/S3 statuses, and Phase A–E roadmap.
  - `HANDOFF.md`: Full alignment with Phase A1–A5 closeout priorities and immediate next tasks.
  - `DECISIONS.md`: Addition of DEC-029 formally ratifying v4, withdrawing the "open lane" novelty claim, and recording the H-A/S3 reconciliation.
  - `EXPERIMENTS.md`: Alignment of EXP-019 summary and research trajectory with v4.
  - `research_memory.json`: Update of metadata pointers to DEC-029 and v4.
  - `research_knowledge/README.md`: Pointer update to v4 and Layer 1 reconciliation.
- **Correct but Potentially Risky:**
  - Leaving DEC-028 without an inline pointer to DEC-029 risked local confusion; mitigated by adding the cross-reference noted in Item 4 above.
- **Incorrect:**
  - None. Zero incorrect factual statements or invalid changes were introduced.
- **Missing:**
  - Items 1–5 in Section 9 above represent minor residual synchronizations to clean up before brainstorming.

---

## 11. Recommended Next Actions

1. **Remediate Residual Contradictions (Immediate, Non-Experimental)**:
   - Apply textual fixes to `concepts_brief.md`, `novelty_registry.md`, `P007`, and `DECISIONS.md` (DEC-028 inline cross-references).
   - Append the Phase A1 formal reconciliation erratum to `experiments/EXP-019/EXP019_FINAL_REPORT.md`.
2. **Execute Phase A Close-Out (Sequenced)**:
   - **Phase A2**: Run the diagnostic code/parameter audit on `UA-2` to close the no-op question.
   - **Phase A3**: Extract and report the $k=1–4$ calibration learning curve and clinical sleep summaries.
   - **Phase A4/A5**: Complete S1 donor-harm documentation and NAS erratum.
3. **Execute Phase B Evidence Base**:
   - Apply KB round-2 corrections and execute systematic forward-citation searches.
4. **Authorize Brainstorming**:
   - Only after Phase A1–A3 are documented should `egm-brainstorm` be invoked for Phase D protocol designs.
