# Goldfish Review Report: Round 1 (EXP-018 Design Proposal)

**Target File**: `experiments/EXP-018/design_draft.md`  
**Audited By**: 4-Agent Goldfish Swarm (`egm-design`, `egm-implementation-readiness`, `egm-codebase-details`, `egm-structural`)  
**Standard**: `feature-spec-standard.md` & `implementation-readiness-standard.md`  
**Review Status**: ❌ REJECTED (Round 1 — 7 Blockers, 4 Majors, 1 Nit)

---

## Consolidated Findings Matrix

| Finding ID | Reviewer | Severity | Scope | Core Issue | Resolution |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `DESIGN-R1-1` | `egm-design`, `egm-structural` | 🔴 Blocker | Global | Complete omission of mandatory "Alternatives Considered" section | `PROPOSE FIX` |
| `DESIGN-R1-2` | `egm-design`, `egm-impl` | 🔴 Blocker | Line 16 | Flawed `batch * L = constant`: gradient noise disparity & BiGRU degeneration at $L=1$ | `PROPOSE FIX` |
| `DESIGN-R1-3` | `egm-design`, `egm-impl` | 🔴 Blocker | Lines 19-20 | Underestimated runtime and unstaged 105-run monolithic 5-fold CV | `PROPOSE FIX` |
| `CODE-R1-1` | `egm-codebase-details` | 🔴 Blocker | Line 43 | Circadian `time_feat` already in `x_meta`, confounding context isolation | `PROPOSE FIX` |
| `CODE-R1-2` | `egm-codebase-details`, `egm-impl` | 🔴 Blocker | Line 26 | Clinical metrics (TST, SOL, WASO, SE) do not exist in `evaluator.py` | `PROPOSE FIX` |
| `IMPL-R1-1` | `egm-impl`, `egm-design` | 🔴 Blocker | Line 17 | Underspecified sliding inference, margin discarding, boundary handling | `PROPOSE FIX` |
| `IMPL-R1-2` | `egm-impl` | 🔴 Blocker | Global | Complete absence of file boundaries, code paths, and output schema | `PROPOSE FIX` |
| `STRUCT-R1-1` | `egm-structural` | 🟡 Major | Lines 1-8 | Missing formal Problem Statement, Goals, and Non-Goals | `PROPOSE FIX` |
| `STRUCT-R1-2` | `egm-structural`, `egm-design` | 🟡 Major | Lines 29-37 | Asymmetric statistical decision rule with dead zones | `PROPOSE FIX` |
| `IMPL-R1-3` | `egm-impl`, `egm-codebase` | 🟡 Major | Line 19 | Undefined baseline architectures (SLAMSS variant and time prior) | `PROPOSE FIX` |
| `IMPL-R1-4` | `egm-impl`, `egm-codebase` | 🟡 Major | Line 18 | Unreferenced `EXP014_MANIFESTS` and unspecified checkpointing rule | `PROPOSE FIX` |
| `STRUCT-R1-3` | `egm-structural` | 🟢 Nit | Lines 13, 20, 40 | Conversational phrasing ("Your CNN-BiGRU") and missing metadata | `PROPOSE FIX` |

---

## Detailed Resolutions & Planned Action Items

1. **Alternatives Considered**: Add formal section evaluating strided sliding windows vs random crops, transformer/state-space models vs CNN-BiGRU, and staged discovery vs monolithic CV.
2. **Compute Control**: Replace `batch * L = constant` with fixed total step budget ($S_{\text{total}} = 3,200$) and constant physical trajectory batch size ($B=4$) with gradient accumulation. Treat $L=1$ as a distinct non-recurrent ablation (or bound $L \ge 5$).
3. **Staged Execution**: Partition into **Phase 1** (Horizon discovery & curve validation: 7 $L$ values $\times$ 3 seeds on 31 train / 5 val subjects = 21 runs) and **Phase 2** (Cohort-wide 5-fold CV across 47 subjects, gated on Phase 1 results).
4. **Metadata Disentanglement**: Explicitly zero out `time_feat` (`x_meta[:, :, 2] = 0.0`) in the primary context-horizon condition to prevent circadian leakage.
5. **Clinical Metrics**: Formulate exact mathematical definitions for TST, SOL, WASO, SE, and specify the implementation of `ClinicalHypnogramEvaluator`.
6. **Sliding Inference Engine**: Specify window size $W=L$, stride $S = \max(1, \lfloor L/2 \rfloor)$, margin discard fraction $m=0.25$ ($m=0$ for $L=1$), boundary condition padding with masking, and mean probability aggregation.
7. **File Boundaries & Governance**: Add explicit write allowlist, code paths, artifact directories (`experiments/EXP-018/`), and test plans.
8. **Decision Framework**: Establish signed 4-quadrant outcome matrix and pre-register parametric exponential saturation curve fit $F_1(L) = A - B e^{-k L}$.


---

# Goldfish Review Report: Round 2 (EXP-018 Design Proposal)

**Target File**: `experiments/EXP-018/design_draft.md`  
**Audited By**: 4-Agent Goldfish Swarm (`egm-design`, `egm-implementation-readiness`, `egm-codebase-details`, `egm-structural`)  
**Standard**: `feature-spec-standard.md` & `implementation-readiness-standard.md`  
**Review Status**: 🟡 CONDITIONAL PASS $\to$ 🟢 **ALL RESOLVED** (1 Blocker, 6 Majors, 5 Nits across swarm; 0 unresolved remaining)

---

## Round 2 Review Matrix

| Finding ID | Reviewer | Severity | Scope | Core Issue | Resolution |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `IMPL-R2-1` | `egm-impl` | 🔴 Blocker | Sec 4.2 | Ambiguous micro-batch size ("B=4 or B=2") and unspecified $L=1$ schedule | **Resolved**: Added explicit $(L, B, N_{\text{accum}}, \text{Tokens/step})$ configuration table maintaining exact $2,400$ token exposure across all $L$. |
| `IMPL-R2-2` | `egm-impl`, `egm-design` | 🟡 Major | Sec 4.4 | Destructive margin trimming and negative indexing crash when $N \le L$ | **Resolved**: Added explicit branching for $N \le L$ (bypass margin discard $m=0$, right-pad with zeros, pass native $N$ to `pack_padded_sequence`) and non-negative boundary indexing. |
| `DESIGN-R2-1` | `egm-design` | 🟡 Major | Sec 4.2 | Micro-batch cap of $B=16$ at $L=1$ causes 480k-pass IPC bottleneck | **Resolved**: Decoupled $L=1$ flat epoch batching ($B=240, N_{\text{accum}}=10$), cutting forward passes to 32k while preserving exact token exposure. |
| `STRUCT-R2-1` | `egm-structural`, `egm-design` | 🟡 Major | Sec 7 | Statistical decision matrix dead zones (sub-threshold impairment, bounded straddles) | **Resolved**: Restructured decision matrix into mutually exclusive and collectively exhaustive (MECE) partitions with explicit operational routing for all cases. |
| `DESIGN-R2-2` | `egm-design` | 🟡 Major | Sec 3 | Unmapped gating disposition for `SUB-THRESHOLD GAIN` in Phase 1 flowchart | **Resolved**: Formally routed `SUB-THRESHOLD GAIN` to conclude at Phase 1 and document findings (no Phase 2 CV authorized). |
| `IMPL-R2-3` | `egm-impl` | 🟡 Major | Sec 3, 4.3 | Circadian Ablation Series execution scope unbudgeted in Phase 1 / Phase 2 | **Resolved**: Designated Circadian Ablation as Phase 1b (9 runs), executed only post-gating. |
| `CODE-R2-1` | `egm-codebase` | 🟢 Nit | Global | Verification of cached feature dimensions and fold manifests | **Verified**: 100% fidelity with codebase contracts and cache fingerprints. |
| `STRUCT-R2-2` | `egm-structural` | 🟢 Nit | Sec 7 | Trailing colon in section header | **Resolved**: Formatted cleanly. |
| `IMPL-R2-4` | `egm-impl` | 🟢 Nit | Sec 5 | SLAMSS-IFS execution role ambiguous | **Resolved**: Explicitly marked as a frozen external comparator (0 new training runs required). |
| `IMPL-R2-5` | `egm-impl` | 🟢 Nit | Sec 6, 10 | Clinical metric zero-sleep edge cases and missing unit test enumeration | **Resolved**: Defined edge cases ($\text{SOL}=\text{TIB}, \text{WASO}=0.0, \text{SE}=0.0$) and enumerated 6 explicit unit tests. |
| `DESIGN-R2-3` | `egm-design` | 🟢 Nit | Sec 6 | Ambiguous mathematical formula for saturation threshold $L^*$ | **Resolved**: Disambiguated to $F_1(L^*) = A - 0.05 B \implies L^* = \frac{\ln(20)}{k} \approx \frac{2.996}{k}$. |
| `DESIGN-R2-4` | `egm-design` | 🟢 Nit | Sec 4.3 | Metadata channel naming precision | **Resolved**: Naming aligned to `[raw_hr_rate, motion_count, time_feat, cov_hr, cov_motion]`. |

---

## Convergence Assessment
With all Round 2 findings resolved:
- **Blockers**: 0
- **Majors**: 0
- **Nits**: 0
- **Architectural Soundness**: Verified across all 4 specialist reviewer roles.
- **Implementation Readiness**: 100% of algorithmic parameters, batch schedules, inference edge cases, and unit tests are explicitly defined with zero guesswork.
