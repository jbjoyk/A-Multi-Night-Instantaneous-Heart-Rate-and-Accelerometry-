# PHASE 3 INFRASTRUCTURE SMOKE TEST SPECIFICATION

> [!IMPORTANT]
> **EXECUTION STATUS: EXECUTED & PASSED (2026-11-14)**  
> Governed Phase-3 infrastructure smoke test `EXP-016-SMOKE` executed under explicit human authorization.
> - **Execution Mode**: `INFRASTRUCTURE_SMOKE_TEST`
> - **Status**: `COMPLETED`
> - **Scientific Evidence**: `FALSE` (strictly classified as operational non-evidence)
> - **Runtime**: 1.20s epoch execution (2.38s total cycle) on `cuda:0` (NVIDIA GeForce RTX 3060 12GB).
> - **Memory**: Peak VRAM allocated was 267.1 MB (well below the 10,240 MB limit).
> - **Leakage Check**: 0 outer test subjects accessed (`test_set_accessed = false`).
> - **Metrics**: Train Loss = 1.3409, Val Loss = 1.2476, Val Macro-F1 = 0.2855, Accuracy = 0.4496, Kappa = 0.1255.
> - **Decision**: `SIMULATION_NON_EVIDENCE`.
> - **Registry**: Successfully persisted as 13th entry in `research_memory.json`. All 76 automated repository tests pass cleanly.

---

## 1. PURPOSE & SCIENTIFIC SCOPE

The objective of the Phase-3 smoke test is strictly **engineering verification** of the autonomous execution loop:
1. Verify that the runtime environment successfully instantiates PyTorch on `cuda:0`.
2. Verify that 1 real training epoch runs: dataloading from the audited cache (`prep_57738d534b`), forward pass through `SleepChannelNet`, loss computation, backward pass (gradient calculation), and 1 optimizer step.
3. Verify that memory consumption remains strictly within the 10,240 MB VRAM ceiling on an RTX 3060.
4. Verify that the output metrics are evaluated only on the inner validation split (5 subjects), with **zero access to held-out test data**.
5. Verify that the execution token lifecycle, state machine transitions, and memory persistence function end-to-end under live conditions.

### Scientific Classification
- **Execution Mode**: `INFRASTRUCTURE_SMOKE_TEST`
- **Scientific Evidence**: `FALSE` (non-evidence)
- **Eligible for Leaderboard**: `FALSE`
- **Eligible for Model Selection / Hypotheses**: `FALSE`
- **Role**: Operational pipeline validation only.

---

## 2. HARDWARE & COMPUTE CONSTRAINTS

| Parameter | Specification | Enforcement Mechanism |
|---|---|---|
| **GPU Device** | `cuda:0` (Single NVIDIA GeForce RTX 3060 12GB) | `GPUResourceManager` exclusive lock |
| **VRAM Ceiling** | 10,240 MB maximum allocation | Monitored by `torch.cuda.max_memory_allocated` |
| **Compute Budget** | `SMOKE` (1 epoch) | Hard Governor ceiling ($\le 2$ epochs) |
| **Time Limit** | 600 seconds (10 minutes) | Governor execution timeout |
| **Multi-GPU / DDP** | Disabled | Single-device execution |

---

## 3. MODEL & TRAINING CONFIGURATION

| Hyperparameter | Value | Description |
|---|---|---|
| **Model Architecture** | `SleepChannelNet` (Model A baseline configuration) | Matched conventional CNN-BiGRU |
| **Input Channels** | 12 instantaneous HR/accelerometry features + 5 temporal metadata features | Standard audited input representation |
| **Sequence Length ($T$)** | 600 epochs (5 hours) | Standard window duration |
| **Batch Size** | 4 sequences | Standard batch size from EXP-012/013/014 |
| **Epochs** | 1 epoch exactly | Minimum required to verify forward + backward + step |
| **Optimizer** | Adam (`torch.optim.Adam`) | Standard project optimizer |
| **Learning Rate** | $1 \times 10^{-4}$ | Standard learning rate |
| **Weight Decay** | $0.0$ | No weight decay |
| **Loss Function** | Cross-Entropy Loss with class weights / RCC | Standard project loss |
| **Random Seed** | `42` | Frozen deterministic seed (`torch.manual_seed(42)`) |

---

## 4. DATASET & SPLIT INTEGRITY

| Gate / Property | Requirement | Verification Method |
|---|---|---|
| **Dataset Cache** | `prep_57738d534b` | SHA-256 fingerprint verified (`8ae7a916a4cd5709`) |
| **Training Split** | 31 official training subjects | `dataset_gate.py` disjointness assertion |
| **Validation Split** | 5 official validation subjects | `dataset_gate.py` inner validation role |
| **Test Set Protection** | 11 held-out test subjects | `TestSetProtectionGate` (hard exception on test access) |
| **Data Leakage Check** | `assert_no_test_leakage` called before dataloader instantiation | Automated pre-flight check |

---

## 5. REPLAY PREVENTION & TOKEN AUTHORIZATION

Execution requires a cryptographically signed HMAC token issued by `ResearchGovernor` at Level 4/5:
- **Token Format**: `<timestamp>:<content_hash>:<hmac_signature>`
- **Token Validity**: 3600 seconds maximum lifespan.
- **Single-Use Replay Protection**: Cached in `ExperimentEngine.used_tokens`.
- **Payload Invariance**: Binds `experiment_id = "EXP-016-SMOKE"`, `budget = "SMOKE"`, exact change content hash, and dataset fingerprint `8ae7a916a4cd5709`.

---

## 6. EXPECTED OUTCOMES & VERIFICATION CRITERIA

The smoke test will be deemed a **PASS** if and only if all of the following conditions are met:
1. PyTorch initializes on `cuda:0` without CUDA out-of-memory or driver error.
2. 1 epoch completes over the training split, with logged loss decreasing across mini-batches.
3. 1 validation evaluation completes over the 5 validation subjects, producing valid metric values ($0 \le \text{Accuracy} \le 1$, $0 \le \text{Macro-F1} \le 1$, $-1 \le \kappa \le 1$).
4. Evaluator confirms `test_set_accessed = False`.
5. Result is recorded in `research_memory.json` as:
   ```json
   {
     "experiment_id": "EXP-016-SMOKE",
     "execution_mode": "INFRASTRUCTURE_SMOKE_TEST",
     "lifecycle_status": "COMPLETED",
     "real_training": true,
     "scientific_evidence": false,
     "eligible_for_leaderboard": false,
     "simulation_tag": "INFRASTRUCTURE SMOKE TEST — NON-EVIDENCE"
   }
   ```
6. The GPU lock on `cuda:0` is cleanly released in the resource manager context manager.

---

## 7. HUMAN AUTHORIZATION GATE

> [!CAUTION]
> Execution of this smoke test is **STRICTLY BLOCKED** until explicit written human authorization is provided in the chat interface. No agent may trigger this execution autonomously.
