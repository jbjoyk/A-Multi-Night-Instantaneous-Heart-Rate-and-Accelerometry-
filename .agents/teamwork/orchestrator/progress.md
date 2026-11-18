## Current Status
Last visited: 2026-11-15T08:51:15Z
- [x] Initialized DISPATCH.md, BRIEFING.md, SCOPE.md, DEAD_ENDS.md
- [x] Survey codebase and prior iterations (Explorers r2_1, r2_2, r2_3 completed)
- [x] NAS Iteration 2: Candidate B (Cross-Modal FiLM, 193,380 params) -> Reverted per proxy failure
- [x] NAS Iteration 3: Candidate Alpha (CACMF, 190,502 params) -> Inner-val 72.53%, outer-test 63.09% -> Reverted per proxy failure
- [x] NAS Iteration 4: Candidate Beta CACA (186,870 params) -> Inner-val 71.26%, outer-test 62.59% -> Reverted per proxy failure
- [x] NAS Iteration 5: Candidate Gamma (193,156 params) -> Inner-val 69.34%, outer-test 63.16% -> Reverted per proxy failure
- [x] NAS Iteration 6: Candidate Delta (195,346 params) -> Inner-val 70.60%, outer-test 62.11% (Ep 50: 63.09%) -> Reverted per proxy failure
- [x] NAS Iteration 7: Candidate Epsilon (195,708 params) -> Inner-val 70.42%, outer-test 64.27% Acc / 0.6435 WF1 (New Best) -> Reverted per proxy filter
- [x] NAS Iteration 8: Candidate Zeta (195,892 params) -> Inner-val 70.66%, outer-test 65.54% Acc / 0.6556 WF1 / Kappa 0.5051 (All-Time Project Best) -> Reverted per proxy filter
- [x] NAS Iteration 9: Candidate Eta (196,644 params) -> Inner-val 71.78%, outer-test 64.57% Acc / 0.6475 WF1 -> Reverted per proxy failure
- [x] NAS Iteration 10: Candidate Theta (196,373 params) -> Inner-val 72.96%, outer-test 66.84% Acc / 0.6677 WF1 (Epoch 50: 67.01% Acc / 0.6684 WF1, Kappa 0.5153, All-Time Project Best) -> Reverted per proxy filter
- [x] NAS Iteration 11: Candidate Iota (196,409 params) -> Inner-val 71.06%, outer-test 65.12% Acc / 0.6513 WF1 -> Reverted per proxy filter
- [ ] NAS Iteration 12: Formulate Candidate Kappa (Bounded Learned-Margin Synergy Manifold + Candidate Theta Foundation)
- [ ] NAS Iteration 12: Implement and run 50-epoch Fold 1 proxy evaluation
- [ ] If proxy passes, execute 5-fold CV across all 47 subjects on dual GPUs

## Iteration Status
Current iteration: 12 / continuous (unbounded)

## Retrospective Notes
- Candidate Theta remains the all-time project champion: 66.84% Accuracy (67.01% at Epoch 50), 0.6677 Weighted-F1, Cohen's Kappa 0.5153 (+1.30 pp Acc over Zeta).
- Candidate Iota proved that removing the sharp zero boundary with softplus causes synergy leakage into genuine Light sleep, blowing Light->REM false alarms back up to 2,412 and degrading accuracy to 65.12%.
- Pathology Analysis for Iteration 12: We MUST retain the sharp zero-cutoff of ReLU to protect Light sleep, but introduce a **learned asymmetric margin** $\tau_{\text{acc}} > 0$ and $\tau_{\text{ihr}} < 0$: $s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr}}) \cdot \text{ReLU}(\tau_{\text{acc}} - u_{\text{acc}})$. This allows small physiological REM twitches (up to $\tau_{\text{acc}} \approx 0.25$) without killing $s_{\text{rem}}$, while strictly cutting off once real body movements occur ($u_{\text{acc}} > \tau_{\text{acc}}$)! In addition, in the classifier head, apply an **Adaptive Angular Margin (ArcMargin)** to enforce angular separation between Light, Deep, and REM representations.



