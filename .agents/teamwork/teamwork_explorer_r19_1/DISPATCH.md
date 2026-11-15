# DISPATCH: NAS Iteration 19 — Candidate Rho Architecture Exploration

## Mission
You are dispatched as `teamwork_explorer_r19_1` to formulate, implement, and verify the structural mutation for **Candidate Rho** (`AutonomicHomeostasisBoundedMarginPyramidCNNBiGRU`) targeting Hypotheses H-F and H-A on the BIDSleep dataset.

## Required Reading
1. `ORIGINAL_REQUEST.md` (project root)
2. `.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md` (critical: review failures of Iterations 2-18)
4. `.agents/teamwork/orchestrator_gen2/GATE_STATUS.md`
5. `experiments/EXP-014/nas_iteration12_model.py` (Candidate Kappa: baseline 73.00% inner-val, 67.04% outer-test)
6. `experiments/EXP-014/nas_iteration17_model.py` (Candidate Omicron: verified BoundedMarginSTE)
7. `experiments/EXP-014/nas_iteration18_model.py` (Candidate Pi)

## Core Architectural Motivation: Candidate Rho
- **The Groundbreaking Discovery**:
  * In Candidate Kappa (and all prior candidates inheriting from Theta/Zeta), the synergy formula for Light sleep was:
    $$s_{\text{light}} = \text{ReLU}(\tau_{\text{ihr\_light}} - u_{\text{ihr}}) \cdot \text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_light}})$$
  * Notice the second factor: $\text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_light}})$.
  * During quiet sleep, accelerometry deviation $u_{\text{acc}} = \Delta z_{\text{acc}}$ is zero or negative ($u_{\text{acc}} \le 0$). With initial $\tau_{\text{acc\_light}} = 0.0$, $u_{\text{acc}} - \tau \le 0 \implies \text{ReLU} = 0.0$!
  * **Consequently, $s_{\text{light}}$ was IDENTICALLY ZERO during genuine Light sleep!** The model possessed zero active synergy signal for Light sleep!
  * To compensate, the network was forced during training to inflate the linear classifier bias $b_{\text{light}}$ to a huge positive value so that whenever $s_{\text{wake}}=0, s_{\text{deep}}=0, s_{\text{rem}}=0$, the model outputs Light sleep.
  * On held-out test subjects, whenever feature norms $\|h\|$ fluctuated or $s_{\text{rem}}/s_{\text{deep}}$ experienced minor physiological attenuation, the massive $b_{\text{light}}$ bias dominated, collapsing 2,657 REM windows (37.4%) and 2,225 Deep windows (28.1%) into Light sleep!
- **The Candidate Rho Solution: Autonomic & Somatomotor Homeostasis Contrast (AH-CASCI)**:
  1. **Physiologically Authentic Light Sleep Synergy**:
     In physiology, Light sleep (N1/N2) is defined by **Autonomic and Somatomotor Homeostasis / Basal Quiescence** (HR near ultradian baseline, absence of active movement, absence of deep slow-wave bradycardia, absence of REM sympathetic storm):
     $$s_{\text{light}} = \text{ReLU}(\tau_{\text{ihr\_light}} - |u_{\text{ihr}}|) \cdot \text{ReLU}(\tau_{\text{acc\_light}} - |u_{\text{acc}}|)$$
     where:
     - $|u_{\text{ihr}}| = |\Delta z_{\text{ihr}}|$ measures deviation from the 91-epoch ultradian HR baseline.
     - $|u_{\text{acc}}| = |\Delta z_{\text{acc}}|$ measures deviation from the 91-epoch accelerometry baseline.
     - $\tau_{\text{ihr\_light}}$ initialized to $0.50$ (HR within $\pm 0.5$ SD of basal mean).
     - $\tau_{\text{acc\_light}}$ initialized to $0.25$ (movement within $\pm 0.25$ SD of basal quietness).
     Now, when the subject is in quiet, stable Light sleep, $s_{\text{light}}$ **ACTIVELY FIRES** as a genuine discriminative feature vector! When HR surges (REM) or drops (Deep) or movement occurs (Wake), $s_{\text{light}}$ cleanly shuts off!
  2. **Harmonized 4-Stage Context Bottleneck**:
     - Increase `hidden_dim` in `bm_casci` from 3 to **4 dimensions** (`fc1 = Linear(8, 4)` and `fc2 = Linear(4, 144)`), providing a full 4-dimensional rank-4 subspace corresponding to the 4 sleep stages [Wake, Light, Deep, REM].
  3. **Bounded Margin Stability Cutoffs (`BoundedMarginSTE`)**:
     - Retain `BoundedMarginSTE` from Candidate Omicron:
       $\tau_{\text{acc\_rem}} \ge 0.05, \tau_{\text{acc\_deep}} \ge 0.05, \tau_{\text{acc\_wake}} \ge 0.0, \tau_{\text{acc\_light}} \ge 0.05$
       $\tau_{\text{ihr\_rem}} \ge -0.05, \tau_{\text{ihr\_deep}} \ge 0.0, \tau_{\text{ihr\_wake}} \ge 0.0, \tau_{\text{ihr\_light}} \ge 0.05$
       preventing negative margin drift while maintaining 100% gradient flow.
  4. **Strict Parameter Budget**:
     - Candidate Kappa: 196,381 params.
     - `bm_casci` with `hidden_dim=4`: $36 + 8 + 576 + 20 + 8 = 648$ params (+159 params over Kappa's 489).
     - Total trainable parameters: **Exactly 196,540 parameters**.
     - Complies strictly with the hard budget cap ($\le 196,728$, 188 parameters headroom).
     - Compression ratio: 96.41% vs 5.47M benchmark.

## Specific Deliverables
1. Implement Candidate Rho in `experiments/EXP-014/nas_iteration19_model.py`.
2. Implement and run a rigorous verification script `experiments/EXP-014/verify_candidate_rho.py` using `./.venv/bin/python` on `cuda:0` verifying:
   - Parameter budget compliance ($\le 196,728$, expected 196,540).
   - Forward pass on `(B=4, C=4, T=600)` on `cuda:0`.
   - Active firing of $s_{\text{light}} > 0$ when $|u_{\text{ihr}}| < \tau$ and $|u_{\text{acc}}| < \tau$.
   - BoundedMarginSTE stability cutoffs enforced.
   - 100% finite, non-zero gradient flow to all trainable parameters during backprop.
   - Multi-task RCCLoss compatibility.
3. Write your final report to `.agents/teamwork/teamwork_explorer_r19_1/handoff.md`.
4. Message the orchestrator upon completion.
