# EXP-020 Regularized & Gated Adaptation Closure Report (Exploratory)

> **Prior-Access Notice:** Design-informed exploratory closure on BIDSleep; no confirmatory claims.

| Arm | Mean Macro-F1 | Δ vs ZS (pp) | 95% CI (pp) | Holm p | Δ vs FT-2 (pp) | N_harm (≤ -5pp) | Min Δ (pp) |
|---|---:|---:|---|---:|---:|---:|---:|
| `FT-2-KL` | 0.6562 | +1.03 | [-0.11, +2.01] | 0.0030 | +0.02 | 3 | -15.14 |
| `GATE_tau` | 0.6591 | +1.33 | [+0.49, +2.12] | 0.0028 | +0.32 | 2 | -6.40 |
| `FT-2-KL_GATE_tau` | 0.6559 | +1.00 | [-0.14, +1.97] | 0.0030 | -0.01 | 3 | -15.14 |
