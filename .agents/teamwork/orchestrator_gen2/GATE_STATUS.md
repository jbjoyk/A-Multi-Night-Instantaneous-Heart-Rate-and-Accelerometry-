## Gate — Iteration 12 (Candidate Kappa)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r12_1 | teamwork_worker | PROXY_FAIL (Acc 66.84% < 70.0%, WF1 0.6678 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 66.84% Acc < 70.0%, 0.6678 WF1 < 0.6900; Epoch 50: 67.04% Acc). Mutation reverted per Requirement R1.3.

## Gate — Iteration 13 (Candidate Lambda)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r13_1 | teamwork_worker | PROXY_FAIL (Acc 61.85% < 70.0%, WF1 0.6130 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 61.85% Acc < 70.0%, 0.6130 WF1 < 0.6900).
Key Empirical Discovery: DA-BMS autonomic dispersion drove REM recall to 69.74% (+12.36 pp) and Deep recall to 78.39% (+8.18 pp). However, asymmetric ACAM margins (m_rem=0.15, m_deep=0.12) over-carved majority Light sleep (47.15% recall, 3,464 REM false alarms). Mutation reverted per Requirement R1.3.

## Gate — Iteration 14 (Candidate Mu)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r14_1 | teamwork_worker | PROXY_FAIL (Acc 64.55% < 70.0%, WF1 0.6458 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 64.55% Acc < 70.0%, 0.6458 WF1 < 0.6900).
Key Empirical Discovery: Restoring the Multi-Stream Linear Head directly recovered Light sleep recall (+17.86 pp, +2,476 rescued windows, Acc +2.70 pp over Lambda). However, additive OR-gating in dual-autonomic synergy ($u_{\text{ihr}} \lor v_{\text{disp}}$) allows transitional Light sleep autonomic fluctuations to trigger false REM (2,453 false alarms) and Deep (1,711 false alarms) synergy, capping outer-test accuracy below 70.0%. Mutation reverted per Requirement R1.3.

## Gate — Iteration 15 (Candidate Nu)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r15_1 | teamwork_worker | PROXY_FAIL (Acc 64.52% < 70.0%, WF1 0.6452 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 64.52% Acc < 70.0%, 0.6452 WF1 < 0.6900).
Key Empirical Discovery: CADM multiplicative dispersion gating validated: eliminated 278 false REM alarms and recovered 282 Light sleep windows over Candidate Mu. However, whole-sequence z-scoring across T=600 (SAN) subtracted global subject means, destroying the subject's macro-structural sleep stage composition signal (forcing Deep-sleep dominant subjects to zero-mean, causing 2,227 Deep->Light and 2,420 REM->Light errors). Mutation reverted per Requirement R1.3.

## Gate — Iteration 16 (Candidate Xi)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r16_1 | teamwork_worker | PROXY_FAIL (Acc 62.93% < 70.0%, WF1 0.6300 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 62.93% Acc < 70.0%, 0.6300 WF1 < 0.6900).
Key Empirical Discovery: Unconstrained margin drift in CADM ($\tau_{\text{ihr\_rem}} = -0.0672, \tau_{\text{acc\_rem}} = +0.3113$) combined with uniform pre-classifier depthwise causal temporal smoothing (TBR) smeared representations across stage transitions, diluting sharp stage boundaries and causing Light->REM false alarms to surge to 2,976 windows. Mutation reverted per Requirement R1.3.

## Gate — Iteration 17 (Candidate Omicron)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r17_1 | teamwork_worker | PROXY_FAIL (Acc 66.54% < 70.0%, WF1 0.6644 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 66.54% Acc < 70.0%, 0.6644 WF1 < 0.6900; Epoch 35 inner-val: 73.04% Acc / 0.7391 MF1).
Key Empirical Discovery: `BoundedMarginSTE` successfully clamped `tau_ihr_rem` at the -0.0500 lower bound, and inner-validation achieved an all-time project record accuracy of 73.04%. However, additive logit prior correction ($\tau_{\text{prior}} \log p$) with a learnable $\tau$ was absorbed by the linear classifier bias parameters ($b_c$) during training and attenuated by cross-entropy gradients, failing to halt outer-test stage collapse into Light sleep (2,657 REM and 2,225 Deep windows misclassified as Light). Mutation reverted per Requirement R1.3.

## Gate — Iteration 18 (Candidate Pi)
| Agent | Role | Verdict | Source |
|---|---|---|---|
| teamwork_worker_r18_1 | teamwork_worker | PROXY_FAIL (Acc 65.24% < 70.0%, WF1 0.6517 < 0.6900) | handoff.md |

Gate Result: **FAIL** (Proxy filter threshold not met: 65.24% Acc < 70.0%, 0.6517 WF1 < 0.6900; Epoch 50: 65.24% Acc).
Key Empirical Discovery: Removing additive linear bias terms in UH-CH boosted minority stage recalls (REM recall rose to 57.11%, Deep to 70.02%), but discarding feature norms $\|h\|$ via hyperspherical normalization on $S^{63}$ stripped physiological amplitude confidence and allowed 3,714 Light sleep windows to disperse into Deep and REM, dropping test accuracy to 65.24%. Mutation reverted per Requirement R1.3.
