# Cover Letter — *IEEE Journal of Biomedical and Health Informatics*

**Date:** November 28, 2026  
**To:** Editor-in-Chief, *IEEE Journal of Biomedical and Health Informatics* (*IEEE J-BHI*)  
**Subject:** Manuscript Submission — *"Training-Recipe Sensitivity, Edge Capacity Scaling, and Personalization Boundaries in Consumer-Smartwatch Sleep Staging: A Pre-Registered, Leakage-Free Benchmark"*

Dear Editor-in-Chief,

We are pleased to submit our manuscript for consideration in the *IEEE Journal of Biomedical and Health Informatics*.

Consumer smartwatches estimate sleep architecture from rate-limited ($\sim$0.20~Hz) instantaneous heart rate (IHR) and 1~Hz tri-axial accelerometry, yet clinical and edge translation has been obscured by unquantified training-recipe sensitivity, confounded architectural comparisons, and unverified personalization claims. In this pre-registered, leakage-free study across multi-night home Dreem~2 EEG (`PhysioNet BIDSleep v1.0.1`, $N=47$, 246 nights, 203,338 valid epochs) and zero-shot external in-lab polysomnography (`PhysioNet SleepAccel v1.0.0`, $n=25$ quality-clean primary; $N=31$ sensitivity), we establish four main findings:

1. **Training-Recipe Consistency Over Architectural Complexity:** Unifying the training recipe (epoch-based early stopping on inner-validation Macro-F1, AMP + FP32 `CE + 0.01*RCCLoss`, logit-stitched full-night $L=1200$ evaluation) raised the exact 163,940-parameter `Model A` architecture from `64.79%` to `66.68%` Macro-F1 (`W100`, $+1.89\text{ pp}$, $p = 2.12 \times 10^{-6}$). A parameter-matched ablation (`full_v5` FiLM + 2-scale temporal pooling, `190,788` params vs.\ plain `P190`, `190,940` params) yielded only $+0.14\text{ pp}$ ($p_{\text{Holm}} = 0.4037$), providing a transparent negative architectural ablation.
2. **Edge Efficiency & Verified 500-Epoch Convergence:** Under $<400\text{ KiB}$ Dynamic INT8 serialized weight payloads (weights only), compact plain CNN-BiGRUs (`p1_100k`: `98,940` params, `182.00 KiB` INT8 weights, `456.1 MMACs`, `41.02 ms` interleaved 4T CPU latency, `66.67%` Macro-F1) narrowed ~70–73% of the in-cohort gap against the `5.47M`-parameter `Official SLAMSS-IFS` server baseline (`67.37%`), while teacher-distilled `M3` (`1.40M` params) matched `SLAMSS-IFS` Macro-F1 (`67.38%`) and improved Accuracy by $+1.00\text{ pp}$ at $35.7\times$ faster 4T FP32 CPU latency. A dedicated 500-epoch convergence audit (`60/60` runs early-stopped) confirmed that `p1_100k` ($+0.01\text{ pp}$) and `W100` ($+0.02\text{ pp}$) were already converged at 100 epochs, proving that the $\sim$94k+ plateau is genuine rather than a 100-epoch ceiling artifact.
3. **Differentiation from EEG Long-Sequence Models:** We explicitly contextualize our rate-limited ($\sim$0.20~Hz) smartwatch IHR+ACC findings against EEG spectrogram models published in *IEEE J-BHI*, notably **L-SeqSleepNet** (Phan, Lorenzen, Heremans et al., *IEEE J-BHI* 2023, DOI `10.1109/JBHI.2023.3303197`), highlighting how autonomic cardiac dynamics impose tighter signal-to-noise and personalization limits than direct cortical EEG.
4. **Pre-Registered Personalization & External PSG Boundaries:** On recipe-consistent `W100` ($N=46$), two-night personalization yielded a small, practically sub-threshold effect (median $+0.75\text{ pp}$ $[+0.11, +1.28]$, `30W/16L`; mean $+0.58\text{ pp}$ $[-0.06, +1.22]$, below the $+1.50\text{ pp}$ threshold) with null 1-to-4-night scaling ($p = 0.5667$). Zero-shot evaluation on external in-lab PSG showed no statistically significant Macro-F1 difference between `W100` (`61.23%` single / `62.93%` ensemble) and `SLAMSS-IFS` (`61.08%` / `62.79%`, $p=0.9578$), with systematic Deep\%/REM\% overestimation reported transparently without clinical-agreement claims.

All pre-registrations, statistical protocols, TRIPOD+AI checklist items, and verification gates passed with zero errors prior to submission. We confirm that this manuscript is original, has not been published previously, and is not under consideration elsewhere.

Sincerely,  
The Authors
