# Architectural Analysis & Proposal: NAS Iteration 5 (Candidate Gamma)
**Target Agent**: `teamwork_explorer_r5_1` (Neural Architecture Explorer)  
**Date**: 2026-11-15T06:38:00Z  
**Document**: `.agents/teamwork/teamwork_explorer_r5_1/analysis.md`  
**Target Hypotheses**: Hypothesis H-F (Autonomic–Somatomotor Cross-Modal Interaction) & Hypothesis H-A (Subject-Invariant Generalization across All 47 Subjects)  
**Model Name**: `InterEpochSkipAttentionCNNBiGRU` (`CandidateGamma_IESA`)  
**Proposed Implementation**: `.agents/teamwork/teamwork_explorer_r5_1/proposed_nas_iteration5_model.py`  
**Verification Suite**: `.agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py` (8/8 tests passed in 1.285s)  

---

## 1. Executive Summary

In NAS Iterations 1 through 4, every architectural mutation was confined exclusively to *intra-epoch* feature smoothing or cross-modal gating/attention prior to recurrent sequence ingestion. Meanwhile, the macro-scale temporal sequence model remained identical to baseline Model A: an unaugmented 2-layer Bidirectional GRU (hidden size 48 per direction $\to$ 96 dims bidirectional) with **no skip connections** and **no sequence-level attention**.

Over a 600-epoch nocturnal sequence ($T = 600$ epochs $\times$ 30 seconds = 18,000 seconds = 5 hours), this 48-dimensional recurrent hidden state acts as an acute information bottleneck. It suffers from exponential gradient decay during backpropagation through time (BPTT) and cannot retain memory of 90-minute ultradian sleep cycles (180 epochs) or preserve high-frequency instantaneous autonomic-somatomotor cues.

To overcome this fundamental bottleneck, **Candidate Gamma** (`InterEpochSkipAttentionCNNBiGRU`) adapts the key structural innovations of the published SLAMSS-IFS benchmark (Song et al., IEEE TBME 2026, 71.04% Acc, 0.7079 WF1) while achieving a **96.47% parameter compression** (193,156 parameters vs 5,471,044 parameters):
1. **Direct Inter-Epoch Feature Skip Highway**: Directly routes the 144-dimensional intra-epoch representation (`z_hr` 64 + `z_acc` 64 + `h_meta` 16) around the recurrent temporal model straight into the classifier head. This eliminates the BPTT Jacobian bottleneck, allowing raw gradients from the classification loss to flow directly into the CNN modality encoders in a single step.
2. **Inter-Epoch Sequence-Level Multi-Head Attention (IESA)**: Implements a 4-head scaled dot-product attention module across the sequence length $T$ that provides every epoch with an unattenuated $O(1)$ receptive field across the entire night.
3. **Sequence Key-Masking & Step-0 Stability**: Incorporates strict boolean key-masking ($-10^9$ before softmax) to eliminate padded token contamination on variable-length recordings, alongside zero-identity output projection initialization (`out_proj == 0`), guaranteeing that the initial forward pass operates stably on the verified baseline manifold.
4. **Strict Parameter Budget Compliance**: The total parameter count is **exactly 193,156 parameters**, maintaining 3,572 parameters of headroom under the hard cap of 196,728 parameters (+17.82% capacity over baseline Model A).

---

## 2. Retrospective Diagnosis: Why Iterations 1 to 4 Failed

A rigorous synthesis of the experimental logs and handoff reports from Iterations 1 through 4 reveals a consistent pathology:

| Iteration | Proposed Mutation | Model Parameters | Inner-Val Result (Fold 1) | Outer-Test Result (Fold 1) | Diagnostic Root Cause |
|---|---|---|---|---|---|
| **Iteration 1** | Multi-Scale Depthwise Smoothing | 192,692 | N/A (Early Proxy) | 61.8% Acc / 0.58 WF1 | Depthwise convolution across concatenated multimodal channels blurred modality boundaries; REM F1 collapsed to 0.1287. |
| **Iteration 2** | Candidate B: Cross-Modal FiLM Gating | 193,380 | 69.56% Acc | 62.40% Acc / 0.6238 WF1 | Unconstrained affine FiLM generators overfit to training subjects' baseline resting HR; failed on held-out test subjects. |
| **Iteration 3** | Candidate Alpha: Masked TIN + Tanh Gating | 190,502 | 72.53% Acc / 0.7256 WF1 | 63.09% Acc / 0.6314 WF1 | Masked TIN achieved record inner-val performance, but static gating MLP could not resolve Light vs REM confusion on held-out test subjects (5,184 misclassified windows). |
| **Iteration 4** | Candidate Beta: Localized Cosine Cross-Attention (W=5) | 186,870 | 71.26% Acc / 0.7105 WF1 (Epoch 45) | 62.59% Acc / 0.6236 WF1 (Epoch 45) | Restoring peak inner-val checkpoint yielded 62.59% outer test (worse than Epoch 50's 62.87%), disproving late-epoch overfitting. Local cross-attention ($W=5$, 2.5 min) cannot overcome inter-subject distribution shifts. |

### The Critical Architectural Blind Spot
In every iteration above, the *temporal sequence model* was left identical to baseline Model A:
```python
self.temporal_gru = nn.GRU(
    input_size=144,
    hidden_size=48,  # Bidirectional -> 96 total
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.2
)
```
Consider the physical and mathematical consequences:
1. **Recurrent Vanishing Memory over Ultradian Horizons**:
   Human sleep architecture is organized into ultradian NREM-REM cycles of approximately 90 minutes. At 30 seconds per epoch, one complete ultradian cycle spans **180 epochs**. In a standard GRU with hidden size 48, the hidden state update equation:
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
   exhibits exponential decay over long time horizons. The Jacobian product across 180 steps:
   $$\frac{\partial h_{t+180}}{\partial h_t} = \prod_{k=t}^{t+179} \frac{\partial h_{k+1}}{\partial h_k} \approx 0$$
   Consequently, the 48-dimensional hidden state at epoch $t$ has virtually zero memory of preceding sleep cycle baselines.
2. **Destruction of High-Frequency Modality Signals**:
   To classify REM sleep accurately, the network must detect the simultaneous occurrence of:
   - Somatomotor atonia: Near-zero accelerometry amplitude and variance ($h_{\text{acc}}$ quiet).
   - Autonomic sympathetic activation: Sudden bursts of heart rate variability and heart rate elevation ($h_{\text{hr}}$ erratic).
   In baseline Model A and Candidates Alpha/Beta, these rich 64-dim modality representations were concatenated into a 144-dim vector and immediately forced through the 48-dim GRU cell. The GRU's recurrent continuous smoothing washes out the sharp, instantaneous discordance between heart rate surges and motor atonia, blurring REM sleep into quiet Light sleep (N1/N2). In Iteration 4's outer test, **44.46% of all REM windows (3,161 out of 7,109) were misclassified as Light sleep**.
3. **The Backpropagation Bottleneck**:
   In baseline Model A, the gradient to `hr_encoder` is:
   $$\frac{\partial \mathcal{L}}{\partial h_{\text{hr}, t}} = \sum_{k=1}^T \frac{\partial \mathcal{L}}{\partial \text{logits}_k} \cdot \frac{\partial \text{logits}_k}{\partial h_{\text{gru}, k}} \cdot \frac{\partial h_{\text{gru}, k}}{\partial e_{\text{intra}, t}} \cdot \frac{\partial e_{\text{intra}, t}}{\partial h_{\text{hr}, t}}$$
   Because every gradient path MUST traverse the recurrent transition weights of the BiGRU across up to 600 time steps, the gradient magnitude reaching the CNN feature extractors is either severely attenuated or distorted by the recurrent dynamics of the training subjects. The CNN feature extractors are prevented from learning sharp, invariant physiological filters.

---

## 3. Dissection of the SLAMSS-IFS Benchmark Architecture

In the published benchmark paper (*Song et al., IEEE Transactions on Biomedical Engineering*, Vol. 73, No. 4, April 2026, Table V: 71.04% Accuracy, 0.7079 Weighted-F1), the authors addressed this exact failure mode through two structural mechanisms (`slamss_ifs_baseline.py` lines 36–38 and 227–229):

```python
# SLAMSS-IFS Architecture:
# 1. Bahdanau Additive Attention: alignment between decoder states and all encoder states
# 2. Skip Connection: 258-dim epoch representation bypassed directly to output stage
# 3. Classification Head:
classifier_in = hidden_dim + hidden_dim + in_dim  # 96 [decoder] + 96 [context] + 258 [skip] = 450
self.classifier = nn.Linear(classifier_in, n_classes)
```

Why did this architecture achieve 71.04% accuracy where conventional models stalled at 62–63%?
1. **The Feature Skip Connection ($S$)**: By directly bypassing the 258-dim intra-epoch representation around the recurrent model into the final classification layer:
   $$\text{logits}_t = W_{\text{dec}} s_t + W_{\text{ctx}} c_t + W_{\text{skip}} e_t + b$$
   The classification layer can make stage predictions directly from raw intra-epoch cues (e.g. instant zero acceleration + high HRV = REM) without needing the recurrent state to preserve them.
   Furthermore:
   $$\frac{\partial \mathcal{L}}{\partial e_t} = W_{\text{skip}}^T \frac{\partial \mathcal{L}}{\partial \text{logits}_t} + \text{RNN terms}$$
   The first term is a **direct 1-step gradient highway** from the classification loss into the CNN/LSTM feature extractors.
2. **Inter-Epoch Sequence Attention**: The additive attention mechanism aligns states across the entire night:
   $$\alpha_{t, i} = \text{Softmax}_i \left( v^T \tanh(W_s s_t + W_h h_i) \right), \quad c_t = \sum_{i=1}^T \alpha_{t, i} h_i$$
   This gives every epoch $t$ direct, unattenuated access to any epoch $i$ across the entire recording, enabling the network to track ultradian sleep cycles and relate the current epoch to nocturnal baseline anchors.

However, the published SLAMSS-IFS model requires **5,471,044 parameters** (over 5.47 million parameters), primarily due to deep intra-epoch CNN stacks (5 Conv1d layers of 128 channels) and intra-epoch BiLSTM layers.
Our mission is to achieve and exceed this performance within the strict **162k–196k parameter envelope** (>90% compression).

---

## 4. Candidate Gamma Architecture Formulation

Candidate Gamma (`InterEpochSkipAttentionCNNBiGRU`) synthesizes the theoretical power of SLAMSS-IFS into an ultra-compact, tensor-safe architecture requiring exactly **193,156 parameters** (96.47% compression).

```
                      Raw Physiological Inputs (1 Hz, 30 samples/epoch)
                      [Instantaneous Heart Rate]      [Tri-axial Accelerometry]
                                  │                               │
                                  ▼                               ▼
                           ┌──────────────┐                ┌──────────────┐
                           │  hr_encoder  │                │ acc_encoder  │
                           │ (CNN, 64 dims)│                │ (CNN, 64 dims)│
                           └──────┬───────┘                └──────┬───────┘
                                  │                               │
                                  ▼                               ▼
                           ┌──────────────┐                ┌──────────────┐
                           │  tin_hr + LN │                │ tin_acc + LN │
                           │ (Masked TIN) │                │ (Masked TIN) │
                           └──────┬───────┘                └──────┬───────┘
                                  │                               │
                   z_hr (64 dims) │               z_acc (64 dims) │          h_meta (16 dims)
                   ───────────────┴───────────────┬───────────────┴───────────────┬──────────────
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Intra-Epoch Fused Features    │
                                 │   e_intra (B, T, 144 dims)      │
                                 └───────────────┬─────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         │                                               │
                         ▼                                               ▼ [Direct Feature Skip Highway]
            ┌───────────────────────────┐                                │ (Bypasses BiGRU & Attention)
            │  2-Layer Bidirectional    │                                │
            │      GRU (96 dims)        │                                │
            │  + pack_padded_sequence   │                                │
            └────────────┬──────────────┘                                │
                         │                                               │
                         ▼ temp_features (96 dims)                       │
            ┌───────────────────────────┐                                │
            │    Inter-Epoch Sequence   │                                │
            │  Multi-Head Attention     │                                │
            │ (IESA: 4 heads, d_attn=48)│                                │
            │  - Key-Masking: -1e9 pad  │                                │
            │  - Zero-init out_proj     │                                │
            └────────────┬──────────────┘                                │
                         │                                               │
         attn_context    │                                               │
           (48 dims)     │                                               │
                 │       │                                               │
                 ▼       ▼                                               ▼
         ┌────────────────────────────────────────────────────────────────────────┐
         │              Multi-Stream Concatenation (B, T, 288 dims)               │
         │           [ temp_features (96) ; attn_context (48) ; e_intra (144) ]   │
         └───────────────────────────────────┬────────────────────────────────────┘
                                             │
                                             ▼
                                 ┌───────────────────────┐
                                 │    Classifier Head    │
                                 │ Linear(288, 64) -> LN │
                                 │ -> GELU -> Dropout    │
                                 │ -> Linear(64, 4)      │
                                 └───────────┬───────────┘
                                             │
                                             ▼
                                 Stage Logits: (B, T, 4)
                           [Wake, Light, Deep, REM]
```

### 4.1 Submodule Specifications

#### 1. Modality-Specific Encoders & Meta Projection (59,232 parameters)
- Preserves absolute modality isolation prior to fusion:
  - `hr_encoder`: 1D CNN with 3 convolutional stages, BatchNorm, LayerNorm, and linear projection to $d_{\text{model}} = 64$ dims (29,568 params).
  - `acc_encoder`: 1D CNN identical architecture to $d_{\text{model}} = 64$ dims (29,568 params).
  - `meta_proj`: Linear(5, 16) + GELU projecting circadian time proxy and sampling parameters (96 params).

#### 2. Subject-Invariant Masked Temporal Instance Normalization (512 parameters)
- Standardizes nocturnal sequences per recording across valid epochs:
  $$\mu_b = \frac{1}{\sum_t M_{b, t}} \sum_{t=1}^T x_{b, t} M_{b, t}, \quad \sigma_b^2 = \frac{1}{\sum_t M_{b, t}} \sum_{t=1}^T (x_{b, t} - \mu_b)^2 M_{b, t}$$
  $$x_{\text{norm}} = \frac{x - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}} \odot \gamma + \beta$$
- Applied independently to $h_{\text{hr}}$ (64 dims) and $h_{\text{acc}}$ (64 dims), followed by residual addition and LayerNorm:
  $$z_{\text{hr}} = \text{LayerNorm}(h_{\text{hr}} + \text{TIN}(h_{\text{hr}})), \quad z_{\text{acc}} = \text{LayerNorm}(h_{\text{acc}} + \text{TIN}(h_{\text{acc}}))$$
- Intra-epoch fused multimodal representation:
  $$e_{\text{intra}} = [z_{\text{hr}} \;;\; z_{\text{acc}} \;;\; h_{\text{meta}}] \in \mathbb{R}^{B \times T \times 144}$$

#### 3. Recurrent Temporal Sequence Modeling (98,112 parameters)
- `temporal_gru`: 2-layer Bidirectional GRU with `input_size=144`, `hidden_size=48` (bidirectional output dimension = 96), `dropout=0.2` (97,920 params).
- Variable sequence length protection via `nn.utils.rnn.pack_padded_sequence` and `pad_packed_sequence`.
- `temp_norm`: `LayerNorm(96)` (192 params).

#### 4. Inter-Epoch Sequence-Level Multi-Head Attention (16,416 parameters)
- `InterEpochSequenceAttention`:
  - Input: $H \in \mathbb{R}^{B \times T \times 96}$ from `temp_norm`.
  - Linear projections: $W_Q, W_K, W_V \in \mathbb{R}^{96 \times 48}$ ($3 \times 4,656 = 13,968$ params).
  - Number of heads: $N_h = 4$, head dimension $d_k = 12$.
  - Scaled dot-product scores:
    $$S = \frac{Q K^T}{\sqrt{12}} \in \mathbb{R}^{B \times 4 \times T \times T}$$
  - **Sequence Padding Key-Masking**: For invalid key positions $j \ge L_b$:
    $$S_{b, h, t, j} = -10^9$$
    $$\alpha = \text{Softmax}(S, \text{dim}=-1)$$
    This mathematically guarantees that $\alpha_{b, h, t, j} = 0.0$ for all padded tokens, preventing padding artifacts from contaminating valid epochs.
  - Context aggregation: $\text{context} = \alpha V \in \mathbb{R}^{B \times T \times 48}$.
  - `out_proj`: `Linear(48, 48)` (2,352 params).
  - `norm`: `LayerNorm(48)` (96 params).
  - **Step-0 Zero-Identity Initialization**: `nn.init.zeros_(self.out_proj.weight)` and `nn.init.zeros_(self.out_proj.bias)`. At initialization, $\text{attn\_context} \equiv 0.0$.

#### 5. Multi-Stream Classifier Head (18,884 parameters)
- Concatenates the three complementary streams:
  $$\text{combined} = [\text{temp\_features} (96) \;;\; \text{attn\_context} (48) \;;\; e_{\text{intra}} (144)] \in \mathbb{R}^{B \times T \times 288}$$
- Classification MLP:
  - `Linear(288, 64)`: $288 \times 64 + 64 = 18,496$ params.
  - `LayerNorm(64)`: 128 params.
  - `GELU()` non-linear activation.
  - `Dropout(0.2)`.
  - `Linear(64, 4)`: $64 \times 4 + 4 = 260$ params.

---

## 5. Analytical Parameter Budget Verification

Every single parameter of Candidate Gamma is accounted for in the table below:

| Submodule | Component Type | Input Dim | Output Dim | Formula | Parameter Count |
|---|---|---|---|---|---|
| `hr_encoder` | 3-stage 1D CNN + Proj | 1 channel | 64 dims | Verified from `ConventionalCNNBiGRU` | 29,568 |
| `acc_encoder` | 3-stage 1D CNN + Proj | 1 channel | 64 dims | Verified from `ConventionalCNNBiGRU` | 29,568 |
| `meta_proj` | Linear + GELU | 5 dims | 16 dims | $5 \times 16 + 16$ | 96 |
| `tin_hr` | Masked TIN | 64 dims | 64 dims | $64 \times 2$ (weight + bias) | 128 |
| `tin_acc` | Masked TIN | 64 dims | 64 dims | $64 \times 2$ (weight + bias) | 128 |
| `norm_hr` | LayerNorm | 64 dims | 64 dims | $64 \times 2$ | 128 |
| `norm_acc` | LayerNorm | 64 dims | 64 dims | $64 \times 2$ | 128 |
| `temporal_gru` | 2-layer BiGRU | 144 dims | $48 \times 2 = 96$ | Layer 1: $6 \times (48 \times 144 + 48^2 + 96) = 55,872$<br>Layer 2: $6 \times (48 \times 96 + 48^2 + 96) = 42,048$ | 97,920 |
| `temp_norm` | LayerNorm | 96 dims | 96 dims | $96 \times 2$ | 192 |
| `inter_epoch_attn.q_proj` | Linear | 96 dims | 48 dims | $96 \times 48 + 48$ | 4,656 |
| `inter_epoch_attn.k_proj` | Linear | 96 dims | 48 dims | $96 \times 48 + 48$ | 4,656 |
| `inter_epoch_attn.v_proj` | Linear | 96 dims | 48 dims | $96 \times 48 + 48$ | 4,656 |
| `inter_epoch_attn.out_proj` | Linear | 48 dims | 48 dims | $48 \times 48 + 48$ | 2,352 |
| `inter_epoch_attn.norm` | LayerNorm | 48 dims | 48 dims | $48 \times 2$ | 96 |
| `classifier_head[0]` | Linear | 288 dims | 64 dims | $288 \times 64 + 64$ | 18,496 |
| `classifier_head[1]` | LayerNorm | 64 dims | 64 dims | $64 \times 2$ | 128 |
| `classifier_head[4]` | Linear | 64 dims | 4 classes | $64 \times 4 + 4$ | 260 |
| **TOTAL MODEL PARAMETERS** | | | | | **193,156** |

### Budget Compliance Assessment:
- Baseline Model A: 163,940 parameters
- Candidate Gamma: **193,156 parameters** (+29,216 parameters / +17.82% over baseline)
- Strict Budget Cap: **196,728 parameters**
- Headroom Remaining: **3,572 parameters** under the cap ($193,156 \le 196,728 \to$ **STRICTLY COMPLIANT**)
- Compression vs Published SLAMSS-IFS Benchmark (5,471,044 params): **96.47% compression ratio** ($> 90\%$ requirement satisfied).

---

## 6. Gradient Flow & Step-0 Mathematical Proofs

### 6.1 Direct 1-Step Backpropagation Highway
In baseline Model A, the gradient to the intra-epoch feature vector $e_{\text{intra}, t}$ at time $t$ was:
$$\frac{\partial \mathcal{L}}{\partial e_{\text{intra}, t}} = \sum_{k=1}^T \frac{\partial \mathcal{L}}{\partial \text{logits}_k} \cdot \frac{\partial \text{logits}_k}{\partial h_{\text{gru}, k}} \cdot \frac{\partial h_{\text{gru}, k}}{\partial e_{\text{intra}, t}}$$
Every term requires multiplying through the $T$-step BPTT Jacobian of the BiGRU:
$$\frac{\partial h_{\text{gru}, k}}{\partial e_{\text{intra}, t}} = \prod_{j=t+1}^k \frac{\partial h_{\text{gru}, j}}{\partial h_{\text{gru}, j-1}} \cdot \frac{\partial h_{\text{gru}, t}}{\partial e_{\text{intra}, t}}$$
In Candidate Gamma, the classification layer directly receives $e_{\text{intra}, t}$:
$$\text{logits}_t = W_{\text{gru}} h_{\text{gru}, t} + W_{\text{attn}} c_t + W_{\text{skip}} e_{\text{intra}, t} + b$$
Applying the chain rule:
$$\frac{\partial \mathcal{L}}{\partial e_{\text{intra}, t}} = W_{\text{skip}}^T \frac{\partial \mathcal{L}}{\partial \text{logits}_t} + \left( \text{BPTT terms through GRU} \right)$$
**Mathematical Consequence**: The first term $W_{\text{skip}}^T \frac{\partial \mathcal{L}}{\partial \text{logits}_t}$ is completely independent of sequence length $T$ and has **zero recurrent Jacobian decay**. The loss directly updates the CNN encoders in a single backpropagation step!

### 6.2 Step-0 Stability Guarantee
In `InterEpochSequenceAttention`:
$$\text{out\_proj.weight} = \mathbf{0}, \quad \text{out\_proj.bias} = \mathbf{0}$$
At step 0:
$$\text{attn\_context} = \text{LayerNorm}(\mathbf{0}) \cdot M = \mathbf{0}$$
Therefore:
$$\text{combined} = [h_{\text{gru}, t} \;;\; \mathbf{0} \;;\; e_{\text{intra}, t}]$$
The attention module injects **zero initial perturbation** into the network. The network's initial loss is $1.4587$, virtually identical to baseline Model A's $1.4534$, eliminating early training instability.

---

## 7. Empirical Unit Test Verification

The unit test suite `.agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py` was executed using `./.venv/bin/python`:

```
Test 01 PASSED: Parameter count = 193,156 (3,572 headroom under 196,728 cap).
.Test 02 PASSED: Exact step-0 zero-identity attention initialization confirmed.
.Test 03 PASSED: Attention padding key-masking strictly eliminates padded token weights.
.Test 04 PASSED: Masked TIN achieves exact zero-mean and unit-variance nocturnal normalization.
.Test 05 PASSED: Full BIDSleep forward shapes and dual (tensor/dict) interface verified.
.Test 06 PASSED: Non-zero, non-NaN backward gradient flow confirmed across all submodules.
.Test 07 PASSED: Variable sequence length packed execution confirmed.
.Test 08 PASSED: Direct gradient highway to raw input tensors confirmed.
.
----------------------------------------------------------------------
Ran 8 tests in 1.285s

OK
```

All 8 tests passed with 100% success in 1.285 seconds on the local Python virtualenv.

---

## 8. Implementation Roadmap for Worker Agent (Proxy Evaluation)

To evaluate Candidate Gamma on Fold 1 fast proxy evaluation:
1. **Target Evaluation Script**: Create `experiments/EXP-014/evaluate_nas_iteration5_proxy.py` modeled after `evaluate_nas_iteration4_proxy.py`.
2. **Model Import**:
   ```python
   from experiments.EXP_014.nas_iteration5_model import build_nas_iteration5_model
   model = build_nas_iteration5_model(concat_context=True, zero_init=True).to(device)
   ```
3. **Training Protocol**:
   - 50 epochs, Adam optimizer, `lr = 1e-4`, batch size 4 on `cuda:0`.
   - Loss function: Cross-Entropy + 0.01 RCC Loss (identical to Baseline Model A and Candidates Alpha/Beta).
   - Epoch-level validation tracking on inner-validation partition (5 subjects, 24 nights, 14,089 windows).
   - Checkpoint protocol: Save model at peak inner-val Macro-F1 epoch and evaluate on the 10 locked outer-test subjects (32,098 windows).
4. **Proxy Gating Thresholds**:
   - Proxy Accuracy $\ge 70.0\%$
   - Proxy Weighted-F1 $\ge 0.6900$
   If thresholds are met, immediately proceed to dual-GPU 15-run 5-fold cross-validation (`experiments/EXP-014/evaluate_nas_iteration5_5fold.py`).
