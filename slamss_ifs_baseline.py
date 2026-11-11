"""
SLAMSS-IFS: Sequence-to-Sequence LSTM for Automated Mobile Sleep Staging with
Intra-epoch LSTM, Frequency, and Skip connections.

Paper reference:
    Song et al., "AI-Driven Sleep Staging Using Instantaneous Heart Rate and Accelerometry:
    Insights From an Apple Watch Study", IEEE Transactions on Biomedical Engineering (TBME),
    Vol. 73, No. 4, pp. 1596-1608, April 2026 (published online Sept 2025).
    DOI: 10.1109/TBME.2025.3612158

Architecture:
    1. Inputs (per 30-second epoch):
       - Instantaneous Heart Rate (IHR) at 1 Hz (30 samples)
       - Accelerometer magnitude (Accel = sqrt(ax^2 + ay^2 + az^2)) at 1 Hz (30 samples)
       - Sampling frequency (Freq) (1 value per epoch)
       - Circadian/time proxy (Time) (1 value per epoch)
       - Sequence length: up to T = 1200 epochs (~10 hours)
    2. Intra-epoch CNN subnetwork:
       - 5 Conv1d layers (128 channels, kernel=5, padding=2, stride=1)
       - LeakyReLU activations on layers 1-4, no activation on layer 5
       - MaxPool1d(6, 6) after Conv1 (reduces 30 -> 5)
       - MaxPool1d(5, 5) after Conv4 (reduces 5 -> 1)
       - Dropout(0.5)
       - Output: 128-dimensional localized feature vector per epoch
    3. Intra-epoch BiLSTM subnetwork:
       - 2-layer Bidirectional LSTM (hidden_size=32, bidirectional output=64)
       - Final sample extracted (t = 29)
       - Fully connected layer: 64 -> 128
       - Dropout(0.5)
       - Output: 128-dimensional temporal feature vector per epoch
    4. Feature Fusion:
       - Concatenation of CNN (128) + LSTM (128) + Freq (1) + Time (1) = 258 dims per epoch
    5. Inter-epoch Seq2Seq Model:
       - Encoder: 1-layer LSTM (hidden_size=96), Dropout(0.2)
       - Decoder: 1-layer LSTM (hidden_size=96), Dropout(0.2)
       - Bahdanau Additive Attention: alignment between decoder states and all encoder states
       - Skip Connection: 258-dim epoch representation bypassed directly to output stage
       - Classification Head: Linear(96 [decoder] + 96 [context] + 258 [skip] = 450 -> 4 classes)
    6. Loss Function:
       - Real-World (RW) Weighting Loss (Ho & Wookey 2020 / Song et al. 2023, 2026)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. INTRA-EPOCH CNN SUBNETWORK
# ==============================================================================

class IntraEpochCNN(nn.Module):
    """
    Intra-epoch CNN subnetwork capturing localized patterns within each 30-s epoch.
    Input: (B * T, 2, 30) - channels: [IHR, Accel_mag]
    Output: (B * T, 128)
    
    Paper specifications (Section III-E):
    - 5 Conv1d layers, each 128 channels, kernel_size=5, padding=2, stride=1.
    - LeakyReLU activation on Conv1-Conv4; linear (no activation) on Conv5.
    - MaxPool1d(kernel_size=6, stride=6) follows Conv1 (30 -> 5).
    - MaxPool1d(kernel_size=5, stride=5) follows Conv4 (5 -> 1).
    - 50% dropout at the end.
    """
    def __init__(self, in_channels: int = 2, out_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=6, stride=6)  # 30 -> 5
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)  # 5 -> 1
        self.conv5 = nn.Conv1d(128, out_dim, kernel_size=5, padding=2, stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2, 30)
        x = self.pool1(self.act(self.conv1(x)))  # (N, 128, 5)
        x = self.act(self.conv2(x))              # (N, 128, 5)
        x = self.act(self.conv3(x))              # (N, 128, 5)
        x = self.pool2(self.act(self.conv4(x)))  # (N, 128, 1)
        x = self.conv5(x)                        # (N, 128, 1) - no nonlinear activation on Conv5
        x = self.dropout(x)
        return x.squeeze(-1)                     # (N, 128)


# ==============================================================================
# 2. INTRA-EPOCH BiLSTM SUBNETWORK
# ==============================================================================

class IntraEpochBiLSTM(nn.Module):
    """
    Intra-epoch BiLSTM subnetwork modeling temporal dynamics within each 30-s epoch.
    Input: (B * T, 30, 2) - sequence length 30, 2 features [IHR, Accel_mag]
    Output: (B * T, 128)
    
    Paper specifications (Section III-E):
    - 2-layer Bidirectional LSTM, hidden_size=32 (bidirectional output dimension = 64).
    - Takes final sample (t = 29) of each epoch.
    - Fully connected layer: 64 -> 128.
    - 50% dropout layer.
    """
    def __init__(self, in_features: int = 2, hidden_size: int = 32, out_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, out_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 30, 2)
        out, _ = self.lstm(x)             # out: (N, 30, 64)
        final_sample = out[:, -1, :]      # (N, 64)
        feat = self.dropout(self.fc(final_sample))  # (N, 128)
        return feat


# ==============================================================================
# 3. BAHADANAU ADDITIVE ATTENTION
# ==============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism for sequence-to-sequence alignment.
    Paper reference [42]: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate", 2016.
    
    Formula:
        score(s_t, h_i) = v_a^T * tanh(W_s * s_t + W_h * h_i + b_a)
        alpha_{t, i} = softmax_i(score(s_t, h_i))
        context_t = sum_i(alpha_{t, i} * h_i)
    """
    def __init__(self, encoder_dim: int = 96, decoder_dim: int = 96, attn_dim: int = 96):
        super().__init__()
        self.w_enc = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(decoder_dim, attn_dim, bias=True)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_states: (B, T_dec, D_dec)
        encoder_states: (B, T_enc, D_enc)
        mask: (B, T_enc) bool tensor, True for valid positions, False for padded
        
        Returns:
            context: (B, T_dec, D_enc)
            attn_weights: (B, T_dec, T_enc)
        """
        # Linear projections
        # w_enc(encoder_states): (B, 1, T_enc, attn_dim)
        # w_dec(decoder_states): (B, T_dec, 1, attn_dim)
        q = self.w_dec(decoder_states).unsqueeze(2)  # (B, T_dec, 1, attn_dim)
        k = self.w_enc(encoder_states).unsqueeze(1)  # (B, 1, T_enc, attn_dim)
        
        # Additive alignment score
        scores = self.v_a(torch.tanh(q + k)).squeeze(-1)  # (B, T_dec, T_enc)
        
        if mask is not None:
            # Mask out invalid encoder positions (B, 1, T_enc)
            mask_expanded = mask.unsqueeze(1)
            fill_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask_expanded, fill_val)
            
        attn_weights = F.softmax(scores, dim=-1)  # (B, T_dec, T_enc)
        context = torch.bmm(attn_weights, encoder_states)  # (B, T_dec, D_enc)
        return context, attn_weights


# ==============================================================================
# 4. INTER-EPOCH SEQ2SEQ MODEL WITH SKIP CONNECTION
# ==============================================================================

class InterEpochSeq2Seq(nn.Module):
    """
    Inter-epoch sequence-to-sequence subnetwork with Bahdanau attention and decoder skip connection.
    
    Paper specifications (Section III-E):
    - Encoder: 1 LSTM layer, hidden_size=96, 20% dropout.
    - Decoder: 1 LSTM layer, hidden_size=96, 20% dropout.
    - Bahdanau attention calculates alignment scores between encoder and decoder hidden states.
    - Skip connection ("S"): directly exposes the low-dimensional epoch-wise representation (258 dims)
      to the decoder / output projection layer.
    - Final FC layer outputs sleep stage predictions (4 classes: Wake, Light, Deep, REM).
    """
    def __init__(
        self,
        in_dim: int = 258,
        hidden_dim: int = 96,
        n_classes: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.attention = BahdanauAttention(
            encoder_dim=hidden_dim,
            decoder_dim=hidden_dim,
            attn_dim=hidden_dim
        )
        self.dropout = nn.Dropout(p=dropout)
        # Skip connection combines:
        # Decoder state (hidden_dim) + Attention context (hidden_dim) + Low-dim epoch rep (in_dim)
        classifier_in = hidden_dim + hidden_dim + in_dim  # 96 + 96 + 258 = 450
        self.classifier = nn.Linear(classifier_in, n_classes)

    def forward(
        self,
        epoch_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        epoch_repr: (B, T, 258)
        mask: Optional (B, T) boolean mask of valid epochs
        
        Returns:
            logits: (B, T, n_classes)
            attn_weights: (B, T, T)
        """
        # Encoder pass
        enc_out, (h_n, c_n) = self.encoder(epoch_repr)  # (B, T, 96)
        enc_out = self.dropout(enc_out)
        
        # Decoder pass (initialized with encoder states)
        dec_out, _ = self.decoder(epoch_repr, (h_n, c_n))  # (B, T, 96)
        dec_out = self.dropout(dec_out)
        
        # Bahdanau Attention
        context, attn_weights = self.attention(dec_out, enc_out, mask=mask)  # context: (B, T, 96)
        
        # Skip Connection: concatenate decoder state, attention context, and input epoch representation
        combined = torch.cat([dec_out, context, epoch_repr], dim=-1)  # (B, T, 450)
        combined = self.dropout(combined)
        
        logits = self.classifier(combined)  # (B, T, n_classes)
        return logits, attn_weights


# ==============================================================================
# 5. FULL SLAMSS-IFS ARCHITECTURE
# ==============================================================================

class SLAMSS_IFS(nn.Module):
    """
    Complete SLAMSS-IFS Model (Song et al., IEEE TBME 2026).
    
    Accepts:
        x_ihr:   (B, T, 30) instantaneous heart rate at 1 Hz
        x_accel: (B, T, 30) acceleration magnitude at 1 Hz
        x_freq:  (B, T, 1)  epoch-wise IHR sampling frequency
        x_time:  (B, T, 1)  epoch-wise circadian/time proxy
        mask:    Optional (B, T) boolean tensor (True = valid epoch, False = pad)
        
    Returns:
        logits: (B, T, 4) unnormalized class scores for Wake, Light, Deep, REM
        attn_weights (optional): (B, T, T) attention alignment matrix
    """
    def __init__(
        self,
        n_classes: int = 4,
        cnn_out_dim: int = 128,
        lstm_out_dim: int = 128,
        seq_hidden_dim: int = 96,
        cnn_dropout: float = 0.5,
        lstm_dropout: float = 0.5,
        seq_dropout: float = 0.2
    ):
        super().__init__()
        self.n_classes = n_classes
        
        # Intra-epoch subnetworks
        self.cnn = IntraEpochCNN(in_channels=2, out_dim=cnn_out_dim, dropout=cnn_dropout)
        self.bilstm = IntraEpochBiLSTM(in_features=2, hidden_size=32, out_dim=lstm_out_dim, dropout=lstm_dropout)
        
        # Feature fusion dimension = CNN (128) + LSTM (128) + Freq (1) + Time (1) = 258
        self.fusion_dim = cnn_out_dim + lstm_out_dim + 1 + 1
        assert self.fusion_dim == 258, f"Expected 258, got {self.fusion_dim}"
        
        # Inter-epoch sequence model with Bahdanau attention and skip connection
        self.seq2seq = InterEpochSeq2Seq(
            in_dim=self.fusion_dim,
            hidden_dim=seq_hidden_dim,
            n_classes=n_classes,
            dropout=seq_dropout
        )

    def extract_epoch_features(
        self,
        x_ihr: torch.Tensor,
        x_accel: torch.Tensor,
        x_freq: torch.Tensor,
        x_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts the 258-dimensional epoch representations.
        Shapes:
            x_ihr:   (B, T, 30)
            x_accel: (B, T, 30)
            x_freq:  (B, T, 1)
            x_time:  (B, T, 1)
        Returns:
            (B, T, 258)
        """
        B, T, L = x_ihr.shape
        assert L == 30, f"Expected 30 samples per epoch, got {L}"
        
        # Combine IHR and Accel for CNN: (B * T, 2, 30)
        # Combine IHR and Accel for BiLSTM: (B * T, 30, 2)
        ihr_flat = x_ihr.reshape(B * T, 1, L)
        acc_flat = x_accel.reshape(B * T, 1, L)
        sig_cnn = torch.cat([ihr_flat, acc_flat], dim=1)        # (B * T, 2, 30)
        sig_lstm = sig_cnn.transpose(1, 2)                     # (B * T, 30, 2)
        
        # Subnetwork representations
        f_cnn = self.cnn(sig_cnn).view(B, T, -1)               # (B, T, 128)
        f_lstm = self.bilstm(sig_lstm).view(B, T, -1)           # (B, T, 128)
        
        # Feature concatenation: [CNN, LSTM, Freq, Time]
        epoch_repr = torch.cat([f_cnn, f_lstm, x_freq, x_time], dim=-1)  # (B, T, 258)
        return epoch_repr

    def forward(
        self,
        x_ihr: torch.Tensor,
        x_accel: torch.Tensor,
        x_freq: torch.Tensor,
        x_time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 1. Intra-epoch extraction & fusion
        epoch_repr = self.extract_epoch_features(x_ihr, x_accel, x_freq, x_time)
        
        # 2. Inter-epoch seq2seq with Bahdanau attention and skip connection
        logits, attn_weights = self.seq2seq(epoch_repr, mask=mask)
        
        if return_attention:
            return logits, attn_weights
        return logits


# ==============================================================================
# 6. REAL-WORLD (RW) WEIGHTING LOSS
# ==============================================================================

class RWLoss(nn.Module):
    """
    Real-World (RW) Weighting Cross-Entropy Loss.
    
    References:
        [47] Yaoshiang Ho and Samuel Wookey, "The Real-World-Weight Cross-Entropy Loss
             Function: Modeling the Costs of Mislabeling", IEEE Access, vol. 8, pp. 4806-4813, 2020.
        [12] Tzu-An Song et al., "AI-Driven Sleep Staging From Actigraphy and Heart Rate",
             PLoS ONE, vol. 18, no. 5, e0285703, May 2023.
        Song et al., "AI-Driven Sleep Staging Using Instantaneous Heart Rate and Accelerometry:
             Insights From an Apple Watch Study", IEEE TBME, 2026.
             
    Mathematical formulation (Ho & Wookey 2020):
        Let y_i in {0, 1}^K be one-hot ground-truth, p_i in [0, 1]^K be predicted probabilities.
        w_{FN} in R^K is the false negative / class weight vector (penalizes omitting class c).
        w_{FP} in R^{K x K} (with zero diagonal) is the false positive penalty matrix
        (penalizes classifying an instance of true class c as false class k != c).
        
        L = - 1 / N_valid * sum_{i=1}^N [ sum_{c=1}^K y_{i,c} * w_{FN,c} * log(p_{i,c})
                                          + sum_{c=1}^K sum_{k!=c} y_{i,c} * w_{FP,c,k} * log(1 - p_{i,k}) ]
                                          
    Paper ambiguity note:
        The 2026 IEEE TBME paper cites [47] and [12], stating:
        "The loss function used in this study is real-world (RW) weighting loss [47],
        which modifies the traditional cross-entropy loss to penalize both the omission
        of positive instances and the misclassification of negatives as positives."
        However, the authors did not publish a specific 4x4 numerical cost matrix for w_FP.
        When w_FP = 0, this loss function exactly matches weighted cross-entropy with weights w_FN.
        Users can provide an explicit w_FP penalty matrix or default to balanced misclassification penalties.
    """
    def __init__(
        self,
        n_classes: int = 4,
        fn_weights: Optional[torch.Tensor] = None,
        fp_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        eps: float = 1e-7
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.eps = eps
        
        # Default fn_weights to ones if not specified
        if fn_weights is None:
            self.register_buffer("fn_weights", torch.ones(n_classes, dtype=torch.float32))
        else:
            self.register_buffer("fn_weights", fn_weights.float().clone())
            
        # Default fp_weights to zeros if not specified (standard weighted CE),
        # or explicit cost matrix if provided
        if fp_weights is None:
            self.register_buffer("fp_weights", torch.zeros((n_classes, n_classes), dtype=torch.float32))
        else:
            assert fp_weights.shape == (n_classes, n_classes)
            # Enforce zero diagonal (no false positive penalty on correct class)
            fp_w = fp_weights.float().clone()
            fp_w.fill_diagonal_(0.0)
            self.register_buffer("fp_weights", fp_w)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        logits: (B, T, K) unnormalized predictions
        targets: (B, T) integer class labels in {0, ..., K-1} or ignore_index (-100)
        mask: Optional (B, T) boolean mask of valid entries
        """
        B, T, K = logits.shape
        flat_logits = logits.reshape(-1, K)
        flat_targets = targets.reshape(-1)
        
        # Valid positions filter
        valid = (flat_targets >= 0) & (flat_targets < K) & (flat_targets != self.ignore_index)
        if mask is not None:
            valid = valid & mask.reshape(-1)
            
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        valid_logits = flat_logits[valid]
        valid_targets = flat_targets[valid]
        
        # Softmax probabilities
        probs = F.softmax(valid_logits, dim=-1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        
        # One-hot encoding of targets
        one_hot = F.one_hot(valid_targets, num_classes=K).float()  # (M, K)
        
        # Term 1: False negative term: y_{i, c} * w_{FN, c} * log(p_{i, c})
        # fn_weights: (K,)
        fn_term = (one_hot * self.fn_weights.unsqueeze(0)) * torch.log(probs)  # (M, K)
        loss_fn = - torch.sum(fn_term, dim=-1)  # (M,)
        
        # Term 2: False positive term: sum_{k != c} y_{i, c} * w_{FP, c, k} * log(1 - p_{i, k})
        # If fp_weights has non-zero entries:
        if self.fp_weights.abs().sum() > 0:
            # effective_fp_weights for each sample: one_hot @ fp_weights -> (M, K)
            sample_fp_weights = torch.matmul(one_hot, self.fp_weights)  # (M, K)
            fp_term = sample_fp_weights * torch.log(1.0 - probs)        # (M, K)
            loss_fp = - torch.sum(fp_term, dim=-1)                      # (M,)
            total_loss = loss_fn + loss_fp
        else:
            total_loss = loss_fn
            
        return total_loss.mean()


# ==============================================================================
# 7. SPLIT CONFIGURATIONS (PAPER 31/5/11 vs CURRENT 33/7/7)
# ==============================================================================

def make_reproducible_split(
    subjects: List[str],
    split_type: str = "paper",
    seed: int = 0
) -> Dict[str, List[str]]:
    """
    Partitions subjects into disjoint train/val/test splits.
    
    Split types:
      - 'paper':   31 train / 5 val / 11 test (Song et al., IEEE TBME 2026, Section III-I)
      - 'current': 33 train / 7 val / 7 test (70/15/15 ratio used in existing notebook)
    """
    subs = sorted(list(set(subjects)))
    n = len(subs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shuffled = [subs[i] for i in perm]
    
    if split_type == "paper":
        assert n == 47, f"Paper split expects 47 subjects, got {n}"
        n_train, n_val, n_test = 31, 5, 11
        train = sorted(shuffled[:n_train])
        val = sorted(shuffled[n_train:n_train + n_val])
        test = sorted(shuffled[n_train + n_val:])
    elif split_type == "current":
        n_val = int(round(0.15 * n))
        n_test = int(round(0.15 * n))
        n_train = n - n_val - n_test
        test = sorted(shuffled[:n_test])
        val = sorted(shuffled[n_test:n_test + n_val])
        train = sorted(shuffled[n_test + n_val:])
    else:
        raise ValueError(f"Unknown split_type: {split_type}; use 'paper' or 'current'")
        
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert not (s_train & s_val), "Train and Val overlap!"
    assert not (s_train & s_test), "Train and Test overlap!"
    assert not (s_val & s_test), "Val and Test overlap!"
    assert s_train | s_val | s_test == set(subs), "Partitions do not cover all subjects!"
    
    return dict(train=train, val=val, test=test)
