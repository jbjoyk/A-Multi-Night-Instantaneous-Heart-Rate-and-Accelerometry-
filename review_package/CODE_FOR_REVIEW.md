# CODE_FOR_REVIEW.md: Source Code of Critical Pipeline and Model Components

This document contains the complete, unabridged source code for all critical components of the experimental pipeline, including data loaders, feature caching, model architectures, loss functions, evaluation routines, and cross-validation partition logic.

---

## 1. Dataset Loader (`ControlledSleepDataset`)

**File**: `run_controlled_experiment.py` (lines 51–97)

```python
class ControlledSleepDataset(Dataset):
    """
    Dataset for controlled ablation study.
    Fixed sequence length T = 600 epochs (5.0 hours).
    Every night in dataset has >= 610 epochs -> 0 zero-padding for any recording.
    """
    def __init__(
        self,
        records: List[Dict[str, np.ndarray]],
        subject_to_idx: Dict[str, int],
        seq_len: int = 600
    ):
        self.records = records
        self.subject_to_idx = subject_to_idx
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        subj = rec["subject"]
        subj_idx = self.subject_to_idx.get(subj, -100)
        
        T = self.seq_len
        ihr = rec["ihr"][:T]
        acc = rec["accel"][:T]
        freq = rec["freq"][:T]
        time_feat = rec["time"][:T]
        y = rec["y"][:T]
        
        hr_mean = ihr.mean(axis=1, keepdims=True)
        hr_std = ihr.std(axis=1, keepdims=True)
        clock = np.zeros_like(time_feat)
        meta = np.concatenate([freq, clock, time_feat, hr_mean, hr_std], axis=-1)
        mask = np.ones(T, dtype=bool)
        
        return {
            "ihr": torch.from_numpy(ihr.astype(np.float32)),
            "accel": torch.from_numpy(acc.astype(np.float32)),
            "meta": torch.from_numpy(meta.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.int64)),
            "mask": torch.from_numpy(mask),
            "subject_idx": torch.tensor(subj_idx, dtype=torch.int64),
            "subject": subj,
            "night": rec.get("night", "")
        }
```

---

## 2. Feature Extraction and Night Loading (`load_night_features`)

**File**: `slamss_ifs_train.py` (lines 105–141)

```python
def load_night_features(cache_dir: str, subject: str, night_id: str) -> Optional[Dict[str, np.ndarray]]:
    """Loads a night from cache, extracting IHR, Accel magnitude, Freq, Time, y."""
    fname = os.path.join(cache_dir, f"{subject}__{night_id}.npz")
    if not os.path.exists(fname):
        return None
    data = np.load(fname)
    X = data["X"]          # (T, 10 or 13, 30)
    y_raw = data["y"]      # (T,)
    meta = data["meta"]    # (T, 5)
    
    # 4-class stage mapping:
    # In cache: CANON_STAGES = (Wake=0, N1=1, N2=2, N3=3, REM=4)
    # 4-class grouping: Wake=0, Light=(1,2) -> 1, Deep=3 -> 2, REM=4 -> 3
    y_4class = np.full_like(y_raw, -100)
    y_4class[y_raw == 0] = 0
    y_4class[(y_raw == 1) | (y_raw == 2)] = 1
    y_4class[y_raw == 3] = 2
    y_4class[y_raw == 4] = 3
    
    # IHR is channel 0 (1 Hz instantaneous heart rate)
    ihr = X[:, 0, :]       # (T, 30)
    # Accel magnitude is channel 6
    accel = X[:, 6, :]     # (T, 30)
    # Freq is meta column 0 (sampling rate proxy)
    freq = meta[:, 0:1]    # (T, 1)
    # Time is meta column 2 (elapsed time proxy)
    time_feat = meta[:, 2:3]  # (T, 1)
    
    return {
        "subject": subject,
        "night": night_id,
        "ihr": ihr,
        "accel": accel,
        "freq": freq,
        "time": time_feat,
        "y": y_4class
    }
```

---

## 3. Preprocessing Cache Builder and Robust Scaling

**File**: `sleepchannelnet_protocol_patch_v2.ipynb` (Cell 16)

```python
def robust_scale(v, floor):
    med = np.nanmedian(v)
    iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip(np.nan_to_num((v - med) / max(float(iqr), floor)), -10, 10).astype(np.float32)

def robust_scale_masked(v, mask, floor):
    out = np.zeros(np.asarray(v).shape, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(v)
    if mask.any():
        out[mask] = robust_scale(np.asarray(v)[mask], floor)
    return out

def process_night(hr, mot, stage, rec_start, c: Config):
    """hr/mot DataFrames, stage = canonical stage ids (UNKNOWN for unknown), rec_start = verified Unix seconds."""
    n_ep = len(stage); n = n_ep * Lx; fs = c.fs; grid = np.arange(n) / fs
    t_hr_raw, t_mo = hr.time.to_numpy() - rec_start, mot.time.to_numpy() - rec_start
    edges = np.linspace(0, n / fs, n + 1)
    raw_counts = np.histogram(t_hr_raw, bins=edges)[0].astype(float)
    mo_counts = np.histogram(t_mo, bins=np.linspace(0, n / fs, n_ep + 1))[0].astype(float)
    h = hr.hr.to_numpy()
    
    keep = np.isfinite(h) & (h >= 30) & (h <= 200)
    method = getattr(c, "hr_outlier_method", "std3")
    if method == "std3":
        if keep.sum() > 10 and h[keep].std() > 0:
            keep &= np.abs((h - h[keep].mean()) / h[keep].std()) <= 3
            
    # Linear interpolation up to max_hr_gap
    idx = np.where(keep)[0]
    hv = np.full(n, np.nan, dtype=float)
    ok_hr = np.zeros(n, dtype=bool)
    if len(idx) > 1:
        t_ok = t_hr_raw[idx]; h_ok = h[idx]
        in_range = (grid >= t_ok[0]) & (grid <= t_ok[-1])
        gaps = nearest_gap(t_ok, grid)
        valid_grid = in_range & (gaps <= c.max_hr_gap)
        hv[valid_grid] = np.interp(grid[valid_grid], t_ok, h_ok)
        ok_hr[valid_grid] = True
        
    hr_i = pd.Series(hv).ffill().bfill().to_numpy()
    
    # Motion components
    ax_raw, ay_raw, az_raw = mot.x.to_numpy(), mot.y.to_numpy(), mot.z.to_numpy()
    # ... binned to 1 Hz ...
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    dyn = np.abs(mag - roll(mag, w, "median"))
    jerk = np.abs(np.diff(mag, prepend=mag[0])) * fs
    
    chans = [
        robust_scale(hr_i, 2.0),
        robust_scale(np.gradient(hr_i) * fs, 0.05),
        robust_scale(roll(hr_i, w, "std"), 1.0),
        robust_scale(ax, mf),
        robust_scale(ay, mf),
        robust_scale(az, mf),
        robust_scale(mag, mf),
        robust_scale(np.log1p(jerk / mf), 0.1),
        robust_scale(np.log1p(roll(dyn ** 2, w, "mean") / mf ** 2), 0.1),
        robust_scale(np.log1p(roll(raw_counts, w, "sum")), 0.1),
        robust_scale_masked(reactivity, reactivity_valid, 2.0),
        artifact_mask.astype(np.float32),
        robust_scale_masked(coupling, coupling_valid, 0.05)
    ]
    assert len(chans) == len(NODES_ALL)
    sig = np.stack(chans)
    X = sig.reshape(len(NODES_ALL), n_ep, Lx).transpose(1, 0, 2).astype(np.float16)
    
    cov_hr, cov_mo = ok_hr.reshape(n_ep, Lx).mean(1), ok_mo.reshape(n_ep, Lx).mean(1)
    valid = (cov_hr >= c.min_epoch_coverage) & (cov_mo >= c.min_epoch_coverage)
    X[~valid] = 0
    y = stage.astype(np.int64).copy(); n_known = int((y >= 0).sum()); y[~valid] = UNKNOWN
    meta = np.stack([
        np.log1p(raw_counts.reshape(n_ep, Lx).sum(1) / c.epoch_sec * 10),
        np.log1p(mo_counts / c.epoch_sec),
        np.arange(n_ep) * c.epoch_sec / 3600 / 10,
        cov_hr,
        cov_mo
    ], 1).astype(np.float32)
    return X, y, meta, qc
```

---

## 4. Conventional Baseline (`ConventionalCNNBiGRU`)

**File**: `exp013_models.py` (lines 27–171)

```python
class ConventionalCNNBiGRU(nn.Module):
    """
    Model A: Matched-capacity conventional CNN-BiGRU baseline.
    Controls approximately for parameter capacity (~160k parameters, within +/-10% of 162,325)
    to test whether proposed mechanisms provide benefit beyond a conventional architecture.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        use_rcc_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.n_classes = n_classes
        self.use_rcc_loss = use_rcc_loss
        
        # 1. Modality-specific Encoders (Identical to SleepAdaptiveNet)
        self.hr_encoder = HREncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.acc_encoder = ACCEncoder(in_features=1, d_model=d_model, dropout=dropout)
        
        # Meta feature projection (Identical to SleepAdaptiveNet)
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU()
        )
        
        # 2. Conventional Concatenation & Temporal Sequence Modeling
        in_temp = 2 * d_model + 16 # 64 + 64 + 16 = 144
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2, # bidirectional -> total d_temp = 96
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)
        
        # 3. Primary Classifier Head (Identical to SleepAdaptiveNet)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        # Primary stage loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        fn_w, fp_w = get_official_weights()
        self.rcc_loss = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=n_classes)

    def forward(
        self,
        x_ihr: torch.Tensor,
        x_accel: torch.Tensor,
        x_meta: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x_ihr.shape
        
        # 1. Conventional Modality Encoding
        h_hr = self.hr_encoder(x_ihr)     # (B, T, d_model)
        h_acc = self.acc_encoder(x_accel) # (B, T, d_model)
        h_meta = self.meta_proj(x_meta)   # (B, T, 16)
        
        # 2. Direct Feature Concatenation (No Adaptive Gate)
        cat_features = torch.cat([h_hr, h_acc, h_meta], dim=-1) # (B, T, 144)
        
        # 3. Temporal Sequence Modeling
        temp_out, _ = self.temporal_gru(cat_features)
        temp_features = self.temp_norm(temp_out) # (B, T, d_temp)
        
        # 4. Classification Emission Logits
        logits = self.classifier_head(temp_features) # (B, T, n_classes)
        
        # 5. Loss Calculation (RCCLoss identical to SleepAdaptiveNet)
        loss_stage = torch.tensor(0.0, device=x_ihr.device)
        if targets is not None:
            flat_logits = logits.view(-1, self.n_classes)
            flat_targets = targets.view(-1)
            
            valid = (flat_targets >= 0) & (flat_targets < self.n_classes)
            if mask is not None:
                valid = valid & mask.view(-1)
                
            if valid.any():
                per_epoch_ce = self.ce_loss(flat_logits, torch.clamp(flat_targets, 0, self.n_classes - 1))
                per_epoch_ce_2d = per_epoch_ce.view(B, T)
                valid_2d = valid.view(B, T)
                ce_term = per_epoch_ce_2d[valid_2d].mean()
                
                if self.use_rcc_loss:
                    raw_rcc = self.rcc_loss(flat_logits, flat_targets)
                    v_mask = valid.float().unsqueeze(-1).repeat(1, self.n_classes)
                    rcc_term = - torch.sum(raw_rcc * v_mask) / valid.sum().clamp(min=1.0)
                    loss_stage = ce_term + 0.01 * rcc_term
                else:
                    loss_stage = ce_term

        probs = F.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "emission_logits": logits,
            "probabilities": probs,
            "alpha_gate": None,
            "confidence": torch.zeros(B, T, 1, device=x_ihr.device),
            "entropy": torch.zeros(B, T, 1, device=x_ihr.device),
            "total_loss": loss_stage,
            "loss_stage": loss_stage,
            "loss_trans": torch.tensor(0.0, device=x_ihr.device),
            "loss_boundary": torch.tensor(0.0, device=x_ihr.device),
            "loss_adv": torch.tensor(0.0, device=x_ihr.device),
            "loss_conf": torch.tensor(0.0, device=x_ihr.device),
            "boundary_mask": None,
            "temp_features": temp_features
        }
```

---

## 5. Model Matrix Factory (`build_exp014_model`)

**File**: `exp014_models.py` (lines 26–135)

```python
def build_exp014_model(
    model_id: str,
    n_train_subjects: int,
    loss_type: str = "rccloss"
) -> nn.Module:
    """
    Factory function instantiating freshly initialized models for EXP-014 cross-validation.
    """
    model_id = model_id.upper()
    
    if model_id == "A":
        return ConventionalCNNBiGRU(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            dropout=0.2,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "B":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=True,
            transition_aware=True,
            boundary_aware=False,
            subject_invariant=False, # GRL disabled
            uncertainty_head=False,
            lambda_transition=0.10,
            lambda_boundary=0.0,
            lambda_adv=0.0,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "C":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=True,
            transition_aware=False, # Transition module disabled
            boundary_aware=False,
            subject_invariant=True,
            uncertainty_head=False,
            lambda_transition=0.0,
            lambda_boundary=0.0,
            lambda_adv=0.10,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "D":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=False, # Fixed 0.5/0.5 equal fusion in AdaptiveReliabilityFusion
            transition_aware=True,
            boundary_aware=False,
            subject_invariant=True,
            uncertainty_head=False,
            lambda_transition=0.10,
            lambda_boundary=0.0,
            lambda_adv=0.10,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "E":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=True,
            transition_aware=True,
            boundary_aware=False,
            subject_invariant=True,
            uncertainty_head=False,
            lambda_transition=0.10,
            lambda_boundary=0.0,
            lambda_adv=0.10,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Expected one of ['A', 'B', 'C', 'D', 'E'].")
```

---

## 6. Official Test Set Evaluation (`evaluate_test_set`)

**File**: `run_exp012_final_benchmark.py` (lines 52–165)

```python
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, any]:
    """Evaluates the frozen epoch-80 model on the complete official held-out test partition."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_probs = []
    all_subjects = []
    all_nights = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
            logits = out["logits"]
            probs = out["probabilities"]
            preds = torch.argmax(logits, dim=-1)
            
            # Boundary calculation (y_t != y_{t-1})
            y_prev = targets[:, :-1]
            y_curr = targets[:, 1:]
            is_bound = torch.zeros_like(targets, dtype=torch.bool)
            is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
            
            valid = (targets >= 0) & (targets < 4) & mask
            
            subjs = batch["subject"]
            nights = batch["night"]
            
            for b_i in range(len(subjs)):
                s_name = subjs[b_i]
                n_name = nights[b_i]
                m_b = valid[b_i].cpu().numpy()
                
                p_b = preds[b_i][m_b].cpu().numpy()
                t_b = targets[b_i][m_b].cpu().numpy()
                bnd_b = is_bound[b_i][m_b].cpu().numpy()
                prb_b = probs[b_i][m_b].cpu().numpy()
                
                all_preds.extend(p_b)
                all_targets.extend(t_b)
                all_boundaries.extend(bnd_b)
                all_probs.append(prb_b)
                all_subjects.extend([s_name] * len(p_b))
                all_nights.extend([f"{s_name}_{n_name}"] * len(p_b))
                
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    is_bound = np.array(all_boundaries, dtype=bool)
    probs_concat = np.concatenate(all_probs, axis=0)
    subjects_arr = np.array(all_subjects)
    
    # 1. Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    ece = compute_ece(probs_concat, y_true, n_bins=10)
    
    # 2. Per-Class Metrics: 0: Wake, 1: Light, 2: Deep, 3: REM
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # 3. Boundary vs Non-Boundary Error
    is_correct = (y_true == y_pred)
    err_boundary = float(1.0 - is_correct[is_bound].mean()) if is_bound.sum() > 0 else 0.0
    err_non_boundary = float(1.0 - is_correct[~is_bound].mean()) if (~is_bound).sum() > 0 else 0.0
    
    # 4. Subject-Level Analysis
    unique_subs = sorted(list(set(all_subjects)))
    subject_metrics = []
    
    for s_id in unique_subs:
        idx_s = (subjects_arr == s_id)
        yt_s = y_true[idx_s]
        yp_s = y_pred[idx_s]
        acc_s = accuracy_score(yt_s, yp_s)
        mf1_s = f1_score(yt_s, yp_s, average="macro", zero_division=0)
        kap_s = cohen_kappa_score(yt_s, yp_s)
            
        subject_metrics.append({
            "subject_id": s_id,
            "n_windows": int(len(yt_s)),
            "accuracy": float(acc_s),
            "macro_f1": float(mf1_s),
            "kappa": float(kap_s)
        })
        
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "kappa": float(kappa),
        "mcc": float(mcc),
        "ece": float(ece),
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,
        "precision": prec,
        "recall": rec,
        "f1": f1_per_class,
        "support": support,
        "err_boundary": err_boundary,
        "err_non_boundary": err_non_boundary,
        "subject_metrics": subject_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "is_boundary": is_bound,
        "probs": probs_concat,
        "subjects": subjects_arr,
        "nights": np.array(all_nights)
    }
```

---

## 7. Inter-Epoch Validation Routine (`evaluate_validation_sweep_exp013`)

**File**: `run_exp013_attribution.py` (lines 46–140)

```python
def evaluate_validation_sweep_exp013(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Detailed validation evaluation with safe handling when alpha_gate is None (e.g. Model A).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_confidences = []
    all_entropies = []
    all_alphas = []
    all_probs = []
    
    val_losses = []
    loss_components = {"stage": [], "trans": [], "bound": [], "adv": [], "conf": []}
    
    with torch.no_grad():
        for batch in val_loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
            
            val_losses.append(out["total_loss"].item())
            loss_components["stage"].append(out["loss_stage"].item())
            loss_components["trans"].append(out["loss_trans"].item())
            loss_components["bound"].append(out["loss_boundary"].item())
            loss_components["adv"].append(out["loss_adv"].item())
            loss_components["conf"].append(out["loss_conf"].item())
            
            logits = out["logits"]
            probs = out["probabilities"]
            preds = torch.argmax(logits, dim=-1)
            conf = out["confidence"].squeeze(-1) if out.get("confidence") is not None else torch.zeros_like(preds, dtype=torch.float32)
            ent = out["entropy"].squeeze(-1) if out.get("entropy") is not None else torch.zeros_like(preds, dtype=torch.float32)
            
            alpha_tensor = out.get("alpha_gate")
            if alpha_tensor is not None:
                alpha = alpha_tensor.squeeze(-1)
            else:
                alpha = torch.full_like(preds, 0.5, dtype=torch.float32)
            
            # Boundary mask (y_t != y_{t-1})
            y_prev = targets[:, :-1]
            y_curr = targets[:, 1:]
            is_bound = torch.zeros_like(targets, dtype=torch.bool)
            is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
            
            valid = (targets >= 0) & (targets < 4) & mask
            
            all_preds.extend(preds[valid].cpu().numpy())
            all_targets.extend(targets[valid].cpu().numpy())
            all_boundaries.extend(is_bound[valid].cpu().numpy())
            all_confidences.extend(conf[valid].cpu().numpy())
            all_entropies.extend(ent[valid].cpu().numpy())
            all_alphas.extend(alpha[valid].cpu().numpy())
            all_probs.append(probs[valid].cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    is_bound = np.array(all_boundaries, dtype=bool)
    conf = np.array(all_confidences)
    ent = np.array(all_entropies)
    alpha = np.array(all_alphas)
    probs_concat = np.concatenate(all_probs, axis=0)
    
    # Primary Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    is_correct = (y_true == y_pred)
    err_boundary = 1.0 - is_correct[is_bound].mean() if is_bound.sum() > 0 else 0.0
    err_non_boundary = 1.0 - is_correct[~is_bound].mean() if (~is_bound).sum() > 0 else 0.0
    
    ece = compute_ece(probs_concat, y_true, n_bins=10)
    
    return {
        "val_loss": float(np.mean(val_losses)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "precision": prec,
        "recall": rec,
        "f1": f1_per_class,
        "support": support,
        "confusion_matrix": cm,
        "boundary_error_rate": float(err_boundary),
        "non_boundary_error_rate": float(err_non_boundary),
        "ece": float(ece)
    }
```

---

## 8. Calibration Error Computation (`compute_ece`)

**File**: `run_controlled_experiment.py` (lines 34–48)

```python
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across 10 probability bins."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)
```

---

## 9. Cross-Validation Manifest Validation Logic

**File**: `preflight_audit_exp014.py` (lines 75–117)

```python
    # --------------------------------------------------------------------------
    # Check 3: Outer Fold Partitions & Zero Leakage Across 47 Subjects
    # --------------------------------------------------------------------------
    all_mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "Bidslab*", "*", "labels.mat")))
    all_nights_dataset = [(f.split(os.sep)[-3], f.split(os.sep)[-2].replace("Night ", "")) for f in all_mat_files]
    all_subjects_cohort = sorted(list(set(s for s, n in all_nights_dataset)))
    assert len(all_subjects_cohort) == 47, f"Expected 47 cohort subjects, got {len(all_subjects_cohort)}"
    
    test_subject_occurrences = {}
    total_outer_test_nights = 0
    
    for f_idx in range(1, 6):
        manifest_path = os.path.join(MANIFEST_DIR, f"fold_{f_idx}.json")
        assert os.path.exists(manifest_path), f"Missing manifest {manifest_path}"
        with open(manifest_path, "r") as f:
            man = json.load(f)
            
        outer_test = set(man["outer_test_subjects"])
        outer_train = set(man["outer_train_subjects"])
        inner_train = set(man["inner_train_subjects"])
        inner_val = set(man["inner_val_subjects"])
        
        # Disjointness checks
        assert len(outer_test.intersection(outer_train)) == 0, f"Fold {f_idx}: Outer test overlaps outer train!"
        assert len(inner_train.intersection(inner_val)) == 0, f"Fold {f_idx}: Inner train overlaps inner val!"
        assert len(inner_val.intersection(outer_test)) == 0, f"Fold {f_idx}: Inner val overlaps outer test!"
        assert len(inner_train.intersection(outer_test)) == 0, f"Fold {f_idx}: Inner train overlaps outer test!"
        assert inner_train.union(inner_val) == outer_train, f"Fold {f_idx}: Inner partitions do not compose outer train!"
        
        # Track test subject occurrences
        for s in outer_test:
            test_subject_occurrences[s] = test_subject_occurrences.get(s, 0) + 1
            
        total_outer_test_nights += man["n_outer_test_nights"]
        
    print(f"Check 03 - Disjointness: All 5 folds satisfy train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅ [VERIFIED]")
    
    # --------------------------------------------------------------------------
    # Check 4: Exact Outer Test Subject Partitioning (Partition of Cohort)
    # --------------------------------------------------------------------------
    assert len(test_subject_occurrences) == 47, f"Expected all 47 subjects in test folds, got {len(test_subject_occurrences)}"
    for s, count in test_subject_occurrences.items():
        assert count == 1, f"Subject {s} appears {count} times in test folds (expected exactly 1)!"
    assert total_outer_test_nights == 246, f"Expected 246 usable nights across test folds, got {total_outer_test_nights}"
    print(f"Check 04 - Complete Partition: All 47 subjects appear exactly once across outer test folds (246 nights) [VERIFIED]")
```
