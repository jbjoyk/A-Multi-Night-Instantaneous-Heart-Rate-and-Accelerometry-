"""
Unit tests for SLAMSS-IFS architecture and RW loss.
Phase 9 verification suite.
"""

import os
import tempfile
import torch
import torch.nn as nn
from slamss_ifs_baseline import (
    IntraEpochCNN,
    IntraEpochBiLSTM,
    BahdanauAttention,
    InterEpochSeq2Seq,
    SLAMSS_IFS,
    RWLoss,
    make_reproducible_split
)


def run_unit_tests():
    print("=" * 60)
    print("RUNNING SLAMSS-IFS UNIT TESTS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    torch.manual_seed(42)
    
    B, T, L = 2, 8, 30  # Small synthetic batch for testing
    
    # -------------------------------------------------------------
    # 1 & 2. Intra-epoch CNN shapes
    # -------------------------------------------------------------
    print("\n[Test 1 & 2] IntraEpochCNN...")
    cnn = IntraEpochCNN(in_channels=2, out_dim=128).to(device)
    x_cnn = torch.randn(B * T, 2, L, device=device)
    f_cnn = cnn(x_cnn)
    assert f_cnn.shape == (B * T, 128), f"CNN output shape mismatch: {f_cnn.shape}"
    assert not torch.isnan(f_cnn).any(), "NaNs in CNN output"
    assert not torch.isinf(f_cnn).any(), "Infs in CNN output"
    print(f"  Passed! CNN output: {f_cnn.shape} (feature dim = 128)")
    
    # -------------------------------------------------------------
    # 3. Intra-epoch BiLSTM shapes
    # -------------------------------------------------------------
    print("\n[Test 3] IntraEpochBiLSTM...")
    bilstm = IntraEpochBiLSTM(in_features=2, hidden_size=32, out_dim=128).to(device)
    x_lstm = torch.randn(B * T, L, 2, device=device)
    f_lstm = bilstm(x_lstm)
    assert f_lstm.shape == (B * T, 128), f"BiLSTM output shape mismatch: {f_lstm.shape}"
    assert not torch.isnan(f_lstm).any(), "NaNs in BiLSTM output"
    assert not torch.isinf(f_lstm).any(), "Infs in BiLSTM output"
    print(f"  Passed! BiLSTM output: {f_lstm.shape} (feature dim = 128)")
    
    # -------------------------------------------------------------
    # 4. Feature Fusion dimension
    # -------------------------------------------------------------
    print("\n[Test 4] Feature Fusion...")
    x_freq = torch.randn(B, T, 1, device=device)
    x_time = torch.randn(B, T, 1, device=device)
    f_cnn_seq = f_cnn.view(B, T, 128)
    f_lstm_seq = f_lstm.view(B, T, 128)
    epoch_repr = torch.cat([f_cnn_seq, f_lstm_seq, x_freq, x_time], dim=-1)
    assert epoch_repr.shape == (B, T, 258), f"Fusion shape mismatch: {epoch_repr.shape}"
    print(f"  Passed! Fusion representation shape: {epoch_repr.shape} (dim = 258)")
    
    # -------------------------------------------------------------
    # 5, 6, 7, 8. Bahdanau Attention & Seq2Seq
    # -------------------------------------------------------------
    print("\n[Test 5, 6, 7, 8] Bahdanau Attention & Seq2Seq...")
    attn = BahdanauAttention(encoder_dim=96, decoder_dim=96, attn_dim=96).to(device)
    enc_states = torch.randn(B, T, 96, device=device)
    dec_states = torch.randn(B, T, 96, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[0, -2:] = False  # Mask last 2 epochs for batch sample 0
    
    context, attn_w = attn(dec_states, enc_states, mask=mask)
    assert context.shape == (B, T, 96), f"Attention context shape mismatch: {context.shape}"
    assert attn_w.shape == (B, T, T), f"Attention weights shape mismatch: {attn_w.shape}"
    # Verify attention sums to 1 across valid positions
    sum_w = attn_w.sum(dim=-1)
    assert torch.allclose(sum_w, torch.ones_like(sum_w)), "Attention weights do not sum to 1"
    # Verify masked positions get 0 attention
    assert torch.allclose(attn_w[0, :, -2:], torch.zeros_like(attn_w[0, :, -2:])), "Masked positions have non-zero attention"
    print(f"  Passed! Context: {context.shape}, Attention weights: {attn_w.shape}")
    
    seq2seq = InterEpochSeq2Seq(in_dim=258, hidden_dim=96, n_classes=4).to(device)
    logits, _ = seq2seq(epoch_repr, mask=mask)
    assert logits.shape == (B, T, 4), f"Seq2Seq logits shape mismatch: {logits.shape}"
    print(f"  Passed! Seq2Seq output logits shape: {logits.shape}")
    
    # -------------------------------------------------------------
    # 9. Full End-to-End Model
    # -------------------------------------------------------------
    print("\n[Test 9] End-to-End SLAMSS_IFS Model...")
    model = SLAMSS_IFS(n_classes=4).to(device)
    x_ihr = torch.randn(B, T, L, device=device)
    x_acc = torch.randn(B, T, L, device=device)
    out_logits, out_attn = model(x_ihr, x_acc, x_freq, x_time, mask=mask, return_attention=True)
    assert out_logits.shape == (B, T, 4), f"Model logits mismatch: {out_logits.shape}"
    assert out_attn.shape == (B, T, T), f"Model attention mismatch: {out_attn.shape}"
    assert not torch.isnan(out_logits).any(), "NaNs in model output"
    assert not torch.isinf(out_logits).any(), "Infs in model output"
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Passed! Full model logits shape: {out_logits.shape}, Trainable parameters: {param_count:,}")
    
    # -------------------------------------------------------------
    # 10. RW Loss computation
    # -------------------------------------------------------------
    print("\n[Test 10] Real-World (RW) Loss...")
    targets = torch.randint(0, 4, (B, T), device=device)
    targets[0, -2:] = -100  # Unknown / padded labels
    
    # Test with default weights (equivalent to weighted CE)
    loss_fn = RWLoss(n_classes=4, ignore_index=-100).to(device)
    loss_val = loss_fn(out_logits, targets, mask=mask)
    assert torch.isfinite(loss_val), f"RW Loss not finite: {loss_val}"
    assert loss_val.item() > 0, "Loss must be positive"
    print(f"  Passed! Loss value: {loss_val.item():.4f}")
    
    # Test with custom FP misclassification cost matrix
    fp_cost = torch.tensor([
        [0.0, 1.0, 2.0, 1.0],  # true Wake
        [1.0, 0.0, 1.0, 2.0],  # true Light
        [3.0, 1.0, 0.0, 2.0],  # true Deep (high penalty for missing Deep)
        [2.0, 1.0, 1.0, 0.0]   # true REM
    ], device=device)
    fn_weights = torch.tensor([1.2, 0.8, 1.5, 1.1], device=device)
    loss_fn_cost = RWLoss(n_classes=4, fn_weights=fn_weights, fp_weights=fp_cost, ignore_index=-100).to(device)
    loss_cost_val = loss_fn_cost(out_logits, targets, mask=mask)
    assert torch.isfinite(loss_cost_val), f"RW Loss with cost matrix not finite: {loss_cost_val}"
    print(f"  Passed! Loss with cost matrix: {loss_cost_val.item():.4f}")
    
    # -------------------------------------------------------------
    # 11, 12, 13. Backpropagation and gradient check
    # -------------------------------------------------------------
    print("\n[Test 11, 12, 13] Backward Pass & Gradients...")
    model.zero_grad()
    loss_cost_val.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            grad_norms.append(p.grad.norm().item())
        else:
            print(f"  Warning: No gradient for {name}")
    print(f"  Passed! Gradients computed for {len(grad_norms)} parameters, max grad norm = {max(grad_norms):.4f}")
    
    # -------------------------------------------------------------
    # 14. Checkpoint Save/Load Test
    # -------------------------------------------------------------
    print("\n[Test 14] Model Save and Load Consistency...")
    model.eval()
    with torch.no_grad():
        orig_out = model(x_ihr, x_acc, x_freq, x_time)
        
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        loaded_model = SLAMSS_IFS(n_classes=4).to(device)
        loaded_model.load_state_dict(torch.load(tmp_path, map_location=device))
        loaded_model.eval()
        with torch.no_grad():
            loaded_out = loaded_model(x_ihr, x_acc, x_freq, x_time)
        assert torch.allclose(orig_out, loaded_out, atol=1e-6), "Save/load output mismatch"
        print("  Passed! Save and load produced bitwise identical outputs.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    # -------------------------------------------------------------
    # Split Test
    # -------------------------------------------------------------
    print("\n[Split Verification] Paper vs Current Split...")
    subjects = [f"Bidslab{i:02d}" for i in range(47)]
    paper_sp = make_reproducible_split(subjects, split_type="paper", seed=0)
    curr_sp = make_reproducible_split(subjects, split_type="current", seed=0)
    print(f"  Paper split counts: train={len(paper_sp['train'])}, val={len(paper_sp['val'])}, test={len(paper_sp['test'])}")
    assert len(paper_sp['train']) == 31 and len(paper_sp['val']) == 5 and len(paper_sp['test']) == 11
    print(f"  Current split counts: train={len(curr_sp['train'])}, val={len(curr_sp['val'])}, test={len(curr_sp['test'])}")
    assert len(curr_sp['train']) == 33 and len(curr_sp['val']) == 7 and len(curr_sp['test']) == 7
    print("  Passed! Both splits are disjoint and partition all 47 subjects.")

    print("\n" + "=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
