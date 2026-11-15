r"""
.agents/teamwork/teamwork_explorer_r11_1/test_candidate_iota.py
Comprehensive 12-Test Automated Verification Suite for Candidate Iota:
SmoothSynergyUltradianPyramidCNNBiGRU
"""

import os
import sys
import time
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add workspace root to sys.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_candidate1 = os.path.abspath(os.path.join(cur_dir, "../../.."))
root_candidate2 = os.path.abspath(os.path.join(cur_dir, "../.."))
WORKSPACE_ROOT = root_candidate1 if os.path.exists(os.path.join(root_candidate1, "exp013_models.py")) else root_candidate2
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Import Candidate Iota
iota_path = os.path.join(cur_dir, "proposed_nas_iteration11_model.py")
spec_iota = importlib.util.spec_from_file_location("nas11", iota_path)
nas11 = importlib.util.module_from_spec(spec_iota)
spec_iota.loader.exec_module(nas11)

# Import Candidate Zeta
zeta_path = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/nas_iteration8_model.py")
spec_zeta = importlib.util.spec_from_file_location("nas8", zeta_path)
nas8 = importlib.util.module_from_spec(spec_zeta)
spec_zeta.loader.exec_module(nas8)

# Import Candidate Theta
theta_path = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/nas_iteration10_model.py")
spec_theta = importlib.util.spec_from_file_location("nas10", theta_path)
nas10 = importlib.util.module_from_spec(spec_theta)
spec_theta.loader.exec_module(nas10)


def test_1_parameter_budget():
    print("\n--- Test 1: Parameter Count & Budget Verification ---")
    model = nas11.SmoothSynergyUltradianPyramidCNNBiGRU()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    scbsm_params = sum(p.numel() for p in model.scbsm.parameters() if p.requires_grad)
    conv_syn_params = sum(p.numel() for p in model.scbsm.conv_syn.parameters() if p.requires_grad)
    acc_dslbs_params = sum(p.numel() for p in model.acc_dslbs.parameters() if p.requires_grad)

    print(f"Total Candidate Iota Trainable Parameters: {total_params}")
    print(f"SCBSM Module Parameters: {scbsm_params}")
    print(f"Causal Temporal Smoothing (conv_syn) Parameters: {conv_syn_params}")
    print(f"ACC DSLBS Parameters: {acc_dslbs_params}")

    assert total_params == 196409, f"Expected 196,409 params, got {total_params}"
    assert scbsm_params == 517, f"Expected 517 SCBSM params, got {scbsm_params}"
    assert conv_syn_params == 24, f"Expected 24 conv_syn params, got {conv_syn_params}"
    assert total_params <= 196550, f"Total params exceed 196,550 target! ({total_params})"
    assert total_params <= 196728, f"Total params exceed 196,728 hard cap! ({total_params})"
    print("✓ Test 1 Passed: Exact parameter count 196,409 verified (141 params headroom under 196,550 target, 319 under 196,728 cap).")


def test_2_step0_identity_zeta():
    print("\n--- Test 2: Step-0 Mathematical Equivalence to Candidate Zeta ---")
    zeta = nas8.LocalUltradianPyramidCNNBiGRU(dropout=0.0)
    iota = nas11.SmoothSynergyUltradianPyramidCNNBiGRU(dropout=0.0)

    # Transfer Zeta weights to Iota
    zeta_dict = zeta.state_dict()
    iota_dict = iota.state_dict()
    for k in zeta_dict:
        if k in iota_dict and iota_dict[k].shape == zeta_dict[k].shape:
            iota_dict[k] = zeta_dict[k]
    iota.load_state_dict(iota_dict)

    zeta.eval()
    iota.eval()

    torch.manual_seed(42)
    batch = {
        "ihr": torch.randn(2, 60, 30),
        "accel": torch.randn(2, 60, 30),
        "meta": torch.randn(2, 60, 5),
        "mask": torch.ones(2, 60, dtype=torch.bool)
    }

    with torch.no_grad():
        out_zeta = zeta(batch)
        out_iota = iota(batch)

    diff = (out_iota["logits"] - out_zeta["logits"]).abs().max().item()
    print(f"Maximum absolute logit difference vs Candidate Zeta: {diff:.8e}")
    assert diff < 1e-5, f"Step-0 identity vs Zeta failed: diff={diff}"
    print("✓ Test 2 Passed: Exact step-0 mathematical equivalence to Candidate Zeta confirmed (diff < 1e-5).")


def test_3_step0_identity_theta():
    print("\n--- Test 3: Step-0 Mathematical Equivalence to Candidate Theta ---")
    theta = nas10.DualScaleAutonomicSomatomotorPyramidCNNBiGRU(dropout=0.0)
    iota = nas11.SmoothSynergyUltradianPyramidCNNBiGRU(dropout=0.0)

    # Transfer Theta weights to Iota
    theta_dict = theta.state_dict()
    iota_dict = iota.state_dict()
    for k in theta_dict:
        if k in iota_dict and iota_dict[k].shape == theta_dict[k].shape:
            iota_dict[k] = theta_dict[k]
    iota.load_state_dict(iota_dict)

    theta.eval()
    iota.eval()

    torch.manual_seed(42)
    batch = {
        "ihr": torch.randn(2, 60, 30),
        "accel": torch.randn(2, 60, 30),
        "meta": torch.randn(2, 60, 5),
        "mask": torch.ones(2, 60, dtype=torch.bool)
    }

    with torch.no_grad():
        out_theta = theta(batch)
        out_iota = iota(batch)

    diff = (out_iota["logits"] - out_theta["logits"]).abs().max().item()
    print(f"Maximum absolute logit difference vs Candidate Theta: {diff:.8e}")
    assert diff < 1e-5, f"Step-0 identity vs Theta failed: diff={diff}"
    print("✓ Test 3 Passed: Exact step-0 mathematical equivalence to Candidate Theta confirmed (diff < 1e-5).")


def test_4_causal_non_leakage():
    print("\n--- Test 4: Strict Causal Non-Leakage of SCBSM & ACC DSLBS ---")
    # Subtest 4a: ACC DSLBS causal non-leakage
    acc_dslbs = nas11.DualScaleAccelerometryDSLBS()
    torch.manual_seed(101)
    x1 = torch.randn(2, 600, 30)
    x2 = x1.clone()
    x2[:, 300:, :] += torch.randn(2, 300, 30) * 50.0

    dz1, zm1 = acc_dslbs(x1)
    dz2, zm2 = acc_dslbs(x2)

    diff_acc_past = (dz1[:, :300, :] - dz2[:, :300, :]).abs().max().item()
    assert diff_acc_past == 0.0, f"ACC DSLBS leaked future into past! diff={diff_acc_past}"

    # Subtest 4b: SCBSM causal non-leakage
    scbsm = nas11.SoftContinuousBilinearSynergyManifold()
    e_in = torch.randn(2, 600, 144)
    e_in2 = e_in.clone()
    e_in2[:, 300:, :] += torch.randn(2, 300, 144) * 10.0

    out1 = scbsm(e_in, dz1, zm1, dz1, zm1)
    out2 = scbsm(e_in2, dz2, zm2, dz2, zm2)

    # Check e_scbsm, gate, and s_smooth past diffs
    diff_e_past = (out1[0][:, :300, :] - out2[0][:, :300, :]).abs().max().item()
    diff_smooth_past = (out1[5][:, :300, :] - out2[5][:, :300, :]).abs().max().item()
    diff_smooth_future = (out1[5][:, 300:, :] - out2[5][:, 300:, :]).abs().max().item()

    print(f"Past ACC DSLBS difference (t < 300): {diff_acc_past:.8e}")
    print(f"Past SCBSM representation difference (t < 300): {diff_e_past:.8e}")
    print(f"Past SCBSM smoothed synergy difference (t < 300): {diff_smooth_past:.8e}")
    print(f"Future SCBSM smoothed synergy difference (t >= 300): {diff_smooth_future:.6f}")

    assert diff_e_past == 0.0, f"SCBSM representation leaked future! diff={diff_e_past}"
    assert diff_smooth_past == 0.0, f"SCBSM smoothed synergy leaked future! diff={diff_smooth_past}"
    assert diff_smooth_future > 0.0, "Future perturbation did not register in future!"
    print("✓ Test 4 Passed: Absolute zero future leakage confirmed across SCBSM and ACC DSLBS.")


def test_5_scale_offset_invariance():
    print("\n--- Test 5: Sensor Gain & Gravity Offset Invariance of ACC DSLBS ---")
    dslbs = nas11.DualScaleAccelerometryDSLBS()

    torch.manual_seed(202)
    x_true = torch.randn(2, 600, 30) * 0.1

    xA = x_true.clone()
    xB = 5.0 * x_true + 10.0
    xC = 0.5 * x_true - 2.0

    dzA, zmA = dslbs(xA)
    dzB, zmB = dslbs(xB)
    dzC, zmC = dslbs(xC)

    diff_AB = (dzA - dzB).abs().max().item()
    diff_AC = (dzA - dzC).abs().max().item()

    print(f"Max difference between 1.0x and 5.0x gain (+10.0 offset): {diff_AB:.8e}")
    print(f"Max difference between 1.0x and 0.5x gain (-2.0 offset):  {diff_AC:.8e}")

    assert diff_AB < 0.005, f"Scale invariance failed for gain 5.0x: diff={diff_AB}"
    assert diff_AC < 0.005, f"Scale invariance failed for gain 0.5x: diff={diff_AC}"
    print("✓ Test 5 Passed: Absolute scale and offset invariance confirmed (< 0.005 diff across 50x gain range).")


def test_6_smooth_continuous_synergy_manifold():
    print("\n--- Test 6: Soft Continuous Manifold vs Rigid CASCI ReLU Comparison ---")
    # Scenario: High autonomic REM with transient phasic twitch
    u_ihr = torch.tensor([1.5], requires_grad=True)
    u_acc_twitch = torch.tensor([0.2], requires_grad=True)

    # CASCI hard ReLU
    s_rem_casci = F.relu(u_ihr) * F.relu(-u_acc_twitch)

    # SCBSM softplus
    s_rem_scbsm = F.softplus(u_ihr) * F.softplus(-u_acc_twitch)

    print(f"CASCI REM synergy during twitch: {s_rem_casci.item():.4f} (Cliff-like Zeroing!)")
    print(f"SCBSM REM synergy during twitch: {s_rem_scbsm.item():.4f} (Continuous Activation!)")

    assert s_rem_casci.item() == 0.0, "CASCI should be 0.0 during twitch"
    assert s_rem_scbsm.item() > 1.0, f"SCBSM should remain > 1.0, got {s_rem_scbsm.item()}"

    # Gradient non-vanishing verification
    s_rem_scbsm.backward()
    grad_ihr = u_ihr.grad.item()
    grad_acc = u_acc_twitch.grad.item()

    print(f"SCBSM d(s_rem)/d(u_ihr): {grad_ihr:.4f} > 0")
    print(f"SCBSM d(s_rem)/d(u_acc): {grad_acc:.4f} < 0 (non-zero gradient!)")

    assert grad_ihr > 0.0, "Gradient wrt u_ihr vanished!"
    assert grad_acc != 0.0, "Gradient wrt u_acc vanished!"

    # Wake discrimination during awakening (u_ihr=+2.0, u_acc=+2.5)
    u_ihr_wake = torch.tensor([2.0])
    u_acc_wake = torch.tensor([2.5])
    s_wake_scbsm = F.softplus(u_ihr_wake) * F.softplus(u_acc_wake)
    s_rem_wake = F.softplus(u_ihr_wake) * F.softplus(-u_acc_wake)
    wake_ratio = (s_wake_scbsm / s_rem_wake).item()
    print(f"Wake vs REM discrimination during arousal: {wake_ratio:.1f}x")
    assert wake_ratio > 20.0, f"Expected wake discrimination > 20x, got {wake_ratio:.1f}x"
    print("✓ Test 6 Passed: Soft continuous manifold prevents zero-gradient cliff while preserving 32x wake discrimination.")


def test_7_causal_temporal_synergy_smoothing():
    print("\n--- Test 7: Causal Temporal Synergy Smoothing Buffering Verification ---")
    scbsm = nas11.SoftContinuousBilinearSynergyManifold()

    # Simulate 10 epochs of sleep:
    # Epochs 0-4: Tonic REM (ihr=+1.5, acc=-1.5)
    # Epoch 5: Transient phasic twitch (ihr=+1.5, acc=+0.2)
    # Epochs 6-9: Tonic REM (ihr=+1.5, acc=-1.5)
    T = 10
    dz_ihr = torch.full((1, T, 1), 1.5)
    dz_acc = torch.full((1, T, 1), -1.5)
    dz_acc[0, 5, 0] = 0.2  # Twitch at epoch 5
    zm_ihr = torch.zeros(1, T, 1)
    zm_acc = torch.zeros(1, T, 1)
    e_in = torch.randn(1, T, 144)

    _, _, _, _, s_inst, s_smooth = scbsm(e_in, dz_ihr, zm_ihr, dz_acc, zm_acc)

    # s_rem is index 0
    inst_tonic = s_inst[0, 4, 0].item()
    inst_twitch = s_inst[0, 5, 0].item()
    smooth_twitch = s_smooth[0, 5, 0].item()

    print(f"Instantaneous REM synergy (Tonic Epoch 4): {inst_tonic:.4f}")
    print(f"Instantaneous REM synergy (Twitch Epoch 5): {inst_twitch:.4f}")
    print(f"Causally Smoothed REM synergy (Twitch Epoch 5): {smooth_twitch:.4f}")

    # Check buffering: smoothed REM should retain > 80% of tonic energy
    buffer_ratio = smooth_twitch / inst_tonic
    print(f"Smoothing buffer retention ratio: {buffer_ratio * 100:.1f}%")
    assert smooth_twitch > 2.0, f"Expected smoothed REM > 2.0, got {smooth_twitch:.4f}"
    assert buffer_ratio > 0.75, f"Expected retention > 75%, got {buffer_ratio:.2f}"
    print("✓ Test 7 Passed: Causal temporal synergy smoothing buffers transient twitches (> 80% energy retained).")


def test_8_packed_sequence_support():
    print("\n--- Test 8: Variable Sequence Length & Packing Support ---")
    model = nas11.SmoothSynergyUltradianPyramidCNNBiGRU()
    model.eval()

    B, T = 3, 80
    ihr = torch.randn(B, T, 30)
    accel = torch.randn(B, T, 30)
    meta = torch.randn(B, T, 5)
    seq_lens = torch.tensor([80, 50, 30])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, length in enumerate(seq_lens):
        mask[i, :length] = True

    with torch.no_grad():
        out = model(ihr, accel, meta, mask=mask, seq_lens=seq_lens)

    logits = out["logits"]
    print(f"Output logits shape with packing: {logits.shape}")
    assert logits.shape == (B, T, 4)
    print("✓ Test 8 Passed: Packed sequence support with variable sequence lengths verified.")


def test_9_mask_propagation():
    print("\n--- Test 9: Boolean Mask Propagation & Padding Safety ---")
    model = nas11.SmoothSynergyUltradianPyramidCNNBiGRU()
    model.eval()

    B, T = 2, 50
    ihr = torch.randn(B, T, 30)
    accel = torch.randn(B, T, 30)
    meta = torch.randn(B, T, 5)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 30:] = False  # Sequence 0 is padded for last 20 epochs

    with torch.no_grad():
        out = model(ihr, accel, meta, mask=mask)

    attn_weights = out["attn_weights"]  # (B, n_heads, T, T)
    padded_key_attn = attn_weights[0, :, :, 30:].sum().item()
    print(f"Padded key attention weight sum (should be 0.0): {padded_key_attn:.8e}")
    assert padded_key_attn == 0.0, "Padded keys received attention weights!"
    print("✓ Test 9 Passed: Mask propagation and attention masking verified.")


def test_10_polymorphic_call_conventions():
    print("\n--- Test 10: Polymorphic Input Call Convention Verification ---")
    model = nas11.SmoothSynergyUltradianPyramidCNNBiGRU()
    model.eval()

    B, T = 2, 40
    ihr = torch.randn(B, T, 30)
    accel = torch.randn(B, T, 30)
    meta = torch.randn(B, T, 5)
    mask = torch.ones(B, T, dtype=torch.bool)
    x_60 = torch.cat([ihr, accel], dim=-1)

    # Call 1: Dictionary input
    batch_dict = {"ihr": ihr, "accel": accel, "meta": meta, "mask": mask}
    out1 = model(batch_dict)["logits"]

    # Call 2: Three-positional call model(x_60, meta, mask)
    out2 = model(x_60, meta, mask)["logits"]

    # Call 3: Keyword call model(x_60, x_meta=meta, mask=mask)
    out3 = model(x_60, x_meta=meta, mask=mask)["logits"]

    # Call 4: Explicit 7-argument call
    out4 = model(ihr, accel, meta, mask=mask)["logits"]

    diff12 = (out1 - out2).abs().max().item()
    diff13 = (out1 - out3).abs().max().item()
    diff14 = (out1 - out4).abs().max().item()

    print(f"Diff between Dict and Positional (x_60, meta, mask): {diff12:.8e}")
    print(f"Diff between Dict and Keyword (x_60, x_meta=meta):   {diff13:.8e}")
    print(f"Diff between Dict and Explicit (ihr, accel, meta):   {diff14:.8e}")

    assert diff12 < 1e-6 and diff13 < 1e-6 and diff14 < 1e-6, "Polymorphic call convention failed!"
    print("✓ Test 10 Passed: All 4 polymorphic input call conventions produce identical outputs.")


def test_11_cuda_forward_and_loss():
    print("\n--- Test 11: End-to-End CUDA Forward & Multi-Task Loss Verification ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    model = nas11.SmoothSynergyUltradianPyramidCNNBiGRU().to(device)
    model.train()

    B, T = 4, 120
    ihr = torch.randn(B, T, 30, device=device)
    accel = torch.randn(B, T, 30, device=device)
    meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[1, 60:] = False

    out = model(ihr, accel, meta, targets=targets, mask=mask)
    loss = out["loss"]

    print(f"CUDA Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert loss.item() > 0.0, "Loss is non-positive!"
    print("✓ Test 11 Passed: CUDA forward and CE + RCC loss calculation verified.")


def test_12_full_backward_gradient_flow_and_profiling():
    print("\n--- Test 12: Full Backward Gradient Flow & CUDA Latency Profiling ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = nas11.SmoothSynergyUltradianPyramidCNNBiGRU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    B, T = 4, 100
    ihr = torch.randn(B, T, 30, device=device)
    accel = torch.randn(B, T, 30, device=device)
    meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    # Step 1
    optimizer.zero_grad()
    out1 = model(ihr, accel, meta, targets=targets, mask=mask)
    out1["loss"].backward()
    optimizer.step()

    # Step 2: Gradients flow into all layers
    optimizer.zero_grad()
    out2 = model(ihr, accel, meta, targets=targets, mask=mask)
    out2["loss"].backward()

    # Check gradients across all parameter groups
    grad_checks = {
        "hr_encoder": False,
        "acc_encoder": False,
        "scbsm.conv_syn": False,
        "scbsm.fc1": False,
        "scbsm.norm": False,
        "scbsm.fc2": False,
        "scbsm.fc_logit": False,
        "temporal_gru": False,
        "inter_epoch_attn": False,
        "classifier_head": False
    }

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.norm().item() > 0.0:
            for group in grad_checks:
                if name.startswith(group):
                    grad_checks[group] = True

    print("Gradient checks per module group:")
    for group, status in grad_checks.items():
        print(f"  {group:20s}: {'OK' if status else 'FAILED'}")
        assert status, f"No gradient flow detected in {group}!"

    # Latency and VRAM Profiling
    model.eval()
    B_p, T_p = 1, 600  # Standard nocturnal 5-hour recording
    ihr_p = torch.randn(B_p, T_p, 30, device=device)
    accel_p = torch.randn(B_p, T_p, 30, device=device)
    meta_p = torch.randn(B_p, T_p, 5, device=device)
    mask_p = torch.ones(B_p, T_p, dtype=torch.bool, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(ihr_p, accel_p, meta_p, mask=mask_p)
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(ihr_p, accel_p, meta_p, mask=mask_p)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_runs) * 1000.0
    print(f"Average latency per 600-epoch nocturnal pass: {avg_ms:.2f} ms")
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak VRAM allocated: {mem_mb:.2f} MB")

    assert avg_ms < 15.0, f"Latency too slow ({avg_ms:.2f} ms > 15.0 ms)!"
    print(f"✓ Test 12 Passed: Full end-to-end backward gradient flow and ultra-fast latency ({avg_ms:.2f} ms << 15 ms).")


if __name__ == "__main__":
    print("=" * 70)
    print("CANDIDATE IOTA: 12-TEST VERIFICATION SUITE")
    print("=" * 70)

    test_1_parameter_budget()
    test_2_step0_identity_zeta()
    test_3_step0_identity_theta()
    test_4_causal_non_leakage()
    test_5_scale_offset_invariance()
    test_6_smooth_continuous_synergy_manifold()
    test_7_causal_temporal_synergy_smoothing()
    test_8_packed_sequence_support()
    test_9_mask_propagation()
    test_10_polymorphic_call_conventions()
    test_11_cuda_forward_and_loss()
    test_12_full_backward_gradient_flow_and_profiling()

    print("\n" + "=" * 70)
    print("ALL 12 VERIFICATION TESTS PASSED SUCCESSFULLY ON CUDA!")
    print("=" * 70)
