import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Official RCCLoss implementation
class RCCLoss(nn.Module):
    def __init__(self, fn_weights=None, fp_weights=None, cls=4):
        super(RCCLoss, self).__init__()
        self.cls = cls
        self.full_fn_weights = torch.Tensor(fn_weights)
        self.full_fp_weights = torch.Tensor(fp_weights)
        
    def forward(self, x, y):
        x = F.softmax(x, dim=1)
        # Clamp to [0, cls-1] to prevent negative index assertions during scatter when padded with -100
        yi = torch.clamp(y, 0, self.cls - 1).unsqueeze(1).long()
        y1 = torch.zeros(x.size()[0], self.cls, device=x.device).scatter_(1, yi, 1)
        if y1.type() != x.data.type():
            y1 = y1.type_as(x.data)
        eps = 1e-5
        x = torch.clamp(x, eps, 1. - eps)
        logs = torch.log(x)
        logs_1_sub = torch.log(1. - x)
        
        if self.full_fn_weights.device != x.device or self.full_fn_weights.type() != x.data.type():
            self.full_fn_weights = self.full_fn_weights.to(x.device).type_as(x.data)
        if self.full_fp_weights.device != x.device or self.full_fp_weights.type() != x.data.type():
            self.full_fp_weights = self.full_fp_weights.to(x.device).type_as(x.data)
            
        m_full_fn_weights = torch.mm(y1, self.full_fn_weights)
        m_full_fp_weights = torch.mm(y1, self.full_fp_weights)
        loss = (0.25 * m_full_fn_weights * logs + 0.5 * m_full_fp_weights * logs_1_sub)
        return loss


def get_official_weights():
    w, l, d, r = 16433.0, 63668.0, 25685.0, 32015.0
    fn_weights = [
        [1.0 / 0.0878, 0.00, 0.00, 0.00],
        [0.00, 1.0 / 0.4636, 0.00, 0.00],
        [0.00, 0.00, 1.0 / 0.1940, 0.00],
        [0.00, 0.00, 0.00, 1.0 / 0.2548]
    ]
    fp_weights4 = [
        [0.00, (w / l)**0.5, (w / d)**0.5, (w / r)**0.5],
        [(l / w)**0.5 * 0.5, 0.00, (l / d)**0.5 * 0.5, (l / r)**0.5],
        [(d / w)**0.5, (d/l)**0.5 * 0.5, 0.00, (r / l)**0.5 * 0.5],
        [(r / w)**0.5 * 0.5, (r / l)**0.5, (r / d)**0.5, 0.00]
    ]
    return fn_weights, fp_weights4


def run_10_loss_tests():
    print("=" * 65)
    print("SECTION 7: UNIT TESTS FOR RCC LOSS (10 DETERMINISTIC TESTS)")
    print("=" * 65)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn_w, fp_w = get_official_weights()
    loss_rccl = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=4).to(device)
    loss_ce_none = nn.CrossEntropyLoss(reduction="none")
    
    results = []
    
    # Test 1: Perfect prediction (confidence -> 1.0 on true class)
    logits_perfect = torch.tensor([[20.0, -20.0, -20.0, -20.0]], device=device)
    y_perfect = torch.tensor([0], device=device)
    raw_loss = loss_rccl(logits_perfect, y_perfect)
    rwl = -torch.sum(raw_loss)
    ce = loss_ce_none(logits_perfect, y_perfect).item()
    t1_pass = rwl.item() < 0.05 and ce < 0.01
    results.append(("1. Perfect Prediction", t1_pass, f"RWL={rwl.item():.5f}, CE={ce:.5f}"))

    # Test 2: False Negative (Severe misclassification: Wake predicted as Light)
    logits_fn = torch.tensor([[-20.0, 20.0, -20.0, -20.0]], device=device)
    y_fn = torch.tensor([0], device=device) # true is wake (minority)
    raw_loss_fn = loss_rccl(logits_fn, y_fn)
    rwl_fn = -torch.sum(raw_loss_fn)
    t2_pass = rwl_fn.item() > 20.0 # heavily penalized by 1/0.0878
    results.append(("2. False Negative (Minority Wake)", t2_pass, f"RWL={rwl_fn.item():.4f}"))

    # Test 3: False Positive penalty matrix verification
    # When true is Wake (class 0), row 0 of fp_weights is [0, (w/l)^0.5, (w/d)^0.5, (w/r)^0.5]
    # (w/l)^0.5 = (16433/63668)^0.5 = 0.5080
    row0_fp = [float(f"{(16433.0 / c)**0.5:.4f}") for c in [63668.0, 25685.0, 32015.0]]
    t3_pass = abs(fp_w[0][1] - (16433.0 / 63668.0)**0.5) < 1e-5
    results.append(("3. False Positive Matrix Structure", t3_pass, f"w_FP[0, 1]={fp_w[0][1]:.4f}, exp={row0_fp[0]:.4f}"))

    # Test 4: Asymmetric Cost Verification
    # Cost of predicting Wake when True is Light vs. predicting Light when True is Wake
    # True=Light (1), pred=Wake (0): fp_weights[1][0] = (l/w)^0.5 * 0.5 = (63668/16433)^0.5 * 0.5 = 0.9841
    # True=Wake (0), pred=Light (1): fp_weights[0][1] = (w/l)^0.5 = 0.5080
    cost_wake_as_light = fp_w[0][1]
    cost_light_as_wake = fp_w[1][0]
    t4_pass = abs(cost_wake_as_light - cost_light_as_wake) > 0.1
    results.append(("4. Cost Asymmetry", t4_pass, f"FP(Wake->Light)={cost_wake_as_light:.4f} != FP(Light->Wake)={cost_light_as_wake:.4f}"))

    # Test 5: Masked Padding Positions (zero loss contribution)
    logits_pad = torch.randn(4, 4, device=device)
    targets_pad = torch.tensor([0, 1, 2, 3], device=device)
    mask = torch.tensor([1, 1, 0, 0], device=device) # last 2 are padded
    ce_loss = loss_ce_none(logits_pad, targets_pad)
    masked_ce = torch.sum(ce_loss * mask)
    rw_loss = loss_rccl(logits_pad, targets_pad)
    masked_rw = -torch.sum(rw_loss * mask.unsqueeze(1).repeat(1, 4))
    
    # check that only positions 0 and 1 contribute
    ce_only_01 = loss_ce_none(logits_pad[:2], targets_pad[:2]).sum()
    rw_only_01 = -torch.sum(loss_rccl(logits_pad[:2], targets_pad[:2]))
    t5_pass = torch.allclose(masked_ce, ce_only_01, atol=1e-5) and torch.allclose(masked_rw, rw_only_01, atol=1e-5)
    results.append(("5. Masked Padding Zero Contribution", t5_pass, f"Masked CE diff={abs(masked_ce.item()-ce_only_01.item()):.2e}"))

    # Test 6: All-masked batch
    mask_all_zero = torch.zeros(4, device=device)
    tot_ce_zero = torch.sum(ce_loss * mask_all_zero)
    tot_rw_zero = -torch.sum(rw_loss * mask_all_zero.unsqueeze(1).repeat(1, 4))
    tot_loss_all_zero = tot_ce_zero + 0.01 * tot_rw_zero
    t6_pass = (tot_loss_all_zero.item() == 0.0)
    results.append(("6. All-Masked Batch Handling", t6_pass, f"Total Loss={tot_loss_all_zero.item():.4f}"))

    # Test 7: Gradient flow to model parameters
    linear_layer = nn.Linear(10, 4).to(device)
    x_input = torch.randn(5, 10, device=device, requires_grad=True)
    out = linear_layer(x_input)
    y_t = torch.tensor([0, 1, 2, 3, 0], device=device)
    loss_val = -torch.sum(loss_rccl(out, y_t))
    loss_val.backward()
    grad_norm = linear_layer.weight.grad.norm().item()
    t7_pass = grad_norm > 0.0 and not torch.isnan(linear_layer.weight.grad).any()
    results.append(("7. Gradient Flow & Non-Zero Grads", t7_pass, f"Weight Grad Norm={grad_norm:.4f}"))

    # Test 8: Numerical Stability under extreme values (+100.0, -100.0)
    logits_extreme = torch.tensor([[100.0, -100.0, -100.0, -100.0]], device=device)
    loss_ext = loss_rccl(logits_extreme, torch.tensor([0], device=device))
    t8_pass = not torch.isnan(loss_ext).any() and not torch.isinf(loss_ext).any()
    results.append(("8. Numerical Stability (Extreme Logits)", t8_pass, f"Finite={t8_pass}, val={loss_ext.sum().item():.4f}"))

    # Test 9: Edge Case with Uniform Probabilities (Maximum Entropy)
    logits_uniform = torch.zeros(1, 4, device=device)
    loss_unif = -torch.sum(loss_rccl(logits_uniform, torch.tensor([0], device=device)))
    # With p_c = 0.25, logs = ln(0.25) = -1.386, logs_1_sub = ln(0.75) = -0.2877
    t9_pass = loss_unif.item() > 0.0 and not torch.isnan(loss_unif)
    results.append(("9. Uniform Distribution (Max Entropy)", t9_pass, f"Loss={loss_unif.item():.4f}"))

    # Test 10: Composite Loss Scaling (CE + 0.01 * RWL)
    logits_rnd = torch.randn(100, 4, device=device)
    y_rnd = torch.randint(0, 4, (100,), device=device)
    ce_rnd = torch.sum(loss_ce_none(logits_rnd, y_rnd)).item()
    rw_rnd = (-torch.sum(loss_rccl(logits_rnd, y_rnd))).item()
    composite = ce_rnd + 0.01 * rw_rnd
    # In official code, 0.01 * RWL acts as a regularizer (~1.5% to 3% of CE sum)
    ratio = (0.01 * rw_rnd) / (ce_rnd + 1e-8)
    t10_pass = 0.01 <= ratio <= 0.05
    results.append(("10. Composite Loss Balance (0.01*RWL / CE)", t10_pass, f"CE={ce_rnd:.1f}, 0.01*RWL={0.01*rw_rnd:.1f}, Ratio={ratio:.4f} (expected ~0.015-0.030)"))

    all_passed = True
    for name, p, msg in results:
        status = "PASSED" if p else "FAILED"
        if not p:
            all_passed = False
        print(f"  [{status}] {name:40s} | {msg}")
    print("=" * 65)
    print(f"TOTAL UNIT TESTS: 10/10 | ALL PASSED: {all_passed}")
    print("=" * 65)
    return all_passed

if __name__ == "__main__":
    run_10_loss_tests()
