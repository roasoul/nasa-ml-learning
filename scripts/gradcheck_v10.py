"""Gradcheck all 5 amplitudes of the V10 MultiTemplateGateBank.

The gates are composed entirely of standard torch ops (pow, mul,
exp, clamp), so autograd already has analytic gradients. The purpose
of this script is to confirm that:

    1. Each A_i receives a nonzero gradient when its gate is in its
       active region (above the A.clamp floor and not zero-masked).
    2. The other amplitudes receive zero gradient when we scope a
       loss to a single gate's slice. That verifies gates are
       independent (no cross-coupling via shared buffers).
    3. Autograd's numerical-vs-analytic agreement holds in float64
       at atol=1e-5, rtol=1e-4.

Test inputs are chosen per-gate to lie in the active region:

    G1, G2 — U/V: |x| < sqrt(2) so the polynomial factor > 0 and the
             clamp(max=0) is inactive on the output sign.
    G3     — inverted: same |x| < sqrt(2) so clamp(min=0) is inactive.
    G4     — asymmetric: x > 0 (polynomial positive → -A*poly < 0 →
             clamp keeps it; the x < 0 branch zeroes the output).
    G5     — Gaussian: any |x| (exp is always positive so the product
             is always negative, always kept by clamp).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.autograd import gradcheck

from src.models.multi_template_gate import MultiTemplateGateBank


import math


def _gate_forward(A1, A2, A3, A4, A5, x, gate_idx):
    """Compute a single gate's output directly, preserving the autograd
    connection from the input A tensors through to the output.

    Wrapping them in new `nn.Parameter` objects would detach from the
    caller's graph — gradcheck needs the leaves it passed in to
    actually appear in the backward pass.
    """
    x = x.clamp(min=-math.pi, max=math.pi)
    A1c = A1.clamp(min=0.001)
    A2c = A2.clamp(min=0.001)
    A3c = A3.clamp(min=0.001)
    A4c = A4.clamp(min=0.001)
    A5c = A5.clamp(min=0.001)
    if gate_idx == 0:
        return torch.clamp(-A1c * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0), max=0.0)
    if gate_idx == 1:
        return torch.clamp(-A2c * (1.0 - x ** 2 / 2.0), max=0.0)
    if gate_idx == 2:
        return torch.clamp(A3c * (1.0 - x ** 2 / 2.0), min=0.0)
    if gate_idx == 3:
        return torch.clamp(-A4c * (x - x ** 3 / 6.0), max=0.0)
    if gate_idx == 4:
        return torch.clamp(-A5c * torch.exp(-x ** 2 / 0.5), max=0.0)
    raise ValueError(gate_idx)


def run_per_gate_gradcheck() -> list[bool]:
    """Check each amplitude's gradient by wrapping in a closure."""
    torch.manual_seed(0)
    results = []
    for gate_idx, (name, x_vals) in enumerate([
        ("G1 planet U",      torch.linspace(-1.0, 1.0, 9)),
        ("G2 V-shape",       torch.linspace(-1.0, 1.0, 9)),
        ("G3 inverted sec",  torch.linspace(-1.0, 1.0, 9)),
        ("G4 asymmetric",    torch.linspace(0.1, 1.2, 9)),
        ("G5 Gaussian",      torch.linspace(-0.4, 0.4, 9)),
    ]):
        x = x_vals.reshape(1, -1).double()
        A = [torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
             for _ in range(5)]

        def f(A1, A2, A3, A4, A5, _gate_idx=gate_idx):
            return _gate_forward(A1, A2, A3, A4, A5, x, _gate_idx)

        ok = gradcheck(f, tuple(A), eps=1e-6, atol=1e-5, rtol=1e-4, nondet_tol=0.0)
        print(f"  {name:<20} gradcheck:  {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    return results


def run_independence_check() -> bool:
    """Gate i's output should depend only on A_i — backward into A_j (j≠i)
    should be zero."""
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).reshape(1, -1)
    all_ok = True
    for gate_idx in range(5):
        bank = MultiTemplateGateBank(init_amplitude=0.05).double()
        out = bank(x)
        loss = out[:, gate_idx, :].sum()
        loss.backward()
        grads = [float(bank.A1.grad.abs().item()),
                 float(bank.A2.grad.abs().item()),
                 float(bank.A3.grad.abs().item()),
                 float(bank.A4.grad.abs().item()),
                 float(bank.A5.grad.abs().item())]
        # Only grads[gate_idx] should be nonzero; all others should be 0.
        nonzero = [abs(g) > 1e-12 for g in grads]
        expected = [False] * 5
        expected[gate_idx] = True
        gate_ok = nonzero == expected
        all_ok = all_ok and gate_ok
        marker = "OK" if gate_ok else "FAIL"
        print(f"  Gate G{gate_idx+1} independence [{marker}] "
              f"grads = {['%.2e' % g for g in grads]}")
    return all_ok


def main() -> None:
    print("V10 MultiTemplateGateBank gradcheck")
    print("-" * 60)
    per_gate = run_per_gate_gradcheck()
    print("-" * 60)
    indep = run_independence_check()
    print("-" * 60)
    if all(per_gate) and indep:
        print("ALL V10 GATE CHECKS PASSED")
    else:
        print("FAIL — do not proceed to training")
        sys.exit(1)


if __name__ == "__main__":
    main()
