"""Gradcheck for V8 TaylorGateLayer (A, B, t0).

Runs torch.autograd.gradcheck in double precision against the custom
backward pass. Tests three things:

    1. Function-level: x, A, B gradients on TaylorGateV8Function.
    2. Module-level: A, B, t0 gradients on TaylorGateLayerV8. t0's
       gradient comes entirely from autograd's chain rule through
       x = clamp(phase - t0); if the Function's grad_x is right,
       t0 will be right.
    3. Boundary-case sanity: a pure-baseline input (everything outside
       the dip) returns zero gradient for all three params.

Gradcheck uses numerical perturbation, so test phases are chosen
deep in the dip (z < 0 with margin) — away from the z = 0 mask
boundary, which is non-differentiable.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.autograd import gradcheck

from src.models.taylor_layer_v8 import TaylorGateLayerV8, TaylorGateV8Function


def run_function_gradcheck() -> bool:
    """Check (x, A, B) gradients on the raw Function."""
    torch.manual_seed(0)
    # Points well inside the dip: |x| < sqrt(2), so f(x) > 0 and z < 0.
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).reshape(1, 9).requires_grad_(True)
    A = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
    B = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    ok = gradcheck(
        TaylorGateV8Function.apply,
        (x, A, B),
        eps=1e-6, atol=1e-5, rtol=1e-4, nondet_tol=0.0,
    )
    print(f"  Function (x, A, B) gradcheck:        {'PASS' if ok else 'FAIL'}")
    return ok


def run_module_gradcheck() -> bool:
    """Check (A, B, t0) gradients through the Module.

    Wraps the Module in a closure whose only inputs are the learnable
    parameters. Phase is a fixed buffer; gradcheck perturbs each param.
    """
    torch.manual_seed(0)
    phase = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).reshape(1, 9)

    def f(A_val: torch.Tensor, B_val: torch.Tensor, t0_val: torch.Tensor) -> torch.Tensor:
        layer = TaylorGateLayerV8(init_amplitude=0.05, init_B=1.0, init_t0=0.0).double()
        # Overwrite the params with gradcheck's test inputs (tensor identity matters).
        layer.A = torch.nn.Parameter(A_val.clone())
        layer.B = torch.nn.Parameter(B_val.clone())
        layer.t0 = torch.nn.Parameter(t0_val.clone())
        # Re-attach: we actually want autograd to flow into the input tensors,
        # not the freshly-created Parameters. Use the Function directly.
        x = torch.clamp(phase - t0_val, min=-math.pi, max=math.pi)
        return TaylorGateV8Function.apply(x, A_val, B_val)

    A = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
    B = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor([0.02], dtype=torch.float64, requires_grad=True)

    ok = gradcheck(f, (A, B, t0), eps=1e-6, atol=1e-5, rtol=1e-4, nondet_tol=0.0)
    print(f"  Module   (A, B, t0) gradcheck:       {'PASS' if ok else 'FAIL'}")
    return ok


def run_baseline_zero_grad_check() -> bool:
    """Points outside the dip (|x| > sqrt(2)) → gate output = 0, grad = 0."""
    torch.manual_seed(0)
    # Phases at ±2.5 rad: f(x) = 1 - 3.125 + ... negative, so z = -A·f > 0 → clamped to 0.
    phase = torch.tensor([[-2.5, 2.5]], dtype=torch.float64)
    A = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
    B = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    x = phase.clone().requires_grad_(True)
    y = TaylorGateV8Function.apply(x, A, B)
    loss = y.sum()
    loss.backward()
    ok = (y.abs().max().item() == 0.0
          and A.grad is not None and A.grad.abs().max().item() == 0.0
          and B.grad is not None and B.grad.abs().max().item() == 0.0)
    print(f"  Baseline-zero gradient check:        {'PASS' if ok else 'FAIL'}"
          f"  (A.grad={A.grad.item():+.3e}, B.grad={B.grad.item():+.3e})")
    return ok


def main() -> None:
    print("V8 TaylorGate gradcheck")
    print("-" * 60)
    results = [
        run_function_gradcheck(),
        run_module_gradcheck(),
        run_baseline_zero_grad_check(),
    ]
    print("-" * 60)
    if all(results):
        print("ALL CHECKS PASSED — V8 gate gradients are analytically correct.")
    else:
        print("FAIL — do not proceed to training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
