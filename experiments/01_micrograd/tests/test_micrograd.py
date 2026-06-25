"""Tests for the solo micrograd re-implementation.

If these pass against `micrograd_solo.py`, Day 4 is done.
"""

from __future__ import annotations

import math
import random

import pytest


# Lazy import — lets us write this file before micrograd_solo.py exists.
def _import_solo():
    try:
        from micrograd_solo import MLP, Value, fit  # type: ignore
        return Value, MLP, fit
    except ImportError as e:
        pytest.skip(f"micrograd_solo.py not implemented yet: {e}")


# ---------------------------------------------------------------------------
# Gradient correctness vs PyTorch
# ---------------------------------------------------------------------------

def _torch_grads(expr_fn, a_val: float, b_val: float, c_val: float):
    """Evaluate `expr_fn(a, b, c)` in PyTorch and return (a.grad, b.grad, c.grad)."""
    import torch

    ta = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
    tb = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
    tc = torch.tensor(c_val, requires_grad=True, dtype=torch.float64)
    out = expr_fn(ta, tb, tc)
    out.backward()
    return ta.grad.item(), tb.grad.item(), tc.grad.item()


def _micrograd_grads(Value, expr_fn, a_val, b_val, c_val):
    a = Value(a_val)
    b = Value(b_val)
    c = Value(c_val)
    out = expr_fn(a, b, c)
    out.backward()
    return a.grad, b.grad, c.grad


@pytest.mark.parametrize(
    "name,expr_torch,expr_micro",
    [
        ("add_mul",
         lambda a, b, c: (a + b) * c,
         lambda a, b, c: (a + b) * c),
        ("tanh_chain",
         lambda a, b, c: (a * b + c).tanh(),
         lambda a, b, c: (a * b + c).tanh()),
        ("relu_pow",
         lambda a, b, c: (a * b - c).relu() + a ** 2,
         lambda a, b, c: (a * b - c).relu() + a ** 2),
    ],
)
def test_gradient_matches_pytorch(name, expr_torch, expr_micro):
    Value, _, _ = _import_solo()
    a_val, b_val, c_val = 2.0, -3.0, 0.5
    t_grads = _torch_grads(expr_torch, a_val, b_val, c_val)
    m_grads = _micrograd_grads(Value, expr_micro, a_val, b_val, c_val)
    for tg, mg in zip(t_grads, m_grads):
        assert math.isclose(tg, mg, abs_tol=1e-6, rel_tol=1e-6), f"{name}: {tg} vs {mg}"


# ---------------------------------------------------------------------------
# Diamond accumulation — using the same Value twice must not lose gradient
# ---------------------------------------------------------------------------

def test_diamond_accumulation():
    Value, _, _ = _import_solo()
    a = Value(3.0)
    b = a + a       # b = 2a, db/da = 2
    c = b * b       # c = 4a^2, dc/da = 8a = 24
    c.backward()
    assert math.isclose(a.grad, 24.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# MLP trains on Karpathy's 4-point dataset
# ---------------------------------------------------------------------------

def test_mlp_trains():
    _, MLP, fit = _import_solo()
    random.seed(42)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP(3, [4, 4, 1])
    final_loss = fit(mlp, xs, ys, steps=200, lr=0.05)
    assert final_loss < 0.01, f"final_loss={final_loss} (expected <0.01 in 200 steps)"
