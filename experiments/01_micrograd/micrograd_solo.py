"""micrograd — a tiny autograd engine, from scratch.

Read this file top-to-bottom. The point isn't the final result — it's seeing
how a Value tracks its parents, how each operation knows its own gradient
rule, and how `backward()` walks the graph in the right order to apply the
chain rule.

If you finish reading and `pytest tests/test_micrograd.py` is green, you
have a working autograd engine that matches PyTorch's gradients to 1e-6 on
small expressions.

Reference: Andrej Karpathy's `micrograd` (https://github.com/karpathy/micrograd).
This implementation mirrors his choices intentionally so the video lecture
will feel familiar when you watch it.
"""

from __future__ import annotations

import math
import random
from typing import Callable


# =============================================================================
# Value — the unit of an autograd graph
# =============================================================================

class Value:
    """A scalar wrapped in metadata so we can run backprop on expressions.

    Three things a Value tracks:
      data:  the actual number (a float)
      grad:  d(loss)/d(self), filled in by .backward()
      _prev: the set of Values that this Value was computed from
      _op:   a label like "+" or "*tanh" so the graph is debuggable
      _backward: a closure that knows how to push gradient *backwards*
                 from this node to its parents, using the chain rule

    The closure pattern is the key trick: each operation defines a tiny
    backward function at the moment of forward computation, capturing the
    operands. When .backward() is later called on the final node, it walks
    the graph in reverse topological order and calls each closure.
    """

    def __init__(self, data: float, _prev: tuple["Value", ...] = (), _op: str = "") -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: set["Value"] = set(_prev)
        self._op: str = _op
        # Default backward does nothing — leaf nodes have nothing upstream.
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    # -------------------------------------------------------------------------
    # Operations. Each one:
    #   1. Computes the forward value
    #   2. Wraps it in a new Value that remembers `self` and `other` as parents
    #   3. Defines a _backward closure that knows the local derivative
    # -------------------------------------------------------------------------

    def __add__(self, other: "Value | float") -> "Value":
        # If `other` is a plain number, lift it into a Value so the graph is uniform.
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(out)/d(self) = 1   and   d(out)/d(other) = 1
            # Chain rule: each parent gets out.grad * (local derivative).
            # We *accumulate* with +=, not assign — a Value can appear in
            # multiple expressions and each contributes a gradient term.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(self * other)/d(self) = other,   d(.)/d(other) = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Value":
        # Only supports power by a constant scalar (no Value ** Value).
        # That's what Karpathy does and it covers everything we need for an MLP.
        assert isinstance(exponent, (int, float)), "only scalar exponents supported"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            # d(self**n)/d(self) = n * self**(n-1)
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward() -> None:
            # d(exp(x))/dx = exp(x) — and that's exactly out.data
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - out.data^2
            # Note we read the cached `t` (== out.data) — cheaper than recomputing.
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            # d(relu(x))/dx = 1 if x > 0 else 0
            # Strictly, the derivative at x=0 is undefined; we follow PyTorch
            # and pick the right-side derivative (0). Doesn't matter in practice.
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Convenience operators — wrappers so we can write natural Python expressions.
    # -------------------------------------------------------------------------

    def __neg__(self) -> "Value":             # -self
        return self * -1

    def __radd__(self, other: float) -> "Value":  # other + self  (right-add)
        return self + other

    def __sub__(self, other: "Value | float") -> "Value":  # self - other
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other: float) -> "Value":  # other - self
        return Value(other) - self

    def __rmul__(self, other: float) -> "Value":  # other * self
        return self * other

    def __truediv__(self, other: "Value | float") -> "Value":  # self / other
        # Implemented via x / y == x * y^(-1) so we reuse __pow__ and __mul__
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other: float) -> "Value":  # other / self
        return Value(other) * self ** -1

    # -------------------------------------------------------------------------
    # backward — the only "smart" method. Topologically sorts the graph from
    # `self` (the final output) backwards, then calls each node's _backward
    # closure in reverse order so gradients flow from output to inputs.
    # -------------------------------------------------------------------------

    def backward(self) -> None:
        # Step 1: build a topological order of the graph rooted at self.
        # Post-order DFS gives us nodes from leaves up to root; we reverse
        # at the end to get root-first (which is the order we want for
        # backward).
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v in visited:
                return
            visited.add(v)
            for parent in v._prev:
                build(parent)
            topo.append(v)

        build(self)

        # Step 2: seed the root's gradient. d(self)/d(self) == 1.
        self.grad = 1.0

        # Step 3: walk in reverse, calling each node's _backward closure.
        # Each call pushes gradient from this node to its parents using +=.
        for v in reversed(topo):
            v._backward()


# =============================================================================
# Neuron, Layer, MLP — built only out of Value operations
# =============================================================================

class Neuron:
    """A single neuron: n weights + 1 bias + tanh nonlinearity.

    Forward:  out = tanh(sum_i(w_i * x_i) + b)
    """

    def __init__(self, nin: int) -> None:
        # Initialize each weight from Uniform(-1, 1). Different per-neuron
        # init breaks symmetry — otherwise every neuron in a layer would
        # compute the same output and gradient.
        self.w: list[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0.0)

    def __call__(self, x: list[Value | float]) -> Value:
        # sum(generator, start=...) lets us begin with self.b as the accumulator
        # so the initial value is a Value, not a Python int.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    """A list of neurons sharing the same input. Output is a list of activations."""

    def __init__(self, nin: int, nout: int) -> None:
        self.neurons: list[Neuron] = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Stack of layers. nin -> hidden_1 -> hidden_2 -> ... -> nout.

    Constructed as MLP(3, [4, 4, 1]) for: 3 inputs, two hidden layers of 4,
    then a 1-output head.
    """

    def __init__(self, nin: int, nouts: list[int]) -> None:
        sizes = [nin] + nouts
        self.layers: list[Layer] = [
            Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))
        ]

    def __call__(self, x: list[Value | float]) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)
        # If the last layer has one output, return the scalar (more natural for losses).
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


# =============================================================================
# fit — a tiny training loop. Mean-squared-error on a list of (x, y) pairs.
# =============================================================================

def fit(
    mlp: MLP,
    xs: list[list[float]],
    ys: list[float],
    *,
    steps: int = 200,
    lr: float = 0.05,
    verbose: bool = False,
) -> float:
    """Run `steps` of vanilla SGD. Returns the final scalar loss."""
    final_loss = float("inf")
    for step in range(steps):
        # Forward — produce a prediction for each example.
        ypred = [mlp(x) for x in xs]

        # Loss — sum of squared errors. Note: target `yt` is a plain float;
        # `ypred_i - yt` triggers __sub__ which lifts yt into a Value.
        loss = sum((ypred_i - yt) ** 2 for ypred_i, yt in zip(ypred, ys))

        # ---- the critical trio ----
        # 1. Zero the grads. Without this, gradients from previous steps
        #    accumulate via the += in _backward and training diverges.
        for p in mlp.parameters():
            p.grad = 0.0
        # 2. Compute gradients by walking the graph backwards from loss.
        loss.backward()
        # 3. Take a step in the direction that *decreases* the loss
        #    (i.e. opposite the gradient, hence the minus sign).
        for p in mlp.parameters():
            p.data -= lr * p.grad

        final_loss = loss.data
        if verbose and step % 10 == 0:
            print(f"step {step:4d}  loss {final_loss:.6f}")
    return final_loss
