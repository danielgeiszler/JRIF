"""Muon optimizer — Momentum Orthogonalized Update.

Based on Keller Jordan's implementation from the NanoGPT speedrun project.
Applies Nesterov momentum in the space of orthogonalized gradients using
Newton-Schulz iteration for efficient matrix orthogonalization.
"""

import torch
from torch.optim import Optimizer


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization via Newton-Schulz iteration.

    Operates on 2D matrices. Returns an orthogonalized version of G.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X


class Muon(Optimizer):
    """Muon optimizer for matrix-shaped parameters (conv filters, etc).

    Args:
        params: Parameters to optimize (should be 2D+ weight tensors).
        lr: Learning rate.
        momentum: Nesterov momentum coefficient.
        ns_steps: Number of Newton-Schulz orthogonalization iterations.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Reshape to 2D for orthogonalization
                orig_shape = g.shape
                if g.ndim == 1:
                    # Skip 1D params (biases) — shouldn't be passed to Muon
                    p.add_(g, alpha=-lr)
                    continue
                g_2d = g.reshape(orig_shape[0], -1)

                # Newton-Schulz orthogonalization
                g_orth = _newton_schulz(g_2d, steps=ns_steps)
                g_orth = g_orth.reshape(orig_shape)

                # Nesterov momentum
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g_orth)
                p.add_(buf, alpha=-lr)
                # Nesterov lookahead
                p.add_(g_orth, alpha=-lr * momentum)
