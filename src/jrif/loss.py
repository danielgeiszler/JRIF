"""Robust regression losses for RT prediction.

`LaplaceUniformMixture` is a contamination mixture loss intended to replace
Huber on prefiltered training data where the *RT-label* outlier rate is
unknown. Residuals are modelled as

    r ~ (1 - eps) * Laplace(0, b)  +  eps * Uniform(-W, W)

with `b > 0` and `eps in (0, 1)` learned jointly with the model. The clean
component scale `b` adapts to the data noise floor; `eps` is the learned
contamination rate (analogous to Chronologer's hand-set FDR mask, but fit
from the data instead of fixed). `W` is the contaminant support and is
fixed at construction (in the same units as the residuals — z-space here);
it just needs to be wide enough to cover plausible outliers.

Both `b` and `eps` are stored in unconstrained form (raw real scalars) and
mapped onto their valid ranges via softplus / sigmoid; this keeps gradient
descent over the loss parameters numerically stable without per-step
projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(x: float) -> float:
    """Return the value u such that softplus(u) == x."""
    return math.log(math.expm1(x))


def _inv_sigmoid(x: float) -> float:
    """Return the value u such that sigmoid(u) == x."""
    return math.log(x / (1.0 - x))


class LaplaceUniformMixture(nn.Module):
    """Negative log-likelihood under a Laplace + uniform contamination mixture.

    Args:
        init_b:   initial Laplace scale (in residual units). Default 1.0.
        init_eps: initial contamination weight. Default 0.05.
        W:        half-width of the uniform contaminant support. Default 10.0.
    """

    def __init__(self, init_b: float = 1.0, init_eps: float = 0.05, W: float = 10.0):
        super().__init__()
        if init_b <= 0:
            raise ValueError("init_b must be > 0")
        if not (0.0 < init_eps < 1.0):
            raise ValueError("init_eps must be in (0, 1)")
        self._b_raw = nn.Parameter(torch.tensor(_inv_softplus(init_b), dtype=torch.float32))
        self._eps_raw = nn.Parameter(torch.tensor(_inv_sigmoid(init_eps), dtype=torch.float32))
        self.W = float(W)
        self._b_floor = 1e-3

    @property
    def b(self) -> torch.Tensor:
        return F.softplus(self._b_raw) + self._b_floor

    @property
    def eps(self) -> torch.Tensor:
        return torch.sigmoid(self._eps_raw)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r = pred - target
        b = self.b
        eps = self.eps

        # Component log-pdfs (per-element).
        log_p_laplace = -torch.log(2.0 * b) - r.abs() / b
        log_2W = math.log(2.0 * self.W)
        in_support = (r.abs() < self.W)
        # Outside the support the uniform component contributes zero to the
        # mixture density; represent its log-pdf as a very large negative
        # number so logsumexp falls back to the Laplace component cleanly.
        log_p_uniform = torch.where(
            in_support,
            torch.full_like(r, -log_2W),
            torch.full_like(r, -1e10),
        )

        log_w_laplace = torch.log1p(-eps) + log_p_laplace
        log_w_uniform = torch.log(eps) + log_p_uniform
        log_mix = torch.logsumexp(
            torch.stack([log_w_laplace, log_w_uniform], dim=0), dim=0,
        )
        return -log_mix.mean()


class NormalNormalMixture(nn.Module):
    """NLL under a scale mixture of two zero-mean Gaussians.

    Models residuals as
        r ~ (1 - eps) * N(0, sigma_clean^2)  +  eps * N(0, sigma_outlier^2)

    where `sigma_clean`, `sigma_outlier` and `eps` are all learned jointly
    with the model. To keep the components identifiable we parameterize the
    outlier scale relatively as `sigma_outlier = sigma_clean * (1 + softplus(d))`
    so that sigma_outlier > sigma_clean by construction. This fits empirical
    PSM-residual distributions which are Gaussian-bulk + heavier-tail-Gaussian
    (a classic scale-mixture-of-Gaussians, equivalent to heteroscedastic
    Gaussian noise where ~`eps` of peptides have higher prediction variance).

    Args:
        init_sigma_clean: initial sigma of the dominant component.
        init_sigma_ratio: initial ratio sigma_outlier / sigma_clean (>= 1).
        init_eps:         initial contamination weight in (0, 1).
    """

    def __init__(
        self,
        init_sigma_clean: float = 0.4,
        init_sigma_ratio: float = 2.0,
        init_eps: float = 0.05,
    ):
        super().__init__()
        if init_sigma_clean <= 0:
            raise ValueError("init_sigma_clean must be > 0")
        if init_sigma_ratio < 1.0:
            raise ValueError("init_sigma_ratio must be >= 1 (outlier wider than clean)")
        if not (0.0 < init_eps < 1.0):
            raise ValueError("init_eps must be in (0, 1)")
        self._sigma_clean_raw = nn.Parameter(
            torch.tensor(_inv_softplus(init_sigma_clean), dtype=torch.float32)
        )
        # sigma_outlier = sigma_clean * (1 + softplus(_ratio_extra_raw))
        # init: 1 + softplus(x) = ratio  ->  softplus(x) = ratio - 1
        extra = init_sigma_ratio - 1.0
        self._ratio_extra_raw = nn.Parameter(
            torch.tensor(_inv_softplus(extra) if extra > 0 else -10.0, dtype=torch.float32)
        )
        self._eps_raw = nn.Parameter(
            torch.tensor(_inv_sigmoid(init_eps), dtype=torch.float32)
        )
        self._sigma_floor = 1e-3

    @property
    def sigma_clean(self) -> torch.Tensor:
        return F.softplus(self._sigma_clean_raw) + self._sigma_floor

    @property
    def sigma_outlier(self) -> torch.Tensor:
        return self.sigma_clean * (1.0 + F.softplus(self._ratio_extra_raw))

    @property
    def eps(self) -> torch.Tensor:
        return torch.sigmoid(self._eps_raw)

    # Names mirror LaplaceUniformMixture so bench logging works unchanged.
    @property
    def b(self) -> torch.Tensor:
        """Alias for sigma_clean (so bench logging that prints `b` keeps working)."""
        return self.sigma_clean

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r = pred - target
        sigma_c = self.sigma_clean
        sigma_o = self.sigma_outlier
        eps = self.eps

        log_p_clean = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma_c)
            - 0.5 * (r / sigma_c) ** 2
        )
        log_p_outlier = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma_o)
            - 0.5 * (r / sigma_o) ** 2
        )

        log_w_clean = torch.log1p(-eps) + log_p_clean
        log_w_outlier = torch.log(eps) + log_p_outlier
        log_mix = torch.logsumexp(
            torch.stack([log_w_clean, log_w_outlier], dim=0), dim=0,
        )
        return -log_mix.mean()
