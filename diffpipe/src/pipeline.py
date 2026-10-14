"""
Differentiable 8-stage HBM nano-CT degradation pipeline in PyTorch.

Mirrors the legacy numpy/OpenCV pipeline at:
  HBM_paper2/diffpipe/legacy/generate_canonical.py

All optimizable parameters are stored as nn.Parameter in raw (unconstrained)
space; physical positivity is enforced via softplus at forward time.
EXCEPTION: N (Poisson photon count) uses sigmoid-box [50, 1000] so that
display value == pipeline value (v4+).
EXCEPTION: sigma_post is stored as a direct nn.Parameter (no softplus).
           Physical value = sigma_post directly; clamped externally during
           optimization. σ_post=0 → stage is a no-op (bit-identical to
           the legacy 7-stage pipeline).

Stage order:
  1. Poisson noise
  2. Gaussian noise
  3. Salt & Pepper
  4. PSF blur
  5. Vignetting
  5.5. Post-blur detector noise (S5.5, HANDOFF v6.4 §1.1)
  6. Contrast
  7. Ring artifact
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softplus_inverse(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1).  Used to initialize raw params."""
    return math.log(math.expm1(x))


# N sigmoid-box constants (v4+)
_N_LO: float = 50.0
_N_HI: float = 1000.0


def sigmoid_box_inverse(N: float) -> float:
    """Inverse of N = _N_LO + (_N_HI - _N_LO) * sigmoid(raw).  Init helper."""
    ratio = (N - _N_LO) / (_N_HI - _N_LO)
    ratio = max(1e-6, min(1.0 - 1e-6, ratio))
    return math.log(ratio / (1.0 - ratio))


# ---------------------------------------------------------------------------
# DegradationPipeline
# ---------------------------------------------------------------------------

class DegradationPipeline(nn.Module):
    """
    Differentiable degradation pipeline.

    Parameters
    ----------
    All physical θ values with positivity constraints are stored as θ_raw
    (nn.Parameter).  At forward time: θ = F.softplus(θ_raw).

    Contrast params (in_low, in_high, out_low, out_high) are unconstrained
    nn.Parameter (no softplus).

    S&P density p is stored as nn.Parameter with requires_grad=False.

    min_brightness is a plain Python float constant (not a Parameter).

    PSF kernel size _psf_k is fixed at __init__ from the initial fwhm value
    and never recomputed dynamically (dynamic k would break autograd at the
    discrete step).
    """

    # ------------------------------------------------------------------
    # Initial physical values (θ_manual)
    # ------------------------------------------------------------------
    _INIT = dict(
        fwhm=2.5,
        N=300.0,
        sigma_g=0.0408,        # 0.0272 * 1.5
        p=0.001,
        alpha=0.35,
        in_low=0.05,
        in_high=0.90,
        out_low=0.10,
        out_high=0.90,
        sigma_post=0.01,       # S5.5: post-blur detector noise (HANDOFF v6.4 §1.1)
        ring_strength=0.12,
        ring_n=8.0,
    )

    def __init__(self):
        super().__init__()

        init = self._INIT

        # ---- Stage 1: Poisson (sigmoid-box [50, 1000], v4+) ----
        self.N_raw = nn.Parameter(
            torch.tensor(sigmoid_box_inverse(init['N']), dtype=torch.float32)
        )

        # ---- Stage 2: Gaussian ----
        self.sigma_g_raw = nn.Parameter(
            torch.tensor(softplus_inverse(init['sigma_g']), dtype=torch.float32)
        )

        # ---- Stage 3: Salt & Pepper (no gradient) ----
        self.p = nn.Parameter(
            torch.tensor(init['p'], dtype=torch.float32),
            requires_grad=False,
        )

        # ---- Stage 4: PSF blur ----
        # Kernel size fixed at init; dynamic k breaks gradient at discrete step.
        init_fwhm = init['fwhm']
        init_sigma = init_fwhm / 2.355
        # Use round() to match legacy generate_canonical.py formula exactly:
        #   k = int(2 * round(3 * sigma) + 1); k = max(3, k if k%2==1 else k+1)
        _k = int(2 * round(3 * init_sigma) + 1)
        if _k % 2 == 0:
            _k += 1
        self._psf_k: int = max(3, _k)

        self.fwhm_raw = nn.Parameter(
            torch.tensor(softplus_inverse(init_fwhm), dtype=torch.float32)
        )

        # ---- Stage 5: Vignetting ----
        self.alpha_raw = nn.Parameter(
            torch.tensor(softplus_inverse(init['alpha']), dtype=torch.float32)
        )
        self.min_brightness: float = 0.50  # constant, not a Parameter

        # ---- Stage 5.5: Post-blur detector noise (HANDOFF v6.4 §1.1) ----
        # Direct nn.Parameter in [1e-4, 0.1]; no softplus — clamped externally.
        # σ_post=0 → no-op → bit-identical to legacy 7-stage pipeline.
        self.sigma_post = nn.Parameter(
            torch.tensor(init['sigma_post'], dtype=torch.float32)
        )

        # ---- Stage 6: Contrast (unconstrained) ----
        self.in_low = nn.Parameter(
            torch.tensor(init['in_low'], dtype=torch.float32)
        )
        self.in_high = nn.Parameter(
            torch.tensor(init['in_high'], dtype=torch.float32)
        )
        self.out_low = nn.Parameter(
            torch.tensor(init['out_low'], dtype=torch.float32)
        )
        self.out_high = nn.Parameter(
            torch.tensor(init['out_high'], dtype=torch.float32)
        )

        # ---- Stage 7: Ring artifact ----
        self.ring_strength_raw = nn.Parameter(
            torch.tensor(softplus_inverse(init['ring_strength']), dtype=torch.float32)
        )
        self.ring_n_raw = nn.Parameter(
            torch.tensor(softplus_inverse(init['ring_n']), dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Individual stage methods
    # ------------------------------------------------------------------

    def _stage1_poisson(self, x: torch.Tensor, eps1: torch.Tensor) -> torch.Tensor:
        """Poisson noise via Gaussian approximation. N via sigmoid-box [50,1000]."""
        N = _N_LO + (_N_HI - _N_LO) * torch.sigmoid(self.N_raw)
        std = torch.sqrt(torch.clamp(x, min=1e-6) / N)
        return x + std * eps1

    def _stage2_gaussian(self, x: torch.Tensor, eps2: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian background noise."""
        sigma_g = F.softplus(self.sigma_g_raw)
        return x + sigma_g * eps2

    def _stage3_salt_pepper(self, x: torch.Tensor, eps3: torch.Tensor) -> torch.Tensor:
        """Salt & Pepper noise.  p has no gradient."""
        p_val = self.p  # scalar tensor, requires_grad=False
        return torch.where(
            eps3 < p_val / 2,
            torch.ones_like(x),
            torch.where(
                eps3 > 1.0 - p_val / 2,
                torch.zeros_like(x),
                x,
            ),
        )

    def _stage4_psf_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable Gaussian PSF blur with fixed kernel size."""
        fwhm = F.softplus(self.fwhm_raw)
        sigma = fwhm / 2.355
        k = self._psf_k  # fixed int

        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)).view(1, 1, k, k)

        pad = k // 2
        x4d = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        x_padded = F.pad(x4d, [pad, pad, pad, pad], mode='reflect')
        return F.conv2d(x_padded, kernel).squeeze(0).squeeze(0)

    def _stage5_vignetting(self, x: torch.Tensor) -> torch.Tensor:
        """Vignetting: alpha * d² / max_d² darkening toward corners."""
        alpha = F.softplus(self.alpha_raw)
        H, W = x.shape[-2], x.shape[-1]
        cy, cx = H / 2.0, W / 2.0

        y_c = torch.arange(H, dtype=x.dtype, device=x.device)
        x_c = torch.arange(W, dtype=x.dtype, device=x.device)
        yy, xx = torch.meshgrid(y_c, x_c, indexing='ij')

        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        max_d2 = cx ** 2 + cy ** 2

        vig = 1.0 - alpha * d2 / max_d2
        vig = torch.clamp(vig, min=self.min_brightness)
        return x * vig

    def _stage55_post_noise(self, x: torch.Tensor, eps_post: torch.Tensor) -> torch.Tensor:
        """S5.5: Post-blur detector noise (HANDOFF v6.4 §1.1).
        x = x + σ_post · ε_post
        σ_post=0 → no-op (bit-identical to legacy pipeline).
        """
        return x + self.sigma_post * eps_post

    def _stage6_contrast(self, x: torch.Tensor) -> torch.Tensor:
        """Linear contrast rescaling."""
        r = (x - self.in_low) / (self.in_high - self.in_low + 1e-8)
        return torch.clamp(r * (self.out_high - self.out_low) + self.out_low, 0.0, 1.0)

    def _stage7_ring(self, x: torch.Tensor) -> torch.Tensor:
        """Concentric ring artifact via sinusoidal radial modulation."""
        ring_strength = F.softplus(self.ring_strength_raw)
        ring_n = F.softplus(self.ring_n_raw)
        H, W = x.shape[-2], x.shape[-1]
        cy, cx = H / 2.0, W / 2.0

        y_c = torch.arange(H, dtype=x.dtype, device=x.device)
        x_c = torch.arange(W, dtype=x.dtype, device=x.device)
        yy, xx = torch.meshgrid(y_c, x_c, indexing='ij')

        r_map = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rmax = r_map.max() + 1e-8
        mod = ring_strength * torch.sin(2.0 * math.pi * ring_n * r_map / rmax)
        return torch.clamp(x * (1.0 + mod), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        eps: dict,
        max_stage: int = 7,
    ) -> torch.Tensor:
        """
        Apply degradation pipeline up to ``max_stage`` (1-indexed, inclusive).

        Parameters
        ----------
        x : torch.Tensor
            Input image, float32, shape (H, W), values in [0, 1].
        eps : dict
            Pre-sampled noise tensors (same keys → bit-identical output):
              eps['eps1'] ~ N(0, 1)   shape (H, W)   — Poisson stage
              eps['eps2'] ~ N(0, 1)   shape (H, W)   — Gaussian stage
              eps['eps3'] ~ U(0, 1)   shape (H, W)   — S&P stage
              eps['eps_post'] ~ N(0,1) shape (H,W)   — S5.5 post-noise (optional;
                                                        σ_post=0 → no effect)
        max_stage : int
            Stages 1..max_stage are applied in order.  0 → clean passthrough.
            Stage 5.5 maps to max_stage >= 6 (applied before contrast).

        Returns
        -------
        torch.Tensor  shape (H, W), values in [0, 1].
        """
        if max_stage == 0:
            return x.clone()

        y = x
        if max_stage >= 1:
            y = self._stage1_poisson(y, eps['eps1'])
        if max_stage >= 2:
            y = self._stage2_gaussian(y, eps['eps2'])
        if max_stage >= 3:
            y = self._stage3_salt_pepper(y, eps['eps3'])
        if max_stage >= 4:
            y = self._stage4_psf_blur(y)
        if max_stage >= 5:
            y = self._stage5_vignetting(y)
        if max_stage >= 6:
            eps_post = eps.get('eps_post')
            if eps_post is None:
                eps_post = torch.zeros_like(y)
            y = self._stage55_post_noise(y, eps_post)
            y = self._stage6_contrast(y)
        if max_stage >= 7:
            y = self._stage7_ring(y)
        return y


# ---------------------------------------------------------------------------
# Standalone stage functions for gradcheck
# ---------------------------------------------------------------------------

def _psf_fn(x: torch.Tensor, fwhm_raw: torch.Tensor, k: int) -> torch.Tensor:
    fwhm = F.softplus(fwhm_raw)
    sigma = fwhm / 2.355
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    gauss_1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)).view(1, 1, k, k)
    pad = k // 2
    x4d = F.pad(x.unsqueeze(0).unsqueeze(0), [pad, pad, pad, pad], mode='reflect')
    return F.conv2d(x4d, kernel).squeeze(0).squeeze(0)


def _poisson_fn(
    x: torch.Tensor, N_raw: torch.Tensor, eps1: torch.Tensor
) -> torch.Tensor:
    N = _N_LO + (_N_HI - _N_LO) * torch.sigmoid(N_raw)
    return x + torch.sqrt(torch.clamp(x, 1e-6) / N) * eps1


def _gaussian_fn(
    x: torch.Tensor, sigma_raw: torch.Tensor, eps2: torch.Tensor
) -> torch.Tensor:
    sigma = F.softplus(sigma_raw)
    return x + sigma * eps2


def _vignette_fn(
    x: torch.Tensor, alpha_raw: torch.Tensor, min_brightness: float = 0.5
) -> torch.Tensor:
    alpha = F.softplus(alpha_raw)
    H, W = x.shape[-2], x.shape[-1]
    y_c = torch.arange(H, dtype=x.dtype)
    x_c = torch.arange(W, dtype=x.dtype)
    yy, xx = torch.meshgrid(y_c, x_c, indexing='ij')
    d2 = (xx - W / 2) ** 2 + (yy - H / 2) ** 2
    max_d2 = (W / 2) ** 2 + (H / 2) ** 2
    vig = torch.clamp(1.0 - alpha * d2 / max_d2, min=min_brightness)
    return x * vig


def _contrast_fn(
    x: torch.Tensor,
    in_low: torch.Tensor,
    in_high: torch.Tensor,
    out_low: torch.Tensor,
    out_high: torch.Tensor,
) -> torch.Tensor:
    r = (x - in_low) / (in_high - in_low + 1e-8)
    return torch.clamp(r * (out_high - out_low) + out_low, 0.0, 1.0)


def _ring_fn(
    x: torch.Tensor, strength_raw: torch.Tensor, n_raw: torch.Tensor
) -> torch.Tensor:
    strength = F.softplus(strength_raw)
    n = F.softplus(n_raw)
    H, W = x.shape[-2], x.shape[-1]
    y_c = torch.arange(H, dtype=x.dtype)
    x_c = torch.arange(W, dtype=x.dtype)
    yy, xx = torch.meshgrid(y_c, x_c, indexing='ij')
    r_map = torch.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
    rmax = r_map.max() + 1e-8
    mod = strength * torch.sin(2.0 * math.pi * n * r_map / rmax)
    return torch.clamp(x * (1.0 + mod), 0.0, 1.0)


def _sigma_post_fn(
    x: torch.Tensor, sigma_post: torch.Tensor, eps_post: torch.Tensor
) -> torch.Tensor:
    """S5.5 standalone for gradcheck. σ_post=0 → no-op."""
    return x + sigma_post * eps_post
