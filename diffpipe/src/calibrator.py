"""
HBM pipeline calibrator module.

Provides:
  estimate_sp_density  — closed-form S&P density estimator (no gradient)
  warm_start           — Foi 2008 initial parameter estimates
  calibrate            — Adam-based main calibration loop

All functions are self-contained; DegradationPipeline is imported lazily so
this module can be imported even when torch is not installed (for unit tests of
the numpy-only functions).
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# θ_manual reference values (fallback / initial guesses)
# ---------------------------------------------------------------------------
THETA_MANUAL: Dict[str, float] = dict(
    fwhm=2.5,
    N=300.0,
    sigma_g=0.0408,
    p=0.001,
    alpha=0.35,
    in_low=0.05,
    in_high=0.90,
    out_low=0.10,
    out_high=0.90,
    ring_strength=0.12,
    ring_n=8.0,
)


# ---------------------------------------------------------------------------
# 1. estimate_sp_density
# ---------------------------------------------------------------------------

def estimate_sp_density(image: np.ndarray) -> float:
    """
    Closed-form Salt-and-Pepper density estimator.

    Algorithm (spec §2.3):
      res  = image − median_filter(image, 3)
      MAD  = median(|res − median(res)|)
      outliers = |res| > 6 * MAD
      p_hat = mean(outliers)

    Parameters
    ----------
    image : np.ndarray  shape (H, W), float32, values in [0, 1]

    Returns
    -------
    float  — estimated S&P pixel fraction
    """
    image = np.asarray(image, dtype=np.float32)

    # Residual after removing local median
    med_img = median_filter(image, size=3)
    res = image - med_img

    # Median Absolute Deviation
    med_res = float(np.median(res))
    MAD = float(np.median(np.abs(res - med_res)))

    # Robust threshold: 6-sigma rule on MAD
    threshold = 6.0 * MAD
    outliers = np.abs(res) > threshold

    p_hat = float(np.mean(outliers))
    return p_hat


# ---------------------------------------------------------------------------
# 2. warm_start
# ---------------------------------------------------------------------------

def warm_start(real_images: List[np.ndarray], pipeline) -> Dict[str, float]:
    """
    Initial parameter estimates before gradient optimisation (Foi 2008).

    For N and sigma_g: fit v(μ) = μ/N + σ_g² to patch statistics.
    For p:             average estimate_sp_density over all real images.
    For all others:    fall back to θ_manual.

    Parameters
    ----------
    real_images : list of np.ndarray  shape (H, W), float32
    pipeline    : DegradationPipeline instance (unused currently; reserved for
                  future inference of spatial parameters)

    Returns
    -------
    dict with keys: N, sigma_g, p, fwhm, alpha, in_low, in_high,
                    out_low, out_high, ring_strength, ring_n
    """
    patch_size = 8

    all_mu: List[float] = []
    all_var: List[float] = []

    for img in real_images:
        img = np.asarray(img, dtype=np.float32)
        H, W = img.shape

        # Tile into non-overlapping 8×8 patches
        n_row = H // patch_size
        n_col = W // patch_size
        for r in range(n_row):
            for c in range(n_col):
                patch = img[r * patch_size:(r + 1) * patch_size,
                            c * patch_size:(c + 1) * patch_size]
                all_mu.append(float(patch.mean()))
                all_var.append(float(patch.var()))

    mu_arr = np.asarray(all_mu, dtype=np.float64)
    var_arr = np.asarray(all_var, dtype=np.float64)

    # Bin by brightness into 16 bins, compute (μ_b, v_b) per bin
    n_bins = 16
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(mu_arr, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mu_bins: List[float] = []
    var_bins: List[float] = []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() < 5:
            continue
        mu_bins.append(float(mu_arr[mask].mean()))
        var_bins.append(float(var_arr[mask].mean()))

    mu_bins_arr = np.asarray(mu_bins, dtype=np.float64)
    var_bins_arr = np.asarray(var_bins, dtype=np.float64)

    # Linear least-squares: v = a*μ + b  →  a=1/N, b=σ_g²
    N_est = THETA_MANUAL['N']
    sigma_g_est = THETA_MANUAL['sigma_g']

    if len(mu_bins_arr) >= 2:
        coeffs = np.polyfit(mu_bins_arr, var_bins_arr, 1)
        a, b = float(coeffs[0]), float(coeffs[1])
        if a > 0:
            N_est = 1.0 / a
        # Clip absurdly large N (numerical issues in very flat scenes)
        N_est = float(np.clip(N_est, 10.0, 1e5))
        sigma_g_est = float(np.sqrt(max(b, 0.0)))

    # Estimate p from all images
    p_estimates = [estimate_sp_density(img) for img in real_images]
    p_est = float(np.mean(p_estimates))

    return dict(
        N=N_est,
        sigma_g=sigma_g_est,
        p=p_est,
        fwhm=THETA_MANUAL['fwhm'],
        alpha=THETA_MANUAL['alpha'],
        in_low=THETA_MANUAL['in_low'],
        in_high=THETA_MANUAL['in_high'],
        out_low=THETA_MANUAL['out_low'],
        out_high=THETA_MANUAL['out_high'],
        ring_strength=THETA_MANUAL['ring_strength'],
        ring_n=THETA_MANUAL['ring_n'],
    )


# ---------------------------------------------------------------------------
# Internal: simple calibration loss (used when CalibrationLoss is unavailable)
# ---------------------------------------------------------------------------

def _default_loss(synth: "torch.Tensor", real: "torch.Tensor"):
    """
    Lightweight multi-component loss used when no dedicated CalibrationLoss
    class is available.

    Components
    ----------
    L_vm   : variance matching  — MSE of local variance maps (8×8 blocks)
    L_nps  : noise power spectrum — L1 of radially-binned FFT magnitudes
    L_rad  : radial mean profile — MSE of radial brightness profiles
    L_hist : histogram matching  — L1 of 64-bin histograms
    """
    import torch
    import torch.nn.functional as F_nn

    eps = 1e-8

    # -- L_hist: 64-bin histogram (soft, via kde with gaussian kernel) ----------
    def soft_hist(x: "torch.Tensor", n_bins: int = 64) -> "torch.Tensor":
        x_flat = x.reshape(-1)
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
        centers = (edges[:-1] + edges[1:]) / 2.0
        bw = 1.0 / n_bins
        # Gaussian KDE
        diffs = x_flat.unsqueeze(1) - centers.unsqueeze(0)   # (N, n_bins)
        weights = torch.exp(-0.5 * (diffs / bw) ** 2)
        hist = weights.sum(0)
        return hist / (hist.sum() + eps)

    h_synth = soft_hist(synth)
    h_real = soft_hist(real)
    L_hist = torch.mean(torch.abs(h_synth - h_real))

    # -- L_vm: variance matching over 8×8 tiles --------------------------------
    def block_var(x: "torch.Tensor", block: int = 8) -> "torch.Tensor":
        H, W = x.shape
        x4d = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        nH, nW = H // block, W // block
        x_crop = x4d[:, :, :nH * block, :nW * block]
        # Unfold into blocks
        unfolded = x_crop.unfold(2, block, block).unfold(3, block, block)
        # shape: (1, 1, nH, nW, block, block)
        var_map = unfolded.var(dim=(-1, -2))  # (1, 1, nH, nW)
        return var_map.squeeze()

    bv_synth = block_var(synth)
    bv_real = block_var(real)
    L_vm = F_nn.mse_loss(bv_synth, bv_real)

    # -- L_nps: radial noise power spectrum ------------------------------------
    def radial_psd(x: "torch.Tensor", n_bins: int = 32) -> "torch.Tensor":
        H, W = x.shape
        # Subtract local mean via average pool to isolate noise
        x4d = x.unsqueeze(0).unsqueeze(0)
        lp = F_nn.avg_pool2d(x4d, kernel_size=8, stride=1, padding=4)[:, :, :H, :W]
        noise = x - lp.squeeze()
        # 2D FFT magnitude
        fft = torch.fft.rfft2(noise)
        mag = torch.abs(fft)
        # Radial binning
        fx = torch.fft.rfftfreq(W, device=x.device)
        fy = torch.fft.fftfreq(H, device=x.device)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(FX ** 2 + FY ** 2)
        r_bins = torch.linspace(0.0, 0.5, n_bins + 1, device=x.device)
        psd = torch.zeros(n_bins, device=x.device, dtype=x.dtype)
        for i in range(n_bins):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if mask.sum() > 0:
                psd[i] = mag[mask].mean()
        return psd / (psd.sum() + eps)

    nps_synth = radial_psd(synth)
    nps_real = radial_psd(real)
    L_nps = torch.mean(torch.abs(nps_synth - nps_real))

    # -- L_rad: radial mean brightness profile ---------------------------------
    def radial_profile(x: "torch.Tensor", n_bins: int = 32) -> "torch.Tensor":
        H, W = x.shape
        cy, cx = H / 2.0, W / 2.0
        y_c = torch.arange(H, dtype=x.dtype, device=x.device)
        x_c = torch.arange(W, dtype=x.dtype, device=x.device)
        yy, xx = torch.meshgrid(y_c, x_c, indexing='ij')
        r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r_max = r.max() + eps
        r_norm = r / r_max
        edges = torch.linspace(0.0, 1.0, n_bins + 1, device=x.device, dtype=x.dtype)
        profile = torch.zeros(n_bins, device=x.device, dtype=x.dtype)
        for i in range(n_bins):
            mask = (r_norm >= edges[i]) & (r_norm < edges[i + 1])
            if mask.sum() > 0:
                profile[i] = x[mask].mean()
        return profile

    rad_synth = radial_profile(synth)
    rad_real = radial_profile(real)
    L_rad = F_nn.mse_loss(rad_synth, rad_real)

    L_total = L_hist + L_vm + L_nps + L_rad
    return L_total, L_vm, L_nps, L_rad, L_hist


# ---------------------------------------------------------------------------
# 3. calibrate
# ---------------------------------------------------------------------------

def calibrate(
    pipeline,
    real_images: List[np.ndarray],
    config: Optional[Dict[str, Any]] = None,
    template_images: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Main Adam calibration loop (spec §2.5).

    Parameters
    ----------
    pipeline         : DegradationPipeline  (nn.Module with _raw parameters)
    real_images      : list of np.ndarray   float32 in [0, 1]
                       Calibration targets for CalibrationLoss (pre-compute statistics).
    config           : dict, optional calibration settings:
                         lr          float   learning rate      (default 1e-2)
                         n_steps     int     max steps          (default 2000)
                         batch_size  int     ε sets per step    (default 32)
                         conv_window int     MA window          (default 200)
                         conv_tol    float   relative tolerance (default 0.001)
                         out_dir     str     output directory   (default '.')
    template_images  : list of np.ndarray, optional
                       Clean template images passed as input to pipeline.
                       If None, uses real_images as pipeline input (backward-compat).

    Returns
    -------
    dict:
        final_params  : dict of physical parameter values
        converged     : bool
        n_steps       : int
        calib_trace   : list of dicts (one per step)
    """
    import torch
    import torch.nn.functional as F_nn

    # Try to import optional dedicated CalibrationLoss
    try:
        from losses import CalibrationLoss  # type: ignore
        _use_calib_loss = True
    except ImportError:
        _use_calib_loss = False

    # ---- Config defaults ----------------------------------------------------
    cfg = dict(
        lr=1e-2,
        n_steps=2000,
        batch_size=32,
        conv_window=200,
        conv_tol=1e-3,
        out_dir='.',
    )
    if config is not None:
        cfg.update(config)

    lr = float(cfg['lr'])
    n_steps = int(cfg['n_steps'])
    batch_size = int(cfg['batch_size'])
    conv_window = int(cfg['conv_window'])
    conv_tol = float(cfg['conv_tol'])
    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Warm-start p (no-gradient parameter) --------------------------------
    ws = warm_start(real_images, pipeline)
    with torch.no_grad():
        pipeline.p.fill_(ws['p'])

    # ---- Convert real images to tensors -------------------------------------
    real_tensors = [
        torch.tensor(np.asarray(img, dtype=np.float32))
        for img in real_images
    ]

    # Template tensors: clean images fed to pipeline as input x
    _tmpl_source = template_images if template_images is not None else real_images
    template_tensors = [
        torch.tensor(np.asarray(img, dtype=np.float32))
        for img in _tmpl_source
    ]

    # ---- Loss function -------------------------------------------------------
    if _use_calib_loss:
        loss_fn = CalibrationLoss(real_images=[np.asarray(img, dtype=np.float32) for img in real_images])
    else:
        loss_fn = None  # use _default_loss

    def compute_loss(synth_img: "torch.Tensor", real_img: "torch.Tensor"):
        if loss_fn is not None:
            # CalibrationLoss.forward takes only synth; real stats are pre-computed
            result = loss_fn(synth_img)
            if isinstance(result, tuple) and len(result) == 2:
                # returns (L_total, {'L_vm': ..., 'L_nps': ..., 'L_rad': ..., 'L_hist': ...})
                L_total, parts = result
                return L_total, parts['L_vm'], parts['L_nps'], parts['L_rad'], parts['L_hist']
            return result, result, result, result, result
        return _default_loss(synth_img, real_img)

    # ---- Optimiser — only parameters with requires_grad=True -----------------
    opt_params = [p for p in pipeline.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # ---- Calibration loop ---------------------------------------------------
    calib_trace: List[Dict[str, Any]] = []
    loss_history: List[float] = []
    converged = False
    actual_steps = n_steps

    def _get_physical_params() -> Dict[str, float]:
        with torch.no_grad():
            return dict(
                N=float(F_nn.softplus(pipeline.N_raw).item()),
                sigma_g=float(F_nn.softplus(pipeline.sigma_g_raw).item()),
                p=float(pipeline.p.item()),
                fwhm=float(F_nn.softplus(pipeline.fwhm_raw).item()),
                alpha=float(F_nn.softplus(pipeline.alpha_raw).item()),
                in_low=float(pipeline.in_low.item()),
                in_high=float(pipeline.in_high.item()),
                out_low=float(pipeline.out_low.item()),
                out_high=float(pipeline.out_high.item()),
                ring_strength=float(F_nn.softplus(pipeline.ring_strength_raw).item()),
                ring_n=float(F_nn.softplus(pipeline.ring_n_raw).item()),
            )

    t0 = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sample a random template (pipeline input) and real image (loss target)
        real_img = real_tensors[step % len(real_tensors)]
        template = template_tensors[step % len(template_tensors)]
        H, W = template.shape

        # Accumulate loss over batch of ε sets
        L_total_acc = torch.tensor(0.0)
        L_vm_acc = torch.tensor(0.0)
        L_nps_acc = torch.tensor(0.0)
        L_rad_acc = torch.tensor(0.0)
        L_hist_acc = torch.tensor(0.0)

        for _ in range(batch_size):
            eps = {
                'eps1': torch.randn(H, W),
                'eps2': torch.randn(H, W),
                'eps3': torch.rand(H, W),
            }
            synth = pipeline(template, eps)
            L_t, L_vm, L_nps, L_rad, L_hist = compute_loss(synth, real_img)
            L_total_acc = L_total_acc + L_t
            L_vm_acc = L_vm_acc + L_vm
            L_nps_acc = L_nps_acc + L_nps
            L_rad_acc = L_rad_acc + L_rad
            L_hist_acc = L_hist_acc + L_hist

        loss_step = L_total_acc / batch_size
        loss_step.backward()
        optimizer.step()

        loss_val = float(loss_step.item())
        loss_history.append(loss_val)

        phys = _get_physical_params()
        row = dict(
            step=step,
            L_total=loss_val,
            L_vm=float((L_vm_acc / batch_size).item()),
            L_nps=float((L_nps_acc / batch_size).item()),
            L_rad=float((L_rad_acc / batch_size).item()),
            L_hist=float((L_hist_acc / batch_size).item()),
            elapsed=time.time() - t0,
            **phys,
        )
        calib_trace.append(row)

        # Convergence check: last conv_window steps loss MA change < conv_tol
        if step >= conv_window:
            recent = loss_history[-conv_window:]
            half = conv_window // 2
            ma_old = float(np.mean(recent[:half]))
            ma_new = float(np.mean(recent[half:]))
            if ma_old > 1e-12:
                rel_change = abs(ma_new - ma_old) / ma_old
                if rel_change < conv_tol:
                    converged = True
                    actual_steps = step + 1
                    break

        if (step + 1) % 100 == 0:
            print(f'[calibrate] step {step+1}/{n_steps}  L={loss_val:.6f}  '
                  f'N={phys["N"]:.1f}  sg={phys["sigma_g"]:.4f}')

    # ---- Save trace ---------------------------------------------------------
    trace_path = out_dir / 'calib_trace.csv'
    if calib_trace:
        fieldnames = list(calib_trace[0].keys())
        with open(trace_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(calib_trace)

    final_params = _get_physical_params()

    return dict(
        final_params=final_params,
        converged=converged,
        n_steps=actual_steps,
        calib_trace=calib_trace,
    )
