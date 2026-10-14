#!/usr/bin/env python3
"""
Phase 3b — Self-Recovery Experiment (RQ1, no-early-convergence re-run)
=========================================
Protocol:
  1. 10 pre-registered GT parameter sets (gt_params.json, committed before run)
  2. For each θ*: generate N_FAKE synthetic "fake-real" images
  3. Build CalibrationLoss from these fake images
  4. Calibrate 3 times (different ε seeds) from θ_manual init
  5. Evaluate |θ̂ - θ*| / θ*

Key design choices vs spec §2.5:
  - Adam with parameter groups: N_raw uses lr=1.0 (effective step ~0.28/step;
    covers max |N*-N_init|=233 within ~833 steps of 2000-step budget)
  - All params initialized from θ_manual (legitimate §2.4 "prior knowledge" warm start,
    no GT leakage; PSF-corrected warm start is logged but not used for init because
    fake images pass through vignetting+contrast stages causing a ~14× N bias)
  - 2000 steps
  - batch_size=8 ε samples per step

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python phase3/run_phase3b.py 2>&1 | tee phase3/run_phase3b.log
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse
from losses import CalibrationLoss

# =====================================================================
# Config
# =====================================================================
REAL_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
PHASE3_DIR = Path(__file__).parent / 'phase3b'
PHASE3_DIR.mkdir(exist_ok=True)

N_TEMPLATES = 50    # clean templates for both fake-gen and calibration
N_STEPS     = 2000  # optimization steps per run
BATCH_SIZE  = 8     # ε batches per step
N_REPS      = 3     # calibration repetitions per GT set
LR_DEFAULT  = 0.01  # learning rate for most params
LR_N        = 1.0   # learning rate for N_raw; effective step ~lr/sqrt(grad_var_ratio) ≈ 0.28/step,
                    # covers max |N*-N_init|=233 in ~833 steps within 2000-step budget
CONV_WINDOW = 200   # convergence moving-average window
CONV_TOL    = 1e-3  # relative change tolerance

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

THETA_MANUAL = dict(fwhm=2.5, N=300.0, sigma_g=0.0408, p=0.001, alpha=0.35,
                    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
                    ring_strength=0.12, ring_n=8.0)

# =====================================================================
# Load templates
# =====================================================================
print(f"Loading {N_TEMPLATES} templates …")
img_paths = sorted(REAL_DIR.glob('*.png'))[:N_TEMPLATES]
assert len(img_paths) >= N_TEMPLATES, f"Need {N_TEMPLATES} images in {REAL_DIR}"
templates_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                for p in img_paths]
H, W = templates_np[0].shape
templates_t = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"  shape {H}×{W}, device={DEVICE}")


# =====================================================================
# Helpers
# =====================================================================

def set_pipeline_to_theta(pipe: DegradationPipeline, theta: dict, device='cpu') -> None:
    """Set all pipeline parameters to physical θ values."""
    with torch.no_grad():
        pipe.N_raw.fill_(softplus_inverse(theta['N']))
        pipe.sigma_g_raw.fill_(softplus_inverse(theta['sigma_g']))
        pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta.get('ring_strength', 0.12)))
        pipe.ring_n_raw.fill_(softplus_inverse(theta.get('ring_n', 8.0)))
        pipe.in_low.fill_(theta.get('in_low', 0.05))
        pipe.in_high.fill_(theta.get('in_high', 0.90))
        pipe.out_low.fill_(theta.get('out_low', 0.10))
        pipe.out_high.fill_(theta.get('out_high', 0.90))
        pipe.p.fill_(theta.get('p', 0.001))


def get_theta_physical(pipe: DegradationPipeline) -> dict:
    """Extract physical parameter values from pipeline."""
    with torch.no_grad():
        return dict(
            N=float(F.softplus(pipe.N_raw).item()),
            sigma_g=float(F.softplus(pipe.sigma_g_raw).item()),
            p=float(pipe.p.item()),
            fwhm=float(F.softplus(pipe.fwhm_raw).item()),
            alpha=float(F.softplus(pipe.alpha_raw).item()),
            in_low=float(pipe.in_low.item()),
            in_high=float(pipe.in_high.item()),
            out_low=float(pipe.out_low.item()),
            out_high=float(pipe.out_high.item()),
            ring_strength=float(F.softplus(pipe.ring_strength_raw).item()),
            ring_n=float(F.softplus(pipe.ring_n_raw).item()),
        )


def compute_K_PSF(fwhm: float) -> float:
    """Compute L2-norm squared of Gaussian PSF kernel (Parseval factor)."""
    sigma = fwhm / 2.355
    k = int(2 * round(3 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    ax = np.arange(k, dtype=np.float64) - k // 2
    g1 = np.exp(-ax ** 2 / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    g2d = np.outer(g1, g1)
    return float(np.sum(g2d ** 2))


def warm_start_psf_corrected(images_np: list, fwhm_init: float = 2.5) -> dict:
    """
    PSF-corrected warm start for N and sigma_g.

    Uses flat patches (low gradient) and accounts for the K_PSF reduction
    factor: var_observed ≈ (μ/N + σ_g²) × K_PSF in flat patches.
    Applying the correction gives unbiased estimates.
    """
    K_PSF = compute_K_PSF(fwhm_init)

    patch_size = 8
    flat_mus, flat_vars = [], []

    for img in images_np:
        img_np = np.asarray(img, np.float32)
        HH, WW = img_np.shape
        gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx ** 2 + gy ** 2)

        nHH, nWW = HH // patch_size, WW // patch_size
        per_img_mus, per_img_vars, per_img_gm = [], [], []
        for r in range(nHH):
            for c in range(nWW):
                patch = img_np[r * patch_size:(r+1) * patch_size,
                               c * patch_size:(c+1) * patch_size]
                gm_patch = gm[r * patch_size:(r+1) * patch_size,
                              c * patch_size:(c+1) * patch_size]
                per_img_mus.append(float(patch.mean()))
                per_img_vars.append(float(patch.var()))
                per_img_gm.append(float(gm_patch.mean()))

        per_img_gm_arr = np.array(per_img_gm)
        thr = np.percentile(per_img_gm_arr, 50)
        flat_mask = per_img_gm_arr < thr

        flat_mus.extend(np.array(per_img_mus)[flat_mask].tolist())
        flat_vars.extend(np.array(per_img_vars)[flat_mask].tolist())

    flat_mus_arr = np.array(flat_mus, dtype=np.float64)
    flat_vars_arr = np.array(flat_vars, dtype=np.float64)

    # Bin and fit v = K_PSF * (μ/N + σ_g²)
    n_bins = 16
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(flat_mus_arr, bin_edges) - 1, 0, n_bins - 1)

    mu_b, var_b = [], []
    for b in range(n_bins):
        mask_b = bin_idx == b
        if mask_b.sum() >= 5:
            mu_b.append(float(flat_mus_arr[mask_b].mean()))
            var_b.append(float(flat_vars_arr[mask_b].mean()))

    N_est = THETA_MANUAL['N']
    sg_est = THETA_MANUAL['sigma_g']

    if len(mu_b) >= 2:
        mu_b_arr = np.array(mu_b, dtype=np.float64)
        var_b_arr = np.array(var_b, dtype=np.float64)
        coeffs = np.polyfit(mu_b_arr, var_b_arr, 1)
        a_obs, b_obs = float(coeffs[0]), float(coeffs[1])
        # a_obs ≈ K_PSF / N*  →  N* ≈ K_PSF / a_obs
        # b_obs ≈ K_PSF × σ_g*²  →  σ_g* ≈ sqrt(b_obs / K_PSF)
        if a_obs > 1e-12:
            N_est = float(np.clip(K_PSF / a_obs, 10.0, 1e5))
        sg_est = float(np.clip(np.sqrt(max(b_obs / K_PSF, 0.0)), 0.001, 0.5))

    return dict(N=N_est, sigma_g=sg_est)


def _seed_all(seed: int) -> None:
    """Seed both CPU and all CUDA RNG streams for full reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_fake_real(pipe_gt: DegradationPipeline, theta_star: dict,
                       templates: list, seed_offset: int = 100000) -> list:
    """Generate N_FAKE fake-real images using θ* (different ε from calibration)."""
    set_pipeline_to_theta(pipe_gt, theta_star)
    fake_real_np = []
    for i, x in enumerate(templates):
        _seed_all(seed_offset + i)
        eps = {
            'eps1': torch.randn_like(x),
            'eps2': torch.randn_like(x),
            'eps3': torch.rand_like(x),
        }
        with torch.no_grad():
            y = pipe_gt(x, eps)
        fake_real_np.append(y.cpu().numpy())
    return fake_real_np


def run_one_calibration(
    pipe: DegradationPipeline,
    loss_fn: CalibrationLoss,
    templates: list,
    init_theta: dict,
    rep_seed: int,
    out_dir: Path,
    run_label: str,
) -> tuple[dict, list]:
    """
    One calibration run.

    Args:
        init_theta: starting parameter values (warm_start result, NOT θ*)
        rep_seed:   master seed for this repetition's ε generation
        out_dir:    directory for calib_trace.csv
        run_label:  string label for logging

    Returns:
        (final_theta, trace)
    """
    set_pipeline_to_theta(pipe, init_theta)

    # Parameter groups with different learning rates
    optimizer = torch.optim.Adam([
        {'params': [pipe.N_raw], 'lr': LR_N},
        {'params': [pipe.sigma_g_raw, pipe.fwhm_raw, pipe.alpha_raw,
                    pipe.ring_strength_raw, pipe.ring_n_raw], 'lr': LR_DEFAULT},
        {'params': [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high],
         'lr': LR_DEFAULT},
    ])

    trace = []
    loss_history = []
    converged = False
    actual_steps = N_STEPS

    t0 = time.time()
    _seed_all(rep_seed)

    for step in range(N_STEPS):
        optimizer.zero_grad()

        x = templates[step % len(templates)]
        batch_loss = torch.tensor(0.0, device=DEVICE)
        batch_parts = {'L_vm': 0.0, 'L_nps': 0.0, 'L_rad': 0.0, 'L_hist': 0.0}

        for b in range(BATCH_SIZE):
            eps = {
                'eps1': torch.randn(H, W, device=DEVICE),
                'eps2': torch.randn(H, W, device=DEVICE),
                'eps3': torch.rand(H, W, device=DEVICE),
            }
            syn = pipe(x, eps)
            L, parts = loss_fn(syn)
            batch_loss = batch_loss + L
            for k in batch_parts:
                batch_parts[k] += parts[k].item()

        (batch_loss / BATCH_SIZE).backward()
        optimizer.step()

        loss_val = (batch_loss / BATCH_SIZE).item()
        loss_history.append(loss_val)

        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val,
                   elapsed=time.time() - t0)
        row.update({k: v / BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        # No early convergence — always run N_STEPS.
        # Phase 3a showed that CONV_TOL=1e-3 on loss triggers when loss plateaus but
        # parameters (esp. sigma_g) are still slowly moving toward the true minimum.
        # Running the full 2000 steps allows sigma_g to reach lower GT values.

        if (step + 1) % 200 == 0:
            print(f"    [{run_label}] step {step+1}  L={loss_val:.5f}  "
                  f"N={phys['N']:.1f}  sg={phys['sigma_g']:.4f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}")

    print(f"    [{run_label}] done: converged={converged}, steps={actual_steps}")

    # Save trace
    if trace:
        trace_path = out_dir / f'calib_trace.csv'
        with open(trace_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)

    return get_theta_physical(pipe), trace


def rel_error(hat: float, star: float) -> float:
    if abs(star) < 1e-12:
        return float('inf')
    return abs(hat - star) / abs(star)


# =====================================================================
# Main
# =====================================================================

# Load pre-registered GT sets
gt_params_path = Path(__file__).parent / 'gt_params.json'
with open(gt_params_path) as f:
    gt_data = json.load(f)
gt_sets = gt_data['gt_sets']
print(f"Loaded {len(gt_sets)} GT sets from {gt_params_path}")

pipe_gt  = DegradationPipeline().to(DEVICE)
pipe_opt = DegradationPipeline().to(DEVICE)

RECOVERY_PARAMS = ['N', 'sigma_g', 'fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength']

all_results = []

t_total = time.time()

for gt_set in gt_sets:
    set_id = gt_set['set_id']
    theta_star = {k: v for k, v in gt_set.items() if k != 'set_id'}
    print(f"\n{'='*60}")
    print(f"GT set {set_id}: N={theta_star['N']:.0f}, sg={theta_star['sigma_g']:.4f}, "
          f"fwhm={theta_star['fwhm']:.2f}, alpha={theta_star['alpha']:.3f}")

    set_dir = PHASE3_DIR / f'set_{set_id:02d}'
    set_dir.mkdir(exist_ok=True)

    # ---- Step 1: Generate fake-real images using θ* ----
    print(f"  Generating {N_TEMPLATES} fake-real images …")
    fake_real_np = generate_fake_real(pipe_gt, theta_star, templates_t, seed_offset=100000)

    # ---- Step 2: Build CalibrationLoss from fake-real images ----
    print("  Building CalibrationLoss …")
    loss_fn = CalibrationLoss(fake_real_np).to(DEVICE)

    # ---- Step 3: Warm start (PSF-corrected, using fake-real images) ----
    ws = warm_start_psf_corrected(fake_real_np, fwhm_init=THETA_MANUAL['fwhm'])
    init_theta = THETA_MANUAL.copy()
    # Warm-start for N and sigma_g is biased because fake images pass through
    # vignetting and contrast stages.  Use θ_manual for all parameters as the
    # calibration starting point (legitimate §2.4 "prior knowledge" init, no GT leakage).
    # Warm-start estimates are logged for diagnostic purposes only.
    print(f"  Init: θ_manual for all params.  WS diag: N_ws={ws['N']:.1f} (GT={theta_star['N']:.0f}), "
          f"sg_ws={ws['sigma_g']:.4f} (GT={theta_star['sigma_g']:.4f})")

    # ---- Step 4: 3 calibration repetitions ----
    rep_results = []
    for rep in range(N_REPS):
        rep_seed = set_id * 10000 + rep * 1000
        rep_dir = set_dir / f'rep_{rep}'
        rep_dir.mkdir(exist_ok=True)
        print(f"\n  [Rep {rep}] seed={rep_seed}")

        theta_hat, trace = run_one_calibration(
            pipe_opt, loss_fn, templates_t, init_theta,
            rep_seed=rep_seed,
            out_dir=rep_dir,
            run_label=f"set{set_id}r{rep}",
        )
        rep_results.append(theta_hat)

        # Per-param relative error
        errs = {p: rel_error(theta_hat[p], theta_star[p]) for p in RECOVERY_PARAMS}
        print(f"    Relative errors: " + "  ".join(f"{p}={errs[p]:.3f}" for p in RECOVERY_PARAMS))

    # ---- Step 5: Aggregate ----
    means = {}
    stds = {}
    for param in RECOVERY_PARAMS:
        vals = [r[param] for r in rep_results]
        means[param] = float(np.mean(vals))
        stds[param] = float(np.std(vals))

    rel_errors = {p: rel_error(means[p], theta_star[p]) for p in RECOVERY_PARAMS}

    print(f"\n  SUMMARY set {set_id}:")
    for p in RECOVERY_PARAMS:
        print(f"    {p}: GT={theta_star[p]:.4f} | mean={means[p]:.4f} | "
              f"std={stds[p]:.4f} | relerr={rel_errors[p]:.3f}")

    set_result = {
        'set_id': set_id,
        'theta_star': theta_star,
        'theta_hat_mean': means,
        'theta_hat_std': stds,
        'theta_hat_all': rep_results,
        'rel_error': rel_errors,
        'warm_start': {'N_start': init_theta['N'], 'sigma_g_start': init_theta['sigma_g'],
                       'ws_N_diagnostic': ws['N'], 'ws_sg_diagnostic': ws['sigma_g']},
    }
    all_results.append(set_result)

    # Save intermediate result
    with open(set_dir / 'set_result.json', 'w') as f:
        json.dump(set_result, f, indent=2)

print(f"\n{'='*60}")
print(f"Total time: {(time.time() - t_total)/60:.1f} min")

# =====================================================================
# Final evaluation: check success criteria (spec §3.3)
# =====================================================================
print("\n=== Phase 3 Success Criteria Check ===")

sigma_g_errors = [r['rel_error']['sigma_g'] for r in all_results]
fwhm_errors    = [r['rel_error']['fwhm'] for r in all_results]
N_errors       = [r['rel_error']['N'] for r in all_results]
alpha_errors   = [r['rel_error']['alpha'] for r in all_results]
inlow_errors   = [r['rel_error']['in_low'] for r in all_results]
inhigh_errors  = [r['rel_error']['in_high'] for r in all_results]
ring_errors    = [r['rel_error']['ring_strength'] for r in all_results]

print(f"sigma_g median relerr: {np.median(sigma_g_errors):.3f}  (spec: < 0.10)")
print(f"fwhm    median relerr: {np.median(fwhm_errors):.3f}  (spec: < 0.10)")
print(f"N       median relerr: {np.median(N_errors):.3f}  (spec: < 0.20)")
print(f"alpha   median relerr: {np.median(alpha_errors):.3f}  (spec: < 0.25)")
print(f"in_low  median relerr: {np.median(inlow_errors):.3f}  (spec: < 0.25 or func equiv)")
print(f"in_high median relerr: {np.median(inhigh_errors):.3f}  (spec: < 0.25 or func equiv)")
print(f"ring_str median relerr: {np.median(ring_errors):.3f}  (spec: func equiv only)")

sg_pass    = np.median(sigma_g_errors) < 0.10
fwhm_pass  = np.median(fwhm_errors) < 0.10
N_pass     = np.median(N_errors) < 0.20
alpha_pass = np.median(alpha_errors) < 0.25

gate_status = 'PASS' if (sg_pass and fwhm_pass) else 'FAIL'
print(f"\n>>> Phase 3 Gate (sigma_g + fwhm): {gate_status}")

# Save recovery_results.json
recovery_output = {
    'gate': gate_status,
    'results': all_results,
    'summary': {
        'sigma_g_median_relerr': float(np.median(sigma_g_errors)),
        'fwhm_median_relerr':    float(np.median(fwhm_errors)),
        'N_median_relerr':       float(np.median(N_errors)),
        'alpha_median_relerr':   float(np.median(alpha_errors)),
        'ring_median_relerr':    float(np.median(ring_errors)),
    },
    'criteria': {
        'sigma_g_pass': bool(sg_pass),
        'fwhm_pass':    bool(fwhm_pass),
        'N_pass':       bool(N_pass),
        'alpha_pass':   bool(alpha_pass),
    },
    'config': {
        'n_templates': N_TEMPLATES,
        'n_steps': N_STEPS,
        'batch_size': BATCH_SIZE,
        'n_reps': N_REPS,
        'lr_default': LR_DEFAULT,
        'lr_N': LR_N,
        'device': DEVICE,
    }
}
with open(PHASE3_DIR / 'recovery_results.json', 'w') as f:
    json.dump(recovery_output, f, indent=2)
print(f"\nSaved: {PHASE3_DIR}/recovery_results.json")
