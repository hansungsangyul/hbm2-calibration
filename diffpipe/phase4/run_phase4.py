#!/usr/bin/env python3
"""
Phase 4 — Real Image Calibration (RQ2 setup)
=============================================
Protocol (spec §4.1):
  1. Load clean HBM templates as pipeline input (HBM_clean)
  2. Load real degraded CT images as calibration target (HBM_dirty2)
  3. Build CalibrationLoss from real images (pre-compute statistics once)
  4. Calibrate 3 times (different ε seeds) from θ_manual init
  5. Report θ̂_real (mean ± std) → theta_real.json (paper Table 2)
  6. Generate comparison images: θ̂_real synth vs real vs θ_manual synth

Design notes:
  - Same calibration loop as Phase 3b (Adam, 2000 steps, no early convergence)
  - CalibrationLoss statistics built from HBM_dirty2 (real degraded CT scans)
  - Pipeline applied to HBM_clean templates to produce synthetic images
  - Phase 4 only runs AFTER Phase 3 gate PASSES

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python phase4/run_phase4.py 2>&1 | tee phase4/run_phase4.log
"""

from __future__ import annotations

import csv
import json
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
TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = Path(__file__).parent
PHASE4_DIR.mkdir(exist_ok=True)

N_TEMPLATES = 50    # clean templates (pipeline input)
N_REAL      = 50    # real dirty images (CalibrationLoss target)
N_STEPS     = 2000  # optimization steps per run (no early convergence)
BATCH_SIZE  = 8     # ε batches per step
N_REPS      = 3     # calibration repetitions

LR_DEFAULT = 0.01
LR_N       = 1.0   # higher lr for N_raw (same as Phase 3b)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

THETA_MANUAL = dict(
    fwhm=2.5, N=300.0, sigma_g=0.0408, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)

# =====================================================================
# Load images
# =====================================================================
print(f"Loading {N_TEMPLATES} clean templates from {TEMPLATE_DIR} …")
template_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]
assert len(template_paths) >= N_TEMPLATES, f"Need {N_TEMPLATES} images in {TEMPLATE_DIR}"
templates_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                for p in template_paths]
H, W = templates_np[0].shape
templates_t = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"  shape {H}×{W}, device={DEVICE}")

print(f"Loading {N_REAL} real (dirty2) images from {REAL_DIR} …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
assert len(real_paths) >= N_REAL, f"Need {N_REAL} images in {REAL_DIR}"
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
           for p in real_paths]
print(f"  shape {real_np[0].shape[0]}×{real_np[0].shape[1]}")

# Log real image statistics for diagnostic
real_means = [float(img.mean()) for img in real_np]
real_stds  = [float(img.std()) for img in real_np]
print(f"  real stats: mean={np.mean(real_means):.4f}±{np.std(real_means):.4f}  "
      f"std={np.mean(real_stds):.4f}±{np.std(real_stds):.4f}")


# =====================================================================
# Helpers
# =====================================================================

def set_pipeline_to_theta(pipe: DegradationPipeline, theta: dict) -> None:
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


def _seed_all(seed: int) -> None:
    """Seed both CPU and all CUDA RNG streams for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    One calibration run against real image statistics.

    The loss_fn has pre-computed statistics from the real (dirty2) images.
    Templates (clean) are fed to the pipeline to produce synthetic images.

    Returns (final_theta_physical, trace_list)
    """
    set_pipeline_to_theta(pipe, init_theta)

    optimizer = torch.optim.Adam([
        {'params': [pipe.N_raw], 'lr': LR_N},
        {'params': [pipe.sigma_g_raw, pipe.fwhm_raw, pipe.alpha_raw,
                    pipe.ring_strength_raw, pipe.ring_n_raw], 'lr': LR_DEFAULT},
        {'params': [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high],
         'lr': LR_DEFAULT},
    ])

    trace = []
    t0 = time.time()
    _seed_all(rep_seed)

    for step in range(N_STEPS):
        optimizer.zero_grad()

        x = templates[step % len(templates)]
        batch_loss = torch.tensor(0.0, device=DEVICE)
        batch_parts = {'L_vm': 0.0, 'L_nps': 0.0, 'L_rad': 0.0, 'L_hist': 0.0}

        for _ in range(BATCH_SIZE):
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
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time() - t0)
        row.update({k: v / BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step + 1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"N={phys['N']:.1f}  sg={phys['sigma_g']:.4f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}")

    print(f"  [{run_label}] done: 2000 steps")

    if trace:
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = out_dir / 'calib_trace.csv'
        with open(trace_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
            writer.writeheader()
            writer.writerows(trace)
        print(f"  [{run_label}] trace saved → {trace_path}")

    return get_theta_physical(pipe), trace


def generate_synth_sample(pipe: DegradationPipeline, template_np: np.ndarray,
                           seed: int, n_samples: int = 8) -> np.ndarray:
    """
    Generate averaged synthetic image sample for visual comparison.
    Averages n_samples realizations to reduce noise variability.
    """
    x = torch.from_numpy(template_np).to(DEVICE)
    acc = torch.zeros_like(x)
    for i in range(n_samples):
        _seed_all(seed + i)
        eps = {
            'eps1': torch.randn(H, W, device=DEVICE),
            'eps2': torch.randn(H, W, device=DEVICE),
            'eps3': torch.rand(H, W, device=DEVICE),
        }
        with torch.no_grad():
            acc += pipe(x, eps)
    return (acc / n_samples).cpu().numpy()


# =====================================================================
# Main
# =====================================================================
print(f"\n{'='*60}")
print("Phase 4: Real Image Calibration")
print(f"  Templates   : {TEMPLATE_DIR} ({N_TEMPLATES} images)")
print(f"  Real target : {REAL_DIR} ({N_REAL} images)")
print(f"  Init        : θ_manual for all params")
print(f"  Steps       : {N_STEPS} (no early convergence)")
print(f"  Batch       : {BATCH_SIZE} ε/step, {N_REPS} reps")
print(f"{'='*60}")

# Build CalibrationLoss from real dirty2 images (statistics pre-computed once)
print(f"\nBuilding CalibrationLoss from {N_REAL} real images …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  CalibrationLoss built.")

pipe_opt = DegradationPipeline().to(DEVICE)

rep_results = []
t_total = time.time()

for rep in range(N_REPS):
    rep_seed = 900000 + rep * 1000   # Phase 4 seed space (avoid collision with Phase 3)
    rep_dir = PHASE4_DIR / f'rep_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")

    theta_hat, trace = run_one_calibration(
        pipe_opt, loss_fn, templates_t,
        init_theta=THETA_MANUAL.copy(),
        rep_seed=rep_seed,
        out_dir=rep_dir,
        run_label=f"r{rep}",
    )
    rep_results.append(theta_hat)

    print(f"  θ̂_real rep {rep}: N={theta_hat['N']:.1f}  sg={theta_hat['sigma_g']:.4f}  "
          f"fwhm={theta_hat['fwhm']:.3f}  alpha={theta_hat['alpha']:.3f}")

print(f"\nTotal calibration time: {(time.time()-t_total)/60:.1f} min")

# =====================================================================
# Aggregate: θ̂_real = mean ± std across 3 reps
# =====================================================================
PARAMS_TO_REPORT = ['N', 'sigma_g', 'fwhm', 'alpha', 'in_low', 'in_high',
                    'ring_strength', 'ring_n', 'p']

theta_real_mean = {}
theta_real_std  = {}
for param in PARAMS_TO_REPORT:
    vals = [r[param] for r in rep_results]
    theta_real_mean[param] = float(np.mean(vals))
    theta_real_std[param]  = float(np.std(vals))

print(f"\n{'='*60}")
print("θ̂_real vs θ_manual (paper Table 2):")
print(f"{'Param':<16} {'θ_manual':>12} {'θ̂_real mean':>14} {'θ̂_real std':>12} {'rel diff':>10}")
print('-'*68)
for param in PARAMS_TO_REPORT:
    man = THETA_MANUAL.get(param, float('nan'))
    hat = theta_real_mean[param]
    std = theta_real_std[param]
    rel = abs(hat - man) / abs(man) if abs(man) > 1e-12 else float('nan')
    print(f"  {param:<14} {man:>12.4f} {hat:>14.4f} {std:>12.4f} {rel:>10.3f}")

# =====================================================================
# Save theta_real.json
# =====================================================================
theta_real_output = {
    'theta_real_mean': theta_real_mean,
    'theta_real_std':  theta_real_std,
    'theta_hat_all':   rep_results,
    'theta_manual':    THETA_MANUAL,
    'config': {
        'template_dir': str(TEMPLATE_DIR),
        'real_dir':     str(REAL_DIR),
        'n_templates':  N_TEMPLATES,
        'n_real':       N_REAL,
        'n_steps':      N_STEPS,
        'batch_size':   BATCH_SIZE,
        'n_reps':       N_REPS,
        'lr_default':   LR_DEFAULT,
        'lr_N':         LR_N,
        'device':       DEVICE,
    }
}
theta_real_path = PHASE4_DIR / 'theta_real.json'
with open(theta_real_path, 'w') as f:
    json.dump(theta_real_output, f, indent=2)
print(f"\nSaved: {theta_real_path}")

# =====================================================================
# Generate visual comparison images (spec §4.1 qualitative comparison)
# =====================================================================
print("\nGenerating comparison images …")
comparison_dir = PHASE4_DIR / 'comparison_images'
comparison_dir.mkdir(exist_ok=True)

# Use first 3 templates for comparison
pipe_manual = DegradationPipeline().to(DEVICE)
set_pipeline_to_theta(pipe_manual, THETA_MANUAL)
set_pipeline_to_theta(pipe_opt, theta_real_mean)

for i in range(min(3, len(templates_np))):
    tmpl = templates_np[i]
    real_img = real_np[i]

    synth_manual = generate_synth_sample(pipe_manual, tmpl, seed=999000 + i)
    synth_real   = generate_synth_sample(pipe_opt,   tmpl, seed=999100 + i)

    def to_uint8(arr):
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    cv2.imwrite(str(comparison_dir / f'img{i:02d}_template.png'),  to_uint8(tmpl))
    cv2.imwrite(str(comparison_dir / f'img{i:02d}_real_dirty2.png'), to_uint8(real_img))
    cv2.imwrite(str(comparison_dir / f'img{i:02d}_synth_manual.png'), to_uint8(synth_manual))
    cv2.imwrite(str(comparison_dir / f'img{i:02d}_synth_real.png'),   to_uint8(synth_real))
    print(f"  Comparison images {i:02d} saved.")

print(f"\nPhase 4 complete. Results in {PHASE4_DIR}")
print(f"  theta_real.json  : {theta_real_path}")
print(f"  comparison_images: {comparison_dir}")
