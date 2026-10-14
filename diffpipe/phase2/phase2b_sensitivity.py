#!/usr/bin/env python3
"""
Phase 2b Sensitivity Matrix Re-run
===================================
Fixes applied to losses.py:
  - L_vm: flat-patch filter (50th-pct gradient threshold) so within-patch
           variance reflects Poisson+Gaussian noise, not image structure.
  - L_rad: normalized by mean before MSE so the loss is insensitive to
           contrast scale (in_high) and captures only vignetting shape.

Perturbation protocol changes vs Phase 2:
  - Contrast params (in_low, in_high): absolute ±0.05 (not ±20%) to avoid
    in_high > 1.0 clipping which was the main confound in the prior run.
  - All other params: ±20% in physical space (same as Phase 2).

Sensitivity metric: |L(θ+) - L(θ-)| / (Δθ_frac × L_base)
  where Δθ_frac = 0.40 for ±20%, or absolute delta for contrast.

Gate criteria (updated):
  - L_vm: N or sigma_g must be rank 1 or 2
  - L_nps: fwhm must be rank 1
  - L_rad: alpha must be rank 1 or 2
  - L_hist: any contrast param (in_low, in_high, out_low, out_high) must be rank 1

Usage:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python phase2/phase2b_sensitivity.py
"""

import sys
import os
import json
import math
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import cv2
from pipeline import DegradationPipeline
from losses import CalibrationLoss

REAL_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
N_REAL = 20   # images for loss precomputation

print("Loading real images …")
img_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
assert len(img_paths) >= N_REAL, f"Need {N_REAL} images in {REAL_DIR}"
real_images = []
for p in img_paths:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Could not read {p}"
    real_images.append(img.astype(np.float32) / 255.0)
H, W = real_images[0].shape
print(f"  {len(real_images)} images, shape {H}×{W}")

print("Building CalibrationLoss …")
loss_fn = CalibrationLoss(real_images)
print("  done.")

pipeline = DegradationPipeline()
pipeline.eval()

# Use a real image as "clean" input for the pipeline.
# This gives realistic structure so that flat patches in real images
# correspond to flat regions in the synthetic output too.
x_input = torch.from_numpy(real_images[0])

torch.manual_seed(42)
eps_fixed = {
    'eps1': torch.randn(H, W),
    'eps2': torch.randn(H, W),
    'eps3': torch.rand(H, W),
}


def run_loss(pipe, x, eps):
    with torch.no_grad():
        syn = pipe(x, eps, max_stage=7)
    _, parts = loss_fn(syn)
    return {k: v.item() for k, v in parts.items()}


def softplus_inv(phys):
    if phys > 20.0:
        return phys
    return math.log(math.expm1(phys))


# Parameter specs: (attr_name, is_softplus_raw, is_contrast_abs)
PARAMS = {
    'N':            ('N_raw',            True,  False),
    'sigma_g':      ('sigma_g_raw',      True,  False),
    'p':            ('p',                False, False),
    'fwhm':         ('fwhm_raw',         True,  False),
    'alpha':        ('alpha_raw',        True,  False),
    'in_low':       ('in_low',           False, True),
    'in_high':      ('in_high',          False, True),
    'ring_strength':('ring_strength_raw',True,  False),
}

CONTRAST_ABS_DELTA = 0.05   # absolute perturbation for contrast params
REL_DELTA = 0.20            # ±20% for other params
CONTRAST_BOUNDS = {
    'in_low':  (0.01, 0.85),
    'in_high': (0.10, 0.98),
}

LOSS_NAMES = ['L_vm', 'L_nps', 'L_rad', 'L_hist']


def perturb_run(pname, sign, x, eps):
    """Return loss dict after perturbing pname by ±delta."""
    attr, is_raw, is_contrast = PARAMS[pname]
    with torch.no_grad():
        orig = getattr(pipeline, attr).data.clone()

    if is_contrast:
        val = orig.item()
        new_val = val + sign * CONTRAST_ABS_DELTA
        lo, hi = CONTRAST_BOUNDS.get(pname, (0.0, 1.0))
        new_val = float(np.clip(new_val, lo, hi))
        getattr(pipeline, attr).data.fill_(new_val)
    else:
        if is_raw:
            phys = float(F.softplus(orig))
            phys_new = phys * (1.0 + sign * REL_DELTA)
            phys_new = max(phys_new, 1e-6)
            raw_new = softplus_inv(phys_new)
            getattr(pipeline, attr).data.fill_(raw_new)
        else:
            val = orig.item()
            new_val = val * (1.0 + sign * REL_DELTA)
            getattr(pipeline, attr).data.fill_(float(np.clip(new_val, 1e-9, None)))

    result = run_loss(pipeline, x, eps)
    getattr(pipeline, attr).data.copy_(orig)
    return result


print("\nBaseline losses …")
L_base = run_loss(pipeline, x_input, eps_fixed)
print(f"  {L_base}")

print("\nComputing sensitivity matrix …")
sensitivity = {}
for pname in PARAMS:
    L_plus  = perturb_run(pname, +1.0, x_input, eps_fixed)
    L_minus = perturb_run(pname, -1.0, x_input, eps_fixed)
    row = {}
    for lname in LOSS_NAMES:
        lb = max(L_base[lname], 1e-12)
        delta = abs(L_plus[lname] - L_minus[lname])
        if PARAMS[pname][2]:  # contrast: absolute perturbation
            # Normalize by L_base (delta / lb gives dimensionless sensitivity)
            row[lname] = round(delta / lb, 4)
        else:
            row[lname] = round(delta / (2 * REL_DELTA * lb), 4)
    sensitivity[pname] = row
    print(f"  {pname:20s}: {row}")

# ---- Print matrix ----
print("\n=== Sensitivity Matrix ===")
header = f"{'parameter':20s}" + "".join(f"  {l:>10s}" for l in LOSS_NAMES)
print(header)
for pname, row in sensitivity.items():
    vals = "".join(f"  {row[l]:10.4f}" for l in LOSS_NAMES)
    print(f"  {pname:20s}{vals}")

# ---- Diagonal dominance check (updated criterion) ----
NOISE_PARAMS   = {'N', 'sigma_g'}
BLUR_PARAMS    = {'fwhm'}
VIG_PARAMS     = {'alpha'}
CONTRAST_PARAMS = {'in_low', 'in_high', 'out_low', 'out_high'}

checks = {}

# L_vm: N or sigma_g in top-2
col_vm = {p: sensitivity[p]['L_vm'] for p in sensitivity}
rank_vm = sorted(col_vm, key=col_vm.get, reverse=True)
top2_vm = set(rank_vm[:2])
checks['L_vm'] = {
    'target':   'N or sigma_g in top-2',
    'top2':     rank_vm[:2],
    'pass':     bool(top2_vm & NOISE_PARAMS),
}

# L_nps: fwhm rank-1
col_nps = {p: sensitivity[p]['L_nps'] for p in sensitivity}
rank_nps = sorted(col_nps, key=col_nps.get, reverse=True)
checks['L_nps'] = {
    'target':   'fwhm rank-1',
    'top2':     rank_nps[:2],
    'pass':     rank_nps[0] == 'fwhm',
}

# L_rad: alpha in top-2
col_rad = {p: sensitivity[p]['L_rad'] for p in sensitivity}
rank_rad = sorted(col_rad, key=col_rad.get, reverse=True)
top2_rad = set(rank_rad[:2])
checks['L_rad'] = {
    'target':   'alpha in top-2',
    'top2':     rank_rad[:2],
    'pass':     'alpha' in top2_rad,
}

# L_hist: any contrast param in top-1
col_hist = {p: sensitivity[p]['L_hist'] for p in sensitivity}
rank_hist = sorted(col_hist, key=col_hist.get, reverse=True)
checks['L_hist'] = {
    'target':   'any contrast param rank-1',
    'top2':     rank_hist[:2],
    'pass':     rank_hist[0] in CONTRAST_PARAMS,
}

print("\n=== Diagonal Dominance Check ===")
all_pass = True
for lname, chk in checks.items():
    status = "PASS" if chk['pass'] else "FAIL"
    if not chk['pass']:
        all_pass = False
    print(f"  {lname}: {status} | target={chk['target']} | top-2={chk['top2']}")

gate = "PASS" if all_pass else "FAIL"
print(f"\n>>> Phase 2b Gate: {gate}")

# ---- Save results ----
OUT = ROOT / 'phase2'

with open(OUT / 'sensitivity_matrix_v2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['parameter'] + LOSS_NAMES)
    for pname, row in sensitivity.items():
        writer.writerow([pname] + [row[l] for l in LOSS_NAMES])

result_json = {
    'baseline_losses': L_base,
    'sensitivity': sensitivity,
    'param_order': list(PARAMS.keys()),
    'loss_order': LOSS_NAMES,
    'diagonal_dominance': checks,
    'phase2b_gate': gate,
    'fixes_applied': [
        'L_vm: flat-patch filter (50th-pct gradient threshold)',
        'L_rad: normalized by mean (contrast-invariant)',
        'contrast perturbation: absolute ±0.05 (was ×1.2/×0.8)',
    ],
}
with open(OUT / 'sensitivity_summary_v2.json', 'w') as f:
    json.dump(result_json, f, indent=2)

print(f"\nSaved: {OUT}/sensitivity_matrix_v2.csv")
print(f"Saved: {OUT}/sensitivity_summary_v2.json")
