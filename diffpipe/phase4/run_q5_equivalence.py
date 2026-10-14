#!/usr/bin/env python3
"""
지시 25 Q5 — out_low/out_high 등가성 검증
==========================================
v5 θ̂ 전체(out_low≈0.2515, out_high≈1.413) vs out_*를 manual(0.10, 0.90)로 복원.
두 설정으로 이미지 생성 후 4개 통계량 거리 → §M 기준선(1.5 × d_realization) 비교.

d_realization (set0) = 0.01382  →  1.5 × d_realization = 0.02073

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_q5_equivalence.py

Output: phase4/q5_equivalence.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

# ── Paths ────────────────────────────────────────────────────────────
CLEAN_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4    = ROOT / 'phase4'

PRENORM_A = 0.501834
PRENORM_B = 0.173320
N_IMGS    = 50        # template images for generation
N_REAL    = 50        # real images for CalibrationLoss reference
N_EVAL    = 100       # synthetic images to evaluate per config
SEED_EVAL = 980000

D_REALIZATION = 0.01382237  # phase3/d_realization_details.json set 0
THRESHOLD     = 1.5 * D_REALIZATION  # §M criterion: 0.02073

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── v5 θ̂ mean (theta_real_v5.json, out_* from theta_hat_main mean) ─
V5_THETA = dict(
    N=300.0, sigma_g=0.0408, p=0.001,
    fwhm=1.4248, alpha=0.4973, in_low=0.0039, in_high=0.7021,
    out_low=0.2515,   # mean of rep0/1/2: (0.2519+0.2511+0.2514)/3
    out_high=1.4133,  # mean of rep0/1/2: (1.4127+1.4139+1.4132)/3
    ring_strength=0.1012, ring_n=7.987,
)

# ── manual out_* restoration ─────────────────────────────────────────
V5_MANUAL_OUT = dict(V5_THETA)
V5_MANUAL_OUT['out_low']  = 0.10
V5_MANUAL_OUT['out_high'] = 0.90


def build_pipe(theta: dict) -> DegradationPipeline:
    pipe = DegradationPipeline().to(DEVICE)
    with torch.no_grad():
        pipe.N_raw.fill_(sigmoid_box_inverse(theta['N']))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], 1e-4)))
        pipe.p.fill_(theta['p'])
        pipe.fwhm_raw.fill_(softplus_inverse(max(theta['fwhm'], 1e-4)))
        pipe.alpha_raw.fill_(softplus_inverse(max(theta['alpha'], 1e-4)))
        pipe.in_low.fill_(theta['in_low'])
        pipe.in_high.fill_(theta['in_high'])
        pipe.out_low.fill_(theta['out_low'])
        pipe.out_high.fill_(theta['out_high'])
        pipe.ring_strength_raw.fill_(softplus_inverse(max(theta['ring_strength'], 1e-4)))
        pipe.ring_n_raw.fill_(softplus_inverse(max(theta['ring_n'], 1e-4)))
    pipe.eval()
    return pipe


def gen_images(pipe: DegradationPipeline, templates_t: list, n: int, seed: int) -> torch.Tensor:
    """Generate n synthetic images, return [n, H, W] float32 cpu tensor."""
    torch.manual_seed(seed)
    H, W = templates_t[0].shape
    imgs = []
    with torch.no_grad():
        for i in range(n):
            t = templates_t[i % len(templates_t)]
            eps = {'eps1': torch.randn(H, W, device=DEVICE),
                   'eps2': torch.randn(H, W, device=DEVICE),
                   'eps3': torch.rand(H, W, device=DEVICE),
                   'eps4': torch.rand(H, W, device=DEVICE)}
            y = pipe(t, **eps)
            imgs.append(y.cpu())
    return torch.stack(imgs)


def compute_nps_1d(imgs: torch.Tensor) -> np.ndarray:
    """Mean radially-averaged NPS from [n, H, W] images."""
    n, H, W = imgs.shape
    npss = []
    for i in range(n):
        img_np = imgs[i].numpy()
        f = np.fft.fft2(img_np - img_np.mean())
        psd = np.abs(np.fft.fftshift(f))**2 / (H * W)
        cy, cx = H // 2, W // 2
        y_idx, x_idx = np.indices((H, W))
        r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(int)
        max_r = min(cx, cy)
        profile = np.zeros(max_r)
        for ri in range(max_r):
            mask = r == ri
            if mask.sum() > 0:
                profile[ri] = psd[mask].mean()
        npss.append(profile)
    return np.array(npss).mean(axis=0)


def compute_radial_profile(imgs: torch.Tensor) -> np.ndarray:
    """Mean radial intensity profile (center-to-edge)."""
    n, H, W = imgs.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(int)
    max_r = min(cx, cy)
    profiles = []
    for i in range(n):
        img_np = imgs[i].numpy()
        profile = np.zeros(max_r)
        for ri in range(max_r):
            mask = r == ri
            if mask.sum() > 0:
                profile[ri] = img_np[mask].mean()
        profiles.append(profile)
    return np.array(profiles).mean(axis=0)


def compute_histogram(imgs: torch.Tensor, bins: int = 64) -> np.ndarray:
    """Mean histogram across images."""
    all_vals = imgs.numpy().flatten()
    hist, _ = np.histogram(all_vals, bins=bins, range=(0.0, 1.0), density=True)
    return hist.astype(np.float64)


def stat_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two 1-D arrays."""
    return float(np.sqrt(np.mean((a - b)**2)))


# ── Load images ──────────────────────────────────────────────────────
print("Loading templates …")
template_paths = sorted(CLEAN_DIR.glob('*.png'))[:N_IMGS]
templates_np = [PRENORM_A * cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 + PRENORM_B
                for p in template_paths]
templates_t = [torch.from_numpy(t).to(DEVICE) for t in templates_np]

print("Loading real images for CalibrationLoss …")
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in sorted(REAL_DIR.glob('*.png'))[:N_REAL]]
loss_fn = CalibrationLoss(real_np).to(DEVICE)

# ── Build both pipelines ─────────────────────────────────────────────
print(f"\nBuilding pipelines …")
print(f"  θ_v5_full  : out_low={V5_THETA['out_low']:.4f}  out_high={V5_THETA['out_high']:.4f}")
print(f"  θ_v5_manual: out_low={V5_MANUAL_OUT['out_low']:.4f}  out_high={V5_MANUAL_OUT['out_high']:.4f}")
pipe_v5   = build_pipe(V5_THETA)
pipe_man  = build_pipe(V5_MANUAL_OUT)

# ── Generate evaluation images ───────────────────────────────────────
print(f"\nGenerating {N_EVAL} images per config …")
imgs_v5  = gen_images(pipe_v5,  templates_t, N_EVAL, SEED_EVAL)
imgs_man = gen_images(pipe_man, templates_t, N_EVAL, SEED_EVAL)

print(f"  v5_full  : mean={imgs_v5.mean():.4f}  std={imgs_v5.std():.4f}")
print(f"  v5_manual: mean={imgs_man.mean():.4f}  std={imgs_man.std():.4f}")

# ── CalibrationLoss per config ───────────────────────────────────────
print("\nComputing CalibrationLoss per config …")
with torch.no_grad():
    L_v5  = loss_fn(imgs_v5[0].to(DEVICE)).item()
    L_man = loss_fn(imgs_man[0].to(DEVICE)).item()
print(f"  L_vm(v5_full)  = {L_v5:.5f}")
print(f"  L_vm(v5_manual)= {L_man:.5f}")
L_vm_dist = abs(L_v5 - L_man)
print(f"  |ΔL_vm| = {L_vm_dist:.5f}")

# ── NPS, radial profile, histogram ───────────────────────────────────
print("\nComputing NPS …")
nps_v5  = compute_nps_1d(imgs_v5)
nps_man = compute_nps_1d(imgs_man)
nps_dist = stat_distance(nps_v5, nps_man)

print("Computing radial profiles …")
rad_v5  = compute_radial_profile(imgs_v5)
rad_man = compute_radial_profile(imgs_man)
rad_dist = stat_distance(rad_v5, rad_man)

print("Computing histograms …")
hist_v5  = compute_histogram(imgs_v5)
hist_man = compute_histogram(imgs_man)
hist_dist = stat_distance(hist_v5, hist_man)

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"§M 기준선: 1.5 × d_realization = {THRESHOLD:.5f}")
print(f"통계량 거리:")
print(f"  |ΔL_vm|    = {L_vm_dist:.5f}  {'[TRIGGER]' if L_vm_dist > THRESHOLD else 'OK'}")
print(f"  NPS        = {nps_dist:.5f}  {'[TRIGGER]' if nps_dist > THRESHOLD else 'OK'}")
print(f"  Radial     = {rad_dist:.5f}  {'[TRIGGER]' if rad_dist > THRESHOLD else 'OK'}")
print(f"  Histogram  = {hist_dist:.5f}  {'[TRIGGER]' if hist_dist > THRESHOLD else 'OK'}")
trigger = any(d > THRESHOLD for d in [L_vm_dist, nps_dist, rad_dist, hist_dist])
print(f"  트리거 발동: {'YES → v5/v5b PENDING_CONSULTATION' if trigger else 'NO'}")

out = {
    "handoff_ref": "HANDOFF_v6.1 지시 25 Q5",
    "theta_v5_full": V5_THETA,
    "theta_v5_manual_out": V5_MANUAL_OUT,
    "d_realization": D_REALIZATION,
    "threshold_1_5x": THRESHOLD,
    "distances": {
        "L_vm": L_vm_dist,
        "nps": nps_dist,
        "radial": rad_dist,
        "histogram": hist_dist,
    },
    "trigger": trigger,
    "trigger_details": {
        "L_vm": L_vm_dist > THRESHOLD,
        "nps": nps_dist > THRESHOLD,
        "radial": rad_dist > THRESHOLD,
        "histogram": hist_dist > THRESHOLD,
    },
    "n_eval_images": N_EVAL,
    "seed_eval": SEED_EVAL,
}

out_path = PHASE4 / 'q5_equivalence.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
