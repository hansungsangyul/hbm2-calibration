#!/usr/bin/env python3
"""
Phase 4 v3 — Real Image Calibration with Pre-normalization + N prior
=====================================================================
Changes from v2 (per HANDOFF_verification_feedback_v2.md §B):

1. Global intensity pre-normalization (조건 3, pre-authorized):
   x' = a·x + b, where a=0.501834, b=0.173320
   Pre-computed (closed-form) from HBM_clean/dirty2 (mean, std) matching.
   NOT in optimization loop — separates domain gap from degradation parameters.

2. N prior: N ∈ [50, 1000] via sigmoid box reparameterization
   N = N_lo + (N_hi - N_lo) * sigmoid(N_raw)
   Guards against N→5.6 degenerate attractor after σ_g constraint.

3. σ_g ≤ 0.042, contrast box [0,0.3]/[0.7,1.0] — same as v2.

4. Gate redefinition (§B):
   - Flat-noise: per single image (NO rep averaging)
   - Flat mask: defined on real image, applied to BOTH real and synth
   - Success: median ratio ∈ [0.5, 2.0] + all params in physical range +
     N NOT pinned to prior boundary (±5% of [50,1000]=boundary ±47.5)

Pre-normalization derivation:
  HBM_clean:  mean=0.37551, std=0.25087
  HBM_dirty2: mean=0.36176, std=0.12590
  a = std_d / std_c = 0.12590/0.25087 = 0.501834
  b = mean_d - a*mean_c = 0.36176 - 0.501834*0.37551 = 0.173320
  Mapped pixel range (sample): [0.1733, 0.6752]

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python phase4/run_phase4_v3.py 2>&1 | tee phase4/run_phase4_v3.log
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse
from losses import CalibrationLoss

# ── Config ────────────────────────────────────────────────────────────
TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = Path(__file__).parent

N_TEMPLATES = 50
N_REAL      = 50
N_STEPS     = 2000
BATCH_SIZE  = 8
N_REPS      = 3

LR_DEFAULT = 0.01
LR_N       = 1.0

# Pre-normalization (pre-computed, fixed)
PRENORM_A = 0.501834
PRENORM_B = 0.173320

# Physical constraints
SIGMA_G_MAX = 0.042
SIGMA_G_MIN = 1e-4
IN_LOW_MIN, IN_LOW_MAX   = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00
N_LO, N_HI = 50.0, 1000.0   # sigmoid box for N prior

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Pre-normalization: x' = {PRENORM_A:.6f}·x + {PRENORM_B:.6f}")
print(f"N prior: [{N_LO}, {N_HI}]  σ_g ∈ [{SIGMA_G_MIN}, {SIGMA_G_MAX}]")
print(f"in_low ∈ [{IN_LOW_MIN}, {IN_LOW_MAX}]  in_high ∈ [{IN_HIGH_MIN}, {IN_HIGH_MAX}]")

THETA_MANUAL = dict(
    fwhm=2.5, N=300.0, sigma_g=0.0408, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)

# ── N sigmoid-box reparameterization ─────────────────────────────────
def n_sigmoid_inverse(N_val: float) -> float:
    """Inverse of N = N_lo + (N_hi - N_lo) * sigmoid(raw)"""
    ratio = (N_val - N_LO) / (N_HI - N_LO)
    ratio = float(np.clip(ratio, 1e-6, 1 - 1e-6))
    return float(np.log(ratio / (1 - ratio)))   # logit

def n_from_raw(raw_tensor: torch.Tensor) -> torch.Tensor:
    return N_LO + (N_HI - N_LO) * torch.sigmoid(raw_tensor)

# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_to_theta(pipe: DegradationPipeline, theta: dict) -> None:
    with torch.no_grad():
        N_clamped = float(np.clip(theta['N'], N_LO + 1, N_HI - 1))
        pipe.N_raw.fill_(n_sigmoid_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], SIGMA_G_MIN)))
        pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta.get('ring_strength', 0.12)))
        pipe.ring_n_raw.fill_(softplus_inverse(theta.get('ring_n', 8.0)))
        pipe.in_low.fill_(np.clip(theta.get('in_low', 0.05), IN_LOW_MIN, IN_LOW_MAX))
        pipe.in_high.fill_(np.clip(theta.get('in_high', 0.90), IN_HIGH_MIN, IN_HIGH_MAX))
        pipe.out_low.fill_(theta.get('out_low', 0.10))
        pipe.out_high.fill_(theta.get('out_high', 0.90))
        pipe.p.fill_(theta.get('p', 0.001))


def project_constraints(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        # σ_g clamp
        sg = float(F.softplus(pipe.sigma_g_raw).item())
        sg_c = float(np.clip(sg, SIGMA_G_MIN, SIGMA_G_MAX))
        if abs(sg - sg_c) > 1e-9:
            pipe.sigma_g_raw.fill_(softplus_inverse(sg_c))
        # contrast box
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)
        # N is already constrained by sigmoid: clamp raw to ±6 for numerical safety
        pipe.N_raw.clamp_(-6.0, 6.0)


def get_theta_physical(pipe: DegradationPipeline) -> dict:
    with torch.no_grad():
        return dict(
            N=float(n_from_raw(pipe.N_raw).item()),
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Load images ───────────────────────────────────────────────────────
print(f"\nLoading {N_TEMPLATES} clean templates …")
template_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]
templates_np_raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                    for p in template_paths]
H, W = templates_np_raw[0].shape

# Apply pre-normalization
templates_np = [PRENORM_A * img + PRENORM_B for img in templates_np_raw]
templates_t  = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"  raw:    mean={np.mean([t.mean() for t in templates_np_raw]):.4f}, std={np.mean([t.std() for t in templates_np_raw]):.4f}")
print(f"  mapped: mean={np.mean([t.mean() for t in templates_np]):.4f}, std={np.mean([t.std() for t in templates_np]):.4f}")

print(f"Loading {N_REAL} real (dirty2) images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]
print(f"  real stats: mean={np.mean([i.mean() for i in real_np]):.4f}±{np.std([i.mean() for i in real_np]):.4f}  "
      f"std={np.mean([i.std() for i in real_np]):.4f}±{np.std([i.std() for i in real_np]):.4f}")

# ── Build flat masks (defined on real images, applied to both) ────────
print("Building per-image real flat masks …")
flat_masks = []
for i, img in enumerate(real_np[:5]):
    mask = np.zeros((H, W), dtype=bool)
    for r in range(0, H-8, 4):
        for c in range(0, W-8, 4):
            p = img[r:r+8, c:c+8]
            if 0.15 < p.mean() < 0.75 and p.std() < 0.04:
                mask[r:r+8, c:c+8] = True
    flat_masks.append(mask)
    n_px = mask.sum()
    print(f"  img{i:02d}: {n_px} flat pixels ({100*n_px/(H*W):.1f}%)")

# ── CalibrationLoss ───────────────────────────────────────────────────
print(f"\nBuilding CalibrationLoss from {N_REAL} real images …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")

# ── Training loop ─────────────────────────────────────────────────────
def run_one_calibration(pipe, loss_fn, templates, init_theta, rep_seed, out_dir, run_label):
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
        project_constraints(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            n_boundary = abs(phys['N']-N_LO) < 0.05*(N_HI-N_LO) or abs(phys['N']-N_HI) < 0.05*(N_HI-N_LO)
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"N={phys['N']:.1f}{'!BOUND' if n_boundary else ''}  sg={phys['sigma_g']:.5f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'calib_trace.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"  [{run_label}] done: 2000 steps")
    return get_theta_physical(pipe), trace


# ── Gate measurement function ─────────────────────────────────────────
def measure_flat_noise_gate(pipe, templates_np, real_np, flat_masks, seed_offset):
    """
    Per-image, single synth (no averaging), real flat mask applied to both.
    Returns: list of {img, real_flat, synth_flat, ratio}
    """
    rows = []
    for i in range(len(flat_masks)):
        tmpl  = templates_np[i]
        real  = real_np[i]
        mask  = flat_masks[i]
        H_, W_ = real.shape

        # Real noise
        real_patches = []
        for r in range(0, H_-8, 4):
            for c in range(0, W_-8, 4):
                if mask[r:r+8, c:c+8].all():
                    real_patches.append(real[r:r+8, c:c+8].std())

        # Synth — single image
        _seed_all(999000 + seed_offset + i)
        x = torch.from_numpy(tmpl).to(DEVICE)
        eps = {'eps1': torch.randn(H_, W_, device=DEVICE),
               'eps2': torch.randn(H_, W_, device=DEVICE),
               'eps3': torch.rand(H_, W_, device=DEVICE)}
        with torch.no_grad():
            synth = pipe(x, eps).cpu().numpy()

        synth_patches = []
        for r in range(0, H_-8, 4):
            for c in range(0, W_-8, 4):
                if mask[r:r+8, c:c+8].all():
                    synth_patches.append(synth[r:r+8, c:c+8].std())

        real_noise  = float(np.median(real_patches))  if real_patches  else float('nan')
        synth_noise = float(np.median(synth_patches)) if synth_patches else float('nan')
        ratio = synth_noise / real_noise if real_noise > 0 else float('nan')
        rows.append({'img': i, 'real_flat': real_noise, 'synth_flat_single': synth_noise, 'ratio': ratio})
    return rows


# ── Main ──────────────────────────────────────────────────────────────
pipe_opt = DegradationPipeline().to(DEVICE)
rep_results = []
t_total = time.time()

print(f"\n{'='*65}")
print("Phase 4 v3: Real Image Calibration (pre-norm + N prior + constraints)")
print(f"  Pre-norm a={PRENORM_A:.6f}, b={PRENORM_B:.6f}")
print(f"  N prior [{N_LO}, {N_HI}] (sigmoid box)")
print(f"  σ_g ≤ {SIGMA_G_MAX}  |  in_low ∈ [{IN_LOW_MIN},{IN_LOW_MAX}]  |  in_high ∈ [{IN_HIGH_MIN},{IN_HIGH_MAX}]")
print(f"{'='*65}")

for rep in range(N_REPS):
    rep_seed = 920000 + rep * 1000
    rep_dir  = PHASE4_DIR / f'rep_v3_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")
    theta_hat, trace = run_one_calibration(
        pipe_opt, loss_fn, templates_t,
        init_theta=THETA_MANUAL.copy(),
        rep_seed=rep_seed,
        out_dir=rep_dir,
        run_label=f"r{rep}",
    )
    rep_results.append(theta_hat)
    print(f"  θ̂_real v3 rep {rep}: N={theta_hat['N']:.1f}  sg={theta_hat['sigma_g']:.5f}  "
          f"fwhm={theta_hat['fwhm']:.3f}  alpha={theta_hat['alpha']:.3f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}")

print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
PARAMS = ['N','sigma_g','fwhm','alpha','in_low','in_high','ring_strength','ring_n','p']
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS}

print(f"\n{'='*65}")
print("θ̂_real v3 vs θ_manual:")
print(f"{'Param':<16} {'θ_manual':>10} {'θ̂_real_v3':>12} {'±std':>8} {'|Δ|/θm':>9}")
print('-'*58)
for p in PARAMS:
    man = THETA_MANUAL.get(p, float('nan'))
    hat = theta_mean[p]
    std = theta_std[p]
    rel = abs(hat-man)/abs(man) if abs(man) > 1e-12 else float('nan')
    print(f"  {p:<14} {man:>10.5f} {hat:>12.5f} {std:>8.5f} {rel:>9.3f}")

# N boundary check
N_hat = theta_mean['N']
N_bound_tol = 0.05 * (N_HI - N_LO)  # ±47.5
n_at_lo = abs(N_hat - N_LO) < N_bound_tol
n_at_hi = abs(N_hat - N_HI) < N_bound_tol
print(f"\n  N prior check: N̂={N_hat:.1f}  bounds=[{N_LO},{N_HI}]  "
      f"tol=±{N_bound_tol:.1f}  pinned={'YES (boundary)' if n_at_lo or n_at_hi else 'NO ✓'}")

# ── Gate measurement (v3 corrected method) ────────────────────────────
print("\n=== Flat-noise gate (v3 corrected: single image, real mask) ===")
set_pipeline_to_theta(pipe_opt, theta_mean)
gate_rows = measure_flat_noise_gate(pipe_opt, templates_np, real_np, flat_masks, seed_offset=0)
print(f"{'img':>5}  {'real_flat':>10}  {'synth_flat_single':>18}  {'ratio':>6}")
print('-'*50)
for row in gate_rows:
    print(f"  img{row['img']:02d}  {row['real_flat']:>10.5f}  {row['synth_flat_single']:>18.5f}  {row['ratio']:>6.3f}")
med_ratio = float(np.nanmedian([r['ratio'] for r in gate_rows]))
gate_pass = 0.5 <= med_ratio <= 2.0 and not (n_at_lo or n_at_hi)
in_range = all(IN_LOW_MIN <= theta_mean['in_low'] <= IN_LOW_MAX and
               IN_HIGH_MIN <= theta_mean['in_high'] <= IN_HIGH_MAX
               for _ in [None])
print(f"\n  median ratio: {med_ratio:.3f}  (target [0.5, 2.0])")
print(f"  N not pinned: {'YES ✓' if not (n_at_lo or n_at_hi) else 'NO — boundary pinned'}")
print(f"  contrast in range: {'YES ✓' if in_range else 'NO'}")
print(f"  → Gate: {'PASS ✅' if gate_pass else 'FAIL ✗'}")

# ── Save identifiability_trajectory.csv ──────────────────────────────
v1 = json.load(open(PHASE4_DIR/'theta_real_v1_deprecated.json'))
v2 = json.load(open(PHASE4_DIR/'theta_real_v2.json'))
v1_m = v1.get('theta_hat_all', [v1])[0] if 'theta_hat_all' not in v1 else {}
# v1 was old format — read directly
try:
    v1_m = json.load(open(PHASE4_DIR/'theta_real_v1_deprecated.json'))
    # old format has direct keys
    if 'N' in v1_m:
        v1_mean = v1_m
    else:
        v1_mean = v1_m.get('theta_real_mean', {})
except:
    v1_mean = {}
v2_mean = v2.get('theta_real_mean', {})

import csv as csv_mod
traj_path = PHASE4_DIR / 'identifiability_trajectory.csv'
with open(traj_path, 'w', newline='') as f:
    writer = csv_mod.writer(f)
    writer.writerow(['param', 'theta_manual', 'v1', 'v2', 'v3', 'v1_rel_err', 'v2_rel_err', 'v3_rel_err'])
    for p in PARAMS:
        man = THETA_MANUAL.get(p, float('nan'))
        v1v = v1_mean.get(p, float('nan'))
        v2v = v2_mean.get(p, float('nan'))
        v3v = theta_mean.get(p, float('nan'))
        def rel(hat, ref): return abs(hat-ref)/abs(ref) if abs(ref)>1e-12 else float('nan')
        writer.writerow([p, f"{man:.6f}", f"{v1v:.6f}", f"{v2v:.6f}", f"{v3v:.6f}",
                         f"{rel(v1v,man):.4f}", f"{rel(v2v,man):.4f}", f"{rel(v3v,man):.4f}"])
print(f"\nSaved: {traj_path}")

# ── Save theta_real_v3.json ───────────────────────────────────────────
output = {
    "version": "v3",
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B,
                         "clean_mean": 0.37551, "clean_std": 0.25087,
                         "dirty2_mean": 0.36176, "dirty2_std": 0.12590,
                         "formula": "x' = a*x + b, closed-form moments matching (not optimized)"},
    "constraints": {
        "N_prior": [N_LO, N_HI], "N_reparameterization": "sigmoid box",
        "sigma_g_max": SIGMA_G_MAX, "sigma_g_min": SIGMA_G_MIN,
        "in_low_range": [IN_LOW_MIN, IN_LOW_MAX],
        "in_high_range": [IN_HIGH_MIN, IN_HIGH_MAX],
    },
    "theta_real_mean": theta_mean,
    "theta_real_std":  theta_std,
    "theta_hat_all":   rep_results,
    "theta_manual":    THETA_MANUAL,
    "gate_v3": {
        "method": "single synth (no rep avg), real flat mask applied to both",
        "per_image": gate_rows,
        "median_ratio": med_ratio,
        "n_pinned": bool(n_at_lo or n_at_hi),
        "contrast_in_range": bool(in_range),
        "pass": bool(gate_pass),
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "lr_default": LR_DEFAULT, "lr_N": LR_N,
        "device": DEVICE,
        "seeds": [920000 + i*1000 for i in range(N_REPS)],
    }
}
v3_path = PHASE4_DIR / 'theta_real_v3.json'
with open(v3_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {v3_path}")
print(f"\nPhase 4 v3 complete. Gate: {'PASS' if gate_pass else 'FAIL'}")
