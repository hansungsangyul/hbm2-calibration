#!/usr/bin/env python3
"""
Phase 4 v2 — Real Image Calibration with Physical Constraints
=============================================================
Changes from v1:
  1. σ_g physical prior upper bound: σ_max = 0.042 (2×p90 of dirty2 flat-region Gaussian residual)
     Projected gradient: after each step, clamp σ_g to [1e-4, σ_max] and write back σ_g_raw.
  2. Contrast box constraint: in_low ∈ [0, 0.3], in_high ∈ [0.7, 1.0]
     Projected gradient after each step.
  3. v1 result preserved as theta_real_v1_deprecated.json (see HANDOFF_verification_feedback_v1.md)

Physical justification:
  - σ_max = 0.042: dirty2 flat noise p90=0.021, ×2 for safety.
    Flat noise ≈ Poisson (σ_Poisson=0.032 at μ=0.29, N=280); Gaussian residual median=0.
  - in_low ∈ [0, 0.3]: black point cannot exceed 30% gray physically.
  - in_high ∈ [0.7, 1.0]: white point must be at least 70% gray.

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python phase4/run_phase4_v2.py 2>&1 | tee phase4/run_phase4_v2.log
"""
from __future__ import annotations
import csv, json, shutil, sys, time
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
PHASE4_DIR.mkdir(exist_ok=True)

N_TEMPLATES = 50
N_REAL      = 50
N_STEPS     = 2000
BATCH_SIZE  = 8
N_REPS      = 3

LR_DEFAULT = 0.01
LR_N       = 1.0

# Physical constraints (지시 2)
SIGMA_G_MAX = 0.042        # 2 × p90 of Gaussian residual from dirty2 flat patches
SIGMA_G_MIN = 1e-4
IN_LOW_MIN, IN_LOW_MAX   = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Physical constraints: σ_g ∈ [{SIGMA_G_MIN}, {SIGMA_G_MAX}]  "
      f"in_low ∈ [{IN_LOW_MIN}, {IN_LOW_MAX}]  in_high ∈ [{IN_HIGH_MIN}, {IN_HIGH_MAX}]")

THETA_MANUAL = dict(
    fwhm=2.5, N=300.0, sigma_g=0.0408, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)

# ── Preserve v1 ───────────────────────────────────────────────────────
v1_path = PHASE4_DIR / 'theta_real.json'
v1_deprecated = PHASE4_DIR / 'theta_real_v1_deprecated.json'
if v1_path.exists() and not v1_deprecated.exists():
    shutil.copy(v1_path, v1_deprecated)
    print(f"Preserved v1 → {v1_deprecated.name}")

# ── Load images ───────────────────────────────────────────────────────
print(f"\nLoading {N_TEMPLATES} clean templates …")
template_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]
templates_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                for p in template_paths]
H, W = templates_np[0].shape
templates_t = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"  shape {H}×{W}")

print(f"Loading {N_REAL} real (dirty2) images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]
real_means = [float(img.mean()) for img in real_np]
real_stds  = [float(img.std())  for img in real_np]
print(f"  real stats: mean={np.mean(real_means):.4f}±{np.std(real_means):.4f}  "
      f"std={np.mean(real_stds):.4f}±{np.std(real_stds):.4f}")

# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_to_theta(pipe: DegradationPipeline, theta: dict) -> None:
    with torch.no_grad():
        pipe.N_raw.fill_(softplus_inverse(theta['N']))
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
    """Projected gradient: clamp physical parameters to valid ranges after optimizer step."""
    with torch.no_grad():
        # σ_g: clamp physical value, write back through softplus_inverse
        sg = float(F.softplus(pipe.sigma_g_raw).item())
        sg_clamped = float(np.clip(sg, SIGMA_G_MIN, SIGMA_G_MAX))
        if abs(sg - sg_clamped) > 1e-9:
            pipe.sigma_g_raw.fill_(softplus_inverse(sg_clamped))

        # in_low, in_high: direct clamp
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)


def get_theta_physical(pipe: DegradationPipeline) -> dict:
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        project_constraints(pipe)  # ← physical prior projection

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"N={phys['N']:.1f}  sg={phys['sigma_g']:.5f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}")

    print(f"  [{run_label}] done: 2000 steps")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / 'calib_trace.csv'
    with open(trace_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    return get_theta_physical(pipe), trace


def generate_synth_sample(pipe, template_np, seed, n_samples=8):
    x = torch.from_numpy(template_np).to(DEVICE)
    acc = torch.zeros_like(x)
    for i in range(n_samples):
        _seed_all(seed+i)
        eps = {
            'eps1': torch.randn(H, W, device=DEVICE),
            'eps2': torch.randn(H, W, device=DEVICE),
            'eps3': torch.rand(H, W, device=DEVICE),
        }
        with torch.no_grad():
            acc += pipe(x, eps)
    return (acc/n_samples).cpu().numpy()


# ── Main ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Phase 4 v2: Real Image Calibration (physical constraints)")
print(f"  σ_g prior: ∈ [0, {SIGMA_G_MAX}] (2×p90 flat-region Gaussian residual)")
print(f"  in_low:    ∈ [{IN_LOW_MIN}, {IN_LOW_MAX}]")
print(f"  in_high:   ∈ [{IN_HIGH_MIN}, {IN_HIGH_MAX}]")
print(f"{'='*60}")

print(f"\nBuilding CalibrationLoss from {N_REAL} real images …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")

pipe_opt = DegradationPipeline().to(DEVICE)
rep_results = []
t_total = time.time()

for rep in range(N_REPS):
    rep_seed = 910000 + rep * 1000  # v2 seed space
    rep_dir = PHASE4_DIR / f'rep_v2_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")
    theta_hat, trace = run_one_calibration(
        pipe_opt, loss_fn, templates_t,
        init_theta=THETA_MANUAL.copy(),
        rep_seed=rep_seed,
        out_dir=rep_dir,
        run_label=f"r{rep}",
    )
    rep_results.append(theta_hat)
    print(f"  θ̂_real v2 rep {rep}: N={theta_hat['N']:.1f}  sg={theta_hat['sigma_g']:.5f}  "
          f"fwhm={theta_hat['fwhm']:.3f}  alpha={theta_hat['alpha']:.3f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}")

print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
PARAMS = ['N','sigma_g','fwhm','alpha','in_low','in_high','ring_strength','ring_n','p']
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS}

print(f"\n{'='*60}")
print("θ̂_real v2 vs θ_manual:")
print(f"{'Param':<16} {'θ_manual':>10} {'θ̂_real_v2':>12} {'±std':>8} {'|Δ|/θm':>9}")
print('-'*58)
for p in PARAMS:
    man = THETA_MANUAL.get(p, float('nan'))
    hat = theta_mean[p]
    std = theta_std[p]
    rel = abs(hat-man)/abs(man) if abs(man) > 1e-12 else float('nan')
    print(f"  {p:<14} {man:>10.5f} {hat:>12.5f} {std:>8.5f} {rel:>9.3f}")

# ── Flat-noise verification ───────────────────────────────────────────
print("\n=== Flat-noise verification ===")
set_pipeline_to_theta(pipe_opt, theta_mean)
flat_preds = []
for i in range(5):
    tmpl = templates_np[i]
    # Generate 8-rep averaged synth
    synth = generate_synth_sample(pipe_opt, tmpl, seed=991000+i)
    # Flat-region noise in synth
    stds = []
    for r in range(0, H-8, 4):
        for c in range(0, W-8, 4):
            p = synth[r:r+8, c:c+8]
            pm = p.mean()
            ps = p.std()
            if 0.15 < pm < 0.75 and ps < 0.04:
                stds.append(ps)
    pred_noise = float(np.median(stds)) if stds else float('nan')
    flat_preds.append(pred_noise)
    print(f"  img{i:02d}: synth_v2 flat noise = {pred_noise:.5f}")

# Measure real dirty2 flat noise on same 5 images
real_flat_noises = []
for i in range(5):
    img_r = real_np[i]
    stds_r = []
    for r in range(0, H-8, 4):
        for c in range(0, W-8, 4):
            pr = img_r[r:r+8, c:c+8]
            if 0.15 < pr.mean() < 0.75 and pr.std() < 0.04:
                stds_r.append(pr.std())
    real_flat_noises.append(float(np.median(stds_r)) if stds_r else float('nan'))
    print(f"  img{i:02d}: real dirty2 flat noise = {real_flat_noises[-1]:.5f}")
print(f"\n  real dirty2 flat noise (measured): median={np.nanmedian(real_flat_noises):.5f}")
print(f"  synth_v2 flat noise:               median={np.median(flat_preds):.5f}")
ratio = np.nanmedian(flat_preds) / np.nanmedian(real_flat_noises)
print(f"  ratio synth/real = {ratio:.2f}  (success if 0.5x–2.0x)")
success = 0.5 <= ratio <= 2.0
print(f"  → {'PASS ✅' if success else 'FAIL ✗'}")

# ── Save θ̂_real_v2.json ──────────────────────────────────────────────
theta_v1 = json.load(open(PHASE4_DIR/'theta_real_v1_deprecated.json')) if (PHASE4_DIR/'theta_real_v1_deprecated.json').exists() else {}

output = {
    "version": "v2",
    "constraints": {
        "sigma_g_max": SIGMA_G_MAX,
        "sigma_g_min": SIGMA_G_MIN,
        "in_low_range": [IN_LOW_MIN, IN_LOW_MAX],
        "in_high_range": [IN_HIGH_MIN, IN_HIGH_MAX],
        "justification": (
            f"sigma_g_max={SIGMA_G_MAX}: 2×p90 of Gaussian residual from dirty2 flat patches "
            f"(Poisson sigma≈0.032 at mu=0.29,N=280 explains most of observed flat noise≈0.030; "
            f"Gaussian residual p90=0.021, ×2 safety margin). "
            f"in_low/in_high: physically meaningful black/white point range."
        )
    },
    "theta_real_mean": theta_mean,
    "theta_real_std":  theta_std,
    "theta_hat_all":   rep_results,
    "theta_manual":    THETA_MANUAL,
    "flat_noise_verification": {
        "real_dirty2_median":   float(np.nanmedian(real_flat_noises)),
        "synth_v2_median":      float(np.nanmedian(flat_preds)),
        "ratio":                float(ratio),
        "pass":                 bool(success),
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR),
        "real_dir":     str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "lr_default": LR_DEFAULT, "lr_N": LR_N,
        "device": DEVICE,
    }
}

v2_path = PHASE4_DIR / 'theta_real_v2.json'
with open(v2_path, 'w') as f:
    json.dump(output, f, indent=2)
# Also update theta_real.json to v2
with open(PHASE4_DIR / 'theta_real.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {v2_path}")
print(f"Flat-noise gate: {'PASS' if success else 'FAIL'}")
print(f"\nPhase 4 v2 complete.")
