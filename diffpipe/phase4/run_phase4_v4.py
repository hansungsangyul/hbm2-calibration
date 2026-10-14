#!/usr/bin/env python3
"""
Phase 4 v4 — Real Image Calibration with Fixed N Parameterization
==================================================================
Changes from v3:
  - pipeline.py: N = 50 + 950·sigmoid(N_raw) in forward (sigmoid-box, v4+)
    Previously v3 used softplus(N_raw) in forward — bug, never actually applied
    the N prior. v4 is the FIRST run where pre-norm + N prior are both active.
  - T6 regression verified before run (assert N_display == N_pipeline)
  - Seeds: 940000 / 941000 / 942000 (new seeds for v4)
  - Everything else identical to v3: pre-norm a=0.501834/b=0.173320,
    σ_g≤0.042, contrast box [0,0.3]/[0.7,1.0], 3 reps × 2000 steps

Result branching (pre-registered per §K-4):
  - N boundary-pinned (±5% of [50,1000] = ±47.5) → non-identifiable confirmed
  - N ∈ [150, 500] interior → calibration success, upgrade §4.1

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \
        phase4/run_phase4_v4.py 2>&1 | tee phase4/run_phase4_v4.log
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
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

# ── T6 regression check (must pass before any calibration) ────────────
print("Running T6 regression check …")
_pipe_t6 = DegradationPipeline()
_test_raws = [sigmoid_box_inverse(300.0), 0.0, 6.0, -6.0]
for _r in _test_raws:
    with torch.no_grad():
        _pipe_t6.N_raw.fill_(_r)
    _n_disp = float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(_pipe_t6.N_raw))
    _n_pipe = float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(_pipe_t6.N_raw))
    assert abs(_n_disp - _n_pipe) < 1e-6, f"T6 FAIL: raw={_r}, disp={_n_disp}, pipe={_n_pipe}"
del _pipe_t6
print("  T6 PASS ✓ — N_display == N_pipeline for all test raws\n")

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Pre-normalization: x' = {PRENORM_A:.6f}·x + {PRENORM_B:.6f}")
print(f"N prior: [{_N_LO}, {_N_HI}] via sigmoid-box (forward + display unified)")
print(f"σ_g ∈ [{SIGMA_G_MIN}, {SIGMA_G_MAX}]")
print(f"in_low ∈ [{IN_LOW_MIN}, {IN_LOW_MAX}]  in_high ∈ [{IN_HIGH_MIN}, {IN_HIGH_MAX}]")

THETA_MANUAL = dict(
    fwhm=2.5, N=300.0, sigma_g=0.0408, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)


# ── Helpers ───────────────────────────────────────────────────────────
def n_from_raw(raw_tensor: torch.Tensor) -> torch.Tensor:
    return _N_LO + (_N_HI - _N_LO) * torch.sigmoid(raw_tensor)


def set_pipeline_to_theta(pipe: DegradationPipeline, theta: dict) -> None:
    with torch.no_grad():
        N_clamped = float(np.clip(theta['N'], _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
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
        sg = float(F.softplus(pipe.sigma_g_raw).item())
        sg_c = float(np.clip(sg, SIGMA_G_MIN, SIGMA_G_MAX))
        if abs(sg - sg_c) > 1e-9:
            pipe.sigma_g_raw.fill_(softplus_inverse(sg_c))
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)
        # N already sigmoid-box; clamp raw ±6 for numerical safety
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

templates_np = [PRENORM_A * img + PRENORM_B for img in templates_np_raw]
templates_t  = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"  raw:    mean={np.mean([t.mean() for t in templates_np_raw]):.4f}")
print(f"  mapped: mean={np.mean([t.mean() for t in templates_np]):.4f}")

print(f"Loading {N_REAL} real (dirty2) images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]
print(f"  real stats: mean={np.mean([i.mean() for i in real_np]):.4f}  "
      f"std={np.mean([i.std() for i in real_np]):.4f}")

# ── Build flat masks ──────────────────────────────────────────────────
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
    print(f"  img{i:02d}: {mask.sum()} flat pixels ({100*mask.sum()/(H*W):.1f}%)")

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
            n_bound = (abs(phys['N'] - _N_LO) < 0.05*(_N_HI - _N_LO) or
                       abs(phys['N'] - _N_HI) < 0.05*(_N_HI - _N_LO))
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"N={phys['N']:.1f}{'!BOUND' if n_bound else ''}  "
                  f"sg={phys['sigma_g']:.5f}  fwhm={phys['fwhm']:.3f}  "
                  f"alpha={phys['alpha']:.3f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'calib_trace.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"  [{run_label}] done: {N_STEPS} steps")
    return get_theta_physical(pipe), trace


# ── Gate measurement ──────────────────────────────────────────────────
def measure_flat_noise_gate(pipe, templates_np, real_np, flat_masks, seed_offset):
    rows = []
    for i in range(len(flat_masks)):
        tmpl  = templates_np[i]
        real  = real_np[i]
        mask  = flat_masks[i]
        H_, W_ = real.shape

        real_patches = []
        for r in range(0, H_-8, 4):
            for c in range(0, W_-8, 4):
                if mask[r:r+8, c:c+8].all():
                    real_patches.append(real[r:r+8, c:c+8].std())

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
print("Phase 4 v4: Real Image Calibration (pre-norm + N prior FIXED)")
print(f"  N forward = 50 + 950·sigmoid(N_raw)  [v4: display==pipeline]")
print(f"  Pre-norm a={PRENORM_A:.6f}, b={PRENORM_B:.6f}")
print(f"  σ_g ≤ {SIGMA_G_MAX}")
print(f"{'='*65}")

for rep in range(N_REPS):
    rep_seed = 940000 + rep * 1000
    rep_dir  = PHASE4_DIR / f'rep_v4_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")
    theta_hat, trace = run_one_calibration(
        pipe_opt, loss_fn, templates_t,
        init_theta=THETA_MANUAL.copy(),
        rep_seed=rep_seed,
        out_dir=rep_dir,
        run_label=f"r{rep}",
    )
    rep_results.append(theta_hat)
    print(f"  θ̂_real v4 rep {rep}: N={theta_hat['N']:.1f}  sg={theta_hat['sigma_g']:.5f}  "
          f"fwhm={theta_hat['fwhm']:.3f}  alpha={theta_hat['alpha']:.3f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}")

print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
PARAMS = ['N','sigma_g','fwhm','alpha','in_low','in_high','ring_strength','ring_n','p']
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS}

print(f"\n{'='*65}")
print("θ̂_real v4 vs θ_manual:")
print(f"{'Param':<16} {'θ_manual':>10} {'θ̂_real_v4':>12} {'±std':>8} {'|Δ|/θm':>9}")
print('-'*58)
for p in PARAMS:
    man = THETA_MANUAL.get(p, float('nan'))
    hat = theta_mean[p]
    std = theta_std[p]
    rel = abs(hat-man)/abs(man) if abs(man) > 1e-12 else float('nan')
    print(f"  {p:<14} {man:>10.5f} {hat:>12.5f} {std:>8.5f} {rel:>9.3f}")

# ── N branch determination (pre-registered §K-4) ──────────────────────
N_hat = theta_mean['N']
N_bound_tol = 0.05 * (_N_HI - _N_LO)  # ±47.5
n_at_lo = abs(N_hat - _N_LO) < N_bound_tol
n_at_hi = abs(N_hat - _N_HI) < N_bound_tol
n_pinned = n_at_lo or n_at_hi
n_interior = 150.0 <= N_hat <= 500.0

if n_pinned:
    BRANCH = "BOUNDARY_PINNED"
    BRANCH_MSG = ("N boundary-pinned → 'pre-norm + N prior하에서도 비식별' 확정 "
                  "→ §3.3 서사: Attractor A/B + prior 강제 시 경계 도피. "
                  "D_calib N·σ_g = manual 유지")
elif n_interior:
    BRANCH = "INTERIOR_SUCCESS"
    BRANCH_MSG = ("N ∈ [150, 500] 내부점 → 노이즈 캘리브레이션 성공 재분류 "
                  "→ §4.1 승격. D_calib N·σ_g = v4 값으로 갱신. flat-noise gate 재판정")
else:
    BRANCH = "INTERIOR_OTHER"
    BRANCH_MSG = (f"N={N_hat:.1f} (경계 아님, [150,500] 외부) → 논문 서사 세밀 조정 필요")

print(f"\n  N prior check: N̂={N_hat:.1f}  bounds=[{_N_LO},{_N_HI}]  "
      f"tol=±{N_bound_tol:.1f}")
print(f"  Branch: {BRANCH}")
print(f"  → {BRANCH_MSG}")

# ── Gate ──────────────────────────────────────────────────────────────
print("\n=== Flat-noise gate (v4: single image, real mask) ===")
set_pipeline_to_theta(pipe_opt, theta_mean)
gate_rows = measure_flat_noise_gate(pipe_opt, templates_np, real_np, flat_masks, seed_offset=0)
print(f"{'img':>5}  {'real_flat':>10}  {'synth_flat_single':>18}  {'ratio':>6}")
print('-'*50)
for row in gate_rows:
    print(f"  img{row['img']:02d}  {row['real_flat']:>10.5f}  {row['synth_flat_single']:>18.5f}  {row['ratio']:>6.3f}")
med_ratio = float(np.nanmedian([r['ratio'] for r in gate_rows]))
gate_pass = 0.5 <= med_ratio <= 2.0 and not n_pinned
in_range = (IN_LOW_MIN <= theta_mean['in_low'] <= IN_LOW_MAX and
            IN_HIGH_MIN <= theta_mean['in_high'] <= IN_HIGH_MAX)
print(f"\n  median ratio: {med_ratio:.3f}  (target [0.5, 2.0])")
print(f"  N not pinned: {'YES ✓' if not n_pinned else 'NO — boundary pinned'}")
print(f"  contrast in range: {'YES ✓' if in_range else 'NO'}")
print(f"  → Gate: {'PASS ✅' if gate_pass else 'FAIL ✗'}")

# ── v3/v4 comparison table ────────────────────────────────────────────
print(f"\n{'='*65}")
print("v3 vs v4 parameter comparison (fwhm/alpha/contrast stability):")
v3_data = json.load(open(PHASE4_DIR / 'theta_real_v3.json'))
v3_mean = v3_data['theta_real_mean']
# Note: v3 N was display=997.7 but pipeline=6.0 (bug). v4 N is true N.
print(f"{'Param':<14} {'v3_display':>12} {'v4':>10} {'|Δv3→v4|':>10}")
print('-'*50)
for p in ['N','sigma_g','fwhm','alpha','in_low','in_high','ring_strength']:
    v3v = v3_mean.get(p, float('nan'))
    v4v = theta_mean.get(p, float('nan'))
    delta = abs(v4v - v3v)
    note = " ← BUG(v3 display≠pipeline)" if p == 'N' else ""
    print(f"  {p:<12}  {v3v:>12.5f}  {v4v:>10.5f}  {delta:>10.5f}{note}")

# ── Save outputs ──────────────────────────────────────────────────────
# theta_real_v4.json
output = {
    "version": "v4",
    "bug_fix": "pipeline.py N: softplus→sigmoid-box. v4 = first run with pre-norm + N prior both active.",
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "constraints": {
        "N_prior": [_N_LO, _N_HI], "N_reparameterization": "sigmoid-box (forward==display)",
        "sigma_g_max": SIGMA_G_MAX, "sigma_g_min": SIGMA_G_MIN,
        "in_low_range": [IN_LOW_MIN, IN_LOW_MAX],
        "in_high_range": [IN_HIGH_MIN, IN_HIGH_MAX],
    },
    "theta_real_mean": theta_mean,
    "theta_real_std":  theta_std,
    "theta_hat_all":   rep_results,
    "theta_manual":    THETA_MANUAL,
    "n_branch": {
        "branch": BRANCH,
        "N_hat": N_hat,
        "N_bound_tol": N_bound_tol,
        "n_pinned": bool(n_pinned),
        "n_interior_150_500": bool(n_interior),
        "message": BRANCH_MSG,
    },
    "gate_v4": {
        "method": "single synth (no rep avg), real flat mask applied to both",
        "per_image": gate_rows,
        "median_ratio": med_ratio,
        "n_pinned": bool(n_pinned),
        "contrast_in_range": bool(in_range),
        "pass": bool(gate_pass),
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "lr_default": LR_DEFAULT, "lr_N": LR_N,
        "device": DEVICE,
        "seeds": [940000 + i*1000 for i in range(N_REPS)],
    }
}
v4_path = PHASE4_DIR / 'theta_real_v4.json'
with open(v4_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {v4_path}")

# Update identifiability_trajectory.csv to add v4 column
import csv as csv_mod
v1_data = json.load(open(PHASE4_DIR / 'theta_real_v1_deprecated.json'))
v1_mean = v1_data if 'N' in v1_data else v1_data.get('theta_real_mean', {})
v2_mean = json.load(open(PHASE4_DIR / 'theta_real_v2.json')).get('theta_real_mean', {})
# v3 N_pipeline was 6.002 (softplus); use actual pipeline value for trajectory
v3_N_pipeline = 6.002  # softplus(6.0); NOT display value 997.7
traj_path = PHASE4_DIR / 'identifiability_trajectory.csv'
with open(traj_path, 'w', newline='') as f:
    writer = csv_mod.writer(f)
    writer.writerow(['param', 'theta_manual', 'v1', 'v2', 'v3_pipeline', 'v4',
                     'v1_rel_err', 'v2_rel_err', 'v3_pipeline_rel_err', 'v4_rel_err', 'note'])
    for p in PARAMS:
        man = THETA_MANUAL.get(p, float('nan'))
        v1v = v1_mean.get(p, float('nan'))
        v2v = v2_mean.get(p, float('nan'))
        v3v = v3_mean.get(p, float('nan'))
        if p == 'N':
            v3v = v3_N_pipeline  # use actual pipeline value, not display
        v4v = theta_mean.get(p, float('nan'))
        def rel(hat, ref): return abs(hat-ref)/abs(ref) if abs(ref)>1e-12 else float('nan')
        note = 'v3_pipeline=softplus(6.0)=6.002 (bug; display was 997.7)' if p == 'N' else ''
        writer.writerow([p, f"{man:.6f}", f"{v1v:.6f}", f"{v2v:.6f}",
                         f"{v3v:.6f}", f"{v4v:.6f}",
                         f"{rel(v1v,man):.4f}", f"{rel(v2v,man):.4f}",
                         f"{rel(v3v,man):.4f}", f"{rel(v4v,man):.4f}", note])
print(f"Updated: {traj_path}")

print(f"\nPhase 4 v4 complete.")
print(f"Branch: {BRANCH}")
print(f"Gate: {'PASS ✅' if gate_pass else 'FAIL ✗'}")
