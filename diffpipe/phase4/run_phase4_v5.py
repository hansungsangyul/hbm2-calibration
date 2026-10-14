#!/usr/bin/env python3
"""
Phase 4 v5 — Tier 1 Freeze Calibration (N·σ_g 동결, fwhm/alpha 조건부 식별)
=============================================================================
v4 결과: BOUNDARY_PINNED → (N, σ_g, fwhm) 3D 비식별 능선 확인.
v5 설계 (§O 3층 구조):
  Tier 1 (동결): N=300, σ_g=0.0408, p=0.001 — 전작 실사 nano-CT 실측값
  Tier 2 (자유): fwhm, alpha, in_low, in_high, ring_strength, ring_n
  → N·σ_g 고정 후 Tier 2 조건부 식별 여부 검증

실행 구성:
  - 메인 3 reps: seeds 950000/951000/952000 (fwhm_init = THETA_MANUAL 2.5)
  - 앵커 2 runs: seed 953000 (fwhm_init=1.5) / seed 954000 (fwhm_init=3.5)
  → 총 5 runs

분기 판정 (§O 사전 등록):
  | v5 fwhm 수렴 | 판정 | 논문 |
  | [2.3, 2.7] + 양방향 수렴 | 전작 수동값 검증 성립 | §4.1: fwhm ≈ manual |
  | 기타 내부점 + 양방향 수렴 | dirty2 실제 PSF ≠ manual | §4.1: 측정값 보고 |
  | 경계/불일치 | Tier 2 재분류 실패 | §3.3에 한계 기재 |

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \\
        phase4/run_phase4_v5.py 2>&1 | tee phase4/run_phase4_v5.log
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

# Pre-normalization (same as v4)
PRENORM_A = 0.501834
PRENORM_B = 0.173320

# Tier 1 frozen values (전작 실사 nano-CT 실측)
TIER1_N     = 300.0
TIER1_SG    = 0.0408
TIER1_P     = 0.001

# Tier 2 physical ranges
IN_LOW_MIN, IN_LOW_MAX   = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Pre-normalization: x' = {PRENORM_A:.6f}·x + {PRENORM_B:.6f}")
print(f"Tier 1 FROZEN: N={TIER1_N}, σ_g={TIER1_SG}, p={TIER1_P}")
print(f"Tier 2 FREE: fwhm, alpha, in_low, in_high, ring_strength, ring_n")

THETA_MANUAL = dict(
    fwhm=2.5, N=TIER1_N, sigma_g=TIER1_SG, p=TIER1_P, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)


# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v5(pipe: DegradationPipeline, fwhm_init: float, freeze: bool = True) -> None:
    """Set pipeline to v5 init. Freeze Tier 1 params if freeze=True."""
    with torch.no_grad():
        # Tier 1: frozen
        N_clamped = float(np.clip(TIER1_N, _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        # Tier 2: free, initialized from THETA_MANUAL except fwhm (may be perturbed)
        pipe.fwhm_raw.fill_(softplus_inverse(fwhm_init))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_MANUAL['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_MANUAL['ring_n']))
        pipe.in_low.fill_(np.clip(THETA_MANUAL['in_low'], IN_LOW_MIN, IN_LOW_MAX))
        pipe.in_high.fill_(np.clip(THETA_MANUAL['in_high'], IN_HIGH_MIN, IN_HIGH_MAX))
        pipe.out_low.fill_(THETA_MANUAL['out_low'])
        pipe.out_high.fill_(THETA_MANUAL['out_high'])

    if freeze:
        pipe.N_raw.requires_grad_(False)
        pipe.sigma_g_raw.requires_grad_(False)
        pipe.p.requires_grad_(False)
    return pipe


def project_constraints_v5(pipe: DegradationPipeline) -> None:
    """Only Tier 2 constraints: in_low/in_high bounds."""
    with torch.no_grad():
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)


def get_theta_physical(pipe: DegradationPipeline) -> dict:
    with torch.no_grad():
        return dict(
            N=float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(pipe.N_raw)),
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
print("Building per-image real flat masks (same as v4) …")
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
def run_one_calibration_v5(pipe, loss_fn, templates, fwhm_init, rep_seed, out_dir, run_label):
    set_pipeline_v5(pipe, fwhm_init=fwhm_init, freeze=True)

    # Only Tier 2 params in optimizer
    tier2_params = [pipe.fwhm_raw, pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw]
    tier2_direct = [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high]
    optimizer = torch.optim.Adam(
        [{'params': tier2_params, 'lr': LR_DEFAULT},
         {'params': tier2_direct, 'lr': LR_DEFAULT}]
    )

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
        project_constraints_v5(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}  "
                  f"ring_s={phys['ring_strength']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'calib_trace.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"  [{run_label}] done: {N_STEPS} steps")
    return get_theta_physical(pipe), trace


# ── Gate measurement (v5: interior + std, no n_pinned) ──────────────
def measure_flat_noise_gate_v5(pipe, templates_np, real_np, flat_masks, seed_offset):
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
print("Phase 4 v5: Tier 1 Freeze (N=300, σ_g=0.0408 동결)")
print(f"  Tier 2 free: fwhm, alpha, in_low, in_high, ring_strength, ring_n")
print(f"  Pre-norm a={PRENORM_A:.6f}, b={PRENORM_B:.6f}")
print(f"{'='*65}")

# Main 3 reps
MAIN_SEEDS = [950000, 951000, 952000]
for rep, rep_seed in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / f'rep_v5_{rep}'
    print(f"\n[Main Rep {rep}] seed={rep_seed}  fwhm_init=2.5 (THETA_MANUAL)")
    theta_hat, trace = run_one_calibration_v5(
        pipe_opt, loss_fn, templates_t,
        fwhm_init=THETA_MANUAL['fwhm'],
        rep_seed=rep_seed,
        out_dir=rep_dir,
        run_label=f"r{rep}",
    )
    rep_results.append(theta_hat)
    print(f"  θ̂_v5 rep {rep}: fwhm={theta_hat['fwhm']:.4f}  alpha={theta_hat['alpha']:.4f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}  "
          f"ring_s={theta_hat['ring_strength']:.4f}")

print(f"\nMain reps done. Total: {(time.time()-t_total)/60:.1f} min")

# Aggregate main reps
PARAMS_TIER2 = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS_TIER2}
# Tier 1 stays frozen
theta_mean['N']   = TIER1_N
theta_mean['sigma_g'] = TIER1_SG
theta_mean['p']   = TIER1_P
theta_std['N']    = 0.0
theta_std['sigma_g'] = 0.0
theta_std['p']    = 0.0

print(f"\n{'='*65}")
print("θ̂_v5 (main 3 reps) vs θ_manual:")
print(f"{'Param':<16} {'θ_manual':>10} {'θ̂_v5_mean':>12} {'±std':>8} {'rel_std%':>9}")
print('-'*58)
for p in PARAMS_TIER2:
    man = THETA_MANUAL.get(p, float('nan'))
    hat = theta_mean[p]
    std = theta_std[p]
    rel_std = 100 * std / (abs(hat) + 1e-12)
    print(f"  {p:<14} {man:>10.5f} {hat:>12.5f} {std:>8.5f} {rel_std:>8.2f}%")

# ── Perturbed-init anchor test ─────────────────────────────────────────
print(f"\n{'='*65}")
print("Perturbed-init anchor test (§O): fwhm_init ∈ {{1.5, 3.5}}")
PERTURBED = [
    (953000, 1.5, 'lo'),
    (954000, 3.5, 'hi'),
]
anchor_results = {}
for pseed, fwhm_init, label in PERTURBED:
    rep_dir = PHASE4_DIR / f'rep_v5_perturb_{label}'
    print(f"\n[Perturbed-{label}] seed={pseed}  fwhm_init={fwhm_init}")
    theta_hat, _ = run_one_calibration_v5(
        pipe_opt, loss_fn, templates_t,
        fwhm_init=fwhm_init,
        rep_seed=pseed,
        out_dir=rep_dir,
        run_label=f"p{label}",
    )
    anchor_results[label] = theta_hat
    print(f"  θ̂_v5 perturb-{label}: fwhm={theta_hat['fwhm']:.4f}  alpha={theta_hat['alpha']:.4f}")

print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")

# ── Branch determination (§O 사전 등록) ───────────────────────────────
fwhm_hat    = theta_mean['fwhm']
fwhm_lo     = anchor_results['lo']['fwhm']
fwhm_hi     = anchor_results['hi']['fwhm']
fwhm_spread = abs(fwhm_lo - fwhm_hi)
fwhm_std    = theta_std['fwhm']
fwhm_std_pct = 100 * fwhm_std / (fwhm_hat + 1e-12)

# Interior check: fwhm not at F.softplus boundaries (>0.1 and <10.0 is safe)
fwhm_interior = 0.3 <= fwhm_hat <= 8.0
alpha_interior = 0.1 <= theta_mean['alpha'] <= 2.0
rep_std_ok = fwhm_std_pct < 5.0  # rep std < 5% of mean

# Perturbed-init convergence: both perturbed runs converge within 10% of mean
lo_close = abs(fwhm_lo - fwhm_hat) / (fwhm_hat + 1e-12) < 0.10
hi_close = abs(fwhm_hi - fwhm_hat) / (fwhm_hat + 1e-12) < 0.10
bidir_converge = lo_close and hi_close

in_manual_range = 2.3 <= fwhm_hat <= 2.7
success = fwhm_interior and alpha_interior and rep_std_ok and bidir_converge

print(f"\n{'='*65}")
print("§O Branch determination:")
print(f"  fwhm_hat={fwhm_hat:.4f}  std={fwhm_std:.4f} ({fwhm_std_pct:.1f}%)")
print(f"  perturb-lo fwhm={fwhm_lo:.4f}  perturb-hi fwhm={fwhm_hi:.4f}  spread={fwhm_spread:.4f}")
print(f"  fwhm interior ({0.3:.1f},{8.0:.1f}): {'YES' if fwhm_interior else 'NO'}")
print(f"  alpha interior ({0.1:.1f},{2.0:.1f}): {'YES' if alpha_interior else 'NO'}")
print(f"  rep std < 5%: {'YES' if rep_std_ok else 'NO'} ({fwhm_std_pct:.1f}%)")
print(f"  bidir convergence (within 10%): lo={'YES' if lo_close else 'NO'}  hi={'YES' if hi_close else 'NO'}")

if not success:
    BRANCH_V5 = "TIER2_FAIL"
    BRANCH_MSG_V5 = "fwhm/alpha not interior or rep std ≥ 5% or bidir not converge → Tier 2 재분류 실패. D_calib fwhm=manual."
elif in_manual_range:
    BRANCH_V5 = "VALIDATES_MANUAL"
    BRANCH_MSG_V5 = f"fwhm={fwhm_hat:.3f} ∈ [2.3, 2.7] + 양방향 수렴 → 전작 수동값 검증 성립 (§4.1: calibrated fwhm ≈ manual)."
else:
    BRANCH_V5 = "DIFFERENT_FROM_MANUAL"
    BRANCH_MSG_V5 = f"fwhm={fwhm_hat:.3f} ∉ [2.3, 2.7] but interior + bidir converge → dirty2 실제 PSF ≠ manual. §4.1: 측정값 보고."
print(f"  Branch: {BRANCH_V5}")
print(f"  → {BRANCH_MSG_V5}")

# ── Gate measurement (v5: interior + rep std) ─────────────────────────
print("\n=== Flat-noise gate (v5: interior + rep std < 5%, no n_pinned) ===")
set_pipeline_v5(pipe_opt, fwhm_init=fwhm_hat, freeze=True)
with torch.no_grad():
    pipe_opt.fwhm_raw.fill_(softplus_inverse(fwhm_hat))
    pipe_opt.alpha_raw.fill_(softplus_inverse(theta_mean['alpha']))
    pipe_opt.in_low.fill_(theta_mean['in_low'])
    pipe_opt.in_high.fill_(theta_mean['in_high'])
    pipe_opt.ring_strength_raw.fill_(softplus_inverse(theta_mean['ring_strength']))
    pipe_opt.ring_n_raw.fill_(softplus_inverse(theta_mean['ring_n']))
gate_rows = measure_flat_noise_gate_v5(pipe_opt, templates_np, real_np, flat_masks, seed_offset=0)
med_ratio = float(np.nanmedian([r['ratio'] for r in gate_rows]))
noise_ratio_ok = 0.5 <= med_ratio <= 2.0
gate_pass_v5 = noise_ratio_ok and success

print(f"{'img':>5}  {'real_flat':>10}  {'synth_flat':>10}  {'ratio':>6}")
print('-'*40)
for row in gate_rows:
    print(f"  img{row['img']:02d}  {row['real_flat']:>10.5f}  {row['synth_flat_single']:>10.5f}  {row['ratio']:>6.3f}")
print(f"\n  median ratio: {med_ratio:.3f}  (target [0.5, 2.0]): {'OK' if noise_ratio_ok else 'FAIL'}")
print(f"  Tier 2 success: {'YES' if success else 'NO'}")
print(f"  → Gate v5: {'PASS ✅' if gate_pass_v5 else 'FAIL ✗'}")

# ── Save outputs ──────────────────────────────────────────────────────
all_params = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
              'ring_strength', 'ring_n']
theta_mean_full = {p: theta_mean.get(p, TIER1_P if p=='p' else float('nan')) for p in all_params}
theta_std_full  = {p: theta_std.get(p, 0.0) for p in all_params}

output = {
    "version": "v5",
    "design": "Tier 1 freeze (N, σ_g, p = 전작 실측). Tier 2 = fwhm, alpha, in_low, in_high, ring.",
    "tier1_frozen": {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std":  theta_std_full,
    "theta_hat_main":  rep_results,
    "theta_hat_perturb": anchor_results,
    "theta_manual":    THETA_MANUAL,
    "branch_v5": {
        "branch": BRANCH_V5,
        "fwhm_hat": fwhm_hat,
        "fwhm_std_pct": fwhm_std_pct,
        "fwhm_perturb_lo": fwhm_lo,
        "fwhm_perturb_hi": fwhm_hi,
        "fwhm_spread": fwhm_spread,
        "bidir_converge": bool(bidir_converge),
        "rep_std_ok": bool(rep_std_ok),
        "in_manual_range": bool(in_manual_range),
        "message": BRANCH_MSG_V5,
    },
    "gate_v5": {
        "method": "single synth (no rep avg), real flat mask; success = interior + rep std < 5%",
        "per_image": gate_rows,
        "median_ratio": med_ratio,
        "noise_ratio_ok": bool(noise_ratio_ok),
        "tier2_success": bool(success),
        "pass": bool(gate_pass_v5),
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps_main": N_REPS,
        "seeds_main": MAIN_SEEDS,
        "seeds_perturb": {"lo": 953000, "hi": 954000},
        "fwhm_init_perturb": {"lo": 1.5, "hi": 3.5},
        "lr_default": LR_DEFAULT,
        "device": DEVICE,
    }
}
v5_path = PHASE4_DIR / 'theta_real_v5.json'
with open(v5_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {v5_path}")

# ── D_calib re-issue ──────────────────────────────────────────────────
if BRANCH_V5 != "TIER2_FAIL":
    d_calib = dict(
        N=TIER1_N,
        sigma_g=TIER1_SG,
        p=TIER1_P,
        fwhm=fwhm_hat,
        alpha=theta_mean['alpha'],
        in_low=theta_mean['in_low'],
        in_high=theta_mean['in_high'],
        out_low=THETA_MANUAL['out_low'],
        out_high=THETA_MANUAL['out_high'],
        ring_strength=theta_mean['ring_strength'],
        ring_n=theta_mean['ring_n'],
    )
    ring_v1v2 = 0.009  # v1/v2 consensus
    ring_diverge = theta_mean['ring_strength'] / ring_v1v2 > 3.0
    if ring_diverge:
        print(f"\n[WARN] ring_strength={theta_mean['ring_strength']:.4f} > 3× v1/v2 ({ring_v1v2}). 보고 필요, 협의 대기.")
else:
    d_calib = dict(
        N=TIER1_N, sigma_g=TIER1_SG, p=TIER1_P,
        fwhm=THETA_MANUAL['fwhm'],  # fallback to manual
        alpha=THETA_MANUAL['alpha'],
        in_low=THETA_MANUAL['in_low'],
        in_high=THETA_MANUAL['in_high'],
        out_low=THETA_MANUAL['out_low'],
        out_high=THETA_MANUAL['out_high'],
        ring_strength=0.009,
        ring_n=8.6,
    )
    ring_diverge = False
    print(f"\n[TIER2_FAIL] D_calib fwhm=manual ({THETA_MANUAL['fwhm']})")

existing = json.load(open(PHASE4_DIR / 'd_calib_config.json'))
existing.update(d_calib)
existing['v5_branch'] = BRANCH_V5
existing['v5_fwhm_hat'] = fwhm_hat
existing['v5_ring_diverge_warn'] = bool(ring_diverge)
existing['prenorm'] = {"a": PRENORM_A, "b": PRENORM_B}
existing['note'] = (
    f"v5 D_calib: Tier 1 frozen (N={TIER1_N}, σ_g={TIER1_SG}). "
    f"Tier 2 from v5 ({BRANCH_V5}). "
    f"fwhm={d_calib['fwhm']:.4f}. "
    f"ring_diverge_warn={ring_diverge}."
)
existing['status'] = 'FINAL'
with open(PHASE4_DIR / 'd_calib_config.json', 'w') as f:
    json.dump(existing, f, indent=2)
print(f"Updated d_calib_config.json → FINAL (was REVOKED)")
print(f"  Branch: {BRANCH_V5}")
print(f"  D_calib fwhm={d_calib['fwhm']:.4f}  alpha={d_calib['alpha']:.4f}")

print(f"\n{'='*65}")
print("Phase 4 v5 complete")
print(f"  Total: {(time.time()-t_total)/60:.1f} min")
print(f"  Branch: {BRANCH_V5}")
print(f"  {BRANCH_MSG_V5}")
print(f"\nPhase 4 v5 complete — D_calib {'FINAL' if BRANCH_V5 != 'TIER2_FAIL' else 'FINAL (fallback manual)'}")
