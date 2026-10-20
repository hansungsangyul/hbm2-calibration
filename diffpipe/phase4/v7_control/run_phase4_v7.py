#!/usr/bin/env python3
"""
Phase 4 v7 — Causal Closure Calibration  (HANDOFF v7.10 §1 지시 73)
=====================================================================
v5c + σ_post 자유 추가. 인과 폐쇄 실험.

Tier 1 동결: N=300, σ_g=0.0408, p=0.001
out_* 동결: 0.10 / 0.90
Tier 2 자유: fwhm, alpha, in_low, in_high, ring_strength, ring_n  (v5c와 동일)
추가 자유:   σ_post  (v7 신규)

주 3 rep:   seeds 980000, 981000, 982000
M-3 check:  seed 983000 (λ×2), 984000 (λ÷2)

사전 판정표 (HANDOFF v7.10 §1):
  f ≥ 1.7      → 인과 폐쇄 (기여 2 완성)
  1.45~1.7     → 부분 이동
  < 1.45       → 인과 미폐쇄

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/v0.7/run_phase4_v7.py \\
        2>&1 | tee /mnt/gstore/home/xiangyue/HBM_paper2/v0.7/run_phase4_v7.log
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

# ── Config ────────────────────────────────────────────────────────────────────
TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
OUT_DIR      = Path('/mnt/gstore/home/xiangyue/HBM_paper2/v0.7')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TEMPLATES = 50
N_REAL      = 50
N_STEPS     = 2000
BATCH_SIZE  = 8
LR          = 0.01

PRENORM_A = 0.501834
PRENORM_B = 0.173320

TIER1_N  = 300.0
TIER1_SG = 0.0408
TIER1_P  = 0.001
OUT_LOW_FROZEN  = 0.10
OUT_HIGH_FROZEN = 0.90

IN_LOW_MIN,  IN_LOW_MAX  = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00
SIGMA_POST_MIN = 1e-4
SIGMA_POST_MAX = 0.10

THETA_MANUAL = dict(
    fwhm=2.5, N=TIER1_N, sigma_g=TIER1_SG, p=TIER1_P, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=OUT_LOW_FROZEN, out_high=OUT_HIGH_FROZEN,
    ring_strength=0.12, ring_n=8.0, sigma_post=0.01,
)

V5C_FWHM = 1.356261   # reference: v5c 3-rep mean

# v5c Tier2 params + sigma_post
PARAMS_V7 = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n', 'sigma_post']

# Run configs: (seed, label, lambda_weights)
MAIN_REPS = [
    (980000, 'v7_r0', (1.0, 1.0, 1.0, 1.0)),
    (981000, 'v7_r1', (1.0, 1.0, 1.0, 1.0)),
    (982000, 'v7_r2', (1.0, 1.0, 1.0, 1.0)),
]
# M-3 λ scaling check
M3_REPS = [
    (983000, 'v7_m3_lambda2x', (2.0, 2.0, 2.0, 2.0)),   # λ×2
    (984000, 'v7_m3_lambda05', (0.5, 0.5, 0.5, 0.5)),   # λ÷2
]
ALL_RUNS = MAIN_REPS + M3_REPS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Tier 1 FROZEN: N={TIER1_N}, σ_g={TIER1_SG}, p={TIER1_P}")
print(f"out_* FROZEN: {OUT_LOW_FROZEN}/{OUT_HIGH_FROZEN}")
print(f"Tier 2 FREE: fwhm, alpha, in_low, in_high, ring_strength, ring_n, σ_post (v7 추가)")


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_pipeline_v7(pipe: DegradationPipeline, fwhm_init: float = 2.5) -> None:
    with torch.no_grad():
        N_c = float(np.clip(TIER1_N, _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        pipe.out_low.fill_(OUT_LOW_FROZEN)
        pipe.out_high.fill_(OUT_HIGH_FROZEN)
        # Tier 2 init
        pipe.fwhm_raw.fill_(softplus_inverse(max(fwhm_init, 1e-4)))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.in_low.fill_(np.clip(THETA_MANUAL['in_low'], IN_LOW_MIN, IN_LOW_MAX))
        pipe.in_high.fill_(np.clip(THETA_MANUAL['in_high'], IN_HIGH_MIN, IN_HIGH_MAX))
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_MANUAL['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_MANUAL['ring_n']))
        pipe.sigma_post.fill_(THETA_MANUAL['sigma_post'])  # free, init=0.01

    # Freeze Tier 1 + out_*
    pipe.N_raw.requires_grad_(False)
    pipe.sigma_g_raw.requires_grad_(False)
    pipe.p.requires_grad_(False)
    pipe.out_low.requires_grad_(False)
    pipe.out_high.requires_grad_(False)
    # sigma_post: FREE (v7 change from v5c)
    pipe.sigma_post.requires_grad_(True)


def project_constraints_v7(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)
        pipe.sigma_post.clamp_(SIGMA_POST_MIN, SIGMA_POST_MAX)


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
            sigma_post=float(pipe.sigma_post.item()),
        )


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Load images ───────────────────────────────────────────────────────────────
print(f"\nLoading {N_TEMPLATES} templates …")
tpl_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]
templates_np_raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    for p in tpl_paths]
H, W = templates_np_raw[0].shape
templates_norm = [PRENORM_A * t + PRENORM_B for t in templates_np_raw]
templates_t = [torch.from_numpy(t).to(DEVICE) for t in templates_norm]

print(f"Loading {N_REAL} real images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
           for p in real_paths]
print(f"  real stats: mean={np.mean([i.mean() for i in real_np]):.4f}")

print(f"\nBuilding CalibrationLoss …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")


# ── One calibration run ───────────────────────────────────────────────────────
def run_one_v7(pipe, loss_fn, templates, rep_seed, label, lambda_weights,
               out_dir, fwhm_init=2.5):
    set_pipeline_v7(pipe, fwhm_init=fwhm_init)
    _seed_all(rep_seed)

    # Optimizer: Tier 2 + sigma_post
    tier2_raws   = [pipe.fwhm_raw, pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw]
    tier2_direct = [pipe.in_low, pipe.in_high, pipe.sigma_post]
    optimizer = torch.optim.Adam(
        [{'params': tier2_raws,   'lr': LR},
         {'params': tier2_direct, 'lr': LR}]
    )

    trace = []
    t0 = time.time()

    for step in range(N_STEPS):
        optimizer.zero_grad()
        x = templates[step % len(templates)]
        batch_loss = torch.tensor(0.0, device=DEVICE)
        batch_parts = {'L_vm': 0.0, 'L_nps': 0.0, 'L_rad': 0.0, 'L_hist': 0.0}

        for _ in range(BATCH_SIZE):
            eps = {
                'eps1':     torch.randn(H, W, device=DEVICE),
                'eps2':     torch.randn(H, W, device=DEVICE),
                'eps3':     torch.rand(H, W, device=DEVICE),
                'eps_post': torch.randn(H, W, device=DEVICE),
            }
            syn = pipe(x, eps)
            L, parts = loss_fn(syn, lambda_weights=lambda_weights)
            batch_loss = batch_loss + L
            for k in batch_parts:
                batch_parts[k] += parts[k].item()

        (batch_loss / BATCH_SIZE).backward()
        optimizer.step()
        project_constraints_v7(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time() - t0)
        row.update({k: v / BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step + 1) % 200 == 0:
            print(f"  [{label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"fwhm={phys['fwhm']:.4f}  σ_post={phys['sigma_post']:.5f}  "
                  f"alpha={phys['alpha']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'calib_trace.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        w.writeheader()
        w.writerows(trace)
    elapsed = time.time() - t0
    print(f"  [{label}] done: {N_STEPS} steps  elapsed={elapsed:.0f}s ({elapsed/60:.1f}min)")
    return get_theta_physical(pipe), trace


# ── Main ──────────────────────────────────────────────────────────────────────
pipe_opt = DegradationPipeline().to(DEVICE)
all_results = {}
t_total = time.time()

print(f"\n{'='*65}")
print("Phase 4 v7: Causal Closure Calibration (HANDOFF v7.10 §1)")
print(f"{'='*65}")

main_rep_thetas = []

for seed, label, lambdas in ALL_RUNS:
    rep_dir = OUT_DIR / f'rep_{label}'
    is_main = label.startswith('v7_r')
    print(f"\n[{'Main' if is_main else 'M3'}] seed={seed}  label={label}  λ={lambdas}")
    theta_hat, trace = run_one_v7(
        pipe_opt, loss_fn, templates_t,
        rep_seed=seed, label=label, lambda_weights=lambdas,
        out_dir=rep_dir, fwhm_init=2.5,
    )
    all_results[label] = theta_hat
    if is_main:
        main_rep_thetas.append(theta_hat)
    print(f"  θ̂_{label}: fwhm={theta_hat['fwhm']:.4f}  σ_post={theta_hat['sigma_post']:.5f}  "
          f"alpha={theta_hat['alpha']:.4f}")

print(f"\nAll runs done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate main 3 reps ─────────────────────────────────────────────────────
main_fwhm   = [r['fwhm'] for r in main_rep_thetas]
main_sigmap = [r['sigma_post'] for r in main_rep_thetas]

fwhm_median = float(np.median(main_fwhm))
fwhm_mean   = float(np.mean(main_fwhm))
fwhm_std    = float(np.std(main_fwhm))
sigmap_median = float(np.median(main_sigmap))
sigmap_mean   = float(np.mean(main_sigmap))

print(f"\n{'='*65}")
print("v7 Aggregate (3 main reps):")
print(f"  fwhm:    {main_fwhm[0]:.4f}, {main_fwhm[1]:.4f}, {main_fwhm[2]:.4f}")
print(f"  fwhm median = {fwhm_median:.4f}  mean = {fwhm_mean:.4f}  std = {fwhm_std:.4f}")
print(f"  σ_post:  {main_sigmap[0]:.5f}, {main_sigmap[1]:.5f}, {main_sigmap[2]:.5f}")
print(f"  σ_post median = {sigmap_median:.5f}  (ref v5e: 0.01695)")
print(f"  v5c reference fwhm: {V5C_FWHM:.4f}")

# Judgment
if fwhm_median >= 1.7:
    judgment = "인과 폐쇄 — σ_post 도입이 NPS 기반 f 추정을 ESF 방향으로 이동"
elif fwhm_median >= 1.45:
    judgment = "부분 이동"
else:
    judgment = "인과 미폐쇄 — f 수렴점 v5c 근방 잔류"

print(f"\n  판정: {judgment}")

# M-3 λ comparison
m3_lambda2x_fwhm = all_results.get('v7_m3_lambda2x', {}).get('fwhm', float('nan'))
m3_lambda05_fwhm = all_results.get('v7_m3_lambda05', {}).get('fwhm', float('nan'))
print(f"\n  M-3 λ 스케일링 fwhm 수렴:")
print(f"    λ=(1,1,1,1): mean={fwhm_mean:.4f}")
print(f"    λ=(2,2,2,2): {m3_lambda2x_fwhm:.4f}  Δ={m3_lambda2x_fwhm - fwhm_mean:+.4f}")
print(f"    λ=(0.5,...): {m3_lambda05_fwhm:.4f}  Δ={m3_lambda05_fwhm - fwhm_mean:+.4f}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
theta_mean_full = {p: float(np.mean([r.get(p, float('nan')) for r in main_rep_thetas]))
                   for p in PARAMS_V7}
theta_std_full  = {p: float(np.std( [r.get(p, float('nan')) for r in main_rep_thetas]))
                   for p in PARAMS_V7}
# Add frozen params
for k, v in [('N', TIER1_N), ('sigma_g', TIER1_SG), ('p', TIER1_P),
             ('out_low', OUT_LOW_FROZEN), ('out_high', OUT_HIGH_FROZEN)]:
    theta_mean_full[k] = v
    theta_std_full[k] = 0.0

output = {
    "version": "v7",
    "handoff_ref": "HANDOFF_v7.10 §1 지시 73",
    "purpose": "v5c + σ_post 자유 → 인과 폐쇄 실험",
    "tier1_frozen": {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "out_frozen":   {"out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN},
    "tier2_free":   PARAMS_V7,
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std":  theta_std_full,
    "theta_hat_reps":  main_rep_thetas,
    "v7_fwhm_median":   fwhm_median,
    "v7_sigma_post_median": sigmap_median,
    "v5c_fwhm_reference": V5C_FWHM,
    "judgment": judgment,
    "m3_lambda_check": {
        "lambda_1111_mean_fwhm":  fwhm_mean,
        "lambda_2222_fwhm":       m3_lambda2x_fwhm,
        "lambda_0505_fwhm":       m3_lambda05_fwhm,
        "lambda_2222_delta":      m3_lambda2x_fwhm - fwhm_mean,
        "lambda_0505_delta":      m3_lambda05_fwhm - fwhm_mean,
    },
    "all_run_thetas": all_results,
    "config": {
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE, "lr": LR,
        "device": DEVICE, "n_templates": N_TEMPLATES, "n_real": N_REAL,
    },
}

json_path = OUT_DIR / 'theta_real_v7.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {json_path}")

# ── Markdown report ───────────────────────────────────────────────────────────
md_path = OUT_DIR / 'calib_v7_report.md'
with open(md_path, 'w') as f:
    f.write("# Calibration v7 — Causal Closure Report\n\n")
    f.write("## Setup\n\n")
    f.write("v5c + σ_post free (Tier-1 + out_* frozen, same protocol)\n\n")
    f.write("| Parameter | Status | Value |\n|-----------|--------|-------|\n")
    f.write(f"| N | FROZEN | {TIER1_N} |\n")
    f.write(f"| σ_g | FROZEN | {TIER1_SG} |\n")
    f.write(f"| p | FROZEN | {TIER1_P} |\n")
    f.write(f"| out_low/high | FROZEN | {OUT_LOW_FROZEN}/{OUT_HIGH_FROZEN} |\n")
    f.write(f"| fwhm | FREE | init=2.5 |\n")
    f.write(f"| σ_post | **FREE (v7 추가)** | init=0.01 |\n")
    f.write(f"| other Tier2 | FREE | same as v5c |\n\n")
    f.write("## 3-Rep Results\n\n")
    f.write("| Rep | seed | fwhm | σ_post |\n|-----|------|------|--------|\n")
    for i, (seed, label, _), theta in zip(range(3), MAIN_REPS, main_rep_thetas):
        f.write(f"| {i} | {seed} | {theta['fwhm']:.4f} | {theta['sigma_post']:.5f} |\n")
    f.write(f"\nfwhm median = **{fwhm_median:.4f}**  (mean={fwhm_mean:.4f}, std={fwhm_std:.4f})\n")
    f.write(f"σ_post median = **{sigmap_median:.5f}**  (v5e ref: 0.01695)\n\n")
    f.write("## 판정\n\n")
    f.write(f"**{judgment}**\n\n")
    f.write("| 기준 | fwhm 범위 | 판정 | 원고 처리 |\n")
    f.write("|------|-----------|------|----------|\n")
    f.write("| ≥ 1.7 | ESF 방향 | 인과 폐쇄 | 기여 2 완성 |\n")
    f.write("| 1.45-1.7 | 부분 이동 | 부분 이동 | 이동량 보고 |\n")
    f.write("| < 1.45 | v5c 근방 | 인과 미폐쇄 | L-1(B) 채택 |\n\n")
    f.write(f"v7 fwhm median = **{fwhm_median:.4f}** → **{judgment}**\n\n")
    f.write("## M-3 λ Scaling Check\n\n")
    f.write("| λ | fwhm | Δ vs λ=(1,1,1,1) |\n|---|------|------------------|\n")
    f.write(f"| (1,1,1,1) | {fwhm_mean:.4f} | baseline |\n")
    f.write(f"| (2,2,2,2) | {m3_lambda2x_fwhm:.4f} | {m3_lambda2x_fwhm - fwhm_mean:+.4f} |\n")
    f.write(f"| (0.5,...) | {m3_lambda05_fwhm:.4f} | {m3_lambda05_fwhm - fwhm_mean:+.4f} |\n")

print(f"Saved: {md_path}")
print("\n=== run_phase4_v7.py done ===")
