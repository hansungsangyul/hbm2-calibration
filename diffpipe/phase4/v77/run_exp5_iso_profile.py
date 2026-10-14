#!/usr/bin/env python3
"""
실험 5 — f 고정 스윕 등손실 프로파일 (HANDOFF v7.7 §5)
=======================================================
f ∈ {1.4, 1.6, 1.8, 2.0, 2.2} 각각 고정하고
(N, σ_g)만 자유로 캘리브레이션. 나머지는 θ_manual 동결.

동결 파라미터 (v5c θ_manual):
  p=0.001, alpha=0.35, in_low=0.05, in_high=0.90
  out_low=0.10, out_high=0.90
  ring_strength=0.12, ring_n=8.0, sigma_post=0.0

캘리브레이션: Adam(lr=0.01), 2000 steps, CalibrationLoss, batch=8
이미지: CLEAN_DIR (N_TEMPLATES=50) + REAL_DIR (N_REAL=50)
각 f값 × 3 reps: seeds 930000, 930001, 930002

출력:
  v77/exp5_results.json — f별 × rep별 final_loss
  v77/exp5_summary.json — f별 loss 평균 + 판정
판정:
  (max_loss - min_loss) / min_loss < 0.25 → RIDGE_F_DIRECT
  ≥ 0.25                                  → F_COUPLING_WEAK

Usage:
    /home/xiangyue/.conda/envs/unet/bin/python run_exp5_iso_profile.py
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
V77_DIR      = DIFFPIPE_DIR / 'phase4' / 'v77'
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
CLEAN_DIR    = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

from exp_config import PRENORM_A, PRENORM_B

# ── Config ────────────────────────────────────────────────────────────────────
N_TEMPLATES = 50
N_REAL      = 50
N_STEPS     = 2000
BATCH_SIZE  = 8
LR          = 0.01

# θ_manual 동결 값 (v5c, ring_strength=0.12 포함)
THETA_MANUAL_FROZEN = dict(
    p=0.001,
    alpha=0.35,
    in_low=0.05,
    in_high=0.90,
    out_low=0.10,
    out_high=0.90,
    ring_strength=0.12,
    ring_n=8.0,
    sigma_post=0.0,
)

# N, sigma_g 초기값
N_INIT      = 300.0
SIGMA_G_INIT = 0.0408

# f 스윕 그리드
F_VALUES = [1.4, 1.6, 1.8, 2.0, 2.2]
REP_SEEDS = [930000, 930001, 930002]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'f sweep: {F_VALUES}')
print(f'Rep seeds: {REP_SEEDS}')
print(f'Free params: N, sigma_g  |  Fixed: {list(THETA_MANUAL_FROZEN.keys())}')

# ── Skip guard ────────────────────────────────────────────────────────────────
results_path = V77_DIR / 'exp5_results.json'
if results_path.exists():
    print(f'[SKIP] {results_path} already exists')
    sys.exit(0)

V77_DIR.mkdir(parents=True, exist_ok=True)

# ── Load images ───────────────────────────────────────────────────────────────
print(f'\nLoading {N_TEMPLATES} template images from {CLEAN_DIR} ...')
template_paths = sorted(CLEAN_DIR.glob('*.png'))[:N_TEMPLATES]
if len(template_paths) == 0:
    raise RuntimeError(f'No PNG files found in {CLEAN_DIR}')
templates_raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                 for p in template_paths]
H, W = templates_raw[0].shape
print(f'  Template size: H={H}, W={W}')
# prenorm 적용 (v5e 패턴 동일)
templates_t = [torch.from_numpy(PRENORM_A * t + PRENORM_B).to(DEVICE) for t in templates_raw]

print(f'Loading {N_REAL} real images from {REAL_DIR} ...')
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
if len(real_paths) == 0:
    raise RuntimeError(f'No PNG files found in {REAL_DIR}')
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
           for p in real_paths]

print('Building CalibrationLoss ...')
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print('  Done.')


# ── Pipeline 초기화 ───────────────────────────────────────────────────────────
def set_pipeline_exp5(pipe: DegradationPipeline, fwhm_val: float) -> None:
    """f 고정, N/sigma_g 자유, 나머지 θ_manual 동결."""
    with torch.no_grad():
        # 자유 파라미터 초기화
        N_c = float(np.clip(N_INIT, _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(SIGMA_G_INIT, 1e-6)))
        # f 고정
        pipe.fwhm_raw.fill_(softplus_inverse(fwhm_val))
        # θ_manual 동결
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL_FROZEN['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_MANUAL_FROZEN['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_MANUAL_FROZEN['ring_n']))
        pipe.in_low.fill_(THETA_MANUAL_FROZEN['in_low'])
        pipe.in_high.fill_(THETA_MANUAL_FROZEN['in_high'])
        pipe.out_low.fill_(THETA_MANUAL_FROZEN['out_low'])
        pipe.out_high.fill_(THETA_MANUAL_FROZEN['out_high'])
        pipe.p.fill_(THETA_MANUAL_FROZEN['p'])
        pipe.sigma_post.fill_(THETA_MANUAL_FROZEN['sigma_post'])

    # 모든 파라미터 동결 후 N_raw, sigma_g_raw만 해제
    for param in pipe.parameters():
        param.requires_grad_(False)
    pipe.N_raw.requires_grad_(True)
    pipe.sigma_g_raw.requires_grad_(True)


def get_theta_physical(pipe: DegradationPipeline) -> dict:
    with torch.no_grad():
        return dict(
            N=float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(pipe.N_raw)),
            sigma_g=float(F.softplus(pipe.sigma_g_raw)),
            fwhm=float(F.softplus(pipe.fwhm_raw)),
        )


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 단일 rep 캘리브레이션 ────────────────────────────────────────────────────
def run_one_rep(pipe: DegradationPipeline, fwhm_val: float,
                rep_seed: int, run_label: str) -> tuple[float, dict]:
    """
    단일 (f, seed) 캘리브레이션.
    Returns: (final_loss, final_theta_physical)
    """
    set_pipeline_exp5(pipe, fwhm_val)
    seed_all(rep_seed)

    optimizer = torch.optim.Adam(
        [pipe.N_raw, pipe.sigma_g_raw], lr=LR
    )

    t0 = time.time()
    final_loss = float('nan')

    for step in range(N_STEPS):
        optimizer.zero_grad()
        x = templates_t[step % len(templates_t)]
        batch_loss = torch.tensor(0.0, device=DEVICE)

        for _ in range(BATCH_SIZE):
            eps = {
                'eps1':     torch.randn(H, W, device=DEVICE),
                'eps2':     torch.randn(H, W, device=DEVICE),
                'eps3':     torch.rand(H, W, device=DEVICE),
                'eps_post': torch.randn(H, W, device=DEVICE),
            }
            syn = pipe(x, eps)
            L, _ = loss_fn(syn)
            batch_loss = batch_loss + L

        (batch_loss / BATCH_SIZE).backward()
        optimizer.step()

        final_loss = (batch_loss / BATCH_SIZE).item()

        if (step + 1) % 400 == 0:
            phys = get_theta_physical(pipe)
            print(f'  [{run_label}] step {step+1:4d}  L={final_loss:.5f}  '
                  f'N={phys["N"]:.1f}  sigma_g={phys["sigma_g"]:.4f}  '
                  f'({time.time()-t0:.0f}s)')

    theta_final = get_theta_physical(pipe)
    elapsed = time.time() - t0
    print(f'  [{run_label}] done: {N_STEPS} steps  L_final={final_loss:.5f}  '
          f'N={theta_final["N"]:.1f}  sigma_g={theta_final["sigma_g"]:.4f}  '
          f'elapsed={elapsed:.0f}s')
    return final_loss, theta_final


# ── Main 스윕 루프 ────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('Exp5: f 고정 스윕 등손실 프로파일 (HANDOFF v7.7 §5)')
print(f'{"="*65}')

pipe_opt = DegradationPipeline().to(DEVICE)

# results[f_val] = list of {rep, seed, final_loss, theta}
all_results: dict[float, list] = {}
t_total = time.time()

for fwhm_val in F_VALUES:
    print(f'\n[f={fwhm_val}] 시작')
    f_reps = []
    for rep, rep_seed in enumerate(REP_SEEDS):
        run_label = f'f{fwhm_val}_r{rep}'
        print(f'\n  [Rep {rep}] seed={rep_seed}  f={fwhm_val}')
        final_loss, theta_final = run_one_rep(
            pipe_opt, fwhm_val, rep_seed, run_label
        )
        f_reps.append({
            'rep':        rep,
            'seed':       rep_seed,
            'fwhm':       fwhm_val,
            'final_loss': final_loss,
            'theta':      theta_final,
        })
    all_results[fwhm_val] = f_reps
    losses = [r['final_loss'] for r in f_reps]
    print(f'  [f={fwhm_val}] rep losses: {[f"{l:.5f}" for l in losses]}  '
          f'mean={float(np.mean(losses)):.5f}')

print(f'\nAll sweeps done. Total: {(time.time()-t_total)/60:.1f} min')

# ── 결과 정리 & 판정 ─────────────────────────────────────────────────────────
# exp5_results.json: 전체 raw 데이터
results_data = {
    'version':    'exp5_v77',
    'experiment': '5_iso_profile',
    'f_values':   F_VALUES,
    'rep_seeds':  REP_SEEDS,
    'n_steps':    N_STEPS,
    'batch_size': BATCH_SIZE,
    'lr':         LR,
    'theta_manual_frozen': THETA_MANUAL_FROZEN,
    'prenorm': {'a': PRENORM_A, 'b': PRENORM_B},
    'results': {
        str(fwhm_val): reps
        for fwhm_val, reps in all_results.items()
    },
}

with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f'\nSaved: {results_path}')

# exp5_summary.json: f별 통계 + 판정
summary_rows = []
all_means = []
for fwhm_val in F_VALUES:
    reps = all_results[fwhm_val]
    losses = [r['final_loss'] for r in reps]
    mean_loss = float(np.mean(losses))
    std_loss  = float(np.std(losses))
    all_means.append(mean_loss)
    summary_rows.append({
        'fwhm':      fwhm_val,
        'rep_losses': losses,
        'mean_loss':  mean_loss,
        'std_loss':   std_loss,
    })

max_mean = max(all_means)
min_mean = min(all_means)
spread   = (max_mean - min_mean) / (min_mean + 1e-12)
verdict  = 'RIDGE_F_DIRECT' if spread < 0.25 else 'F_COUPLING_WEAK'

summary = {
    'version':    'exp5_v77',
    'experiment': '5_iso_profile',
    'per_f':      summary_rows,
    'spread_metric': {
        'max_mean_loss': max_mean,
        'min_mean_loss': min_mean,
        'relative_spread': spread,
        'threshold': 0.25,
    },
    'verdict': verdict,
}

summary_path = V77_DIR / 'exp5_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Saved: {summary_path}')

# ── 콘솔 최종 출력 ────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('Exp5 결과 요약:')
print(f'  {"f":>5}  {"mean_loss":>12}  {"std_loss":>10}')
print(f'  {"-"*35}')
for row in summary_rows:
    print(f'  {row["fwhm"]:>5.1f}  {row["mean_loss"]:>12.5f}  {row["std_loss"]:>10.5f}')
print(f'\n  spread = (max-min)/min = ({max_mean:.5f} - {min_mean:.5f}) / {min_mean:.5f} = {spread:.4f}')
print(f'  판정 (threshold=0.25): {verdict}')
print(f'\n{"="*65}')
print(f'Exp5 complete  ({(time.time()-t_total)/60:.1f} min)')
