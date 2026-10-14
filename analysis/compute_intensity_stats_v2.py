#!/usr/bin/env python3
"""
intensity_stats_v2.csv — D_calib 파이프라인 파라미터 수정 재측정

v1 오류: calib in_low=0.169958, in_high=0.70 (v5e_no_spost 체제값 — 잘못된 출처)
v2 수정: calib in_low=0.05, in_high=0.90  (run_p42r_v2.py THETA_BASE 실제 학습값)

확인 근거:
  run_p42r_v2.py:
    THETA_BASE = dict(N=300.0, ..., in_low=0.05, in_high=0.90, ...)
    CONDITIONS = {'calib': dict(fwhm=2.0956, prenorm=True, ...), ...}
  → D_calib 학습은 THETA_BASE in_low/in_high를 그대로 사용

측정 지점: model(x)에 넣기 직전 텐서 x — TrainDataset.__getitem__의 img_3ch
  = pipeline(prenorm(clean))의 출력, 패딩+크롭(320×512) 포함
  source: model 입력 = img_deg = pipeline 출력 (패딩 제외 후 = 314×512)

열: config, source, n_images, n_pixels, mean, std, ...
    source: 'train_tensor' (학습 루프 텐서), 'real_tensor' (실영상 텐서)
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

import numpy as np
import cv2
import torch

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
OUT_DIR      = Path('/mnt/gstore/home/xiangyue/HBM_paper2')

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI

PRENORM_A = 0.501834
PRENORM_B = 0.173320

# ── 실제 학습 파라미터 (run_p42r_v2.py THETA_BASE + CONDITIONS 동일) ──────────
# sigma_post는 학습에 없음 (eps_post=zeros로 고정, TrainDataset.__getitem__ 참조)
THETA_TRAIN_BASE = dict(
    N=300.0, sigma_g=0.0408, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)

CONFIGS = [
    # (name, prenorm, fwhm)
    # in_low/in_high: THETA_TRAIN_BASE (v1의 오류값 제거)
    ('calib',  True,  2.0956),
    ('manual', False, 2.5),
]

N_TRAIN_SAMPLES = 50


def apply_prenorm(img: np.ndarray) -> np.ndarray:
    return PRENORM_A * img + PRENORM_B


def make_pipeline(fwhm: float) -> DegradationPipeline:
    theta = dict(fwhm=fwhm, **THETA_TRAIN_BASE)
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_c = float(np.clip(theta['N'], _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], 1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(theta['ring_n']))
        pipe.in_low.fill_(theta['in_low'])
        pipe.in_high.fill_(theta['in_high'])
        pipe.out_low.fill_(theta['out_low'])
        pipe.out_high.fill_(theta['out_high'])
        pipe.p.fill_(theta['p'])
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


def compute_stats(pixels: np.ndarray) -> dict:
    p = pixels.astype(np.float64).ravel()
    q = np.percentile(p, [1, 5, 25, 50, 75, 95, 99])
    return {
        'n_pixels': int(len(p)),
        'mean':  round(float(p.mean()), 6),
        'std':   round(float(p.std()),  6),
        'min':   round(float(p.min()),  6),
        'p1':    round(float(q[0]),     6),
        'p5':    round(float(q[1]),     6),
        'p25':   round(float(q[2]),     6),
        'p50':   round(float(q[3]),     6),
        'p75':   round(float(q[4]),     6),
        'p95':   round(float(q[5]),     6),
        'p99':   round(float(q[6]),     6),
        'max':   round(float(p.max()),  6),
    }


def load_real_tensor_pixels(path: Path, prenorm: bool) -> np.ndarray:
    """
    실영상 → model 입력 텐서 화소 (source: real_tensor)
    run_exp2_real_fpr_v2.py와 동일:
      load_dirty2 → copyMakeBorder 3,3,0,0 → [:, 4:516] = 320×512
      prenorm 적용(D_calib) 또는 미적용(D_manual)
      패딩 제외: [3:317, :] = 314×512
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img_f = img.astype(np.float32) / 255.0
    img_p = cv2.copyMakeBorder(img_f, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
    if prenorm:
        img_p = apply_prenorm(img_p)
    img_p = img_p[3:317, :]   # 패딩 제외 → 314×512
    return img_p.ravel()


def sample_train_tensor_pixels(pipeline: DegradationPipeline, prenorm: bool,
                                n_images: int) -> np.ndarray:
    """
    학습 루프 img_3ch 재현 (source: train_tensor)
    run_p42r_v2.py TrainDataset.__getitem__ 동일:
      load PNG → /255 → prenorm?(선택) → pipeline → copyMakeBorder 3,3,0,0 → [:, 4:516]
    패딩 제외: [3:317, :] = 314×512
    sigma_post=0 (eps_post 없음, TrainDataset 동일)
    """
    clean_dir = AUGMENT_DIR / 'train_data' / 'HBM_clean'
    img_paths = sorted(clean_dir.glob('image_*.png'))[:n_images]
    all_pix = []
    for i, path in enumerate(img_paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_f = img.astype(np.float32) / 255.0
        H, W  = img_f.shape
        if prenorm:
            img_f = apply_prenorm(img_f)
        x   = torch.from_numpy(img_f)
        torch.manual_seed(100000 + i)
        eps = {'eps1': torch.randn(H, W), 'eps2': torch.randn(H, W),
               'eps3': torch.rand(H, W)}
        with torch.no_grad():
            y = pipeline(x, eps)
        img_deg = y.numpy()
        img_p = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        img_p = img_p[3:317, :]   # 패딩 제외 → 314×512
        all_pix.append(img_p.ravel())
    return np.concatenate(all_pix)


# ── 실영상 경로 ────────────────────────────────────────────────────────────────
real_paths = sorted(REAL_DIR.glob('*.png'))[:5]
print(f"Real images: {[p.name for p in real_paths]}", flush=True)
print(f"N_train_samples: {N_TRAIN_SAMPLES}", flush=True)

print("\n[v1 오류 파라미터 (참고)]")
print("  calib in_low=0.169958, in_high=0.70  ← v5e_no_spost 체제값 (answers_round3_raw.json:354)")
print("  -> 이 파라미터는 run_p42r_v2.py D_calib 학습에 사용되지 않았음")
print("\n[v2 수정 파라미터]")
print(f"  calib in_low=0.05, in_high=0.90  ← THETA_BASE (run_p42r_v2.py 실제 학습값)")

rows = []
for config_name, prenorm, fwhm in CONFIGS:
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_name}  prenorm={prenorm}  fwhm={fwhm}", flush=True)

    # 실영상 텐서
    real_pix = np.concatenate([load_real_tensor_pixels(p, prenorm) for p in real_paths])
    s_real = compute_stats(real_pix)
    s_real.update({'config': config_name, 'source': 'real_tensor', 'n_images': len(real_paths)})
    print(f"  real_tensor: mean={s_real['mean']:.4f}  std={s_real['std']:.4f}  "
          f"p50={s_real['p50']:.4f}  n_px={real_pix.size}", flush=True)

    # 학습 루프 텐서 재현
    print(f"  Generating {N_TRAIN_SAMPLES} train_tensor samples ...", flush=True)
    pipe = make_pipeline(fwhm)
    train_pix = sample_train_tensor_pixels(pipe, prenorm, N_TRAIN_SAMPLES)
    s_train = compute_stats(train_pix)
    s_train.update({'config': config_name, 'source': 'train_tensor', 'n_images': N_TRAIN_SAMPLES})
    print(f"  train_tensor: mean={s_train['mean']:.4f}  std={s_train['std']:.4f}  "
          f"p50={s_train['p50']:.4f}  n_px={train_pix.size}", flush=True)

    ratio = s_real['std'] / (s_train['std'] + 1e-12)
    print(f"  std(real)/std(train) = {ratio:.4f}  ({(ratio-1)*100:+.1f}%)", flush=True)

    rows.extend([s_real, s_train])

# ── 저장 ─────────────────────────────────────────────────────────────────────
fieldnames = ['config', 'source', 'n_images', 'n_pixels',
              'mean', 'std', 'min', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'max']
out_csv = OUT_DIR / 'intensity_stats_v2.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {out_csv}", flush=True)

# v1과의 비교 요약 출력
print("\n[v1 vs v2 비교 (calib/train)]")
print(f"  v1 calib/train std = 0.1593  (in_low=0.169958, in_high=0.70)")
calib_train = next(r for r in rows if r['config'] == 'calib' and r['source'] == 'train_tensor')
print(f"  v2 calib/train std = {calib_train['std']:.4f}  (in_low=0.05, in_high=0.90)")
calib_real  = next(r for r in rows if r['config'] == 'calib' and r['source'] == 'real_tensor')
print(f"  v2 calib/real  std = {calib_real['std']:.4f}")
print(f"  v2 std(real)/std(train) = {calib_real['std']/(calib_train['std']+1e-12):.4f}")
print(f"\n  T1(vf_remeasure_raw.json) std = 0.1117 (20장 패딩 포함, in_low=0.05)")
print(f"  v2 calib/train             std = {calib_train['std']:.4f} (50장 패딩 제외)")
