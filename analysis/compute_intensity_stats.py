#!/usr/bin/env python3
"""
(2) 실영상 대 학습 배치 화소 강도 통계 — intensity_stats.csv

조건 (handoff 6차 §2):
  - 추론 전처리 적용 후 값 (D_calib: prenorm=True, D_manual: prenorm=False)
  - 실영상 = HBM_dirty2 첫 5매 (FPR 실험 동일)
  - 학습 배치 = prenorm(optional) → 파이프라인 → 화소 (학습 루프 순서 동일)
  - 영값 패딩 6행(3,072화소) 제외, 분모 160,768
  - 29% 수치에 맞추려 하지 말 것

학습 순서 (run_p42r_v2.py TrainDataset.__getitem__ 참조):
  D_calib:  img_f = prenorm(clean) → pipeline(img_f) → 모델 입력
  D_manual: img_f = clean (no prenorm) → pipeline(img_f) → 모델 입력

추론 순서 실영상 (run_exp2_real_fpr_v2.py USES_PRENORM 참조):
  D_calib:  real_img → prenorm → 모델 입력
  D_manual: real_img (no prenorm) → 모델 입력
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
import config as cfg

PRENORM_A = 0.501834
PRENORM_B = 0.173320
N_TRAIN_SAMPLES = 50   # 학습 배치 샘플 수

THETA_BASE = dict(N=300.0, sigma_g=0.0408, ring_strength=0.12, ring_n=8.0,
                  p=0.001, alpha=0.35, out_low=0.1, out_high=0.9)
CONFIGS = [
    ('calib',  True,  dict(fwhm=2.0956, in_low=0.169958, in_high=0.7,   **{k: v for k, v in THETA_BASE.items()})),
    ('manual', False, dict(fwhm=2.5,    in_low=0.05,     in_high=0.9,   **{k: v for k, v in THETA_BASE.items()})),
]


def apply_prenorm(img: np.ndarray) -> np.ndarray:
    return PRENORM_A * img + PRENORM_B


def make_pipeline(theta: dict) -> DegradationPipeline:
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
        'mean':    round(float(p.mean()), 6),
        'std':     round(float(p.std()),  6),
        'min':     round(float(p.min()),  6),
        'p1':      round(float(q[0]),     6),
        'p5':      round(float(q[1]),     6),
        'p25':     round(float(q[2]),     6),
        'p50':     round(float(q[3]),     6),
        'p75':     round(float(q[4]),     6),
        'p95':     round(float(q[5]),     6),
        'p99':     round(float(q[6]),     6),
        'max':     round(float(p.max()),  6),
    }


def load_real_pixels(path: Path, prenorm: bool) -> np.ndarray:
    """실영상 → 320×512 → 패딩 제외 314×512 → 전처리 → float array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    img_f = img.astype(np.float32) / 255.0
    # 학습 동일 패딩+크롭
    img_p = cv2.copyMakeBorder(img_f, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
    img_p = img_p[3:317, :]          # 패딩 6행 제거 → 314×512
    if prenorm:
        img_p = apply_prenorm(img_p)
    return img_p.ravel()


def sample_train_pixels(pipeline: DegradationPipeline, prenorm: bool,
                        n_images: int) -> np.ndarray:
    """
    학습 루프 순서 재현:
      prenorm(optional) → pipeline → 패딩+크롭 → 패딩 행 제거
    prenorm은 파이프라인 입력 전에 적용 (run_p42r_v2.py:141-144 와 동일 순서).
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
        # 학습 순서: prenorm BEFORE pipeline
        if prenorm:
            img_f = apply_prenorm(img_f)
        x   = torch.from_numpy(img_f)
        torch.manual_seed(100000 + i)
        eps = {'eps1': torch.randn(H, W), 'eps2': torch.randn(H, W),
               'eps3': torch.rand(H, W)}
        with torch.no_grad():
            y = pipeline(x, eps)
        img_deg = y.numpy()
        # 패딩+크롭 후 패딩 행 제거
        img_p = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        img_p = img_p[3:317, :]      # 314×512
        all_pix.append(img_p.ravel())
    return np.concatenate(all_pix)


# ── 실영상 경로 ────────────────────────────────────────────────────────────────
real_paths = sorted(REAL_DIR.glob('*.png'))[:5]
print(f"Real images: {[p.name for p in real_paths]}", flush=True)

rows = []
for config_name, prenorm, theta in CONFIGS:
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_name}  prenorm={prenorm}", flush=True)

    # 실영상 통계
    real_pix = np.concatenate([load_real_pixels(p, prenorm) for p in real_paths])
    s_real = compute_stats(real_pix)
    s_real.update({'config': config_name, 'source': 'real', 'n_images': len(real_paths)})
    print(f"  real  : mean={s_real['mean']:.4f}  std={s_real['std']:.4f}  "
          f"p50={s_real['p50']:.4f}  n_px={real_pix.size}", flush=True)

    # 학습 배치 통계
    print(f"  Generating {N_TRAIN_SAMPLES} training samples ...", flush=True)
    pipe = make_pipeline(theta)
    train_pix = sample_train_pixels(pipe, prenorm, N_TRAIN_SAMPLES)
    s_train = compute_stats(train_pix)
    s_train.update({'config': config_name, 'source': 'train', 'n_images': N_TRAIN_SAMPLES})
    print(f"  train : mean={s_train['mean']:.4f}  std={s_train['std']:.4f}  "
          f"p50={s_train['p50']:.4f}  n_px={train_pix.size}", flush=True)

    ratio = s_real['std'] / (s_train['std'] + 1e-12)
    print(f"  std_real / std_train = {ratio:.4f}  ({(ratio-1)*100:+.1f}%)", flush=True)

    rows.extend([s_real, s_train])

# ── 저장 ─────────────────────────────────────────────────────────────────────
fieldnames = ['config', 'source', 'n_images', 'n_pixels',
              'mean', 'std', 'min', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'max']
out_csv = OUT_DIR / 'intensity_stats.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {out_csv}", flush=True)
