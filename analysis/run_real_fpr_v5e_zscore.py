#!/usr/bin/env python3
"""
(3) 미측정 두 구성 실영상 위양성 — real_fpr_v5e_zscore.csv

구성: v5e (exp3b_runs), zscore (exp4_runs)
실영상: HBM_dirty2 첫 5매 (Table 7 동일)
모집단: 20 runs × 5 images = 100 관측 (불안정 run 포함)
패딩 제외: 총 화소 = 314×512 = 160,768 (행 3:317, 열 0:512)
  → 6 패딩 행은 전 조건에서 배경 예측 → 분모에 포함시 FPR 구조적 편향

열:
  config, median_pixel_fpr, pct5_fpr, min_pixel_fpr,
  median_false_components, mean_bg_px
"""
from __future__ import annotations
import csv, gc, sys
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.ndimage import label as nd_label

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
V77          = DIFFPIPE_DIR / 'phase4' / 'v77'
OUT_DIR      = Path('/mnt/gstore/home/xiangyue/HBM_paper2')

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}", flush=True)

PRENORM_A = 0.501834
PRENORM_B = 0.173320

# 패딩 제외 화소 수: 320×512 전체에서 6행(top3+bot3) 제거 = 314×512
VALID_H_SLICE = slice(3, 317)   # 행 3~316 포함 = 314행
TOTAL_VALID   = 314 * 512       # = 160,768

MIN_COMPONENT_SIZE = 50
STRUCT = np.ones((3, 3), dtype=np.int32)


def load_dirty2(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]


def make_tensor_prenorm(raw: np.ndarray) -> torch.Tensor:
    img = PRENORM_A * raw + PRENORM_B
    return torch.from_numpy(np.stack([img]*3, axis=0).astype(np.float32)).unsqueeze(0)


def make_tensor_zscore(raw: np.ndarray) -> torch.Tensor:
    # z-score: 전체 이미지 단위 (패딩 포함, exp4 학습과 동일)
    img = (raw - raw.mean()) / (raw.std() + 1e-8)
    return torch.from_numpy(np.stack([img]*3, axis=0).astype(np.float32)).unsqueeze(0)


def compute_fpr_metrics_valid(pred_class: np.ndarray):
    """
    패딩 제외 (행 3:317) 영역만 분석.
    반환: (pixel_fpr, n_false_components, n_bg_px)
    """
    valid = pred_class[VALID_H_SLICE, :]      # (314, 512)
    non_bg = (valid != 0).astype(np.uint8)
    pixel_fpr = float(non_bg.sum()) / TOTAL_VALID
    labeled, n_comp = nd_label(non_bg, structure=STRUCT)
    n_false = sum(1 for cid in range(1, n_comp + 1)
                  if (labeled == cid).sum() >= MIN_COMPONENT_SIZE)
    n_bg    = int((valid == 0).sum())
    return pixel_fpr, n_false, n_bg


# ── 실영상 로드 ────────────────────────────────────────────────────────────────
dirty2_raw = [(load_dirty2(p), p.name)
              for p in sorted(REAL_DIR.glob('*.png'))[:5]]
print(f"Dirty2 images: {[n for _, n in dirty2_raw]}", flush=True)

CONFIGS = [
    ('v5e',    V77 / 'exp3b_runs', make_tensor_prenorm),
    ('zscore', V77 / 'exp4_runs',  make_tensor_zscore),
]

summary_rows = []

for config_name, runs_dir, make_tensor in CONFIGS:
    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_name}  runs_dir: {runs_dir.name}", flush=True)
    all_fprs, all_nfc, all_nbg = [], [], []

    for run_idx in range(20):
        ckpt = runs_dir / f'run{run_idx}' / 'final.pth'
        if not ckpt.exists():
            print(f"  run{run_idx}: checkpoint not found", flush=True)
            continue
        model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        model.eval()

        run_fprs, run_nfc, run_nbg = [], [], []
        with torch.no_grad():
            for raw_img, img_name in dirty2_raw:
                tensor = make_tensor(raw_img).to(DEVICE)
                logits = model(tensor)
                pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                fpr, nfc, nbg = compute_fpr_metrics_valid(pred)
                run_fprs.append(fpr)
                run_nfc.append(nfc)
                run_nbg.append(nbg)

        print(f"  run{run_idx:02d}: median_fpr={np.median(run_fprs):.4f}  "
              f"mean_nfc={np.mean(run_nfc):.1f}  mean_bg={np.mean(run_nbg):.0f}", flush=True)
        all_fprs.extend(run_fprs)
        all_nfc.extend(run_nfc)
        all_nbg.extend(run_nbg)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    n_obs = len(all_fprs)
    print(f"\n{config_name}: n_obs={n_obs}", flush=True)
    print(f"  median_fpr={np.median(all_fprs):.4f}  "
          f"pct5_fpr={np.percentile(all_fprs, 5):.4f}  "
          f"min_fpr={np.min(all_fprs):.4f}", flush=True)
    print(f"  median_nfc={np.median(all_nfc):.1f}  mean_bg_px={np.mean(all_nbg):.0f}", flush=True)

    summary_rows.append({
        'config':               config_name,
        'n_runs':               20,
        'n_obs':                n_obs,
        'median_pixel_fpr':     round(float(np.median(all_fprs)), 6),
        'pct5_fpr':             round(float(np.percentile(all_fprs, 5)), 6),
        'min_pixel_fpr':        round(float(np.min(all_fprs)), 6),
        'median_false_components': round(float(np.median(all_nfc)), 2),
        'mean_bg_px':           round(float(np.mean(all_nbg)), 1),
        'total_valid_px':       TOTAL_VALID,
        'denominator_note':     'padding_excluded_314x512',
    })

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_csv = OUT_DIR / 'real_fpr_v5e_zscore.csv'
fieldnames = ['config', 'n_runs', 'n_obs',
              'median_pixel_fpr', 'pct5_fpr', 'min_pixel_fpr',
              'median_false_components', 'mean_bg_px',
              'total_valid_px', 'denominator_note']
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nSaved: {out_csv}", flush=True)
