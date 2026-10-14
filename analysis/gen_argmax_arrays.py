#!/usr/bin/env python3
"""
fig8 argmax 배열 저장 스크립트
출력:
  fig8_calib_argmax.png   — D_calib run0, image_0001, 패딩 제거 (314×512), 화소값=클래스 0~3
  fig8_manual_argmax.png  — D_manual run0, image_0001, 패딩 제거 (314×512), 화소값=클래스 0~3
  fig8_orig_image.png     — 원본 실영상 image_0001.png, 패딩 제거 (314×512), uint8 [0,255]
전처리: make_tensor_raw (prenorm 없음)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import cv2
import torch

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
P42R_RUNS    = DIFFPIPE_DIR / 'phase4' / 'p42r_v2_runs'
OUT_DIR      = Path('/mnt/gstore/home/xiangyue/HBM_paper2')

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── 실영상 로딩 (fpr_audit.py 동일 경로) ─────────────────────────────────────
def load_real_padded(path: Path) -> np.ndarray:
    """load_real_raw: 320×512 (패딩 포함)"""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img = cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
    return img  # shape (320, 512)

def make_tensor_raw(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([img] * 3).astype(np.float32)).unsqueeze(0)

# ── 모델 로딩 ─────────────────────────────────────────────────────────────────
print("Loading D_calib run0 ...")
model_calib = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
model_calib.load_state_dict(torch.load(P42R_RUNS / 'calib' / 'run0' / 'final.pth', map_location=DEVICE))
model_calib.eval()

print("Loading D_manual run0 ...")
model_manual = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
model_manual.load_state_dict(torch.load(P42R_RUNS / 'manual' / 'run0' / 'final.pth', map_location=DEVICE))
model_manual.eval()

# ── image_0001.png 처리 ───────────────────────────────────────────────────────
img_path = REAL_DIR / 'image_0001.png'
img_padded = load_real_padded(img_path)  # (320, 512)
tensor = make_tensor_raw(img_padded).to(DEVICE)

with torch.no_grad():
    logits_calib  = model_calib(tensor)
    logits_manual = model_manual(tensor)

argmax_calib_full  = logits_calib.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (320,512)
argmax_manual_full = logits_manual.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

# 패딩 3행 제거 → (314, 512)
PAD = 3
argmax_calib  = argmax_calib_full[PAD:-PAD, :]   # (314, 512)
argmax_manual = argmax_manual_full[PAD:-PAD, :]

# ── 검증 (vf_remeasure.md 표 재확인) ─────────────────────────────────────────
for name, arr_full in [('calib', argmax_calib_full), ('manual', argmax_manual_full)]:
    for c in range(4):
        n = int((arr_full == c).sum())
        print(f"  {name} class_{c}: {n}")

# 클래스별 (패딩 제거 후)
print("\n패딩 제거 후 (314×512 = 160,768 픽셀):")
for name, arr in [('calib', argmax_calib), ('manual', argmax_manual)]:
    for c in range(4):
        n = int((arr == c).sum())
        pct = n / (314 * 512) * 100
        print(f"  {name} class_{c}: {n} ({pct:.3f}%)")

# ── 원본 이미지 저장 (패딩 제거, uint8) ──────────────────────────────────────
orig_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
orig_crop = orig_raw[:, 4:516]  # (314, 512)
orig_uint8 = (orig_crop * 255.0).clip(0, 255).astype(np.uint8)

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_calib  = OUT_DIR / 'fig8_calib_argmax.png'
out_manual = OUT_DIR / 'fig8_manual_argmax.png'
out_orig   = OUT_DIR / 'fig8_orig_image.png'

cv2.imwrite(str(out_calib),  argmax_calib)
cv2.imwrite(str(out_manual), argmax_manual)
cv2.imwrite(str(out_orig),   orig_uint8)

# .npy도 함께 저장
np.save(str(OUT_DIR / 'fig8_calib_argmax.npy'),  argmax_calib)
np.save(str(OUT_DIR / 'fig8_manual_argmax.npy'), argmax_manual)

print(f"\nSaved:")
print(f"  {out_calib}  shape={argmax_calib.shape}  dtype={argmax_calib.dtype}  unique={np.unique(argmax_calib).tolist()}")
print(f"  {out_manual} shape={argmax_manual.shape}  dtype={argmax_manual.dtype}  unique={np.unique(argmax_manual).tolist()}")
print(f"  {out_orig}   shape={orig_uint8.shape}  dtype={orig_uint8.dtype}")
print(f"  fig8_calib_argmax.npy, fig8_manual_argmax.npy")
