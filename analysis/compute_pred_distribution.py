#!/usr/bin/env python3 -u
"""
Q3 — TEST_OOD 예측 분포 계산
출력:
  pred_distribution.csv  — config × pred_class% × gt_class%
목표: D_manual이 test_ood에서 배경(class_0)을 얼마나 예측하는지 확인
방법: 20 runs 평균 (F1 계산과 동일 기준)

OOM 수정: outer=모델(cond×run), inner=testset — 모델을 1회 로드 후 3 testset 평가, 즉시 해제
"""
from __future__ import annotations
import csv, gc, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
P42R_RUNS    = DIFFPIPE_DIR / 'phase4' / 'p42r_v2_runs'
TESTSETS_DIR = DIFFPIPE_DIR / 'phase4' / 'p42r_testsets'
OUT_DIR      = Path('/mnt/gstore/home/xiangyue/HBM_paper2')

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg
from dataset import json_to_mask

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}", flush=True)

TESTSET_NAMES = ['test_id', 'test_ood', 'test_cal']


def load_testset_items(ts_name: str):
    ts_dir = TESTSETS_DIR / ts_name
    meta = json.load(open(ts_dir / 'meta.json'))
    lbl_dir = Path(meta['label_dir'])
    items = []
    for lf in sorted(lbl_dir.glob('label_*.json')):
        num = lf.stem.replace('label_', '')
        np_path = ts_dir / f'image_{num}.npy'
        if np_path.exists():
            items.append((np_path, lf))
    print(f"  {ts_name}: {len(items)} images", flush=True)
    return items


def compute_gt_distribution(items):
    total_pix = 0
    class_counts = np.zeros(4, dtype=np.int64)
    for np_path, lf in items:
        img_deg = np.load(np_path)
        H_orig = img_deg.shape[0] - 6
        mask = json_to_mask(str(lf), H_orig, 520)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        for c in range(4):
            class_counts[c] += int((mask == c).sum())
        total_pix += mask.size
    return (class_counts / total_pix).tolist()


def compute_pred_distribution_single_run(model, items):
    total_pix = 0
    class_counts = np.zeros(4, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for np_path, lf in items:
            img_deg = np.load(np_path).astype(np.float32)
            img_3ch = np.stack([img_deg]*3, axis=0)
            tensor = torch.from_numpy(img_3ch).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            for c in range(4):
                class_counts[c] += int((pred == c).sum())
            total_pix += pred.size
    return (class_counts / total_pix).tolist()


# ── 사전 로드: 각 testset의 items와 GT 분포 ──────────────────────────────────
print("\nLoading testset items and GT distributions...", flush=True)
testset_items = {}
gt_pcts = {}
for ts_name in TESTSET_NAMES:
    testset_items[ts_name] = load_testset_items(ts_name)
    print(f"  Computing GT for {ts_name} ...", flush=True)
    gt_pcts[ts_name] = compute_gt_distribution(testset_items[ts_name])
    g = gt_pcts[ts_name]
    print(f"  GT: c0={g[0]:.4f}  c1={g[1]:.4f}  c2={g[2]:.4f}  c3={g[3]:.4f}", flush=True)

# ── 예측 수집: cond × run → 3 testset ───────────────────────────────────────
# run_results[cond][ts_name] = list of pred_pcts per run
run_results = {
    cond: {ts: [] for ts in TESTSET_NAMES}
    for cond in ['calib', 'manual']
}

for cond in ['calib', 'manual']:
    print(f"\n{'='*60}", flush=True)
    print(f"Condition: {cond}", flush=True)
    for run_idx in range(20):
        ckpt = P42R_RUNS / cond / f'run{run_idx}' / 'final.pth'
        if not ckpt.exists():
            print(f"  run{run_idx}: checkpoint not found, skipping", flush=True)
            continue

        # 모델 로드 1회
        model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))

        for ts_name in TESTSET_NAMES:
            pred_pcts = compute_pred_distribution_single_run(model, testset_items[ts_name])
            run_results[cond][ts_name].append(pred_pcts)
            p = pred_pcts
            print(f"  run{run_idx:02d}/{ts_name}: c0={p[0]:.4f}  c1={p[1]:.4f}  c2={p[2]:.4f}  c3={p[3]:.4f}", flush=True)

        # 모델 즉시 해제
        del model
        torch.cuda.empty_cache()
        gc.collect()

# ── 평균·표준편차 계산 및 CSV 작성 ──────────────────────────────────────────
results = []
for cond in ['calib', 'manual']:
    for ts_name in TESTSET_NAMES:
        pcts_list = run_results[cond][ts_name]
        if not pcts_list:
            continue
        arr = np.array(pcts_list)
        mean_pred = arr.mean(axis=0).tolist()
        std_pred  = arr.std(axis=0).tolist()
        g = gt_pcts[ts_name]
        print(f"\n{cond}/{ts_name} MEAN: c0={mean_pred[0]:.4f}  c1={mean_pred[1]:.4f}  c2={mean_pred[2]:.4f}  c3={mean_pred[3]:.4f}", flush=True)
        results.append({
            'config': cond,
            'testset': ts_name,
            'n_runs': len(pcts_list),
            'pred_class0_pct': round(mean_pred[0], 6),
            'pred_class1_pct': round(mean_pred[1], 6),
            'pred_class2_pct': round(mean_pred[2], 6),
            'pred_class3_pct': round(mean_pred[3], 6),
            'pred_class0_std': round(std_pred[0], 6),
            'pred_class1_std': round(std_pred[1], 6),
            'pred_class2_std': round(std_pred[2], 6),
            'pred_class3_std': round(std_pred[3], 6),
            'gt_class0_pct': round(g[0], 6),
            'gt_class1_pct': round(g[1], 6),
            'gt_class2_pct': round(g[2], 6),
            'gt_class3_pct': round(g[3], 6),
        })

out_csv = OUT_DIR / 'pred_distribution.csv'
fieldnames = list(results[0].keys())
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"\nSaved: {out_csv}", flush=True)
