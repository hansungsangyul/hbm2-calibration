#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 1 — 추론 시 정규화 재평가 (M4)
================================================
100개 체크포인트 (calib×20, manual×20, ab1×20, ab2×20, ab3×20)를
TEST_ID · TEST_OOD에 대해 3가지 모드로 재평가:
  orig    : 정규화 없음
  zscore  : per-image z-score (채널별 독립)
  prenorm : PRENORM_A * img + PRENORM_B

출력:
  phase4/v77/exp1_results.csv
  phase4/v77/exp1_summary.json
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
CLEAN_DIR    = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
PHASE4_DIR   = DIFFPIPE_DIR / 'phase4'
V77_DIR      = PHASE4_DIR / 'v77'
TESTSETS_DIR = PHASE4_DIR / 'p42r_testsets'
P42R_RUNS    = PHASE4_DIR / 'p42r_v2_runs'
P43_RUNS     = PHASE4_DIR / 'p43_runs'

V77_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV  = V77_DIR / 'exp1_results.csv'
OUT_JSON = V77_DIR / 'exp1_summary.json'

if OUT_CSV.exists():
    print(f'[SKIP] {OUT_CSV} already exists')
    sys.exit(0)

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg

from exp_config import PRENORM_A, PRENORM_B

# ── Constants ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[exp1] device={DEVICE}')

# ── Checkpoint registry ────────────────────────────────────────────────────────
# (condition, run_idx, ckpt_path)
CHECKPOINTS = []
for run_idx in range(20):
    CHECKPOINTS.append(('calib',  run_idx, P42R_RUNS / 'calib'  / f'run{run_idx}' / 'final.pth'))
    CHECKPOINTS.append(('manual', run_idx, P42R_RUNS / 'manual' / f'run{run_idx}' / 'final.pth'))
for ab in ('ab1', 'ab2', 'ab3'):
    for run_idx in range(20):
        CHECKPOINTS.append((ab, run_idx, P43_RUNS / ab / f'run{run_idx}' / 'final.pth'))

TESTSETS = ['test_id', 'test_ood']
MODES    = ['orig', 'zscore', 'prenorm']


# ── FixedTestset (base) ────────────────────────────────────────────────────────
class FixedTestset(Dataset):
    def __init__(self, testset_name: str):
        ts_dir = TESTSETS_DIR / testset_name
        meta = json.load(open(ts_dir / 'meta.json'))
        lbl_dir = Path(meta['label_dir'])
        lbl_files = sorted(lbl_dir.glob('*.json'))
        self.images, self.label_files = [], []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            np_path = ts_dir / f'image_{num}.npy'
            if np_path.exists():
                self.images.append(np_path)
                self.label_files.append(lf)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from dataset import json_to_mask
        img_deg = np.load(self.images[idx])
        img_3ch = np.stack([img_deg] * 3, axis=0).astype(np.float32)
        lf = self.label_files[idx]
        H_orig = img_deg.shape[0] - 6
        W_orig = 520
        mask = json_to_mask(str(lf), H_orig, W_orig)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))


class NormWrapperDataset(Dataset):
    """Apply normalization mode on top of FixedTestset images."""
    def __init__(self, base: FixedTestset, mode: str):
        self.base = base
        self.mode = mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img_3ch, mask = self.base[idx]
        # img_3ch: [3, H, W] float32
        if self.mode == 'orig':
            pass
        elif self.mode == 'zscore':
            # per-channel independent z-score
            for c in range(img_3ch.shape[0]):
                ch = img_3ch[c]
                mu = ch.mean()
                sd = ch.std()
                img_3ch[c] = (ch - mu) / (sd + 1e-8)
        elif self.mode == 'prenorm':
            img_3ch = PRENORM_A * img_3ch + PRENORM_B
        return img_3ch, mask


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_f1_per_class(preds: torch.Tensor, targets: torch.Tensor, n_classes: int = 4):
    preds_flat = preds.argmax(dim=1).flatten()
    tgts_flat  = targets.flatten()
    f1s = []
    for c in range(n_classes):
        tp = ((preds_flat == c) & (tgts_flat == c)).sum().float()
        fp = ((preds_flat == c) & (tgts_flat != c)).sum().float()
        fn = ((preds_flat != c) & (tgts_flat == c)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append((2 * prec * rec / (prec + rec + 1e-8)).item())
    return f1s


def eval_on_loader(model: torch.nn.Module, loader: DataLoader) -> tuple[float, list]:
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            outs = model(imgs)
            all_f1.append(compute_f1_per_class(outs, msks))
    per_class = [float(np.mean([b[c] for b in all_f1])) for c in range(4)]
    return float(np.mean(per_class)), per_class


# ── Pre-build testsets ─────────────────────────────────────────────────────────
print('[exp1] Loading base testsets...')
base_testsets = {ts: FixedTestset(ts) for ts in TESTSETS}
print(f'  test_id:  {len(base_testsets["test_id"])} images')
print(f'  test_ood: {len(base_testsets["test_ood"])} images')

# Pre-build loaders for each (testset, mode) combination
loaders = {}
for ts_name, base_ds in base_testsets.items():
    for mode in MODES:
        wrapped = NormWrapperDataset(base_ds, mode)
        loaders[(ts_name, mode)] = DataLoader(
            wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

# ── Main evaluation loop ───────────────────────────────────────────────────────
rows = []
total = len(CHECKPOINTS)

for i, (condition, run_idx, ckpt_path) in enumerate(CHECKPOINTS):
    print(f'\n[exp1] [{i+1}/{total}] condition={condition}  run{run_idx}  ckpt={ckpt_path}')

    if not ckpt_path.exists():
        print(f'  [WARN] checkpoint not found: {ckpt_path}')
        for ts_name in TESTSETS:
            for mode in MODES:
                rows.append({
                    'condition': condition, 'run_idx': run_idx,
                    'testset': ts_name, 'mode': mode,
                    'mean_f1': float('nan'),
                    'f1_c0': float('nan'), 'f1_c1': float('nan'),
                    'f1_c2': float('nan'), 'f1_c3': float('nan'),
                })
        continue

    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    for ts_name in TESTSETS:
        for mode in MODES:
            loader = loaders[(ts_name, mode)]
            mean_f1, per_class = eval_on_loader(model, loader)
            print(f'  {ts_name}/{mode}: mean_f1={mean_f1:.4f}  per-class={[f"{v:.3f}" for v in per_class]}')
            rows.append({
                'condition': condition, 'run_idx': run_idx,
                'testset': ts_name, 'mode': mode,
                'mean_f1': mean_f1,
                'f1_c0': per_class[0], 'f1_c1': per_class[1],
                'f1_c2': per_class[2], 'f1_c3': per_class[3],
            })

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# ── Save CSV ───────────────────────────────────────────────────────────────────
fieldnames = ['condition', 'run_idx', 'testset', 'mode',
              'mean_f1', 'f1_c0', 'f1_c1', 'f1_c2', 'f1_c3']
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'\n[exp1] Saved CSV: {OUT_CSV}')

# ── Summary aggregation ────────────────────────────────────────────────────────
def get_cond_mode_ts(rows, condition, mode, testset):
    vals = [r['mean_f1'] for r in rows
            if r['condition'] == condition and r['mode'] == mode and r['testset'] == testset
            and not np.isnan(r['mean_f1'])]
    return float(np.mean(vals)) if vals else float('nan')


CONDITIONS_ALL = ['calib', 'manual', 'ab1', 'ab2', 'ab3']
summary = {'per_condition_mode': {}, 'verdicts': {}}

for mode in MODES:
    summary['per_condition_mode'][mode] = {}
    for cond in CONDITIONS_ALL:
        entry = {}
        for ts in TESTSETS:
            entry[ts] = get_cond_mode_ts(rows, cond, mode, ts)
        summary['per_condition_mode'][mode][cond] = entry

# ── Verdict per mode ───────────────────────────────────────────────────────────
# gap = calib_mean_id − manual_mean_id
# > -0.01 → ID_RESOLVED; -0.03 < gap <= -0.01 → ID_PARTIAL; <= -0.03 → TRADEOFF_REAL
for mode in MODES:
    calib_id  = summary['per_condition_mode'][mode].get('calib', {}).get('test_id', float('nan'))
    manual_id = summary['per_condition_mode'][mode].get('manual', {}).get('test_id', float('nan'))
    gap = calib_id - manual_id

    if np.isnan(gap):
        verdict = 'MISSING_DATA'
    elif gap > -0.01:
        verdict = 'ID_RESOLVED'
    elif gap > -0.03:
        verdict = 'ID_PARTIAL'
    else:
        verdict = 'TRADEOFF_REAL'

    summary['verdicts'][mode] = {
        'calib_mean_id':  calib_id,
        'manual_mean_id': manual_id,
        'gap_calib_minus_manual': gap,
        'verdict': verdict,
    }
    print(f'  [verdict/{mode}] gap={gap:.4f}  → {verdict}')

with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[exp1] Saved JSON: {OUT_JSON}')

# ── Console verdict summary ────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('EXPERIMENT 1 VERDICT SUMMARY')
print('=' * 65)
for mode in MODES:
    v = summary['verdicts'][mode]
    print(f"  mode={mode:8s}  gap={v['gap_calib_minus_manual']:+.4f}"
          f"  calib_ID={v['calib_mean_id']:.4f}"
          f"  manual_ID={v['manual_mean_id']:.4f}"
          f"  → {v['verdict']}")
print('=' * 65)
print('[exp1] Done.')
