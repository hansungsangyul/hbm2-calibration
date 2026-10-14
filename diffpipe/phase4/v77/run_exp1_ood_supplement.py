#!/usr/bin/env python3
"""
HANDOFF v7.8 §1 (지시 70) — 실험 1 OOD 보완
==============================================
calib 20 + manual 20 체크포인트를 TEST_OOD에 대해
(a) orig, (b) prenorm 추론 정합으로 재평가.

핵심 비교: calib+prenorm OOD  vs  manual+orig OOD
판정표:
  우위 ≥ +0.15 → PARTIAL_RESOLVED_OOD_RETAINED
  +0.05 ~ +0.15 → OOD_PARTIAL
  < +0.05       → OOD_ADVANTAGE_LOST

출력:
  phase4/v77/exp1_ood_supplement.csv
  phase4/v77/exp1_ood_supplement_summary.json
  phase4/v77/report_immediate_v2b.md  (v2 업데이트)
"""
from __future__ import annotations
import json, csv, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
PHASE4_DIR   = DIFFPIPE_DIR / 'phase4'
V77_DIR      = PHASE4_DIR / 'v77'
TESTSETS_DIR = PHASE4_DIR / 'p42r_testsets'
P42R_RUNS    = PHASE4_DIR / 'p42r_v2_runs'

OUT_CSV     = V77_DIR / 'exp1_ood_supplement.csv'
OUT_JSON    = V77_DIR / 'exp1_ood_supplement_summary.json'
OUT_REPORT  = V77_DIR / 'report_immediate_v2b.md'

if OUT_CSV.exists():
    print(f'[SKIP] {OUT_CSV} already exists — delete to re-run')
    sys.exit(0)

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg

from exp_config import PRENORM_A, PRENORM_B

BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[exp1_ood] device={DEVICE}')

CHECKPOINTS = []
for run_idx in range(20):
    CHECKPOINTS.append(('calib',  run_idx, P42R_RUNS / 'calib'  / f'run{run_idx}' / 'final.pth'))
    CHECKPOINTS.append(('manual', run_idx, P42R_RUNS / 'manual' / f'run{run_idx}' / 'final.pth'))


class FixedTestset(Dataset):
    def __init__(self, testset_name: str):
        ts_dir = TESTSETS_DIR / testset_name
        meta   = json.load(open(ts_dir / 'meta.json'))
        lbl_dir = Path(meta['label_dir'])
        lbl_files = sorted(lbl_dir.glob('*.json'))
        self.images, self.label_files = [], []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            np_path = ts_dir / f'image_{num}.npy'
            if np_path.exists():
                self.images.append(np_path)
                self.label_files.append(lf)
        print(f'  FixedTestset({testset_name}): {len(self.images)} images')

    def __len__(self): return len(self.images)

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


class NormWrapper(Dataset):
    def __init__(self, base: FixedTestset, mode: str):
        self.base = base
        self.mode = mode

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img_3ch, mask = self.base[idx]
        if self.mode == 'prenorm':
            img_3ch = PRENORM_A * img_3ch + PRENORM_B
        return img_3ch, mask


def compute_f1_per_class(preds, targets, n_classes=4):
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


def eval_loader(model, loader):
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            outs = model(imgs)
            all_f1.append(compute_f1_per_class(outs, msks))
    per_class = [float(np.mean([b[c] for b in all_f1])) for c in range(4)]
    return float(np.mean(per_class)), per_class


print('[exp1_ood] Loading TEST_OOD...')
base_ood = FixedTestset('test_ood')
loaders = {
    'orig':    DataLoader(NormWrapper(base_ood, 'orig'),    BATCH_SIZE, num_workers=NUM_WORKERS),
    'prenorm': DataLoader(NormWrapper(base_ood, 'prenorm'), BATCH_SIZE, num_workers=NUM_WORKERS),
}

rows = []
total = len(CHECKPOINTS)

for i, (condition, run_idx, ckpt_path) in enumerate(CHECKPOINTS):
    print(f'\n[exp1_ood] [{i+1}/{total}] {condition}/run{run_idx}')
    if not ckpt_path.exists():
        print(f'  [WARN] missing: {ckpt_path}')
        for mode in ('orig', 'prenorm'):
            rows.append({'condition': condition, 'run_idx': run_idx, 'mode': mode,
                         'test_ood_f1': float('nan'), 'missing': True})
        continue

    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    for mode, loader in loaders.items():
        f1, _ = eval_loader(model, loader)
        print(f'  mode={mode}: TEST_OOD={f1:.4f}')
        rows.append({'condition': condition, 'run_idx': run_idx, 'mode': mode,
                     'test_ood_f1': f1, 'missing': False})

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['condition', 'run_idx', 'mode', 'test_ood_f1', 'missing'])
    w.writeheader(); w.writerows(rows)
print(f'\n[exp1_ood] Saved CSV: {OUT_CSV}')


# ── 판정 ────────────────────────────────────────────────────────────────────────
def mean_f1(cond, mode):
    vals = [r['test_ood_f1'] for r in rows
            if r['condition'] == cond and r['mode'] == mode
            and not r['missing'] and not np.isnan(r['test_ood_f1'])]
    return float(np.mean(vals)) if vals else float('nan'), len(vals)

calib_prenorm_ood, n_c = mean_f1('calib', 'prenorm')
calib_orig_ood,    _   = mean_f1('calib', 'orig')
manual_orig_ood,   n_m = mean_f1('manual', 'orig')
manual_prenorm_ood, _  = mean_f1('manual', 'prenorm')

# 핵심 비교: calib+prenorm vs manual+orig
ood_advantage = calib_prenorm_ood - manual_orig_ood

if ood_advantage >= 0.15:
    verdict = 'PARTIAL_RESOLVED_OOD_RETAINED'
    msg = '§4.3 서술: "추론 시 모멘트 정합으로 ID 격차 축소(-0.015)되며 OOD 우위 유지 — 트레이드오프 완화 가능"'
elif ood_advantage >= 0.05:
    verdict = 'OOD_PARTIAL'
    msg = '완화 서술 + 우위 축소 정량 병기'
else:
    verdict = 'OOD_ADVANTAGE_LOST'
    msg = '"추론 정합은 ID를 회복시키나 OOD 우위를 상쇄 — 트레이드오프는 전처리 위치의 선택" 정직 서술'

summary = {
    'calib_prenorm_ood_mean': calib_prenorm_ood,
    'calib_orig_ood_mean':    calib_orig_ood,
    'manual_orig_ood_mean':   manual_orig_ood,
    'manual_prenorm_ood_mean': manual_prenorm_ood,
    'ood_advantage_calib_prenorm_vs_manual_orig': ood_advantage,
    'n_calib_conv': n_c,
    'n_manual_conv': n_m,
    'verdict': verdict,
    'manuscript_note': msg,
}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[exp1_ood] Saved JSON: {OUT_JSON}')

print('\n' + '='*65)
print('EXP1 OOD SUPPLEMENT VERDICT')
print('='*65)
print(f'  calib+prenorm OOD:  {calib_prenorm_ood:.4f}  (n={n_c})')
print(f'  manual+orig   OOD:  {manual_orig_ood:.4f}  (n={n_m})')
print(f'  OOD 우위:           {ood_advantage:+.4f}')
print(f'  → VERDICT: {verdict}')
print('='*65)


# ── report_immediate_v2b.md 생성 ───────────────────────────────────────────────
v2_path = V77_DIR / 'report_immediate_v2.md'
v2_text = v2_path.read_text(encoding='utf-8') if v2_path.exists() else ''

addendum = f"""
---
## 실험 1 OOD 보완 결과 (v7.8 §1, 업데이트 {datetime.now().strftime('%Y-%m-%d %H:%M')})

**핵심 비교 (배치 정직)**: calib+prenorm 추론 vs manual+원본

| 조건 | TEST_OOD F1 | n |
|------|------------|---|
| calib + prenorm 추론 | {calib_prenorm_ood:.4f} | {n_c} |
| calib + orig        | {calib_orig_ood:.4f} | — |
| manual + orig       | {manual_orig_ood:.4f} | {n_m} |
| manual + prenorm 강제 | {manual_prenorm_ood:.4f} | — |

**OOD 우위 (calib+prenorm − manual+orig)**: **{ood_advantage:+.4f}**

**판정**: **{verdict}**
  → {msg}

---
"""

with open(OUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(v2_text.rstrip() + addendum)
print(f'[exp1_ood] Updated report: {OUT_REPORT}')
print('[exp1_ood] Done.')
