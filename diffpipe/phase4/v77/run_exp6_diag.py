#!/usr/bin/env python3
"""
HANDOFF v7.8 §3 (지시 72) — 실험 6 진단: 유능성 기준 달성 config 탐색
=====================================================================
대상 셀: (sigma_g=0.041, fwhm=1.25) = cell2  (유능성 기준 셀)
LR 스윕: {1e-4, 5e-4, 1e-3} × 2 seeds = 6 runs (예산 상한 10)
epochs: 60
모델: DeepLabV3+/ResNet50 (ImageNet 사전학습 백본, 분류기 헤드 교체)
GRV: UNet(Phase5)과 동일 산식  Var(g) / E[|g|]²

유능성 기준 (사전 등록): TEST_ID F1 ≥ 0.80
탈출: 진단 10 runs 내 미달 → "비교 가능 학습 미달성 — M8 수용" 보고

출력:
  phase4/v77/exp6_diag_results.csv
  phase4/v77/exp6_diag_summary.json
  phase4/v77/report_6_diag.md

Usage:
    /home/xiangyue/.conda/envs/unet/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/run_exp6_diag.py
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
from datetime import datetime

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
TESTSETS_DIR = DIFFPIPE_DIR / 'phase4' / 'p42r_testsets'
V77_DIR      = DIFFPIPE_DIR / 'phase4' / 'v77'
DIAG_DIR     = V77_DIR / 'exp6_diag_runs'

OUT_CSV  = V77_DIR / 'exp6_diag_results.csv'
OUT_JSON = V77_DIR / 'exp6_diag_summary.json'
OUT_REP  = V77_DIR / 'report_6_diag.md'

if OUT_CSV.exists():
    print(f'[SKIP] {OUT_CSV} already exists — delete to re-run')
    sys.exit(0)

DIAG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import medfilt as scipy_medfilt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
import config as cfg

from exp_config import PRENORM_A, PRENORM_B, COMPETENCE_THRESHOLD
from exp_utils import compute_grv, compute_f1_per_class

SIGMA_G   = 0.041
FWHM      = 1.25
EPOCHS    = 60
BATCH     = 16

LR_SEEDS = [
    (1e-4,  849000),
    (1e-4,  849001),
    (5e-4,  849100),
    (5e-4,  849101),
    (1e-3,  849200),
    (1e-3,  849201),
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[exp6_diag] device={DEVICE}  sigma_g={SIGMA_G}  fwhm={FWHM}  epochs={EPOCHS}')


# ── Pipeline ──────────────────────────────────────────────────────────────────
def make_pipe():
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_clamped = float(np.clip(300.0, _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(SIGMA_G, 1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(FWHM))
        pipe.alpha_raw.fill_(softplus_inverse(0.35))
        pipe.ring_strength_raw.fill_(softplus_inverse(0.12))
        pipe.ring_n_raw.fill_(softplus_inverse(8.0))
        pipe.in_low.fill_(0.05)
        pipe.in_high.fill_(0.90)
        pipe.out_low.fill_(0.10)
        pipe.out_high.fill_(0.90)
        pipe.p.fill_(0.001)
        pipe.sigma_post.fill_(0.0)
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


# ── Datasets ──────────────────────────────────────────────────────────────────
class TrainDataset(Dataset):
    def __init__(self, data_dir, label_dir, pipe, is_train=True):
        self.pipe = pipe
        self.is_train = is_train
        img_dir = Path(data_dir); lbl_dir = Path(label_dir)
        lbl_files = sorted(lbl_dir.glob('*.json'))
        self.image_files, self.label_files = [], []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            ip = img_dir / f'image_{num}.png'
            if ip.exists():
                self.image_files.append(ip)
                self.label_files.append(lf)
        print(f'  TrainDataset({data_dir}): {len(self.image_files)} pairs')

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        from dataset import json_to_mask
        img = cv2.imread(str(self.image_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = img.shape
        img = PRENORM_A * img + PRENORM_B
        x = torch.from_numpy(img)
        seed_i = idx + (0 if self.is_train else 100000)
        torch.manual_seed(seed_i)
        eps = {'eps1': torch.randn(H, W), 'eps2': torch.randn(H, W), 'eps3': torch.rand(H, W)}
        with torch.no_grad():
            y = self.pipe(x, eps)
        img_deg = y.numpy()
        img_deg = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        img_3ch = np.stack([img_deg] * 3, axis=0).astype(np.float32)
        mask = json_to_mask(str(self.label_files[idx]), H, W)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))


class FixedTestset(Dataset):
    def __init__(self, testset_name):
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


def eval_loader(model, loader):
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            out = model(imgs)['out']
            all_f1.append(compute_f1_per_class(out, msks))
    per_class = [float(np.mean([b[c] for b in all_f1])) for c in range(4)]
    return float(np.mean(per_class)), per_class


def check_diverged(f1_raw):
    f1_arr = np.array(f1_raw)
    k = 3 if len(f1_arr) >= 3 else 1
    f1_filt = scipy_medfilt(f1_arr, kernel_size=k).tolist()
    crit2 = f1_arr[-1] < 0.5
    crit3 = False
    if len(f1_filt) >= 10:
        crit3 = (max(f1_filt[-10:]) - min(f1_filt[-10:])) > 0.1
    return crit2 or crit3


# ── Build model ───────────────────────────────────────────────────────────────
def build_model():
    """DeepLabV3+/ResNet50 with ImageNet pretrained backbone, 4-class head."""
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4]     = nn.Conv2d(256, cfg.NUM_CLASSES, 1)
    model.aux_classifier[4] = nn.Conv2d(256, cfg.NUM_CLASSES, 1)
    return model.to(DEVICE)


# ── Main loop ─────────────────────────────────────────────────────────────────
pipe = make_pipe()
train_ds = TrainDataset(str(AUGMENT_DIR / 'train_data/HBM_clean'),
                        str(AUGMENT_DIR / 'train_data/HBM_labels'), pipe, is_train=True)
val_ds   = TrainDataset(str(AUGMENT_DIR / 'val_data/HBM_clean'),
                        str(AUGMENT_DIR / 'val_data/HBM_labels'),   pipe, is_train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
test_id_loader  = DataLoader(FixedTestset('test_id'),  batch_size=BATCH, num_workers=2)
test_ood_loader = DataLoader(FixedTestset('test_ood'), batch_size=BATCH, num_workers=2)

criterion = nn.CrossEntropyLoss()
rows = []

for run_no, (lr, seed) in enumerate(LR_SEEDS):
    print(f'\n{"="*60}')
    print(f'[diag] run {run_no+1}/{len(LR_SEEDS)}  lr={lr}  seed={seed}')
    print(f'{"="*60}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t_start = time.time()
    f1_raw_list = []

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        model.train()
        tl_sum, n_steps = 0.0, 0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)['out']
            loss = criterion(out, msks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            optimizer.step()
            tl_sum += loss.item(); n_steps += 1

        train_loss = tl_sum / max(n_steps, 1)

        model.eval()
        vf1_sum, vl_sum, n_val = 0.0, 0.0, 0
        grv_batch = []
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                out = model(imgs)['out']
                vl_sum  += criterion(out, msks).item()
                vf1_sum += float(np.mean(compute_f1_per_class(out, msks)))
                n_val   += 1

        # GRV: UNet 동일 산식 — needs backward on a val batch
        model.train()
        model.zero_grad()
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            out = model(imgs)['out']
            loss = criterion(out, msks)
            loss.backward()
            grv_batch.append(compute_grv(model))
            model.zero_grad()
            break  # 1 batch per epoch for GRV
        model.eval()

        val_f1   = vf1_sum / max(n_val, 1)
        val_loss = vl_sum  / max(n_val, 1)
        grv_val  = grv_batch[0] if grv_batch else float('nan')
        f1_raw_list.append(val_f1)

        print(f'  Epoch {epoch:3d}/{EPOCHS}  tr={train_loss:.4f}  vl={val_loss:.4f}  '
              f'F1={val_f1:.3f}  GRV={grv_val:.2f}  ({time.time()-t_ep:.1f}s)')

    # Test eval
    test_id,  _ = eval_loader(model, test_id_loader)
    test_ood, _ = eval_loader(model, test_ood_loader)
    diverged     = check_diverged(f1_raw_list)
    competent    = test_id >= COMPETENCE_THRESHOLD and not diverged
    wall         = time.time() - t_start

    print(f'\n  → TEST_ID={test_id:.4f}  TEST_OOD={test_ood:.4f}  '
          f'diverged={diverged}  COMPETENT={competent}  ({wall/60:.1f}min)')

    rows.append({
        'lr': lr, 'seed': seed, 'epochs': EPOCHS,
        'final_val_f1': f1_raw_list[-1] if f1_raw_list else float('nan'),
        'test_id': test_id, 'test_ood': test_ood,
        'diverged': diverged, 'competent': competent,
        'wall_min': wall / 60,
    })

    # 체크포인트 저장
    run_dir = DIAG_DIR / f'lr{lr}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / 'final.pth')

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    # 조기 탈출: 유능성 기준 달성 시 종료 불필요 (전건 실행)

# ── CSV 저장 ──────────────────────────────────────────────────────────────────
fieldnames = ['lr', 'seed', 'epochs', 'final_val_f1',
              'test_id', 'test_ood', 'diverged', 'competent', 'wall_min']
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
print(f'\n[exp6_diag] Saved CSV: {OUT_CSV}')

# ── 판정 ──────────────────────────────────────────────────────────────────────
competent_rows = [r for r in rows if r['competent']]
any_competent  = len(competent_rows) > 0

if any_competent:
    best = max(competent_rows, key=lambda r: r['test_id'])
    verdict = 'COMPETENT_FOUND'
    best_lr = best['lr']
    print(f'\n[exp6_diag] 유능 config 발견: lr={best_lr}  TEST_ID={best["test_id"]:.4f}')
else:
    verdict = 'COMPETENT_NOT_FOUND'
    best_lr = None
    print(f'\n[exp6_diag] 유능 config 미달 — M8 수용으로 종결')

summary = {
    'sigma_g': SIGMA_G, 'fwhm': FWHM,
    'competence_threshold': COMPETENCE_THRESHOLD,
    'any_competent': any_competent,
    'best_lr': best_lr,
    'verdict': verdict,
    'runs': rows,
}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[exp6_diag] Saved JSON: {OUT_JSON}')

# ── 보고서 ────────────────────────────────────────────────────────────────────
now = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = [
    f'# HANDOFF v7.8 §3 — 실험 6 진단 보고서',
    f'',
    f'**생성일**: {now}',
    f'**셀**: (σ_g={SIGMA_G}, fwhm={FWHM})  epochs={EPOCHS}  모델=DeepLabV3+/ResNet50(ImageNet)',
    f'**GRV 산식**: UNet(Phase5) 동일 — Var(g)/E[|g|]²',
    f'**유능성 기준**: TEST_ID F1 ≥ {COMPETENCE_THRESHOLD}',
    f'',
    f'## 결과',
    f'',
    f'| lr | seed | TEST_ID | TEST_OOD | diverged | COMPETENT |',
    f'|-----|------|---------|----------|----------|-----------|',
]
for r in rows:
    lines.append(
        f'| {r["lr"]:.0e} | {r["seed"]} | {r["test_id"]:.4f} | {r["test_ood"]:.4f} '
        f'| {r["diverged"]} | **{r["competent"]}** |'
    )

lines += ['', f'## 판정', f'']
if verdict == 'COMPETENT_FOUND':
    lines += [
        f'**COMPETENT_FOUND** — best_lr={best_lr}',
        f'  → 본실행 (run_exp6_v2.py --lr {best_lr}) 진행',
    ]
else:
    lines += [
        f'**COMPETENT_NOT_FOUND** (진단 {len(rows)} runs, 전건 유능성 기준 미달)',
        f'  → "비교 가능 학습 미달성으로 전이 검증 미결" — 기여3 하향(M8 수용)으로 종결',
    ]
lines += ['', '*HANDOFF v7.8 §3 진단 완료*']

with open(OUT_REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f'[exp6_diag] Saved report: {OUT_REP}')
print('[exp6_diag] Done.')
