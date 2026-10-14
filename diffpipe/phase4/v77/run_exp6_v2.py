#!/usr/bin/env python3
"""
HANDOFF v7.8 §3 — 실험 6 본실행 (진단 통과 config 사용)
=========================================================
- ImageNet 사전학습 백본 (DeepLabV3+/ResNet50)
- GRV 산식: UNet(Phase5) 동일  Var(g)/E[|g|]²
- LR: 진단 best_lr (--lr 인자로 전달)
- 4 cells × 5 seeds = 20 runs

Usage:
    python run_exp6_v2.py --cell_id 0 --rep 0 --lr 5e-4

판정: v7.7 §6 판정표 그대로 (report_6.py 재실행)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
TESTSETS_DIR = DIFFPIPE_DIR / 'phase4' / 'p42r_testsets'
V77_DIR      = DIFFPIPE_DIR / 'phase4' / 'v77'

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

parser = argparse.ArgumentParser()
parser.add_argument('--cell_id',    type=int, required=True, choices=[0, 1, 2, 3])
parser.add_argument('--rep',        type=int, required=True)
parser.add_argument('--lr',         type=float, required=True)
parser.add_argument('--epochs',     type=int, default=60)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

from exp_config import PRENORM_A, PRENORM_B, CELLS_V2 as CELLS
from exp_utils import compute_grv, compute_f1_per_class

cell    = CELLS[args.cell_id]
seed    = cell['seed_base'] + args.rep
sigma_g = cell['sigma_g']
fwhm    = cell['fwhm']

out_dir  = V77_DIR / 'exp6_v2_runs' / f'cell{args.cell_id}_rep{args.rep}'
out_path = out_dir / 'results.json'

if out_path.exists():
    print(f'[SKIP] {out_path} already exists')
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(seed)
np.random.seed(seed)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f'[exp6_v2] cell_id={args.cell_id}  rep={args.rep}  seed={seed}')
print(f'  sigma_g={sigma_g}  fwhm={fwhm}  lr={args.lr}  epochs={args.epochs}')
print(f'  model=DeepLabV3+/ResNet50(ImageNet)  GRV=Var(g)/E[|g|]²')


def make_pipe():
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_clamped = float(np.clip(300.0, _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(sigma_g, 1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(fwhm))
        pipe.alpha_raw.fill_(softplus_inverse(0.35))
        pipe.ring_strength_raw.fill_(softplus_inverse(0.12))
        pipe.ring_n_raw.fill_(softplus_inverse(8.0))
        pipe.in_low.fill_(0.05); pipe.in_high.fill_(0.90)
        pipe.out_low.fill_(0.10); pipe.out_high.fill_(0.90)
        pipe.p.fill_(0.001); pipe.sigma_post.fill_(0.0)
    for p in pipe.parameters(): p.requires_grad_(False)
    return pipe


class TrainDataset(Dataset):
    def __init__(self, data_dir, label_dir, pipe, is_train=True):
        self.pipe = pipe; self.is_train = is_train
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
        with torch.no_grad(): y = self.pipe(x, eps)
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
                self.images.append(np_path); self.label_files.append(lf)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        from dataset import json_to_mask
        img_deg = np.load(self.images[idx])
        img_3ch = np.stack([img_deg] * 3, axis=0).astype(np.float32)
        lf = self.label_files[idx]
        H_orig = img_deg.shape[0] - 6
        mask = json_to_mask(str(lf), H_orig, 520)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))


def eval_on_loader(model, loader):
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for imgs, msks in loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            out = model(imgs)['out']
            all_f1.append(compute_f1_per_class(out, msks))
    per_class = [float(np.mean([b[c] for b in all_f1])) for c in range(4)]
    return float(np.mean(per_class)), per_class


pipe = make_pipe()
train_ds = TrainDataset(str(AUGMENT_DIR / 'train_data/HBM_clean'),
                        str(AUGMENT_DIR / 'train_data/HBM_labels'), pipe, is_train=True)
val_ds   = TrainDataset(str(AUGMENT_DIR / 'val_data/HBM_clean'),
                        str(AUGMENT_DIR / 'val_data/HBM_labels'),   pipe, is_train=False)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loaders = {
    'test_id':  DataLoader(FixedTestset('test_id'),  batch_size=args.batch_size, num_workers=2),
    'test_ood': DataLoader(FixedTestset('test_ood'), batch_size=args.batch_size, num_workers=2),
}

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4]     = nn.Conv2d(256, cfg.NUM_CLASSES, 1)
model.aux_classifier[4] = nn.Conv2d(256, cfg.NUM_CLASSES, 1)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

t_start = time.time()
f1_raw_list, train_loss_list, grv_history = [], [], []

for epoch in range(1, args.epochs + 1):
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
    train_loss_list.append(train_loss)

    model.eval()
    vf1_sum, n_val = 0.0, 0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            out = model(imgs)['out']
            vf1_sum += float(np.mean(compute_f1_per_class(out, msks)))
            n_val += 1
    val_f1 = vf1_sum / max(n_val, 1)
    f1_raw_list.append(val_f1)

    # GRV: UNet 동일 산식
    model.train()
    model.zero_grad()
    for imgs, msks in val_loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        out = model(imgs)['out']
        loss = criterion(out, msks)
        loss.backward()
        break
    grv = compute_grv(model)
    grv_history.append(grv)
    model.zero_grad()
    model.eval()

    print(f'[cell{args.cell_id}/rep{args.rep}] Epoch {epoch:3d}/{args.epochs}  '
          f'tr={train_loss:.4f}  F1={val_f1:.3f}  GRV={grv:.2f}  ({time.time()-t_ep:.1f}s)')

# ── 사후 판정 ───────────────────────────────────────────────────────────────────
f1_arr  = np.array(f1_raw_list)
k = 3 if len(f1_arr) >= 3 else 1
f1_filt = scipy_medfilt(f1_arr, kernel_size=k).tolist()
crit2 = f1_arr[-1] < 0.5
crit3 = (max(f1_filt[-10:]) - min(f1_filt[-10:])) > 0.1 if len(f1_filt) >= 10 else False
diverged = bool(crit2 or crit3)

if diverged: print(f'\n[POST-HOC DIVERGED]')
else:
    print(f'\n[POST-HOC STABLE] crit3 range={(max(f1_filt[-10:])-min(f1_filt[-10:])):.4f}')

torch.save(model.state_dict(), out_dir / 'final.pth')

test_results = {}
for ts_name, loader in test_loaders.items():
    mean_f1, per_class = eval_on_loader(model, loader)
    test_results[ts_name] = {'mean_f1': mean_f1, 'f1_by_class': per_class}
    print(f'  [{ts_name}] F1={mean_f1:.4f}  per-class={[f"{v:.3f}" for v in per_class]}')

wall_s = time.time() - t_start
grv_mean = float(np.mean([g for g in grv_history if not np.isnan(g)])) if grv_history else 0.0

result = {
    'version': 'exp6_v2',
    'cell_id': args.cell_id, 'rep': args.rep, 'seed': seed,
    'lr': args.lr, 'sigma_g': sigma_g, 'fwhm': fwhm,
    'diverged': diverged,
    'final_f1': f1_raw_list[-1] if f1_raw_list else float('nan'),
    'grv_history': grv_history,
    'test_id':  test_results.get('test_id',  {}).get('mean_f1', float('nan')),
    'test_ood': test_results.get('test_ood', {}).get('mean_f1', float('nan')),
    'wallclock_s': wall_s,
}
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f'\n[cell{args.cell_id}/rep{args.rep}] Done  '
      f'final_f1={result["final_f1"]:.4f}  '
      f'TEST_ID={result["test_id"]:.4f}  TEST_OOD={result["test_ood"]:.4f}  '
      f'GRV_mean={grv_mean:.2f}  '
      f'diverged={diverged}  wall={wall_s/60:.1f}min')
print(f'Saved: {out_path}')
