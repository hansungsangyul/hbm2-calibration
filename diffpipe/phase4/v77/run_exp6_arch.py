#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 6 — DeepLabV3+ 아키텍처 전이 (Architecture Transfer)
======================================================================
fwhm 효과(GRV 신호)가 UNet에 특이적인지, 범용 아키텍처에서도 재현되는지 확인.

4 cells: (sigma_g, fwhm) ∈ {0.0205, 0.041} × {1.25, 5.0}
5 seeds each (rep 0~4) → 총 20 runs

파라미터:
  batch=16, epochs=41, 조기종료 없음
  prenorm=True (PRENORM_A=0.501834, PRENORM_B=0.173320)
  sigma_post=0.0, N=300, p=0.001, alpha=0.35
  in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90
  ring_strength=0.12, ring_n=8.0

모델: DeepLabV3+/ResNet50 (torchvision)
  출력: model(x)['out']

Usage:
    /home/xiangyue/.conda/envs/unet/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/run_exp6_arch.py \\
        --cell_id 0 --rep 0
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
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
import config as cfg

from exp_config import PRENORM_A, PRENORM_B, CELLS_ARCH as CELLS
from exp_utils import compute_grv

# ── Constants ─────────────────────────────────────────────────────────────────
THETA_BASE = dict(
    N=300.0, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
    sigma_post=0.0,
)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cell_id',    type=int, required=True, choices=[0, 1, 2, 3])
parser.add_argument('--rep',        type=int, required=True, help='0-4')
parser.add_argument('--epochs',     type=int, default=41)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

assert 0 <= args.rep < 5, f'rep must be 0-4, got {args.rep}'

cell   = CELLS[args.cell_id]
seed   = cell['seed_base'] + args.rep
sigma_g = cell['sigma_g']
fwhm    = cell['fwhm']

out_dir  = V77_DIR / 'exp6_runs' / f'cell{args.cell_id}_rep{args.rep}'
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

print(f'[exp6] cell_id={args.cell_id}  rep={args.rep}  seed={seed}')
print(f'  sigma_g={sigma_g}  fwhm={fwhm}  epochs={args.epochs}  batch={args.batch_size}')
print(f'  model=DeepLabV3+/ResNet50  device={DEVICE}')
print(f'  out={out_path}')


# ── Pipeline ──────────────────────────────────────────────────────────────────
def make_pipe() -> DegradationPipeline:
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_clamped = float(np.clip(THETA_BASE['N'], _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(sigma_g, 1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(fwhm))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_BASE['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_BASE['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_BASE['ring_n']))
        pipe.in_low.fill_(THETA_BASE['in_low'])
        pipe.in_high.fill_(THETA_BASE['in_high'])
        pipe.out_low.fill_(THETA_BASE['out_low'])
        pipe.out_high.fill_(THETA_BASE['out_high'])
        pipe.p.fill_(THETA_BASE['p'])
        pipe.sigma_post.fill_(THETA_BASE['sigma_post'])
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


# ── Training Dataset (on-the-fly) ─────────────────────────────────────────────
class TrainDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, pipe: DegradationPipeline, is_train=True):
        self.pipe     = pipe
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
        # prenorm=True
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


# ── Fixed Testset Dataset ─────────────────────────────────────────────────────
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_f1_per_class(preds: torch.Tensor, targets: torch.Tensor, n_classes=4):
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


# ── Divergence check (사후 판정) ───────────────────────────────────────────────
from scipy.signal import medfilt as scipy_medfilt


def check_diverged_posthoc(train_losses, f1_raw):
    f1_arr  = np.array(f1_raw, dtype=np.float64)
    k = 3 if len(f1_arr) >= 3 else 1
    f1_filt = scipy_medfilt(f1_arr, kernel_size=k).tolist()
    crit_1  = any(np.isnan(l) or np.isinf(l) for l in train_losses[-5:])
    crit_2  = f1_arr[-1] < 0.5
    crit_3  = False
    if len(f1_filt) >= 10:
        window = f1_filt[-10:]
        crit_3 = (max(window) - min(window)) > 0.1
    diverged = crit_1 or crit_2 or crit_3
    reason   = 1 if crit_1 else (2 if crit_2 else (3 if crit_3 else None))
    return diverged, reason, f1_filt


# ── DeepLabV3+ Loss (CrossEntropy, same as CombinedLoss baseline) ─────────────
class DeepLabLoss(nn.Module):
    """CrossEntropy loss compatible with DeepLabV3 dict output (applied externally)."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    pipe = make_pipe()

    train_ds = TrainDataset(
        str(AUGMENT_DIR / 'train_data/HBM_clean'),
        str(AUGMENT_DIR / 'train_data/HBM_labels'),
        pipe, is_train=True,
    )
    val_ds = TrainDataset(
        str(AUGMENT_DIR / 'val_data/HBM_clean'),
        str(AUGMENT_DIR / 'val_data/HBM_labels'),
        pipe, is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    test_loaders = {
        'test_id':  DataLoader(FixedTestset('test_id'),  batch_size=args.batch_size,
                               shuffle=False, num_workers=2),
        'test_ood': DataLoader(FixedTestset('test_ood'), batch_size=args.batch_size,
                               shuffle=False, num_workers=2),
    }

    # DeepLabV3+/ResNet50
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(num_classes=cfg.NUM_CLASSES, weights=None).to(DEVICE)

    criterion = DeepLabLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)

    f1_raw_list, train_loss_list, val_loss_list, grv_history = [], [], [], []

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
            tl_sum += loss.item()
            n_steps += 1
        train_loss = tl_sum / max(n_steps, 1)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Validation F1
        model.eval()
        vl_sum, vf1_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                out = model(imgs)['out']
                vl_sum  += criterion(out, msks).item()
                vf1_sum += float(np.mean(compute_f1_per_class(out, msks)))
                n_val   += 1
        val_f1   = vf1_sum / max(n_val, 1)
        val_loss = vl_sum  / max(n_val, 1)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        f1_raw_list.append(val_f1)

        # GRV 계산 (매 epoch) — exp_utils.compute_grv 산식
        model.train()
        model.zero_grad()
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            out = model(imgs)['out']  # DeepLab returns dict
            loss = criterion(out, msks)
            loss.backward()
            break
        grv = compute_grv(model)
        grv_history.append(grv)
        model.zero_grad()
        model.eval()

        print(f'[cell{args.cell_id}/rep{args.rep}] Epoch {epoch:3d}/{args.epochs}  '
              f'tr={train_loss:.4f}  vl={val_loss:.4f}  F1={val_f1:.3f}  '
              f'GRV={grv:.4f}  ({time.time()-t_ep:.1f}s)')

    # ── 사후 판정 ──────────────────────────────────────────────────────────────
    diverged, divergence_reason, f1_filt_list = check_diverged_posthoc(
        train_loss_list, f1_raw_list)
    if diverged:
        print(f'\n[POST-HOC DIVERGED] criterion {divergence_reason}')
    else:
        window = f1_filt_list[-10:] if len(f1_filt_list) >= 10 else f1_filt_list
        print(f'\n[POST-HOC STABLE] crit3 range={max(window)-min(window):.4f}')

    ckpt_path = out_dir / 'final.pth'
    torch.save(model.state_dict(), ckpt_path)

    # Test evaluation
    test_results = {}
    for ts_name, loader in test_loaders.items():
        mean_f1, per_class = eval_on_loader(model, loader)
        test_results[ts_name] = {'mean_f1': mean_f1, 'f1_by_class': per_class}
        print(f'  [{ts_name}] F1={mean_f1:.4f}  per-class={[f"{v:.3f}" for v in per_class]}')

    wallclock_s = time.time() - t_start

    result = {
        'version':          'exp6_v1',
        'cell_id':           args.cell_id,
        'rep':               args.rep,
        'seed':              seed,
        'sigma_g':           sigma_g,
        'fwhm':              fwhm,
        'diverged':          diverged,
        'divergence_reason': divergence_reason,
        'final_f1':          f1_raw_list[-1] if f1_raw_list else 0.0,
        'grv_history':       grv_history,
        'test_id':           test_results.get('test_id', {}).get('mean_f1', float('nan')),
        'test_ood':          test_results.get('test_ood', {}).get('mean_f1', float('nan')),
        'f1_raw':            f1_raw_list,
        'f1_filt':           f1_filt_list,
        'train_loss':        train_loss_list,
        'val_loss':          val_loss_list,
        'wallclock_s':       wallclock_s,
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n[cell{args.cell_id}/rep{args.rep}] Done'
          f'  final_f1={result["final_f1"]:.4f}'
          f'  TEST_ID={result["test_id"]:.4f}'
          f'  TEST_OOD={result["test_ood"]:.4f}'
          f'  GRV_mean={float(np.mean(grv_history)):.4f}'
          f'  diverged={diverged}  wall={wallclock_s/60:.1f}min')
    print(f'Saved: {out_path}  ckpt: {ckpt_path}')


if __name__ == '__main__':
    main()
