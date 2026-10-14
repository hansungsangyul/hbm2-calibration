#!/usr/bin/env python3
"""
Phase 5 Grid Run — single (cell_index, rep) training run.
==========================================================
Grid: sigma_g (5) × fwhm (6, including 0=PSF-skip) = 30 cells × 5 reps = 150 runs.

Usage:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        /home/xiangyue/.conda/envs/unet/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5/run_phase5_grid.py \\
        --cell_index 15 --rep 0

Outputs: phase5/runs/sg{i}_fw{j}/rep{k}/results.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
RUNS_DIR     = DIFFPIPE_DIR / 'phase5' / 'runs'

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.signal import medfilt
from torch.utils.data import Dataset, DataLoader
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from model import UNet, CombinedLoss
import config as cfg

# ── Grid spec (HANDOFF v6.1 §7.2) ─────────────────────────────────────────
SIGMA_G_VALUES = [0.01025, 0.0205, 0.041, 0.082, 0.164]
FWHM_VALUES    = [0.0, 0.625, 1.25, 2.5, 5.0, 10.0]
N_FWHM         = len(FWHM_VALUES)   # 6

# λ_max measurement epochs (fixed schedule, no auto-relaxation)
LAMBDA_MAX_EPOCHS = {5, 10, 20, 30, 40}

# ── Constants ───────────────────────────────────────────────────────────────
PRENORM_A           = 0.501834
PRENORM_B           = 0.173320
LAMBDA_MAX_MICRO_BS = 4
LAMBDA_MAX_POWER_ITER = 10
LAMBDA_MAX_BATCH_SEED = 950000   # fixed: same batch across all 150 runs

# Fixed params = θ_manual (HANDOFF v6.1 §7.2, registry manual values)
FIXED_PARAMS = dict(
    N=300.0, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cell_index', type=int, required=True,
                    help='Grid cell index 0-29: cell_index = i_sigma*6 + j_fwhm')
parser.add_argument('--rep',        type=int, required=True,
                    help='Replica index 0-4')
parser.add_argument('--epochs',     type=int, default=41)
parser.add_argument('--batch_size', type=int, default=16,
                    help='Training batch size (default 16 to fit 32GB V100)')
args = parser.parse_args()

assert 0 <= args.cell_index < 30, f"cell_index must be 0-29, got {args.cell_index}"
assert 0 <= args.rep        < 5,  f"rep must be 0-4, got {args.rep}"

# Derive grid coordinates
i_sigma = args.cell_index // N_FWHM
j_fwhm  = args.cell_index  % N_FWHM
sigma_g = SIGMA_G_VALUES[i_sigma]
fwhm    = FWHM_VALUES[j_fwhm]
seed    = 800000 + args.cell_index * 10 + args.rep
cell_id = f"sg{i_sigma}_fw{j_fwhm}"

# Output path
out_dir  = RUNS_DIR / f"sg{i_sigma}_fw{j_fwhm}" / f"rep{args.rep}"
out_path = out_dir / 'results.json'

# ── Early exit if already done ──────────────────────────────────────────────
if out_path.exists():
    print(f"[SKIP] {out_path} already exists")
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True)

# ── Seed ────────────────────────────────────────────────────────────────────
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Phase 5 Grid Run")
print(f"  cell_index={args.cell_index}  cell_id={cell_id}  rep={args.rep}")
print(f"  sigma_g={sigma_g}  fwhm={fwhm}  seed={seed}")
print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lambda_max_epochs={sorted(LAMBDA_MAX_EPOCHS)}")
print(f"  device={DEVICE}")
print(f"  out={out_path}\n")


# ── Build DegradationPipeline ───────────────────────────────────────────────
def make_pipe(sg: float, fw: float) -> DegradationPipeline:
    """CPU pipeline (safe for DataLoader workers)."""
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_clamped = float(np.clip(FIXED_PARAMS['N'], _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(sg, 1e-6)))
        if fw > 0:
            pipe.fwhm_raw.fill_(softplus_inverse(fw))
        pipe.alpha_raw.fill_(softplus_inverse(FIXED_PARAMS['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(FIXED_PARAMS['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(FIXED_PARAMS['ring_n']))
        pipe.in_low.fill_(FIXED_PARAMS['in_low'])
        pipe.in_high.fill_(FIXED_PARAMS['in_high'])
        pipe.out_low.fill_(FIXED_PARAMS['out_low'])
        pipe.out_high.fill_(FIXED_PARAMS['out_high'])
        pipe.p.fill_(FIXED_PARAMS['p'])
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


# ── Dataset ─────────────────────────────────────────────────────────────────
class Phase5Dataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str, pipe: DegradationPipeline,
                 cell_fwhm: float, train: bool = True):
        self.pipe       = pipe
        self.cell_fwhm  = cell_fwhm
        self.train      = train

        img_dir  = Path(data_dir)
        lbl_dir  = Path(label_dir)
        lbl_files = sorted(lbl_dir.glob('*.json'))

        self.image_files, self.label_files = [], []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            ip  = img_dir / f'image_{num}.png'
            if ip.exists():
                self.image_files.append(ip)
                self.label_files.append(lf)

        print(f"  Dataset ({data_dir}): {len(self.image_files)} pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from dataset import json_to_mask
        img = cv2.imread(str(self.image_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = img.shape

        img_norm = PRENORM_A * img + PRENORM_B
        x = torch.from_numpy(img_norm)
        seed_i = idx + (1 if self.train else 100000)
        torch.manual_seed(seed_i)
        eps = {
            'eps1': torch.randn(H, W),
            'eps2': torch.randn(H, W),
            'eps3': torch.rand(H, W),
        }
        with torch.no_grad():
            if self.cell_fwhm == 0.0:
                # Skip PSF stage 4
                y = self.pipe._stage1_poisson(x, eps['eps1'])
                y = self.pipe._stage2_gaussian(y, eps['eps2'])
                y = self.pipe._stage3_salt_pepper(y, eps['eps3'])
                y = self.pipe._stage5_vignetting(y)
                y = self.pipe._stage6_contrast(y)
                y = self.pipe._stage7_ring(y)
            else:
                y = self.pipe(x, eps)
        img_deg = y.numpy()

        img_deg = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        img_deg = img_deg[:, 4:516]

        img_3ch = np.stack([img_deg, img_deg, img_deg], axis=0).astype(np.float32)

        mask = json_to_mask(str(self.label_files[idx]), H, W)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        mask = mask[:, 4:516]

        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))


# ── λ_max via micro-batch HVP ───────────────────────────────────────────────
def compute_lambda_max_micro(model: nn.Module, criterion, micro_batches,
                             n_iter: int = LAMBDA_MAX_POWER_ITER) -> float:
    torch.cuda.empty_cache()
    params = [p for p in model.parameters() if p.requires_grad]

    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / v_norm for vi in v]

    model.train()
    eigenvalue = None
    eigenvalue_prev = None
    n_total = sum(imgs.size(0) for imgs, _ in micro_batches)

    for _it in range(n_iter):
        Hv = [torch.zeros_like(p) for p in params]
        for imgs_k, masks_k in micro_batches:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            outputs_k = model(imgs_k)
            loss_k = criterion(outputs_k, masks_k)
            g_k = torch.autograd.grad(loss_k, params, create_graph=True)
            gv_k = sum((g * vi).sum() for g, vi in zip(g_k, v))
            Hv_k = torch.autograd.grad(gv_k, params, retain_graph=False)
            n_k = imgs_k.size(0)
            for j in range(len(Hv)):
                Hv[j] = Hv[j] + Hv_k[j] * n_k
            del g_k, Hv_k
            torch.cuda.empty_cache()

        Hv = [h / n_total for h in Hv]
        eigenvalue = float(sum((vi * hvi).sum() for vi, hvi in zip(v, Hv)).item())

        if (eigenvalue_prev is not None and
                abs(eigenvalue - eigenvalue_prev) / (abs(eigenvalue_prev) + 1e-12) < 0.01):
            break
        eigenvalue_prev = eigenvalue

        Hv_norm = torch.sqrt(sum((hvi**2).sum() for hvi in Hv))
        if Hv_norm.item() > 1e-12:
            v = [hvi / Hv_norm for hvi in Hv]
        else:
            break

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    return eigenvalue if eigenvalue is not None else float('nan')


# ── Gradient relative variance ───────────────────────────────────────────────
def compute_grad_relative_variance(model: nn.Module) -> float:
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().cpu().float().flatten())
    if not grads:
        return float('nan')
    g = torch.cat(grads)
    mean_abs = g.abs().mean().item()
    if mean_abs < 1e-12:
        return float('nan')
    return float(g.var().item() / (mean_abs**2))


# ── F1 ───────────────────────────────────────────────────────────────────────
def compute_f1(preds: torch.Tensor, targets: torch.Tensor, n_classes: int = 4) -> float:
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
    return float(np.mean(f1s))


# ── Divergence check (§V medfilt revision) ───────────────────────────────────
def check_diverged(train_losses, f1_raw, epoch, n_epochs):
    """Returns (diverged: bool, reason: int|None, f1_filt: list)."""
    f1_arr  = np.array(f1_raw, dtype=np.float32)
    k = 3 if len(f1_arr) >= 3 else 1
    f1_filt = medfilt(f1_arr, kernel_size=k).tolist()

    crit_1 = any(np.isnan(l) or np.isinf(l) for l in train_losses[-5:])
    crit_2 = (epoch >= n_epochs and f1_arr[-1] < 0.5)
    crit_3 = False
    if epoch >= 20 and len(f1_filt) >= 10:
        window = f1_filt[-10:]
        crit_3 = (max(window) - min(window)) > 0.1

    diverged = crit_1 or crit_2 or crit_3
    reason   = 1 if crit_1 else (2 if crit_2 else (3 if crit_3 else None))
    return diverged, reason, f1_filt


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    pipe = make_pipe(sigma_g, fwhm)

    train_img_dir  = str(AUGMENT_DIR / 'train_data/HBM_clean')
    val_img_dir    = str(AUGMENT_DIR / 'val_data/HBM_clean')
    lbl_dir_train  = str(AUGMENT_DIR / 'train_data/HBM_labels')
    lbl_dir_val    = str(AUGMENT_DIR / 'val_data/HBM_labels')

    train_ds = Phase5Dataset(train_img_dir, lbl_dir_train, pipe, fwhm, train=True)
    val_ds   = Phase5Dataset(val_img_dir,   lbl_dir_val,   pipe, fwhm, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # §P fixed λ_max micro-batches (seed 950000, same across all runs)
    LAMBDA_MAX_TOTAL = 32
    torch.manual_seed(LAMBDA_MAX_BATCH_SEED)
    lm_imgs, lm_masks = [], []
    for imgs, msks in train_loader:
        lm_imgs.append(imgs)
        lm_masks.append(msks)
        if sum(x.size(0) for x in lm_imgs) >= LAMBDA_MAX_TOTAL:
            break
    all_imgs  = torch.cat(lm_imgs)[:LAMBDA_MAX_TOTAL]
    all_masks = torch.cat(lm_masks)[:LAMBDA_MAX_TOTAL]
    micro_batches = [
        (all_imgs[i:i+LAMBDA_MAX_MICRO_BS].to(DEVICE),
         all_masks[i:i+LAMBDA_MAX_MICRO_BS].to(DEVICE))
        for i in range(0, LAMBDA_MAX_TOTAL, LAMBDA_MAX_MICRO_BS)
    ]
    # Restore training seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model     = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)

    f1_raw_list    = []
    train_loss_list = []
    val_loss_list  = []
    grad_rv_list   = []
    lambda_max_dict = {}   # epoch_str -> float|None
    diverged       = False
    divergence_reason = None
    f1_filt_list   = []

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        # Training
        model.train()
        tl_sum, n_steps = 0.0, 0
        batch_grad_rv = []
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, msks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            batch_grad_rv.append(compute_grad_relative_variance(model))
            optimizer.step()
            tl_sum  += loss.item()
            n_steps += 1
        train_loss = tl_sum / max(n_steps, 1)
        grad_rv    = float(np.nanmean(batch_grad_rv))
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        vl_sum, vf1_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                outs = model(imgs)
                vl_sum  += criterion(outs, msks).item()
                vf1_sum += compute_f1(outs, msks)
                n_val   += 1
        val_loss = vl_sum  / max(n_val, 1)
        val_f1   = vf1_sum / max(n_val, 1)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        f1_raw_list.append(val_f1)
        grad_rv_list.append(grad_rv)

        # λ_max (fixed schedule: epochs 5,10,20,30,40)
        lm_val = None
        if epoch in LAMBDA_MAX_EPOCHS:
            try:
                lm_val = compute_lambda_max_micro(model, criterion, micro_batches)
            except torch.cuda.OutOfMemoryError:
                print(f"  [λ_max OOM] epoch {epoch}; recording null")
                lm_val = None
            finally:
                import gc; gc.collect()
                torch.cuda.empty_cache()
            lambda_max_dict[str(epoch)] = lm_val

        epoch_wall = time.time() - t_epoch

        lm_str = f"  λ_max={lm_val:.4f}" if lm_val is not None else ""
        print(f"[{cell_id}/rep{args.rep}] Epoch {epoch:3d}/{args.epochs}  "
              f"tr={train_loss:.4f}  vl={val_loss:.4f}  F1={val_f1:.3f}  "
              f"grad_rv={grad_rv:.3f}{lm_str}  ({epoch_wall:.1f}s)")

        # Divergence check
        div, reason, f1_filt_list = check_diverged(
            train_loss_list, f1_raw_list, epoch, args.epochs)
        if div:
            diverged = True
            divergence_reason = reason
            print(f"\n[DIVERGED] criterion {reason} at epoch {epoch}")
            break

    # Fill missing λ_max epochs with None
    for ep in LAMBDA_MAX_EPOCHS:
        if str(ep) not in lambda_max_dict:
            lambda_max_dict[str(ep)] = None

    wallclock_s = time.time() - t_start

    result = {
        "cell_id":           cell_id,
        "i_sigma":           i_sigma,
        "j_fwhm":            j_fwhm,
        "sigma_g":           sigma_g,
        "fwhm":              fwhm,
        "seed":              seed,
        "epochs_completed":  len(f1_raw_list),
        "f1_raw":            f1_raw_list,
        "f1_filt":           f1_filt_list,
        "train_loss":        train_loss_list,
        "val_loss":          val_loss_list,
        "grad_rv":           grad_rv_list,
        "lambda_max":        lambda_max_dict,
        "diverged":          diverged,
        "divergence_reason": divergence_reason,
        "final_f1":          f1_raw_list[-1] if f1_raw_list else 0.0,
        "wallclock_s":       wallclock_s,
    }

    # Save final state_dict (HANDOFF v6.6 §1 지시 32)
    ckpt_path = out_dir / 'final.pth'
    torch.save(model.state_dict(), ckpt_path)

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{cell_id}/rep{args.rep}] Done — final_f1={result['final_f1']:.3f}  "
          f"diverged={diverged}  wallclock={wallclock_s/60:.1f}min")
    print(f"Saved: {out_path}  ckpt: {ckpt_path}")


if __name__ == '__main__':
    main()
