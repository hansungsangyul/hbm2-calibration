#!/usr/bin/env python3
"""
Phase 5 Pilot — UNet Training with DegradationPipeline Augmentation
===================================================================
파일럿 목적:
  1. 진단 로깅 (λ_max HVP, gradient relative variance) 구현 검증
  2. λ_max 계산 오버헤드 실측 → 150 runs 견적 갱신
  3. 실행 경로 확인

파일럿 사양:
  - 셀: σ_g=0.041, fwhm=2.5 (중앙 셀 = θ_manual 중심값)
  - seed=950000 (cell_idx=0, rep_idx=0 기준)
  - 100 epochs
  - λ_max: 5 epoch마다 (HVP, 고정 배치 256장, power iteration 20회)
  - 오버헤드 > 20% 시 자동 완화 → 10 epoch 주기

Run (conda:unet env):
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTHONUNBUFFERED=1 conda run -n unet python \
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5/run_phase5_pilot.py \
        --sigma_g 0.041 --fwhm 2.5 --seed 950000 --epochs 100 \
        2>&1 | tee /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5/pilot_run.log
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DIFFPIPE_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR   = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
PHASE5_DIR    = DIFFPIPE_DIR / 'phase5'

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from model import UNet, CombinedLoss
import config as cfg

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--sigma_g', type=float, required=True)
parser.add_argument('--fwhm',    type=float, required=True)
parser.add_argument('--seed',    type=int,   default=950000)
parser.add_argument('--epochs',  type=int,   default=100)
parser.add_argument('--lambda_max_freq', type=int, default=5,
                    help='Compute lambda_max every N epochs (default: 5)')
parser.add_argument('--lambda_max_batch', type=int, default=4,
                    help='Fixed batch size for lambda_max (default: 4; create_graph doubles memory; 64+ OOM on 32GB GPU with UNet)')
parser.add_argument('--out_dir', type=str,   default=str(PHASE5_DIR / 'pilot_output'))
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_PARAMS = dict(
    N=300.0, alpha=0.525, in_low=0.003, in_high=0.854,
    out_low=0.10, out_high=0.90, ring_strength=0.009, ring_n=8.6, p=0.001,
)
PRENORM_A = 0.501834
PRENORM_B = 0.173320
LAMBDA_MAX_POWER_ITER = 10   # §P: 10 iterations (was 20), early stop at <1% relative change
OVERHEAD_LIMIT = 0.20        # 20%
LAMBDA_MAX_MICRO_BS = 4      # micro-batch size for §P accumulation

print(f"Phase 5 Pilot")
print(f"  σ_g={args.sigma_g:.4f}  fwhm={args.fwhm:.3f}  seed={args.seed}")
print(f"  epochs={args.epochs}  λ_max every {args.lambda_max_freq} epochs")
print(f"  device={DEVICE}")
print(f"  out_dir={OUT_DIR}\n")


# ── Build DegradationPipeline for this cell ────────────────────────────────
def make_pipe(sigma_g: float, fwhm: float) -> DegradationPipeline:
    """Always returns a CPU pipeline (safe for DataLoader workers)."""
    pipe = DegradationPipeline()  # CPU by default
    with torch.no_grad():
        N_clamped = float(np.clip(FIXED_PARAMS['N'], _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(sigma_g, 1e-6)))
        if fwhm > 0:
            pipe.fwhm_raw.fill_(softplus_inverse(fwhm))
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
    return pipe  # CPU — do NOT call .to(DEVICE); workers can't use CUDA


# ── Dataset with DegradationPipeline augmentation ─────────────────────────
class Phase5Dataset(Dataset):
    """Clean HBM images → DegradationPipeline → 3-channel RGB for UNet."""

    def __init__(self, data_dir: str, label_dir: str, pipe: DegradationPipeline,
                 fwhm: float, train: bool = True):
        self.pipe = pipe
        self.fwhm = fwhm
        self.train = train
        self.device_cpu = torch.device('cpu')

        img_dir = Path(data_dir)
        lbl_dir = Path(label_dir)
        lbl_files = sorted(lbl_dir.glob('*.json'))

        self.image_files, self.label_files = [], []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            ip = img_dir / f'image_{num}.png'
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

        # Pre-normalization
        img_norm = PRENORM_A * img + PRENORM_B

        if self.fwhm > 0 or True:
            x = torch.from_numpy(img_norm)
            seed_i = idx + (1 if self.train else 100000)
            torch.manual_seed(seed_i)
            eps = {'eps1': torch.randn(H, W), 'eps2': torch.randn(H, W),
                   'eps3': torch.rand(H, W)}
            with torch.no_grad():
                y = self.pipe(x, eps)
            img_deg = y.numpy()
        else:
            img_deg = img_norm

        # Pad/crop to 320×512
        img_deg = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        img_deg = img_deg[:, 4:516]  # 320×512

        # To 3-channel (repeat grayscale)
        img_3ch = np.stack([img_deg, img_deg, img_deg], axis=0).astype(np.float32)

        # Mask
        mask = json_to_mask(str(self.label_files[idx]), H, W)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        mask = mask[:, 4:516]

        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))


# ── λ_max via micro-batch HVP (§P accumulation) ───────────────────────────
def compute_lambda_max_micro(model: nn.Module, criterion, micro_batches,
                             n_iter: int = LAMBDA_MAX_POWER_ITER) -> float:
    """
    §P: Max eigenvalue of UNet training loss Hessian via power iteration.
    Micro-batch HVP accumulation: Hv = (1/K) Σ_k Hv_k, micro-batch of 4.
    Memory = one micro-batch graph (not full 32-image graph).
    Early stop: |λ - λ_prev| / |λ_prev| < 1%.

    micro_batches: list of (imgs_k, masks_k) each size LAMBDA_MAX_MICRO_BS.
    Returns: float λ_max estimate.
    """
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
        # Accumulate Hv across micro-batches
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

        # Early stop: relative change < 1%
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


def compute_lambda_max(model: nn.Module, criterion, fixed_batch,
                       n_iter: int = LAMBDA_MAX_POWER_ITER) -> float:
    """Legacy single-batch HVP (batch=4). Kept for §P comparison at checkpoint."""
    images, masks = fixed_batch
    torch.cuda.empty_cache()
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / v_norm for vi in v]
    model.train()
    eigenvalue = None
    for _it in range(n_iter):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        outputs = model(images)
        loss = criterion(outputs, masks)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(gv, params, retain_graph=False)
        eigenvalue = float(sum((vi * hvi).sum() for vi, hvi in zip(v, Hv)).item())
        Hv_norm = torch.sqrt(sum((hvi**2).sum() for hvi in Hv))
        if Hv_norm.item() > 1e-12:
            v = [hvi / Hv_norm for hvi in Hv]
        else:
            break
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    return eigenvalue if eigenvalue is not None else float('nan')


# ── Gradient relative variance ─────────────────────────────────────────────
def compute_grad_relative_variance(model: nn.Module) -> float:
    """Relative variance = Var(grad) / E[|grad|]^2 across all grad elements."""
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


# ── F1 from confusion matrix ───────────────────────────────────────────────
def compute_f1(preds: torch.Tensor, targets: torch.Tensor, n_classes: int = 4) -> float:
    """Macro-averaged F1 score."""
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


# ── Divergence check ───────────────────────────────────────────────────────
def check_divergence(history: list, epoch: int) -> tuple:
    """Returns (diverged: bool, reason: str)."""
    losses = [h['train_loss'] for h in history]
    f1s    = [h['val_f1'] for h in history]

    if any(np.isnan(l) or np.isinf(l) for l in losses[-5:]):
        return True, "loss NaN/Inf"
    if epoch >= args.epochs and f1s and f1s[-1] < 0.5:
        return True, f"final F1={f1s[-1]:.3f} < 0.5"
    # F1 variation check: only after epoch 20 (warmup); initial convergence legitimately varies > 0.1
    if epoch >= 20 and len(f1s) >= 10 and max(f1s[-10:]) - min(f1s[-10:]) > 0.1:
        return True, f"F1 variation={max(f1s[-10:]) - min(f1s[-10:]):.3f} > 0.1 in last 10 epochs"
    return False, ""


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Build pipeline
    pipe = make_pipe(args.sigma_g, args.fwhm)  # CPU: DataLoader workers can't use CUDA

    # Datasets
    train_img_dir = str(AUGMENT_DIR / 'train_data/HBM_clean')
    val_img_dir   = str(AUGMENT_DIR / 'val_data/HBM_clean')
    lbl_dir_train = str(AUGMENT_DIR / 'train_data/HBM_labels')
    lbl_dir_val   = str(AUGMENT_DIR / 'val_data/HBM_labels')

    train_ds = Phase5Dataset(train_img_dir, lbl_dir_train, pipe, args.fwhm, train=True)
    val_ds   = Phase5Dataset(val_img_dir,   lbl_dir_val,   pipe, args.fwhm, train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # §P: Build fixed λ_max micro-batches (32 images total, same seed 950000 for all runs)
    # Split into micro-batches of LAMBDA_MAX_MICRO_BS to avoid OOM with create_graph=True
    LAMBDA_MAX_TOTAL = 32   # §P spec: 32 images total, same across all runs
    LAMBDA_MAX_BATCH = args.lambda_max_batch  # kept for legacy single-batch comparison
    lm_seed = 950000        # deterministic seed — same batch for all 150 runs
    torch.manual_seed(lm_seed)
    print(f"Building §P λ_max micro-batches ({LAMBDA_MAX_TOTAL} images total, "
          f"micro_bs={LAMBDA_MAX_MICRO_BS}, seed={lm_seed}) …")
    lambda_imgs, lambda_masks = [], []
    for bi, (imgs, msks) in enumerate(train_loader):
        lambda_imgs.append(imgs)
        lambda_masks.append(msks)
        if sum(x.size(0) for x in lambda_imgs) >= LAMBDA_MAX_TOTAL:
            break
    all_imgs  = torch.cat(lambda_imgs)[:LAMBDA_MAX_TOTAL]
    all_masks = torch.cat(lambda_masks)[:LAMBDA_MAX_TOTAL]
    micro_batches = [
        (all_imgs[i:i+LAMBDA_MAX_MICRO_BS].to(DEVICE),
         all_masks[i:i+LAMBDA_MAX_MICRO_BS].to(DEVICE))
        for i in range(0, LAMBDA_MAX_TOTAL, LAMBDA_MAX_MICRO_BS)
    ]
    # Also keep small fixed_batch for legacy comparison (§P validation)
    fixed_batch = (all_imgs[:LAMBDA_MAX_BATCH].to(DEVICE), all_masks[:LAMBDA_MAX_BATCH].to(DEVICE))
    print(f"  §P micro-batches: {len(micro_batches)} × {LAMBDA_MAX_MICRO_BS} imgs  "
          f"(total {LAMBDA_MAX_TOTAL})\n")
    # Restore training seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Model
    model     = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                                  weight_decay=cfg.WEIGHT_DECAY)

    history = []
    lambda_max_freq = args.lambda_max_freq
    overhead_warned = False
    epoch_time_no_lm = None

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        # Training
        model.train()
        train_loss_sum, n_steps = 0.0, 0
        grad_rv_list = []
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, msks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            grad_rv_list.append(compute_grad_relative_variance(model))
            optimizer.step()
            train_loss_sum += loss.item()
            n_steps += 1

        train_loss = train_loss_sum / max(n_steps, 1)
        grad_rv = float(np.nanmean(grad_rv_list))
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_loss_sum, val_f1_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                outs = model(imgs)
                val_loss_sum += criterion(outs, msks).item()
                val_f1_sum   += compute_f1(outs, msks)
                n_val += 1
        val_loss = val_loss_sum / max(n_val, 1)
        val_f1   = val_f1_sum   / max(n_val, 1)

        # §P: λ_max via micro-batch HVP accumulation (every lambda_max_freq epochs)
        lm_val = float('nan')
        lm_time = 0.0
        if epoch % lambda_max_freq == 0:
            t_lm = time.time()
            try:
                lm_val = compute_lambda_max_micro(model, criterion, micro_batches)
            except torch.cuda.OutOfMemoryError:
                print(f"  [λ_max] §P micro-batch OOM at epoch {epoch}; skipping.")
                lm_val = float('nan')
            lm_time = time.time() - t_lm

            # Overhead check (first computation)
            if not overhead_warned and epoch_time_no_lm is not None:
                overhead_frac = lm_time / max(epoch_time_no_lm, 1e-3)
                if overhead_frac > OVERHEAD_LIMIT:
                    lambda_max_freq = 10
                    print(f"  [WARN] §P λ_max overhead={overhead_frac*100:.0f}% > 20%. "
                          f"Relaxing to every 10 epochs (pre-authorized).")
                    overhead_warned = True
                else:
                    print(f"  [OK] §P λ_max overhead={overhead_frac*100:.0f}% ≤ 20%.")
                    overhead_warned = True

        epoch_wall = time.time() - t_epoch
        if epoch_time_no_lm is None and np.isnan(lm_val):
            epoch_time_no_lm = epoch_wall

        rec = dict(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_f1=val_f1,
            grad_rv=grad_rv,
            lambda_max=lm_val,
            lm_time_sec=lm_time,
            epoch_wall_sec=epoch_wall,
            lambda_max_freq=lambda_max_freq,
        )
        history.append(rec)

        lm_str = f"  λ_max={lm_val:.4f} ({lm_time:.1f}s)" if not np.isnan(lm_val) else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"tr={train_loss:.4f}  vl={val_loss:.4f}  F1={val_f1:.3f}  "
              f"grad_rv={grad_rv:.3f}{lm_str}  ({epoch_wall:.1f}s)")

        # Divergence check
        diverged, reason = check_divergence(history, epoch)
        if diverged:
            print(f"\n[DIVERGED] {reason} at epoch {epoch}")
            break

    # ── Save checkpoint for §P comparison ────────────────────────────────────
    ckpt_path = OUT_DIR / 'pilot_checkpoint.pth'
    torch.save({'model_state_dict': model.state_dict(), 'epoch': len(history)}, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # ── §P: Compare old batch=4 vs new micro-batch=32 at final checkpoint ────
    sp_comparison = {}
    print("\n§P validation: λ_max(batch=4, legacy) vs λ_max(32-img micro-batch, §P)  …")
    try:
        t_old = time.time()
        lm_old = compute_lambda_max(model, criterion, fixed_batch)
        t_old = time.time() - t_old
        t_new = time.time()
        lm_new = compute_lambda_max_micro(model, criterion, micro_batches)
        t_new = time.time() - t_new
        lm_delta_pct = abs(lm_old - lm_new) / (abs(lm_new) + 1e-12) * 100
        print(f"  batch=4  λ_max={lm_old:.2f}  ({t_old:.1f}s)")
        print(f"  §P 32img λ_max={lm_new:.2f}  ({t_new:.1f}s)")
        print(f"  |Δ|/new = {lm_delta_pct:.1f}%  {'→ batch=4 폐기 기준' if lm_delta_pct > 10 else '→ 합리적 근사'}")
        sp_comparison = {
            "lm_batch4_legacy": lm_old, "lm_micro32_new": lm_new,
            "delta_pct": lm_delta_pct, "time_old_s": t_old, "time_new_s": t_new,
            "verdict": "DISCARD_BATCH4" if lm_delta_pct > 10 else "ACCEPTABLE_APPROX",
        }
    except torch.cuda.OutOfMemoryError:
        print("  [OOM] §P comparison failed.")
        sp_comparison = {"error": "OOM"}

    # ── Save results ────────────────────────────────────────────────────────
    # λ_max overhead estimate
    lm_times = [h['lm_time_sec'] for h in history if not np.isnan(h['lambda_max'])]
    epoch_no_lm = [h['epoch_wall_sec'] - h['lm_time_sec'] for h in history]
    lm_overhead_pct = (float(np.mean(lm_times)) / max(float(np.mean(epoch_no_lm)), 1e-3) * 100
                       if lm_times else float('nan'))

    # Updated 150-run estimate
    epoch_total_sec = float(np.mean([h['epoch_wall_sec'] for h in history]))
    run_min = epoch_total_sec * args.epochs / 60
    runs_150_hours = run_min * 150 / 60

    result = {
        "sigma_g": args.sigma_g,
        "fwhm": args.fwhm,
        "seed": args.seed,
        "epochs_completed": len(history),
        "diverged": diverged if history else False,
        "final": {
            "train_loss": history[-1]['train_loss'] if history else float('nan'),
            "val_loss":   history[-1]['val_loss']   if history else float('nan'),
            "val_f1":     history[-1]['val_f1']     if history else float('nan'),
        },
        "lambda_max": {
            "final": history[-1]['lambda_max'] if history else float('nan'),
            "overhead_pct": lm_overhead_pct,
            "relaxed_to_10epochs": lambda_max_freq == 10,
            "n_computed": len(lm_times),
            "method": "§P micro-batch HVP (32 images, micro_bs=4, power_iter=10, early_stop=1%)",
            "sp_comparison": sp_comparison,
        },
        "timing": {
            "epoch_mean_sec": epoch_total_sec,
            "run_min": run_min,
            "runs_150_hours_estimate": runs_150_hours,
            "note": (f"λ_max overhead {lm_overhead_pct:.0f}%. "
                     f"{'Relaxed to 10-epoch.' if lambda_max_freq==10 else 'Kept at 5-epoch.'}")
        },
    }

    out_json = OUT_DIR / f'pilot_result_sg{args.sigma_g:.5f}_fw{args.fwhm:.3f}_s{args.seed}.json'
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    hist_json = OUT_DIR / f'pilot_history_sg{args.sigma_g:.5f}_fw{args.fwhm:.3f}_s{args.seed}.json'
    with open(hist_json, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 5 Pilot Complete")
    print(f"  Final F1: {result['final']['val_f1']:.3f}")
    print(f"  λ_max overhead: {lm_overhead_pct:.0f}%")
    print(f"  Epoch time: {epoch_total_sec:.1f}s → 100-epoch run: {run_min:.0f}min")
    print(f"  150-run estimate (updated): {runs_150_hours:.0f}h")
    print(f"  λ_max freq: every {lambda_max_freq} epochs")
    print(f"  Saved: {out_json}")


if __name__ == '__main__':
    main()
