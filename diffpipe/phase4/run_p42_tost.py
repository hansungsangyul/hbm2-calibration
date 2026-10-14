#!/usr/bin/env python3
"""
P4.2 TOST — D_calib vs D_manual 등가성 검정 (HANDOFF v6.4 §2)
=============================================================
D_calib (fallback) = θ_manual + fwhm=2.0956 + sigma_post=0
D_manual           = θ_manual (fwhm=2.5, 원파이프라인)

- 20 runs × 2 conditions = 40 total
- TOST δ=0.004, 전작 G4 프로토콜
- 산출물: seed별 F1 CSV + TOST 결과

Usage:
    # D_calib condition
    python run_p42_tost.py --condition calib --run_idx 0
    # D_manual condition
    python run_p42_tost.py --condition manual --run_idx 0

    # 분석 only
    python run_p42_tost.py --analyze_only
"""
from __future__ import annotations
import argparse, json, sys, time, csv
from pathlib import Path

import numpy as np

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
P42_DIR      = DIFFPIPE_DIR / 'phase4' / 'p42_runs'

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

# D_calib (fallback): θ_manual + fwhm=2.0956 + sigma_post=0
D_CALIB = dict(
    N=300.0, sigma_g=0.0408, p=0.001,
    fwhm=2.0956,           # ESF 보정값 (HANDOFF v6.4 §1)
    alpha=0.35, in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    sigma_post=0.0,        # 0 = no-op (bit-identical to legacy)
    ring_strength=0.12, ring_n=8.0,
)

# D_manual: θ_manual 전체 (원파이프라인, fwhm=2.5)
D_MANUAL = dict(
    N=300.0, sigma_g=0.0408, p=0.001,
    fwhm=2.5,
    alpha=0.35, in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    sigma_post=0.0,
    ring_strength=0.12, ring_n=8.0,
)

N_RUNS       = 20
EPOCHS       = 41
TOST_DELTA   = 0.004
TOST_ALPHA   = 0.05

# Seed bases: D_calib=700000~700019, D_manual=710000~710019
SEED_BASE = {'calib': 700000, 'manual': 710000}
PRENORM_A = 0.501834
PRENORM_B = 0.173320
LAMBDA_MAX_EPOCHS = {5, 10, 20, 30, 40}
LAMBDA_MAX_BATCH_SEED = 950000
LAMBDA_MAX_TOTAL = 32
LAMBDA_MAX_MICRO_BS = 4
LAMBDA_MAX_POWER_ITER = 10


parser = argparse.ArgumentParser()
parser.add_argument('--condition',    type=str,  choices=['calib', 'manual'])
parser.add_argument('--run_idx',      type=int,  default=0,  help='0~19')
parser.add_argument('--epochs',       type=int,  default=EPOCHS)
parser.add_argument('--batch_size',   type=int,  default=8,
                    help='Training batch size. Default 8 (safe with Phase 5 co-running)')
parser.add_argument('--analyze_only', action='store_true',
                    help='Skip training; run TOST on existing results')
args = parser.parse_args()

if args.analyze_only:
    # ── TOST 분석 ────────────────────────────────────────────────────────
    from scipy import stats

    def load_f1s(condition):
        f1s = []
        for run_idx in range(N_RUNS):
            p = P42_DIR / condition / f'run_{run_idx:02d}' / 'results.json'
            if p.exists():
                d = json.loads(p.read_text())
                f1s.append(d['final_f1'])
        return f1s

    f1_c = load_f1s('calib')
    f1_m = load_f1s('manual')
    print(f"D_calib  n={len(f1_c)}  mean={np.mean(f1_c):.5f}  std={np.std(f1_c):.5f}")
    print(f"D_manual n={len(f1_m)}  mean={np.mean(f1_m):.5f}  std={np.std(f1_m):.5f}")
    if len(f1_c) < N_RUNS or len(f1_m) < N_RUNS:
        print(f"WARNING: 미완료 runs 있음 (calib={len(f1_c)}/{N_RUNS}, manual={len(f1_m)}/{N_RUNS})")
    if len(f1_c) > 1 and len(f1_m) > 1:
        diff = np.array(f1_c) - np.array(f1_m[:len(f1_c)])
        # TOST: two one-sided t-tests
        t_lo, p_lo = stats.ttest_1samp(diff, -TOST_DELTA, alternative='greater')
        t_hi, p_hi = stats.ttest_1samp(diff,  TOST_DELTA, alternative='less')
        equiv = (p_lo < TOST_ALPHA) and (p_hi < TOST_ALPHA)
        print(f"\nTOST (δ={TOST_DELTA}):")
        print(f"  lower test: t={t_lo:.3f}  p={p_lo:.4f}  {'PASS' if p_lo<TOST_ALPHA else 'FAIL'}")
        print(f"  upper test: t={t_hi:.3f}  p={p_hi:.4f}  {'PASS' if p_hi<TOST_ALPHA else 'FAIL'}")
        print(f"  → {'등가성 확인 (FINAL)' if equiv else '등가성 미확인 (PENDING)'}")

        # Save CSV
        out_csv = P42_DIR / 'f1_results.csv'
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run_idx', 'condition', 'final_f1', 'seed'])
            for i, v in enumerate(f1_c):
                w.writerow([i, 'calib',  v, SEED_BASE['calib']  + i])
            for i, v in enumerate(f1_m):
                w.writerow([i, 'manual', v, SEED_BASE['manual'] + i])
        print(f"\nSaved CSV: {out_csv}")

        # Save TOST result
        tost_out = P42_DIR / 'tost_result.json'
        with open(tost_out, 'w') as f:
            json.dump({
                'n_calib': len(f1_c), 'n_manual': len(f1_m),
                'mean_calib': float(np.mean(f1_c)), 'std_calib': float(np.std(f1_c)),
                'mean_manual': float(np.mean(f1_m)), 'std_manual': float(np.std(f1_m)),
                'mean_diff': float(np.mean(diff)),
                'tost_delta': TOST_DELTA, 'tost_alpha': TOST_ALPHA,
                't_lower': float(t_lo), 'p_lower': float(p_lo),
                't_upper': float(t_hi), 'p_upper': float(p_hi),
                'equivalent': bool(equiv),
                'verdict': '등가성 확인' if equiv else '등가성 미확인',
                'approved_by': 'HANDOFF_v6.4 §2',
            }, f, indent=2)
        print(f"Saved TOST: {tost_out}")
    sys.exit(0)

# ── Single run training ───────────────────────────────────────────────
assert args.condition in ('calib', 'manual'), "condition required for training"
assert 0 <= args.run_idx < N_RUNS

seed = SEED_BASE[args.condition] + args.run_idx
theta = D_CALIB if args.condition == 'calib' else D_MANUAL
out_dir = P42_DIR / args.condition / f'run_{args.run_idx:02d}'
out_path = out_dir / 'results.json'

if out_path.exists():
    print(f"[SKIP] {out_path} already exists")
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True)

import cv2, torch, torch.nn as nn
from scipy.signal import medfilt
from torch.utils.data import Dataset, DataLoader
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from model import UNet, CombinedLoss
import config as cfg

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"P4.2 TOST run")
print(f"  condition={args.condition}  run_idx={args.run_idx}  seed={seed}")
print(f"  fwhm={theta['fwhm']}  sigma_post={theta['sigma_post']}")
print(f"  epochs={args.epochs}  batch_size={args.batch_size}  device={DEVICE}")

torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def make_pipe(theta: dict) -> DegradationPipeline:
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_c = float(np.clip(theta['N'], _N_LO+1e-3, _N_HI-1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], 1e-6)))
        if theta['fwhm'] > 0:
            pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(theta['ring_n']))
        pipe.in_low.fill_(theta['in_low'])
        pipe.in_high.fill_(theta['in_high'])
        pipe.out_low.fill_(theta['out_low'])
        pipe.out_high.fill_(theta['out_high'])
        pipe.p.fill_(theta['p'])
        pipe.sigma_post.fill_(theta['sigma_post'])  # 0.0 = no-op
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


class P42Dataset(Dataset):
    def __init__(self, data_dir, label_dir, pipe, theta_fwhm, train=True):
        self.pipe = pipe
        self.theta_fwhm = theta_fwhm
        self.train = train
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
        img_norm = PRENORM_A * img + PRENORM_B
        x = torch.from_numpy(img_norm)
        seed_i = idx + (1 if self.train else 100000)
        torch.manual_seed(seed_i)
        eps = {
            'eps1': torch.randn(H, W),
            'eps2': torch.randn(H, W),
            'eps3': torch.rand(H, W),
            'eps_post': torch.randn(H, W),
        }
        with torch.no_grad():
            y = self.pipe(x, eps)
        img_deg = y.numpy()
        img_deg = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        img_deg = img_deg[:, 4:516]
        img_3ch = np.stack([img_deg]*3, axis=0).astype(np.float32)
        mask = json_to_mask(str(self.label_files[idx]), H, W)
        mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        mask = mask[:, 4:516]
        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))


def compute_f1(preds, targets, n_classes=4):
    preds_flat = preds.argmax(dim=1).flatten()
    tgts_flat  = targets.flatten()
    f1s = []
    for c in range(n_classes):
        tp = ((preds_flat == c) & (tgts_flat == c)).sum().float()
        fp = ((preds_flat == c) & (tgts_flat != c)).sum().float()
        fn = ((preds_flat != c) & (tgts_flat == c)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append((2*prec*rec/(prec+rec+1e-8)).item())
    return float(np.mean(f1s))


def check_diverged(train_losses, f1_raw, epoch, n_epochs):
    f1_arr = np.array(f1_raw, dtype=np.float32)
    k = 3 if len(f1_arr) >= 3 else 1
    f1_filt = medfilt(f1_arr, kernel_size=k).tolist()
    crit_1 = any(np.isnan(l) or np.isinf(l) for l in train_losses[-5:])
    crit_2 = (epoch >= n_epochs and f1_arr[-1] < 0.5)
    crit_3 = False
    if epoch >= 20 and len(f1_filt) >= 10:
        window = f1_filt[-10:]
        crit_3 = (max(window) - min(window)) > 0.1
    diverged = crit_1 or crit_2 or crit_3
    reason = 1 if crit_1 else (2 if crit_2 else (3 if crit_3 else None))
    return diverged, reason, f1_filt


# ── Build & train ─────────────────────────────────────────────────────
t_start = time.time()
pipe = make_pipe(theta)

train_ds = P42Dataset(str(AUGMENT_DIR/'train_data/HBM_clean'),
                      str(AUGMENT_DIR/'train_data/HBM_labels'), pipe, theta['fwhm'], True)
val_ds   = P42Dataset(str(AUGMENT_DIR/'val_data/HBM_clean'),
                      str(AUGMENT_DIR/'val_data/HBM_labels'),   pipe, theta['fwhm'], False)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

model     = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                             weight_decay=cfg.WEIGHT_DECAY)

f1_raw, train_losses, val_losses = [], [], []
diverged = False
divergence_reason = None
f1_filt = []

for epoch in range(1, args.epochs+1):
    model.train()
    tl, n = 0.0, 0
    for imgs, msks in train_loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, msks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
        optimizer.step()
        tl += loss.item(); n += 1
    train_losses.append(tl/max(n,1))
    torch.cuda.empty_cache()

    model.eval()
    vl, vf1, nv = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            out = model(imgs)
            vl  += criterion(out, msks).item()
            vf1 += compute_f1(out, msks)
            nv  += 1
    val_losses.append(vl/max(nv,1))
    f1_raw.append(vf1/max(nv,1))

    div, reason, f1_filt = check_diverged(train_losses, f1_raw, epoch, args.epochs)
    if div and not diverged:
        diverged = True
        divergence_reason = reason

    elapsed = time.time()-t_start
    print(f"  Epoch {epoch:3d}/{args.epochs}  tr={train_losses[-1]:.4f}  "
          f"vl={val_losses[-1]:.4f}  F1={f1_raw[-1]:.3f}  ({elapsed:.0f}s)")
    if diverged:
        print(f"  DIVERGED (reason={divergence_reason})")
        break

final_f1 = float(f1_raw[-1])
wallclock = time.time()-t_start

result = {
    "condition": args.condition, "run_idx": args.run_idx, "seed": seed,
    "theta": theta, "epochs_completed": len(f1_raw),
    "f1_raw": f1_raw, "f1_filt": f1_filt,
    "train_loss": train_losses, "val_loss": val_losses,
    "final_f1": final_f1, "diverged": diverged,
    "divergence_reason": divergence_reason,
    "wallclock_s": wallclock,
    "handoff_ref": "HANDOFF_v6.4 §2 P4.2",
}
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"  final_f1={final_f1:.5f}  elapsed={wallclock/60:.1f}min")
