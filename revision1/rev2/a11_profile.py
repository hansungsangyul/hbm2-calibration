#!/usr/bin/env python3
"""A2(b) N-profile: fix N, free sigma_g, fwhm and Tier2 (v1-free regime), report final loss. Weights (1,1,1,1)."""
import csv, json, sys, time, argparse; from pathlib import Path; import cv2, numpy as np
ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); sys.path.insert(0, str(ROOT/'src'))
import torch, torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss
ap = argparse.ArgumentParser(); ap.add_argument('--N', type=float, required=True); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True); ap.add_argument('--lam', type=str, default='1,1,1,1'); ap.add_argument('--N_init', type=float, default=None); ap.add_argument('--sg_init', type=float, default=0.0408); ap.add_argument('--f_init', type=float, default=2.5); ap.add_argument('--free_N', action='store_true'); a = ap.parse_args(); LAM = tuple(float(v) for v in a.lam.split(','))
A, B = 0.501834, 0.173320; DEV = 'cuda'; STEPS, BS, LR = 2000, 8, 0.01
tpl = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean').glob('*.png'))[:50]]
H, W = tpl[0].shape; T = [torch.from_numpy(A*t+B).to(DEV) for t in tpl]
real = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2').glob('*.png'))[:50]]
loss_fn = CalibrationLoss(real).to(DEV); pipe = DegradationPipeline().to(DEV)
with torch.no_grad():
    pipe.N_raw.fill_(sigmoid_box_inverse(float(np.clip(a.N_init if a.N_init else a.N, _N_LO+1, _N_HI-1)))); pipe.sigma_g_raw.fill_(softplus_inverse(a.sg_init)); pipe.p.fill_(0.001)
    pipe.fwhm_raw.fill_(softplus_inverse(a.f_init)); pipe.alpha_raw.fill_(softplus_inverse(0.35)); pipe.in_low.fill_(0.05); pipe.in_high.fill_(0.90); pipe.out_low.fill_(0.10); pipe.out_high.fill_(0.90)
    pipe.ring_strength_raw.fill_(softplus_inverse(0.12)); pipe.ring_n_raw.fill_(softplus_inverse(8.0)); pipe.sigma_post.fill_(0.0)
for p in (pipe.N_raw, pipe.p, pipe.out_low, pipe.out_high, pipe.sigma_post): p.requires_grad_(False)
if a.free_N: pipe.N_raw.requires_grad_(True)
free = ([pipe.N_raw] if a.free_N else []) + [pipe.sigma_g_raw, pipe.fwhm_raw, pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw, pipe.in_low, pipe.in_high]
torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); opt = torch.optim.Adam(free, lr=LR)
def phys():
    with torch.no_grad(): return dict(N=float(_N_LO+(_N_HI-_N_LO)*torch.sigmoid(pipe.N_raw)), sigma_g=float(F.softplus(pipe.sigma_g_raw)), fwhm=float(F.softplus(pipe.fwhm_raw)), alpha=float(F.softplus(pipe.alpha_raw)), in_low=float(pipe.in_low), in_high=float(pipe.in_high), ring_strength=float(F.softplus(pipe.ring_strength_raw)), ring_n=float(F.softplus(pipe.ring_n_raw)))
tr = []; t0 = time.time()
for s in range(STEPS):
    opt.zero_grad(); x = T[s % len(T)]; bl = torch.tensor(0.0, device=DEV)
    for _ in range(BS):
        e = {'eps1': torch.randn(H, W, device=DEV), 'eps2': torch.randn(H, W, device=DEV), 'eps3': torch.rand(H, W, device=DEV)}
        bl = bl + loss_fn(pipe(x, e), lambda_weights=LAM)[0]
    (bl/BS).backward(); opt.step()
    with torch.no_grad(): pipe.in_low.clamp_(0.0, 0.30); pipe.in_high.clamp_(0.70, 1.0)
    r = dict(step=s, L_total=float(bl/BS)); r.update(phys()); tr.append(r)
    if (s+1) % 400 == 0: print(f"[N={a.N:.0f} s{a.seed}] {s+1} L={r['L_total']:.5f} sg={r['sigma_g']:.4f} f={r['fwhm']:.3f}", flush=True)
o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
with open(o/'calib_trace.csv', 'w', newline='') as fh: w = csv.DictWriter(fh, fieldnames=list(tr[0].keys())); w.writeheader(); w.writerows(tr)
json.dump(dict(N_fixed=None if a.free_N else a.N, free_N=a.free_N, lam=LAM, init=dict(N=a.N_init or a.N, sigma_g=a.sg_init, fwhm=a.f_init), seed=a.seed, theta_hat=phys(), final_loss_mean50=float(np.mean([r['L_total'] for r in tr[-50:]])), elapsed_s=time.time()-t0), open(o/'result.json', 'w'), indent=2)
print(f"[N={a.N:.0f} s{a.seed}] DONE L50={np.mean([r['L_total'] for r in tr[-50:]]):.5f} sg={phys()['sigma_g']:.4f} f={phys()['fwhm']:.3f}", flush=True)
