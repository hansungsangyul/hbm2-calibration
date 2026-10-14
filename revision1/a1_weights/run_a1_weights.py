#!/usr/bin/env python3
"""A1 ??loss-weight sensitivity under the v5c regime. See PREREG_a1.md."""
from __future__ import annotations
import csv, json, sys, time, argparse
from pathlib import Path
import cv2, numpy as np
ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
sys.path.insert(0, str(ROOT / 'src'))
import torch, torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

ap = argparse.ArgumentParser()
ap.add_argument('--scheme', required=True)   # W0..W4
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--fwhm_init', type=float, default=2.5)
args = ap.parse_args()

TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
N_TEMPLATES, N_REAL, N_STEPS, BATCH_SIZE, LR = 50, 50, 2000, 8, 0.01
PRENORM_A, PRENORM_B = 0.501834, 0.173320
TIER1_N, TIER1_SG, TIER1_P = 300.0, 0.0408, 0.001
OUT_LOW, OUT_HIGH = 0.10, 0.90
IN_LOW_MIN, IN_LOW_MAX, IN_HIGH_MIN, IN_HIGH_MAX = 0.00, 0.30, 0.70, 1.00
THETA_MANUAL = dict(fwhm=2.5, alpha=0.35, in_low=0.05, in_high=0.90, ring_strength=0.12, ring_n=8.0)
DEVICE = 'cuda'
FIXED = {'W0': (1.0, 1.0, 1.0, 1.0),
         'W1': (1.0, 0.392/0.138, 0.392/0.0172, 0.392/0.00168),
         'W3': (0.5, 1.0, 1.0, 1.0),
         'W4': (1.0, 2.0, 1.0, 1.0)}

def set_pipeline(pipe, fwhm_init=2.5):
    with torch.no_grad():
        pipe.N_raw.fill_(sigmoid_box_inverse(float(np.clip(TIER1_N, _N_LO+1, _N_HI-1))))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P); pipe.out_low.fill_(OUT_LOW); pipe.out_high.fill_(OUT_HIGH)
        pipe.fwhm_raw.fill_(softplus_inverse(fwhm_init))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.in_low.fill_(THETA_MANUAL['in_low']); pipe.in_high.fill_(THETA_MANUAL['in_high'])
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_MANUAL['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_MANUAL['ring_n']))
    for p in (pipe.N_raw, pipe.sigma_g_raw, pipe.p, pipe.out_low, pipe.out_high):
        p.requires_grad_(False)

def phys(pipe):
    with torch.no_grad():
        return dict(N=float(_N_LO+(_N_HI-_N_LO)*torch.sigmoid(pipe.N_raw)), sigma_g=float(F.softplus(pipe.sigma_g_raw)),
                    p=float(pipe.p), fwhm=float(F.softplus(pipe.fwhm_raw)), alpha=float(F.softplus(pipe.alpha_raw)),
                    in_low=float(pipe.in_low), in_high=float(pipe.in_high), out_low=float(pipe.out_low), out_high=float(pipe.out_high),
                    ring_strength=float(F.softplus(pipe.ring_strength_raw)), ring_n=float(F.softplus(pipe.ring_n_raw)))

tpl = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]]
H, W = tpl[0].shape
templates = [torch.from_numpy(PRENORM_A*t+PRENORM_B).to(DEVICE) for t in tpl]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(REAL_DIR.glob('*.png'))[:N_REAL]]
loss_fn = CalibrationLoss(real_np).to(DEVICE)
pipe = DegradationPipeline().to(DEVICE)
set_pipeline(pipe, fwhm_init=args.fwhm_init)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
tier2 = [pipe.fwhm_raw, pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw, pipe.in_low, pipe.in_high]

def eps_draw():
    return {'eps1': torch.randn(H, W, device=DEVICE), 'eps2': torch.randn(H, W, device=DEVICE), 'eps3': torch.rand(H, W, device=DEVICE)}

# W2: gradient-norm balancing measured at the initial point (fixed seed, before optimization)
if args.scheme == 'W2':
    g = torch.Generator(device=DEVICE).manual_seed(123456)
    norms = np.zeros(4)
    for _ in range(BATCH_SIZE):
        eps = {'eps1': torch.randn(H, W, device=DEVICE, generator=g), 'eps2': torch.randn(H, W, device=DEVICE, generator=g), 'eps3': torch.rand(H, W, device=DEVICE, generator=g)}
        syn = pipe(templates[0], eps); _, parts = loss_fn(syn)
        for i, k in enumerate(['L_vm', 'L_nps', 'L_rad', 'L_hist']):
            grads = torch.autograd.grad(parts[k], tier2, retain_graph=True, allow_unused=True)
            norms[i] += float(torch.sqrt(sum((gg**2).sum() for gg in grads if gg is not None)))
    norms /= BATCH_SIZE
    lam = tuple(float(norms[0]/max(n, 1e-12)) for n in norms)
    grad_norms = norms.tolist()
else:
    lam = FIXED[args.scheme]; grad_norms = None
print(f"[{args.scheme} s{args.seed}] lambda={lam}  grad_norms_init={grad_norms}", flush=True)

opt = torch.optim.Adam([{'params': tier2, 'lr': LR}])
trace = []; t0 = time.time()
for step in range(N_STEPS):
    opt.zero_grad(); x = templates[step % len(templates)]
    bl = torch.tensor(0.0, device=DEVICE); bp = dict(L_vm=0.0, L_nps=0.0, L_rad=0.0, L_hist=0.0)
    for _ in range(BATCH_SIZE):
        syn = pipe(x, eps_draw()); L, parts = loss_fn(syn, lambda_weights=lam)
        bl = bl + L
        for k in bp: bp[k] += parts[k].item()
    (bl/BATCH_SIZE).backward(); opt.step()
    with torch.no_grad():
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX); pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)
    row = dict(step=step, L_total=(bl/BATCH_SIZE).item(), elapsed=time.time()-t0); row.update({k: v/BATCH_SIZE for k, v in bp.items()}); row.update(phys(pipe)); trace.append(row)
    if (step+1) % 200 == 0:
        print(f"  [{args.scheme} s{args.seed}] step {step+1} L={row['L_total']:.5f} fwhm={row['fwhm']:.4f} alpha={row['alpha']:.3f} ring={row['ring_strength']:.4f}", flush=True)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
with open(out/'calib_trace.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(trace[0].keys())); w.writeheader(); w.writerows(trace)
final = phys(pipe); final_loss_mean50 = float(np.mean([r['L_total'] for r in trace[-50:]]))
json.dump(dict(scheme=args.scheme, seed=args.seed, lambda_weights=list(lam), grad_norms_init=grad_norms, theta_hat=final,
               final_loss_mean50=final_loss_mean50, n_steps=N_STEPS, elapsed_s=time.time()-t0), open(out/'result.json', 'w'), indent=2)
print(f"[{args.scheme} s{args.seed}] DONE fwhm={final['fwhm']:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
