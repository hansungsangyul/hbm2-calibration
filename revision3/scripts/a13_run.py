#!/usr/bin/env python3
# A13 (2026-10-17): v5c calibration protocol under three frozen pixel-support conditions (U/N/C; a13_support.py). Usage: a13_run.py <arm> [smoke]
# Protocol copied from diffpipe/phase4/run_phase4_v5c.py: Tier 1 frozen (N=300, sigma_g=0.0408, p=0.001), out_* frozen (0.10/0.90), sigma_post=0,
# Tier 2 free (fwhm, alpha, in_low, in_high, ring_strength, ring_n), init theta_manual (fwhm_init 2.5), Adam lr 0.01, 2000 steps, batch 8,
# template x = templates[step % 50] with prenorm (a, b), seeds {970000, 971000, 972000}. The only change: the loss is evaluated with the per-image
# support of the template's own image index (img_idx = step % 50). Pre-registered in A13_SUPPORT_PREREG.md.
import sys, csv, json, time, numpy as np, cv2, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, '/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2')
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss
from a13_support import SupportCalibrationLoss, support_masks
ARM = sys.argv[1]; SMOKE = len(sys.argv) > 2 and sys.argv[2] == 'smoke'
TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean'); REAL_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
OUT = Path('/mnt/gstore/home/xiangyue/p3stats/a13')/ARM; OUT.mkdir(parents=True, exist_ok=True)
N_IMG, N_STEPS, BATCH, LR = 50, (20 if SMOKE else 2000), 8, 0.01
A, B = 0.501834, 0.173320
T1 = dict(N=300.0, sigma_g=0.0408, p=0.001); OUTL, OUTH = 0.10, 0.90
THETA_MANUAL = dict(fwhm=2.5, alpha=0.35, in_low=0.05, in_high=0.90, ring_strength=0.12, ring_n=8.0)
THETA_CALIB = dict(fwhm=1.356261134147644, alpha=0.31712929407755536, in_low=0.14566862086455026, in_high=0.699999988079071, ring_strength=0.028385845323403675, ring_n=7.931375980377197)
SEEDS = [970000, 971000, 972000]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
def set_pipe(pipe, th):
    with torch.no_grad():
        pipe.N_raw.fill_(sigmoid_box_inverse(float(np.clip(T1['N'], _N_LO+1, _N_HI-1)))); pipe.sigma_g_raw.fill_(softplus_inverse(max(T1['sigma_g'], 1e-4))); pipe.p.fill_(T1['p'])
        pipe.out_low.fill_(OUTL); pipe.out_high.fill_(OUTH); pipe.sigma_post.fill_(0.0)
        pipe.fwhm_raw.fill_(softplus_inverse(max(th['fwhm'], 1e-4))); pipe.alpha_raw.fill_(softplus_inverse(th['alpha']))
        pipe.in_low.fill_(float(np.clip(th['in_low'], 0.0, 0.3))); pipe.in_high.fill_(float(np.clip(th['in_high'], 0.7, 1.0)))
        pipe.ring_strength_raw.fill_(softplus_inverse(th['ring_strength'])); pipe.ring_n_raw.fill_(softplus_inverse(th['ring_n']))
    for q in [pipe.N_raw, pipe.sigma_g_raw, pipe.p, pipe.out_low, pipe.out_high, pipe.sigma_post]: q.requires_grad_(False)
def theta_phys(pipe):
    with torch.no_grad():
        return dict(fwhm=float(F.softplus(pipe.fwhm_raw)), alpha=float(F.softplus(pipe.alpha_raw)), in_low=float(pipe.in_low), in_high=float(pipe.in_high),
                    ring_strength=float(F.softplus(pipe.ring_strength_raw)), ring_n=float(F.softplus(pipe.ring_n_raw)))
tpl = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(TEMPLATE_DIR.glob('*.png'))[:N_IMG]]
real = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(REAL_DIR.glob('*.png'))[:N_IMG]]
H, W = tpl[0].shape
templates = [torch.from_numpy(A*t + B).to(DEV) for t in tpl]
masks, tiles = support_masks(N_IMG, H, W, ARM)
loss_fn = SupportCalibrationLoss(real, masks).to(DEV)
json.dump({'arm': ARM, 'tiles': tiles, 'stats': loss_fn.real_stats()}, open(OUT/'support_and_real_stats.json', 'w'))
print(f'[{ARM}] support pixels/image median {np.median(loss_fn.support_pixel_counts):.0f} of {H*W}; vm patches median {np.median(loss_fn.vm_support_patch_counts):.0f}', flush=True)
# reproduction check of the archived objective (arm U): SupportCalibrationLoss(U).forward(syn, i) vs CalibrationLoss.forward(syn)
if ARM == 'U':
    base = CalibrationLoss(real).to(DEV); pipe0 = DegradationPipeline().to(DEV); set_pipe(pipe0, THETA_MANUAL); torch.manual_seed(1)
    mx = 0.0
    for i in range(3):
        eps = {'eps1': torch.randn(H, W, device=DEV), 'eps2': torch.randn(H, W, device=DEV), 'eps3': torch.rand(H, W, device=DEV)}
        with torch.no_grad():
            syn = pipe0(templates[i], eps); L0, p0 = base(syn); L1, p1 = loss_fn(syn, i)
        mx = max(mx, max(abs(float(p0[k]) - float(p1[k])) for k in p0))
        print(f'[U] repro image {i}: archived ' + ' '.join(f"{k}={float(p0[k]):.6f}" for k in p0) + ' | support ' + ' '.join(f"{k}={float(p1[k]):.6f}" for k in p1), flush=True)
    print(f'[U] max |diff| of loss parts vs archived CalibrationLoss = {mx:.2e}', flush=True)
# loss parts at theta_manual and theta_calib with common draws (4 per image), all 50 images
def eval_parts(th, seed=123):
    pipe0 = DegradationPipeline().to(DEV); set_pipe(pipe0, th); torch.manual_seed(seed); acc = {}
    for i in range(N_IMG):
        for _ in range(4):
            eps = {'eps1': torch.randn(H, W, device=DEV), 'eps2': torch.randn(H, W, device=DEV), 'eps3': torch.rand(H, W, device=DEV)}
            with torch.no_grad(): _, p = loss_fn(pipe0(templates[i], eps), i)
            for k, v in p.items(): acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}
res = {'arm': ARM, 'parts_theta_manual': eval_parts(THETA_MANUAL), 'parts_theta_calib_v5c': eval_parts(THETA_CALIB), 'runs': []}
print(f'[{ARM}] parts@manual {res["parts_theta_manual"]}  parts@calib {res["parts_theta_calib_v5c"]}', flush=True)
def run(seed):
    pipe = DegradationPipeline().to(DEV); set_pipe(pipe, THETA_MANUAL)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    opt = torch.optim.Adam([{'params': [pipe.fwhm_raw, pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw], 'lr': LR}, {'params': [pipe.in_low, pipe.in_high], 'lr': LR}])
    trace = []; t0 = time.time(); rd = OUT/f'seed_{seed}'; rd.mkdir(exist_ok=True)
    for step in range(N_STEPS):
        opt.zero_grad(); i = step % N_IMG; x = templates[i]
        bl = torch.tensor(0.0, device=DEV); bp = {'L_vm': 0.0, 'L_nps': 0.0, 'L_rad': 0.0, 'L_hist': 0.0}
        for _ in range(BATCH):
            eps = {'eps1': torch.randn(H, W, device=DEV), 'eps2': torch.randn(H, W, device=DEV), 'eps3': torch.rand(H, W, device=DEV)}
            L, parts = loss_fn(pipe(x, eps), i); bl = bl + L
            for k in bp: bp[k] += parts[k].item()
        (bl / BATCH).backward(); opt.step()
        with torch.no_grad(): pipe.in_low.clamp_(0.0, 0.3); pipe.in_high.clamp_(0.7, 1.0)
        row = dict(step=step, L_total=(bl/BATCH).item(), elapsed=time.time()-t0); row.update({k: v/BATCH for k, v in bp.items()}); row.update(theta_phys(pipe)); trace.append(row)
        if (step+1) % 200 == 0: print(f'[{ARM} seed {seed}] step {step+1} L={row["L_total"]:.5f} fwhm={row["fwhm"]:.4f} alpha={row["alpha"]:.4f} ring={row["ring_strength"]:.4f} in_low={row["in_low"]:.4f} in_high={row["in_high"]:.4f}', flush=True)
    with open(rd/'calib_trace.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(trace[0].keys())); wr.writeheader(); wr.writerows(trace)
    th = theta_phys(pipe); th['seed'] = seed; th['steps'] = N_STEPS; th['elapsed_s'] = time.time() - t0
    th['fwhm_last200_mean'] = float(np.mean([r['fwhm'] for r in trace[-200:]])); th['fwhm_last200_range'] = [float(min(r['fwhm'] for r in trace[-200:])), float(max(r['fwhm'] for r in trace[-200:]))]
    json.dump(th, open(rd/'theta.json', 'w'), indent=1); return th
for s in SEEDS:
    res['runs'].append(run(s)); json.dump(res, open(OUT/'a13_result.json', 'w'), indent=1)
    print(f'[{ARM}] seed {s} done: ' + ' '.join(f"{k}={v:.4f}" for k, v in res['runs'][-1].items() if isinstance(v, float)), flush=True)
    if SMOKE: break
print(f'[{ARM}] A13_DONE', flush=True)

