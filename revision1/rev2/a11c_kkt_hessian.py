import json, sys, numpy as np, torch, cv2
from pathlib import Path
ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); sys.path.insert(0, str(ROOT/'src'))
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse
from losses import CalibrationLoss
OUT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a2_ident'); DEV = 'cuda'
A, B = 0.501834, 0.173320
tpl = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean').glob('*.png'))[:10]]
H, W = tpl[0].shape; T = [torch.from_numpy(A*t+B).to(DEV) for t in tpl]
real = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0 for p in sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2').glob('*.png'))[:50]]
loss_fn = CalibrationLoss(real).to(DEV)
g = torch.Generator(device=DEV).manual_seed(777)
EPS = [{'eps1': torch.randn(H, W, device=DEV, generator=g), 'eps2': torch.randn(H, W, device=DEV, generator=g), 'eps3': torch.rand(H, W, device=DEV, generator=g), 'eps_post': torch.randn(H, W, device=DEV, generator=g)} for _ in range(16)]
pipe = DegradationPipeline().to(DEV)
for p in pipe.parameters(): p.requires_grad_(False)
def set_theta(th):
    with torch.no_grad():
        pipe.N_raw.fill_(sigmoid_box_inverse(float(np.clip(th['N'], 51, 999)))); pipe.sigma_g_raw.fill_(softplus_inverse(th['sigma_g'])); pipe.p.fill_(th['p'])
        pipe.fwhm_raw.fill_(softplus_inverse(th['fwhm'])); pipe.alpha_raw.fill_(softplus_inverse(th['alpha'])); pipe.in_low.fill_(th['in_low']); pipe.in_high.fill_(th['in_high'])
        pipe.out_low.fill_(th['out_low']); pipe.out_high.fill_(th['out_high']); pipe.ring_strength_raw.fill_(softplus_inverse(th['ring_strength'])); pipe.ring_n_raw.fill_(softplus_inverse(th['ring_n'])); pipe.sigma_post.fill_(th.get('sigma_post', 0.0))
@torch.no_grad()
def loss_at(th):
    set_theta(th); tot = 0.0
    for x in T:
        for e in EPS: tot += float(loss_fn(pipe(x, e))[0])
    return tot/(len(T)*len(EPS))
def hessian(th0, keys, h=0.05):
    u0 = np.log([th0[k] for k in keys]); n = len(keys)
    def L(u): th = dict(th0); th.update({k: float(np.exp(v)) for k, v in zip(keys, u)}); return loss_at(th)
    L0 = L(u0); Hm = np.zeros((n, n)); grad = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h; Lp, Lm = L(u0+ei), L(u0-ei); Hm[i, i] = (Lp-2*L0+Lm)/h**2; grad[i] = (Lp-Lm)/(2*h)
        for j in range(i+1, n):
            ej = np.zeros(n); ej[j] = h
            Hm[i, j] = Hm[j, i] = (L(u0+ei+ej)-L(u0+ei-ej)-L(u0-ei+ej)+L(u0-ei-ej))/(4*h*h)
    w, V = np.linalg.eigh(Hm); return L0, grad, Hm, w, V
def ridge_dir(th, keys):  # gradient of log Var_post ??log((xbar/N+?g?)/f?) wrt (logN, log?g, logf[, log?post])
    xbar = 0.5; a = xbar/th['N']; b = th['sigma_g']**2; d = {'N': -a/(a+b), 'sigma_g': 2*b/(a+b), 'fwhm': -2.0, 'sigma_post': 0.0}
    v = np.array([d[k] for k in keys]); return v/np.linalg.norm(v)

import glob, sys as _s
LB = {'N': 50.0}; UB = {'N': 1000.0}
pts = []
v5c = json.load(open(ROOT/'phase4/theta_real_v5c.json'))['theta_real_mean']; v5c['sigma_post'] = 0.0; pts.append(('v5c', v5c, ['N', 'sigma_g', 'fwhm'], ['fwhm']))
v6 = json.load(open(ROOT/'phase4/theta_real_v6free.json'))['theta_real_mean']; pts.append(('v6free', v6, ['N', 'sigma_g', 'fwhm'], ['N', 'sigma_g', 'fwhm']))
for f in sorted(glob.glob('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a11/joint_*/result.json')):
    d = json.load(open(f)); th = dict(d['theta_hat']); th.update(p=0.001, out_low=0.10, out_high=0.90, sigma_post=0.0)
    pts.append((Path(f).parent.name, th, ['N', 'sigma_g', 'fwhm'], ['N', 'sigma_g', 'fwhm']))
res = {}
for name, th, keys, free in pts:
    out = {}
    for h in (0.025, 0.05, 0.1):
        L0, grad, Hm, w, V = hessian(th, keys, h=h)
        # projected gradient (KKT) in log coords: only free coordinates count; at an active lower bound a positive derivative is admissible
        pg = []
        for k, g in zip(keys, grad):
            if k not in free: continue
            at_lb = k in LB and th[k] <= LB[k]*1.02; at_ub = k in UB and th[k] >= UB[k]*0.98
            pg.append(0.0 if (at_lb and g > 0) or (at_ub and g < 0) else g)
        out[str(h)] = dict(L0=L0, grad=grad.tolist(), proj_grad_norm=float(np.linalg.norm(pg)), eig=w.tolist(), vmin=V[:, 0].tolist(), vmax=V[:, -1].tolist())
        print(name, 'h', h, 'L0 %.5f' % L0, 'grad', np.round(grad, 5), '|pg| %.2e' % out[str(h)]['proj_grad_norm'], 'eig', np.round(w, 4), flush=True)
    res[name] = dict(point={k: th[k] for k in keys}, free=free, by_h=out)
json.dump(res, open('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a11/a11c_kkt_hessian.json', 'w'), indent=2); print('DONE')
