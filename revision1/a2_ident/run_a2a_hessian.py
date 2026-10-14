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
res = {}
v5c = json.load(open(ROOT/'phase4/theta_real_v5c.json'))['theta_real_mean']; v5c['sigma_post'] = 0.0
v6 = json.load(open(ROOT/'phase4/theta_real_v6free.json'))['theta_real_mean']
for name, th, keys in [('v5c_3d', v5c, ['N', 'sigma_g', 'fwhm']), ('v6free_3d', v6, ['N', 'sigma_g', 'fwhm']), ('v6free_4d', v6, ['N', 'sigma_g', 'fwhm', 'sigma_post']), ('v5c_2d_noise', v5c, ['N', 'sigma_g'])]:
    L0, grad, Hm, w, V = hessian(th, keys); vmin = V[:, 0]; rd = ridge_dir(th, keys); cos = float(abs(np.dot(vmin, rd)))
    res[name] = dict(point={k: th[k] for k in keys}, keys=keys, L0=L0, grad=grad.tolist(), H=Hm.tolist(), eig=w.tolist(), cond=float(w[-1]/w[0]) if w[0] > 0 else float('inf'), vmin=vmin.tolist(), ridge_pred_dir=rd.tolist(), abs_cos_vmin_ridgegrad=cos)
    print(name, 'L0=%.5f eig=%s cond=%.1f vmin=%s |cos(vmin,??ogVar)|=%.3f' % (L0, np.round(w, 5), res[name]['cond'], np.round(vmin, 3), cos), flush=True)
json.dump(res, open(OUT/'a2a_hessian.json', 'w'), indent=2); print('DONE')
