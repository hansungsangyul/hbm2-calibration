# A8 paired evaluation, step 1 (2026-10-16): generate the A-degraded counterpart of R's held-out content.
# Content C = download/HBM_clean image_0251..image_2000 (1,750; the same clean images whose R-degraded versions are download/HBM_dirty2).
# For each training configuration, C is degraded by model A with that configuration's own training theta and its training-time prenorm placement
# (prenorm applied to the clean image BEFORE degradation, as in TrainDataset of run_p42r_v2/run_p43_ablation/v77 exp3/exp3b), epsilon seeded per image
# number (seed = 900000 + n, identical across configurations => common random numbers). Output stored after border/crop (320x512) WITHOUT z-score; the
# z-score configuration's per-image z-score is applied at inference by prep('zscore'), mirroring ZScoreFixedTestset. Pre-registered in A8_PAIRED_PREREG.md.
import sys, json, numpy as np, cv2, torch, time
from pathlib import Path
torch.set_num_threads(4)
DP = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); sys.path.insert(0, str(DP/'src'))
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
A, B = 0.501834, 0.173320
F = 2.057 * (2.500 / 2.454)   # FWHM_FROZEN = 2.0956
BASE = dict(N=300.0, sigma_g=0.0408, p=0.001, fwhm=2.5, alpha=0.35, in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90, sigma_post=0.0, ring_strength=0.12, ring_n=8.0)
V5E = dict(N=300.0, sigma_g=0.0408, p=0.001, fwhm=F, alpha=0.295997, in_low=0.169958, in_high=0.700000, out_low=0.10, out_high=0.90, ring_strength=0.030071, ring_n=8.0, sigma_post=0.016949)
CFG = {  # name: (theta, prenorm_before_degradation)
 'D_manual': (dict(BASE), False), 'AB-1': (dict(BASE), True), 'AB-2': ({**BASE, 'fwhm': F}, False), 'AB-3': ({**BASE, 'fwhm': F}, False),
 'D_calib': ({**BASE, 'fwhm': F}, True), 'D_v5e': (V5E, True), 'D_sigma_p': ({**BASE, 'fwhm': F, 'sigma_post': 0.016949}, True), 'z-score': (dict(BASE), False)}
def make_pipe(t):
    pipe = DegradationPipeline()
    with torch.no_grad():
        pipe.N_raw.fill_(sigmoid_box_inverse(float(np.clip(t['N'], _N_LO+1e-3, _N_HI-1e-3)))); pipe.sigma_g_raw.fill_(softplus_inverse(max(t['sigma_g'],1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(t['fwhm'])); pipe.alpha_raw.fill_(softplus_inverse(t['alpha'])); pipe.ring_strength_raw.fill_(softplus_inverse(t['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(t['ring_n'])); pipe.in_low.fill_(t['in_low']); pipe.in_high.fill_(t['in_high']); pipe.out_low.fill_(t['out_low']); pipe.out_high.fill_(t['out_high'])
        pipe.p.fill_(t['p']); pipe.sigma_post.fill_(t['sigma_post'])
    for q in pipe.parameters(): q.requires_grad_(False)
    return pipe
SRC = sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean').glob('image_*.png'))[250:2000]
assert len(SRC) == 1750 and SRC[0].stem == 'image_0251' and SRC[-1].stem == 'image_2000'
OUT = Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired')
meta = {'content': 'download/HBM_clean 0251-2000', 'seed_rule': '900000+n', 'configs': {k: {'theta': v[0], 'prenorm_before_degradation': v[1]} for k, v in CFG.items()}}
json.dump(meta, open(OUT/'a8_gen_meta.json', 'w'), indent=1)
for cname, (theta, pre) in CFG.items():
    od = OUT/cname; od.mkdir(exist_ok=True); pipe = make_pipe(theta); t0 = time.time()
    for p in SRC:
        n = int(p.stem.split('_')[1]); o = od/f'{p.stem}.npy'
        if o.exists(): continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0; H, W = img.shape
        if pre: img = A*img + B
        torch.manual_seed(900000 + n)
        eps = {'eps1': torch.randn(H,W), 'eps2': torch.randn(H,W), 'eps3': torch.rand(H,W), 'eps_post': torch.randn(H,W)}
        with torch.no_grad(): y = pipe(torch.from_numpy(img), eps).numpy()
        y = cv2.copyMakeBorder(y, 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        np.save(o, y.astype(np.float32))
    print(cname, f'{time.time()-t0:.0f}s', flush=True)
print('GEN_DONE')
