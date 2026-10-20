# A12 self-test (before the pre-registered run): (1) all-A composition reproduces the stored A8 output; (2) motion-kernel unit cases;
# (3) timing of the R-port; (4) secondary fidelity statistics V_P vs V_R on S (image statistics only; no detector metrics).
import sys, time, json, numpy as np, cv2, torch
from pathlib import Path
sys.path.insert(0, '/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2')
from a12_rport import draw_params, noise_rng, compose, fspecial_motion, rport_full, A_SPEC, R_SPEC
from a12_run import CFG, make_pipe, PRE_AB, NUMS, CLEANS, bc, OUT
# (1) all-A composition vs stored a8_paired npy (D_manual and D_calib, first 3 images of S)
for cname in ['D_manual', 'D_calib', 'D_sigma_p']:
    theta, pre = CFG[cname]; pipe = make_pipe(theta); spec = dict(A_SPEC); spec['pre'] = pre
    for n in NUMS[:3]:
        ref = np.load(f'/mnt/gstore/home/xiangyue/p3stats/a8_paired/{cname}/image_{n:04d}.npy')
        y = bc(compose(CLEANS[n], n, pipe, PRE_AB, spec, draw_params(n), noise_rng(n)))
        print(f'(1) {cname} image_{n:04d} max|diff| vs a8_paired = {np.abs(y - ref).max():.2e}')
# (1b) all-reference composition reproduces rport_full exactly
for n in NUMS[:3]:
    a = compose(CLEANS[n], n, make_pipe(CFG['D_manual'][0]), PRE_AB, dict(R_SPEC), draw_params(n), noise_rng(n)); b = rport_full(CLEANS[n], draw_params(n), noise_rng(n))
    print(f'(1b) all-R compose vs rport_full image_{n:04d} max|diff| = {np.abs(a-b).max():.2e}')
from a12_rport import variant_specs
print('(1c) variant counts:', {c: len(variant_specs(CFG[c][1], CFG[c][0]['sigma_post'] > 0)) for c in CFG})
# (2) motion kernel unit cases
k = fspecial_motion(5, 0); print('(2) motion(5,0) shape', k.shape, 'row', np.round(k.flatten(), 4))
k = fspecial_motion(5, 90); print('(2) motion(5,90) shape', k.shape, 'col', np.round(k.flatten(), 4))
k = fspecial_motion(6, 45); print('(2) motion(6,45) shape', k.shape, 'sum', k.sum(), '\n', np.round(k, 4))
k = fspecial_motion(1, 30); print('(2) motion(1,30)', k)
# (3) timing of the full R-port
t0 = time.time()
for n in NUMS[:20]: rport_full(CLEANS[n], draw_params(n), noise_rng(n))
print(f'(3) R-port {(time.time()-t0)/20*1000:.1f} ms/image')
# (4) fidelity statistics on S: V_P (port, random) vs V_R (download/HBM_dirty2) vs V_F (fixed)
def stats(x):
    x = x.astype(np.float64)
    gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3); g = np.sqrt(gx**2 + gy**2)
    H, W = x.shape; ps = 8; nH, nW = H//ps, W//ps
    pg = g[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).mean(axis=(1,3)); pv = x[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).std(axis=(1,3))
    flat = pg <= np.quantile(pg, 0.3); flat_std = float(np.median(pv[flat]))
    dh = np.abs(np.diff(x, axis=1)); dv = np.abs(np.diff(x, axis=0))
    bh = dh[:, 7::8].mean() / np.delete(dh, np.s_[7::8], axis=1).mean(); bv = dv[7::8, :].mean() / np.delete(dv, np.s_[7::8], axis=0).mean()
    Jxx, Jyy, Jxy = (gx*gx).mean(), (gy*gy).mean(), (gx*gy).mean()
    l1 = 0.5*(Jxx+Jyy) + 0.5*np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2); l2 = 0.5*(Jxx+Jyy) - 0.5*np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2)
    coh = (l1-l2)/(l1+l2+1e-12); ori = 0.5*np.degrees(np.arctan2(2*Jxy, Jxx-Jyy))
    return dict(mean=float(x.mean()), std=float(x.std()), flat_std=flat_std, block_h=float(bh), block_v=float(bv), coherence=float(coh), orientation=float(ori), grad_mean=float(g.mean()))
R = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
acc = {'V_R': [], 'V_P': [], 'V_F': [], 'clean': []}
for n in NUMS[:200]:
    acc['clean'].append(stats(CLEANS[n]))
    acc['V_R'].append(stats(cv2.imread(str(R/f'image_{n:04d}.png'), cv2.IMREAD_GRAYSCALE)/255.0))
    acc['V_P'].append(stats(rport_full(CLEANS[n], draw_params(n), noise_rng(n))))
    acc['V_F'].append(stats(rport_full(CLEANS[n], draw_params(n, random=False), noise_rng(n))))
summ = {}
for k, L in acc.items():
    summ[k] = {m: [float(np.median([d[m] for d in L])), float(np.quantile([d[m] for d in L], 0.25)), float(np.quantile([d[m] for d in L], 0.75))] for m in L[0]}
json.dump(summ, open(OUT/'a12_fidelity_stats.json', 'w'), indent=1)
for m in acc['V_R'][0]:
    print(f'(4) {m:12s} ' + '  '.join(f"{k}: {summ[k][m][0]:.4f} [{summ[k][m][1]:.4f},{summ[k][m][2]:.4f}]" for k in ['clean','V_R','V_P','V_F']))
# montage source PNGs for the visual check (first 4 images of S): clean, V_R, V_P
for n in NUMS[:4]:
    cv2.imwrite(str(OUT/'montage'/f'clean__image_{n:04d}.png'), (CLEANS[n]*255).round().astype(np.uint8))
    cv2.imwrite(str(OUT/'montage'/f'V_R__image_{n:04d}.png'), cv2.imread(str(R/f'image_{n:04d}.png'), cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(str(OUT/'montage'/f'V_Pfull__image_{n:04d}.png'), (rport_full(CLEANS[n], draw_params(n), noise_rng(n))*255).round().astype(np.uint8))
print('SELFTEST_DONE')

