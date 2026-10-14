import numpy as np, pandas as pd
rng = np.random.default_rng(20260927); B = 4000
base = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/f1_recompute_results/counts/per_image_counts.csv')
ab4 = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a3_esf_low/counts/per_image_counts.csv')
df = pd.concat([base[base.config.isin(['D_calib', 'D_manual', 'AB-1'])], ab4])
def f1s(tp, fp, fn): return 2*tp/np.maximum(2*tp+fp+fn, 1)
T = {}
for (cfg, ts), g in df.groupby(['config', 'test_set']):
    runs = sorted(g.run_id.unique()); imgs = sorted(g.image_id.unique()); ri = {r: i for i, r in enumerate(runs)}; ii = {m: i for i, m in enumerate(imgs)}
    A = np.zeros((len(runs), len(imgs), 4, 3)); A[g.run_id.map(ri).values, g.image_id.map(ii).values, g.class_id.values, :] = g[['tp', 'fp', 'fn']].values
    T[(cfg, ts)] = (A, imgs)
def stat(A, rs, im):
    S = A[rs][:, im].sum(1); f = f1s(S[..., 0], S[..., 1], S[..., 2]); return f.mean(1).mean(), f[:, 1:].mean(1).mean()
rows = []
for ts in ['TEST_ID', 'TEST_OOD', 'TEST_CAL']:
    for a, b in [('D_calib', 'D_manual'), ('AB-4', 'D_calib'), ('AB-4', 'AB-1')]:
        (X, ix), (Y, iy) = T[(a, ts)], T[(b, ts)]; assert ix == iy
        n = len(ix); full = np.arange(n)
        px = stat(X, np.arange(len(X)), full); py = stat(Y, np.arange(len(Y)), full)
        bs = np.empty((B, 2))
        for k in range(B):
            im = rng.integers(0, n, n); rx = rng.integers(0, len(X), len(X)); ry = rng.integers(0, len(Y), len(Y))
            sx, sy = stat(X, rx, im), stat(Y, ry, im); bs[k] = (sx[0]-sy[0], sx[1]-sy[1])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append(dict(test_set=ts, contrast=f'{a}-{b}', d4=px[0]-py[0], d4_lo=lo[0], d4_hi=hi[0], dd=px[1]-py[1], dd_lo=lo[1], dd_hi=hi[1]))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('boot_fix.csv', index=False); print('DONE')
