import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(20260927); B = 4000
base = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/f1_recompute_results/counts/per_image_counts.csv')
ab4 = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a3_esf_low/counts/per_image_counts.csv')
df = pd.concat([base, ab4]); df.loc[df.config.isin(['AB-2', 'AB-3']), 'config'] = 'AB-2+3'
def f1s(tp, fp, fn): return 2*tp/np.maximum(2*tp+fp+fn, 1)
T = {}
for (cfg, ts), g in df.groupby(['config', 'test_set']):
    runs = sorted(g.run_id.unique()); imgs = sorted(g.image_id.unique()); ri = {r: i for i, r in enumerate(runs)}; ii = {m: i for i, m in enumerate(imgs)}
    A = np.zeros((len(runs), len(imgs), 4, 3)); A[g.run_id.map(ri).values, g.image_id.map(ii).values, g.class_id.values, :] = g[['tp', 'fp', 'fn']].values
    T[(cfg, ts)] = (A, runs, imgs)
def runf(A, rs, im, kind):
    S = A[rs][:, im].sum(1); f = f1s(S[..., 0], S[..., 1], S[..., 2]); return (f.mean(1) if kind == '4' else f[:, 1:].mean(1))
fam = {'F1': [('D_calib', 'D_manual', ts, k, False) for ts in ['TEST_ID', 'TEST_OOD'] for k in ['4', 'd']],
       'F2': [(c, 'D_manual', ts, '4', False) for c in ['AB-1', 'AB-2+3', 'D_v5e', 'D_sigma_p', 'z-score'] for ts in ['TEST_ID', 'TEST_OOD']] + [('calib+infer_pre', 'D_calib', ts, '4', True) for ts in ['TEST_ID', 'TEST_OOD']],
       'F3': [(a, b, ts, '4', False) for a, b in [('AB-4', 'D_calib'), ('AB-4', 'AB-1')] for ts in ['TEST_ID', 'TEST_OOD']]}
rows = []
for F, L in fam.items():
    for a, b, ts, k, paired in L:
        (X, rx_, ix), (Y, ry_, iy) = T[(a, ts)], T[(b, ts)]; assert ix == iy; n = len(ix); full = np.arange(n)
        x = runf(X, np.arange(len(X)), full, k); y = runf(Y, np.arange(len(Y)), full, k)
        if paired:
            assert [r for r in rx_] == [r for r in ry_]; t = stats.ttest_rel(x, y)
        else: t = stats.ttest_ind(x, y, equal_var=False)
        bs = np.empty(B)
        for j in range(B):
            im = rng.integers(0, n, n)
            if paired: r = rng.integers(0, len(X), len(X)); bs[j] = runf(X, r, im, k).mean() - runf(Y, r, im, k).mean()
            else: bs[j] = runf(X, rng.integers(0, len(X), len(X)), im, k).mean() - runf(Y, rng.integers(0, len(Y), len(Y)), im, k).mean()
        rows.append(dict(family=F, contrast=f'{a} - {b}', test_set=ts, metric='4-class' if k == '4' else 'defect-only', paired=paired, n_a=len(X), n_b=len(Y), diff=x.mean()-y.mean(), t=t.statistic, p=t.pvalue, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows)
for F, g in R.groupby('family'):
    o = g.p.sort_values(); m = len(o); adj = np.maximum.accumulate([min(1, (m-i)*p) for i, p in enumerate(o.values)])
    R.loc[o.index, 'p_holm'] = adj
R.to_csv('holm_table.csv', index=False); print(R.round(4).to_string()); print('DONE')
