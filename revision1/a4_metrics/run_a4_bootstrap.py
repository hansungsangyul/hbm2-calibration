import numpy as np, pandas as pd, json, itertools
from pathlib import Path
rng = np.random.default_rng(20260925)
df = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/f1_recompute_results/counts/per_image_counts.csv')
out = Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a4_metrics'); out.mkdir(exist_ok=True)
B = 2000
def f1s(tp, fp, fn):  # arrays over class axis last
    p = tp/np.maximum(tp+fp, 1); r = tp/np.maximum(tp+fn, 1); f = 2*tp/np.maximum(2*tp+fp+fn, 1); return p, r, f
rows_pr = []; boot = {}
for (cfg, ts), g in df.groupby(['config', 'test_set']):
    runs = sorted(g.run_id.unique()); imgs = sorted(g.image_id.unique()); C = sorted(g.class_id.unique())
    # tensor [run, image, class, 3]
    T = np.zeros((len(runs), len(imgs), len(C), 3))
    ri = {r: i for i, r in enumerate(runs)}; ii = {m: i for i, m in enumerate(imgs)}
    T[g.run_id.map(ri).values, g.image_id.map(ii).values, g.class_id.values, :] = g[['tp', 'fp', 'fn']].values
    # per-run dataset-level metrics
    S = T.sum(1)  # [run, class, 3]
    p, r, f = f1s(S[..., 0], S[..., 1], S[..., 2])
    for ci, c in enumerate(C):
        rows_pr.append(dict(config=cfg, test_set=ts, class_id=c, precision_mean=p[:, ci].mean(), precision_sd=p[:, ci].std(ddof=1), recall_mean=r[:, ci].mean(), recall_sd=r[:, ci].std(ddof=1), f1_mean=f[:, ci].mean(), f1_sd=f[:, ci].std(ddof=1), n_runs=len(runs)))
    macro4_run = f.mean(1); macrod_run = f[:, 1:].mean(1)
    # bootstraps
    nested4 = np.empty(B); nestedd = np.empty(B); runonly4 = np.empty(B); runonlyd = np.empty(B)
    for b in range(B):
        rs = rng.integers(0, len(runs), len(runs)); im = rng.integers(0, len(imgs), len(imgs))
        Sb = T[rs][:, im].sum(1); _, _, fb = f1s(Sb[..., 0], Sb[..., 1], Sb[..., 2])
        nested4[b] = fb.mean(1).mean(); nestedd[b] = fb[:, 1:].mean(1).mean()
        runonly4[b] = macro4_run[rs].mean(); runonlyd[b] = macrod_run[rs].mean()
    boot[(cfg, ts)] = dict(macro4_mean=float(macro4_run.mean()), macrod_mean=float(macrod_run.mean()), nested4=nested4, nestedd=nestedd, runonly4=runonly4, runonlyd=runonlyd, n_runs=len(runs))
    print(cfg, ts, f"macro4={macro4_run.mean():.4f} nestedCI=[{np.percentile(nested4,2.5):.4f},{np.percentile(nested4,97.5):.4f}] runonlyCI=[{np.percentile(runonly4,2.5):.4f},{np.percentile(runonly4,97.5):.4f}]  defect={macrod_run.mean():.4f}", flush=True)
pd.DataFrame(rows_pr).to_csv(out/'perclass_precision_recall.csv', index=False)
summ = []
for (cfg, ts), d in boot.items():
    row = dict(config=cfg, test_set=ts, n_runs=d['n_runs'], macro4=d['macro4_mean'], macrod=d['macrod_mean'])
    for k in ['nested4', 'nestedd', 'runonly4', 'runonlyd']:
        row[k+'_lo'], row[k+'_hi'] = np.percentile(d[k], [2.5, 97.5])
    if cfg != 'D_manual' and ('D_manual', ts) in boot:
        m = boot[('D_manual', ts)]
        for k in ['nested4', 'nestedd', 'runonly4', 'runonlyd']:
            dlt = d[k] - m[k]; row['delta_'+k+'_lo'], row['delta_'+k+'_hi'] = np.percentile(dlt, [2.5, 97.5])
        row['delta_macro4'] = d['macro4_mean']-m['macro4_mean']; row['delta_macrod'] = d['macrod_mean']-m['macrod_mean']
    summ.append(row)
pd.DataFrame(summ).to_csv(out/'bootstrap_summary.csv', index=False)
print('DONE')
