# A9 analysis: per-defect PR (one-vs-rest on tile score), max recall at FP tiles/image <= {0.01,0.05}, coverage-tercile recall at argmax-free threshold, nested bootstrap
import numpy as np, json; from pathlib import Path
O = Path(__file__).parent/'a9'; gt = np.load(O/'gt.npy'); cov = np.load(O/'coverage.npy'); bins = json.load(open(Path(__file__).parent/'a9_coverage_bins.json'))
names = {1: 'crack', 2: 'hole', 3: 'open'}; rng = np.random.default_rng(20260926); B = 1000
def op_point(s, y, n_img, target):
    # s: scores (flattened tiles), y: 1 if tile is class c; returns max recall with FP tiles/image <= target (ties together)
    o = np.argsort(-s, kind='stable'); s_o, y_o = s[o], y[o]
    tp = np.cumsum(y_o); fp = np.cumsum(1-y_o); last = np.r_[s_o[1:] != s_o[:-1], True]   # only cut at tie boundaries
    tp, fp = tp[last], fp[last]; ok = fp/n_img <= target; P = max(y.sum(), 1)
    return float(tp[ok].max()/P) if ok.any() else 0.0
def ap(s, y):
    o = np.argsort(-s, kind='stable'); y_o = y[o]; tp = np.cumsum(y_o); prec = tp/np.arange(1, len(y_o)+1); return float((prec*y_o).sum()/max(y_o.sum(), 1))
res = {}
for ts in ['test_id', 'test_ood']:
    for cname in ['D_calib', 'D_manual', 'AB-1']:
        S = np.load(O/f'scores_{cname}_{ts}.npy').astype(np.float32)   # [20,500,100,4]
        for c in (1, 2, 3):
            def stats(rs, im):
                y = (gt[im] == c).ravel(); per_run = []
                for r in rs:
                    s = S[r][im][..., c].ravel()
                    per_run.append([ap(s, y), op_point(s, y, len(im), 0.01), op_point(s, y, len(im), 0.05)])
                return np.mean(per_run, 0)
            point = stats(range(20), np.arange(500))
            boot = np.array([stats(rng.integers(0, 20, 20), rng.integers(0, 500, 500)) for _ in range(B)])
            # coverage-tercile recall at argmax class assignment (descriptive)
            tier = np.digitize(cov, bins[names[c]]); rec_t = []
            for k in range(3):
                msk = (gt == c) & (tier == k); rec_t.append(float(np.mean([(S[r].argmax(-1)[msk] == c).mean() for r in range(20)])))
            res[f'{ts}|{cname}|{names[c]}'] = dict(AP=point[0], AP_ci=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(), R_at_fp001=point[1], R_at_fp001_ci=np.percentile(boot[:, 1], [2.5, 97.5]).tolist(), R_at_fp005=point[2], R_at_fp005_ci=np.percentile(boot[:, 2], [2.5, 97.5]).tolist(), recall_by_coverage_tercile=rec_t)
            print(ts, cname, names[c], {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in res[f'{ts}|{cname}|{names[c]}'].items()}, flush=True)
json.dump(res, open(O/'a9_results.json', 'w'), indent=2); print('DONE')
