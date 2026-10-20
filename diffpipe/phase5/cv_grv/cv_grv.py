"""Cross-validated incremental value of five-epoch GRV for predicting instability (manuscript Sec. VII-C).

Input: diffpipe/phase5/phase5_runs_level.csv (150 runs). Outcome: diverged (oscillation event).
Models: logistic regression (L2, C = 1) on standardized features, stratified 5-fold CV, 20 split seeds (0-19);
AUC of the pooled out-of-fold predictions per seed, then mean and SD over seeds.
Feature sets: grid coordinates (ln sigma_g, f) with and without GRV; cell indicators with and without GRV.
Usage: python cv_grv.py [path/to/phase5_runs_level.csv]
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'phase5_runs_level.csv'
df = pd.read_csv(csv)
y = df.diverged.astype(int).values
cells = pd.get_dummies(df.cell_id, drop_first=True).values.astype(float)
sets = {'grid': np.c_[np.log(df.sigma_g), df.fwhm],
        'grid+grv': np.c_[np.log(df.sigma_g), df.fwhm, df.grv5],
        'cell': cells,
        'cell+grv': np.c_[cells, df.grv5]}

def oof_auc(X, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000)).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)

res = {k: [oof_auc(X, s) for s in range(20)] for k, X in sets.items()}
out = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'per_seed': [float(a) for a in v]} for k, v in res.items()}
out['delta_grid'] = out['grid+grv']['mean'] - out['grid']['mean']
out['delta_cell'] = out['cell+grv']['mean'] - out['cell']['mean']
json.dump(out, open(Path(__file__).with_name('cv_grv_results.json'), 'w'), indent=1)
for k in sets: print(f"{k:10s} AUC {out[k]['mean']:.4f} (SD {out[k]['sd']:.4f})")
print(f"delta grid {out['delta_grid']:+.4f}  delta cell {out['delta_cell']:+.4f}")
