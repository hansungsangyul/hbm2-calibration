#!/usr/bin/env python3
"""A14 pre-registered analysis (see A14_PRENORM_DECOMP_PREREG.md).
Run-level dataset-level F1 from TP/FP/FN counts. Conditions: D = D_manual (historical, p42r_v2/manual),
P = AB-1 (historical, p43_runs/ab1), C1, C2 (new). n = 20 each.
Primary outcome: TEST_OOD four-class macro F1. Secondary: TEST_OOD defect-only macro F1, TEST_OOD background F1,
TEST_ID four-class macro F1.
Contrasts: dM = C1 - D, dQ = C2 - D, I = P - C1 - C2 + D, C1 - C2, P - C1, P - C2.
Primary family (Holm): dM, dQ, C1 - C2, I (Welch / HC3-type unequal-variance t on the cell means).
Bootstrap: runs resampled independently within each condition, B = 10000, seed 20261019; percentile 95% CIs for all
contrasts and for the ratios rM = dM/(P - D), rQ = dQ/(P - D) with the denominator resampled jointly.
Descriptive classification (pre-specified): see prereg; 'undetermined' when CIs straddle the boundaries.
Reproduction check: dman_r run0 and ab1_r run0 vs archived run0 of the same seed (reported, not gating).
"""
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
R = Path('/mnt/gstore/home/xiangyue/HBM_paper2')
HIST = R/'f1_recompute_results/counts/per_image_counts.csv'; NEW = R/'revision1/rev2/a14/a14_counts.csv'; OUT = R/'revision1/rev2/a14/a14_results.json'
def f1(tp, fp, fn):
    d = 2*tp+fp+fn; return float('nan') if d == 0 else 2*tp/d
def load(path, keep):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in csv.DictReader(open(path)):
        cfg = keep.get(r['config'])
        if cfg is None: continue
        a = acc[(cfg, r['test_set'].upper(), r['run_id'])][int(r['class_id'])]
        a[0] += int(r['tp']); a[1] += int(r['fp']); a[2] += int(r['fn'])
    out = defaultdict(lambda: defaultdict(dict))
    for (cfg, ts, run), cc in acc.items():
        fs = [f1(*cc[c]) for c in range(4)]
        out[cfg][ts][run] = dict(macro4=float(np.mean(fs)), defect=float(np.mean(fs[1:])), bg=fs[0])
    return out
H = load(HIST, {'D_manual': 'D', 'AB-1': 'P'}); N = load(NEW, {'c1': 'C1', 'c2': 'C2', 'dman_r': 'Dr', 'ab1_r': 'Pr'})
data = {**H, **N}
OUTC = [('TEST_OOD', 'macro4', 'primary'), ('TEST_OOD', 'defect', 'secondary'), ('TEST_OOD', 'bg', 'secondary'), ('TEST_ID', 'macro4', 'secondary')]
rng = np.random.default_rng(20261019); B = 10000
res = {'n': {}, 'outcomes': {}}
for ts, key, role in OUTC:
    v = {c: np.array([data[c][ts][r][key] for r in sorted(data[c][ts])]) for c in ['D', 'P', 'C1', 'C2']}
    for c, values in v.items():
        assert len(values) == 20, (ts, key, c, len(values))
        assert np.isfinite(values).all(), (ts, key, c)
    res['n'] = {c: len(v[c]) for c in v}
    m = {c: float(v[c].mean()) for c in v}; sd = {c: float(v[c].std(ddof=1)) for c in v}
    def con(mm): return dict(dM=mm['C1']-mm['D'], dQ=mm['C2']-mm['D'], I=mm['P']-mm['C1']-mm['C2']+mm['D'],
                             C1_C2=mm['C1']-mm['C2'], P_C1=mm['P']-mm['C1'], P_C2=mm['P']-mm['C2'], P_D=mm['P']-mm['D'])
    pt = con(m)
    w = {'dM': {'C1': 1, 'D': -1}, 'dQ': {'C2': 1, 'D': -1}, 'I': {'P': 1, 'C1': -1, 'C2': -1, 'D': 1}, 'C1_C2': {'C1': 1, 'C2': -1}}
    tests = {}
    for k, ww in w.items():
        se2 = sum((cf**2)*sd[c]**2/len(v[c]) for c, cf in ww.items()); se = math.sqrt(se2)
        dfw = se2**2/sum(((cf**2)*sd[c]**2/len(v[c]))**2/(len(v[c])-1) for c, cf in ww.items())
        t = pt[k]/se; p = 2*stats.t.sf(abs(t), dfw); tests[k] = dict(est=pt[k], se=se, df=dfw, t=t, p=p)
    order = sorted(tests, key=lambda k: tests[k]['p']); prev = 0.0
    for i, k in enumerate(order):
        adj = min(1.0, max(prev, (len(order)-i)*tests[k]['p'])); tests[k]['p_holm'] = adj; prev = adj
    bs = defaultdict(list)
    for _ in range(B):
        mm = {c: float(rng.choice(v[c], len(v[c])).mean()) for c in v}; cc = con(mm)
        for k, x in cc.items(): bs[k].append(x)
        bs['rM'].append(cc['dM']/cc['P_D']); bs['rQ'].append(cc['dQ']/cc['P_D'])
    ci = {k: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] for k, x in bs.items()}
    pt['rM'] = pt['dM']/pt['P_D']; pt['rQ'] = pt['dQ']/pt['P_D']
    res['outcomes'][f'{ts}:{key}'] = dict(role=role, mean=m, sd=sd, contrasts=pt, ci95=ci, tests=tests)
# descriptive classification on the primary outcome
o = res['outcomes']['TEST_OOD:macro4']; ci = o['ci95']
def above(k): return ci[k][0] >= 0.5
def below(k): return ci[k][1] < 0.5
c12 = o['tests']['C1_C2']
if above('rM') and below('rQ') and c12['p_holm'] < 0.05 and o['contrasts']['C1_C2'] > 0: cls = 'mainly affine shift (M)'
elif above('rQ') and below('rM') and c12['p_holm'] < 0.05 and o['contrasts']['C1_C2'] < 0: cls = 'mainly noise law (Q)'
elif above('rM') and above('rQ'): cls = 'each factor alone reproduces at least half'
elif below('rM') and below('rQ'): cls = 'neither factor alone reproduces half'
else: cls = 'undetermined'
res['classification_primary'] = cls
# reproduction check
rep = {}
for nc, hc in [('Dr', 'D'), ('Pr', 'P')]:
    for ts in ['TEST_ID', 'TEST_OOD', 'TEST_CAL']:
        if nc in data and ts in data[nc]:
            for r, val in data[nc][ts].items():
                h = [x for x in data[hc][ts] if x.endswith('_run0')]
                assert len(h) == 1, (hc, ts, h)
                rep[f'{nc}:{ts}:{r}'] = dict(new=val['macro4'], archived=data[hc][ts][h[0]]['macro4'])
res['reproduction'] = rep
# exploratory: post hoc crit3 (range of the median-filtered validation F1 over the last 10 of 41 epochs > 0.1)
instability = {}
srcs = {'D': R/'diffpipe/phase4/p42r_v2_runs/manual', 'P': R/'diffpipe/phase4/p43_runs/ab1',
        'C1': R/'revision1/rev2/a14/runs/c1', 'C2': R/'revision1/rev2/a14/runs/c2'}
for c, root in srcs.items():
    recs = [json.load(open(root / f'run{i}' / 'results.json')) for i in range(20)]
    ev = [max(r['f1_filt'][-10:]) - min(r['f1_filt'][-10:]) > 0.1 for r in recs]
    instability[c] = {'crit3_events': int(sum(ev)), 'n': 20}
res['instability_crit3_posthoc'] = instability
json.dump(res, open(OUT, 'w'), indent=2, default=float)
print(json.dumps({k: res['outcomes'][k]['contrasts'] for k in res['outcomes']}, indent=1, default=float)); print(cls)
