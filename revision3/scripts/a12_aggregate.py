# A12 aggregation per the pre-registration (A12_FACTORIAL_PREREG.md §4–§6). Works on partial CSVs (reports what is complete).
import pandas as pd, numpy as np, json, sys
from pathlib import Path
P3 = Path('/mnt/gstore/home/xiangyue/p3stats')
A12 = P3/'a12'/'a12_per_image.csv'; A8 = P3/'a8_paired'/'a8_paired_per_image.csv'; A7 = P3/'a7_labels'/'a7_labels_v2_per_image.csv'
OUT = P3/'a12'
NUMS = [251 + (1750*k)//500 for k in range(500)]; S = {f'image_{n:04d}' for n in NUMS}
DECL = {'D_manual':'none','AB-1':'none','AB-2':'none','AB-3':'none','D_calib':'none','D_v5e':'none','D_sigma_p':'none','z-score':'zscore'}
ORDER = ['D_manual','AB-1','AB-2','AB-3','D_calib','D_sigma_p','D_v5e','z-score']
MET = ['normal_px_fpr','normal_tile_far_t50','defect_tile_recall_t50','defect_tile_recall_cls_t50','normal_tile_far_t25','defect_tile_recall_t25','defect_tile_recall_cls_t25','normal_tile_far_t75','defect_tile_recall_t75','defect_tile_recall_cls_t75']
GATE = ['normal_tile_far_t50','normal_px_fpr','defect_tile_recall_cls_t50']
a12 = pd.read_csv(A12); a8 = pd.read_csv(A8); a7 = pd.read_csv(A7)
a8 = a8[a8.image.isin(S)].copy(); a8['variant'] = 'V_A'
a7 = a7[(a7.set == 'R_heldout') & a7.image.isin(S)]; a7 = a7[a7.apply(lambda r: r['mode'] == DECL[r['config']], axis=1)].copy(); a7['variant'] = 'V_R'
cols = ['config','run','variant','image'] + MET
df = pd.concat([a12[cols], a8[cols], a7[cols]], ignore_index=True)
# completeness: keep (config, variant) with 20 runs x 500 images
cnt = df.groupby(['config','variant']).size(); complete = {k for k, v in cnt.items() if v == 20*500}
df = df[df.apply(lambda r: (r['config'], r['variant']) in complete, axis=1)]
def F(sub, m):   # median over runs of per-run median over S; IQR over runs
    rm = sub.groupby('run')[m].median(); return float(rm.median()), float(rm.quantile(0.25)), float(rm.quantile(0.75))
res = {'complete_pairs': sorted([f'{c}|{v}' for c, v in complete]), 'F': {}, 'gate': {}, 'shares': {}, 'paired': {}, 'alike': {}}
for c in ORDER:
    d = df[df.config == c]
    vs = sorted(d.variant.unique())
    res['F'][c] = {v: {m: F(d[d.variant == v], m) for m in MET} for v in vs}
    # gate: per-run percentiles over S, median over runs; V_P vs V_R
    if 'V_P' in vs and 'V_R' in vs:
        g = {}
        for m in GATE:
            for q in (0.25, 0.5, 0.75):
                pp = d[d.variant == 'V_P'].groupby('run')[m].quantile(q).median(); rr = d[d.variant == 'V_R'].groupby('run')[m].quantile(q).median()
                g[f'{m}_q{int(q*100)}'] = [float(pp), float(rr), float(abs(pp - rr))]
        g['pass'] = bool(all(v[2] <= 0.10 for v in g.values() if isinstance(v, list)))
        res['gate'][c] = g
    # shares
    if 'V_P' in vs and 'V_A' in vs:
        FA = res['F'][c]['V_A']['normal_tile_far_t50'][0]; FP = res['F'][c]['V_P']['normal_tile_far_t50'][0]; G = FP - FA
        sh = {'G': G, 'defined': G > 0.10}
        for v in vs:
            Fv = res['F'][c][v]['normal_tile_far_t50'][0]
            if v.startswith('I_'): sh[v] = (Fv - FA)/G if G > 0.10 else None
            if v.startswith('L_') or v == 'V_F': sh[v] = (FP - Fv)/G if G > 0.10 else None
        res['shares'][c] = sh
        # A-like outcome
        rcA = res['F'][c]['V_A']['defect_tile_recall_cls_t50'][0]
        res['alike'][c] = {v: bool(abs(res['F'][c][v]['normal_tile_far_t50'][0] - FA) <= 0.05 and res['F'][c][v]['defect_tile_recall_cls_t50'][0] >= rcA - 0.05) for v in vs if v not in ('V_A',)}
    # paired per-image differences: I_k - V_A ; L_k, V_F - V_P
    pr = {}
    for v in vs:
        base = 'V_A' if v.startswith('I_') else ('V_P' if (v.startswith('L_') or v == 'V_F') else None)
        if base is None or base not in vs: continue
        M = d[d.variant == v][['run','image'] + MET].merge(d[d.variant == base][['run','image'] + MET], on=['run','image'], suffixes=('', '_b'))
        pr[v] = {m: F(M.assign(**{m: M[m] - M[m + '_b']}), m) for m in ['normal_tile_far_t50','defect_tile_recall_cls_t50','normal_px_fpr']}
    res['paired'][c] = pr
json.dump(res, open(OUT/'a12_summary.json', 'w'), indent=1)
# markdown
def f3(x): return 'na' if x is None else f'{x:.3f}'
L = ['# A12 summary (partial if pairs missing)', '', f"complete pairs: {len(complete)}", '']
L += ['## Behavioural agreement gate (V_P vs V_R; |diff| <= 0.10 for all summaries)', '', '| config | FAR q50 P/R | pxFPR q50 P/R | recall_cls q50 P/R | max |diff| | pass |', '|---|---|---|---|---|---|']
for c in ORDER:
    if c in res['gate']:
        g = res['gate'][c]; mx = max(v[2] for v in g.values() if isinstance(v, list))
        L.append(f"| {c} | {g['normal_tile_far_t50_q50'][0]:.3f}/{g['normal_tile_far_t50_q50'][1]:.3f} | {g['normal_px_fpr_q50'][0]:.3f}/{g['normal_px_fpr_q50'][1]:.3f} | {g['defect_tile_recall_cls_t50_q50'][0]:.3f}/{g['defect_tile_recall_cls_t50_q50'][1]:.3f} | {mx:.3f} | {g['pass']} |")
L += ['', '## Variants: F = median over runs of per-run median over S [IQR over runs]; shares per §6', '']
for c in ORDER:
    if c not in res['F']: continue
    L += [f'### {c}', '', '| variant | FAR | class-correct recall | px FPR | share | A-like |', '|---|---|---|---|---|---|']
    for v in ['V_A','V_R','V_P','V_F'] + sorted(x for x in res['F'][c] if x.startswith('L_')) + sorted(x for x in res['F'][c] if x.startswith('I_')):
        if v not in res['F'][c]: continue
        Fv = res['F'][c][v]; sh = res['shares'].get(c, {}).get(v); al = res['alike'].get(c, {}).get(v)
        cell = lambda m: f"{Fv[m][0]:.3f} [{Fv[m][1]:.3f}, {Fv[m][2]:.3f}]"
        L.append(f"| {v} | {cell('normal_tile_far_t50')} | {cell('defect_tile_recall_cls_t50')} | {cell('normal_px_fpr')} | {f3(sh) if isinstance(sh, float) else ''} | {'' if al is None else al} |")
    L.append('')
open(OUT/'a12_summary.md', 'w').write('\n'.join(L)); print('\n'.join(L))

