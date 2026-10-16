# Aggregate a7 (tile-label metrics on ID / R_calib / R_heldout) and a8 (A_paired) per the pre-registration: per-run median over images -> median/IQR over runs;
# paired delta = per-image (R - A) -> per-run median -> median/IQR over runs. Declared mode per config for the R side (prenorm configs: 'prenorm'; z-score: 'zscore'; else 'none').
import pandas as pd, numpy as np, json, sys
from pathlib import Path
A7 = Path('/mnt/gstore/home/xiangyue/p3stats/a7_labels/a7_labels_v2_per_image.csv'); A8 = Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired/a8_paired_per_image.csv')
OUT = Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired')
DECL = {'D_manual':'none','AB-1':'none','AB-2':'none','AB-3':'none','D_calib':'none','D_v5e':'none','D_sigma_p':'none','z-score':'zscore'}  # raw input for all (prenorm is applied to templates before degradation; vf_remeasure convention, A7 correction d478fe5); z-score applies its per-image normalization
ORDER = ['D_manual','AB-1','AB-2','AB-3','D_calib','D_sigma_p','D_v5e','z-score']
METRICS = ['normal_px_fpr'] + [f'{m}_t{t}' for t in (25,50,75) for m in ('normal_tile_far','defect_tile_recall','defect_tile_recall_cls')]
a7 = pd.read_csv(A7); a8 = pd.read_csv(A8) if A8.exists() else None
def summarize(df, keys):
    runmed = df.groupby(keys + ['run'])[METRICS].median().reset_index()
    g = runmed.groupby(keys)[METRICS]
    med = g.median(); q1 = g.quantile(0.25); q3 = g.quantile(0.75); n = g.size()
    return med, q1, q3, n, runmed
res = {}
# per set, declared mode
decl_rows = a7[a7.apply(lambda r: r['mode'] == DECL[r['config']], axis=1)]
med, q1, q3, n, runmed = summarize(decl_rows, ['config','set'])
res['a7_declared'] = {f'{c}|{s}': {m: [float(med.loc[(c,s),m]), float(q1.loc[(c,s),m]), float(q3.loc[(c,s),m])] for m in METRICS} | {'n_runs': int(n.loc[(c,s)])} for (c,s) in med.index}
# other mode for prenorm configs (supplementary)
oth = a7[a7.apply(lambda r: r['mode'] != DECL[r['config']], axis=1)]
if len(oth):
    med2, q12, q32, n2, _ = summarize(oth, ['config','mode','set'])
    res['a7_othermode'] = {f'{c}|{md}|{s}': {m: [float(med2.loc[(c,md,s),m]), float(q12.loc[(c,md,s),m]), float(q32.loc[(c,md,s),m])] for m in METRICS} | {'n_runs': int(n2.loc[(c,md,s)])} for (c,md,s) in med2.index}
if a8 is not None and len(a8):
    medA, q1A, q3A, nA, runmedA = summarize(a8, ['config'])
    res['a8_A_paired'] = {c: {m: [float(medA.loc[c,m]), float(q1A.loc[c,m]), float(q3A.loc[c,m])] for m in METRICS} | {'n_runs': int(nA.loc[c])} for c in medA.index}
    # paired deltas
    R = decl_rows[decl_rows['set']=='R_heldout'][['config','run','image']+METRICS]
    M = a8[['config','run','image']+METRICS].merge(R, on=['config','run','image'], suffixes=('_A','_R'))
    for m in METRICS: M[m] = M[m+'_R'] - M[m+'_A']
    medD, q1D, q3D, nD, _ = summarize(M[['config','run','image']+METRICS], ['config'])
    res['paired_delta'] = {c: {m: [float(medD.loc[c,m]), float(q1D.loc[c,m]), float(q3D.loc[c,m])] for m in METRICS} | {'n_runs': int(nD.loc[c]), 'n_pairs_per_run': int(len(M[M.config==c])/max(1,nD.loc[c]))} for c in medD.index}
    # run-level fraction of runs with FAR_R >= 0.9 etc.
json.dump(res, open(OUT/'a8_summary.json','w'), indent=1)
# markdown table (tau=0.5 primary)
def f3(x): return f'{x:.3f}'
lines = ['| Config | A FAR | R FAR | dFAR | A recall | R recall | drecall | A recall_cls | R recall_cls | A px FPR | R px FPR |','|---|---|---|---|---|---|---|---|---|---|---|']
for c in ORDER:
    if 'a8_A_paired' in res and c in res['a8_A_paired']:
        A = res['a8_A_paired'][c]; Rr = res['a7_declared'].get(f'{c}|R_heldout'); D = res['paired_delta'][c]
        def cell(d, m): return f"{f3(d[m][0])} [{f3(d[m][1])}, {f3(d[m][2])}]"
        lines.append(f"| {c} | {cell(A,'normal_tile_far_t50')} | {cell(Rr,'normal_tile_far_t50')} | {cell(D,'normal_tile_far_t50')} | {cell(A,'defect_tile_recall_t50')} | {cell(Rr,'defect_tile_recall_t50')} | {cell(D,'defect_tile_recall_t50')} | {cell(A,'defect_tile_recall_cls_t50')} | {cell(Rr,'defect_tile_recall_cls_t50')} | {cell(A,'normal_px_fpr')} | {cell(Rr,'normal_px_fpr')} |")
    else:
        Rr = res['a7_declared'].get(f'{c}|R_heldout'); I = res['a7_declared'].get(f'{c}|ID')
        if Rr: lines.append(f"| {c} (a7 only) | ID FAR {f3(I['normal_tile_far_t50'][0]) if I else 'na'} | {f3(Rr['normal_tile_far_t50'][0])} | | ID rec {f3(I['defect_tile_recall_t50'][0]) if I else 'na'} | {f3(Rr['defect_tile_recall_t50'][0])} | | | {f3(Rr['defect_tile_recall_cls_t50'][0])} | | {f3(Rr['normal_px_fpr'][0])} |")
open(OUT/'a8_summary.md','w').write('\n'.join(lines)); print('\n'.join(lines))
