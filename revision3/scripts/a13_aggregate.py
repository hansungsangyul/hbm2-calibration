# A13 aggregation: reached Tier-2 parameters per arm and seed; seed-paired differences N-U, C-U, N-C; loss parts at theta_manual / v5c theta.
import json, numpy as np
from pathlib import Path
P = Path('/mnt/gstore/home/xiangyue/p3stats/a13'); arms = ['U', 'N', 'C']
R = {a: json.load(open(P/a/'a13_result.json')) for a in arms if (P/a/'a13_result.json').exists()}
keys = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']
out = {'runs': {a: R[a]['runs'] for a in R}, 'parts': {a: {'manual': R[a]['parts_theta_manual'], 'calib_v5c': R[a]['parts_theta_calib_v5c']} for a in R}, 'paired': {}}
L = ['# A13 summary', '', '| arm | seed | fwhm | last-200 mean | last-200 range | alpha | in_low | in_high | ring_strength | ring_n | elapsed s |', '|---|---|---|---|---|---|---|---|---|---|---|']
for a in R:
    for r in R[a]['runs']:
        L.append(f"| {a} | {r['seed']} | {r['fwhm']:.4f} | {r['fwhm_last200_mean']:.4f} | [{r['fwhm_last200_range'][0]:.4f}, {r['fwhm_last200_range'][1]:.4f}] | {r['alpha']:.4f} | {r['in_low']:.4f} | {r['in_high']:.4f} | {r['ring_strength']:.4f} | {r['ring_n']:.3f} | {r['elapsed_s']:.0f} |")
L += ['', '## Seed-paired differences (reached fwhm)', '', '| pair | seeds | differences | mean |', '|---|---|---|---|']
def by_seed(a): return {r['seed']: r for r in R[a]['runs']}
for x, y in [('N', 'U'), ('C', 'U'), ('N', 'C')]:
    if x in R and y in R:
        bx, byy = by_seed(x), by_seed(y); seeds = sorted(set(bx) & set(byy))
        d = {k: [bx[s][k] - byy[s][k] for s in seeds] for k in keys}
        out['paired'][f'{x}-{y}'] = {'seeds': seeds, **d}
        L.append(f"| {x} − {y} | {seeds} | " + ', '.join(f'{v:+.4f}' for v in d['fwhm']) + f" | {np.mean(d['fwhm']):+.4f} |")
L += ['', '## Loss components (mean over 50 images × 4 common draws)', '', '| arm | theta | L_vm | L_nps | L_rad | L_hist |', '|---|---|---|---|---|---|']
for a in R:
    for th in ['manual', 'calib_v5c']:
        p = out['parts'][a][th]; L.append(f"| {a} | {th} | {p['L_vm']:.4f} | {p['L_nps']:.4f} | {p['L_rad']:.4f} | {p['L_hist']:.4f} |")
L += ['', 'Archived v5c: fwhm mean 1.3563 (std 0.0028 over seeds 970000-972000); alternative-weighting means W1-W4 1.322, 0.970, 1.384, 1.388 (context only).']
json.dump(out, open(P/'a13_summary.json', 'w'), indent=1); open(P/'a13_summary.md', 'w').write('\n'.join(L)); print('\n'.join(L))
