# A12 factorial decomposition (2026-10-17): generate each variant of the 500-image set S in memory and evaluate all 8 configurations x 20 runs
# with the A8 metric code (a8_paired_eval.py). Pre-registered in A12_FACTORIAL_PREREG.md. Resumable: (config, variant) pairs with 20 runs in the CSV are skipped.
import sys, os, csv, json, time, numpy as np, cv2, torch
from pathlib import Path
sys.path.insert(0, '/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2')
from common import *
from dataset import json_to_mask
from a12_rport import draw_params, noise_rng, compose, variant_specs, amendment1_specs, A_SPEC, R_SPEC
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
torch.set_num_threads(8)
PRE_AB = (0.501834, 0.173320)
F = 2.057 * (2.500 / 2.454)   # 2.0956
BASE = dict(N=300.0, sigma_g=0.0408, p=0.001, fwhm=2.5, alpha=0.35, in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90, sigma_post=0.0, ring_strength=0.12, ring_n=8.0)
V5E = dict(N=300.0, sigma_g=0.0408, p=0.001, fwhm=F, alpha=0.295997, in_low=0.169958, in_high=0.700000, out_low=0.10, out_high=0.90, ring_strength=0.030071, ring_n=8.0, sigma_post=0.016949)
CFG = {'D_manual': (dict(BASE), False), 'AB-1': (dict(BASE), True), 'AB-2': ({**BASE, 'fwhm': F}, False), 'AB-3': ({**BASE, 'fwhm': F}, False),
       'D_calib': ({**BASE, 'fwhm': F}, True), 'D_v5e': (V5E, True), 'D_sigma_p': ({**BASE, 'fwhm': F, 'sigma_post': 0.016949}, True), 'z-score': (dict(BASE), False)}
ORDER = ['D_manual', 'AB-1', 'AB-2', 'AB-3', 'D_calib', 'D_sigma_p', 'D_v5e', 'z-score']
def make_pipe(t):
    pipe = DegradationPipeline()
    with torch.no_grad():
        pipe.N_raw.fill_(sigmoid_box_inverse(float(np.clip(t['N'], _N_LO+1e-3, _N_HI-1e-3)))); pipe.sigma_g_raw.fill_(softplus_inverse(max(t['sigma_g'],1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(t['fwhm'])); pipe.alpha_raw.fill_(softplus_inverse(t['alpha'])); pipe.ring_strength_raw.fill_(softplus_inverse(t['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(t['ring_n'])); pipe.in_low.fill_(t['in_low']); pipe.in_high.fill_(t['in_high']); pipe.out_low.fill_(t['out_low']); pipe.out_high.fill_(t['out_high'])
        pipe.p.fill_(t['p']); pipe.sigma_post.fill_(t['sigma_post'])
    for q in pipe.parameters(): q.requires_grad_(False)
    return pipe
# ---- set S, labels, tile grid (identical to a8_paired_eval.py) ----
NUMS = [251 + (1750*k)//500 for k in range(500)]; names = [f'image_{n:04d}' for n in NUMS]
CLEAN = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean'); LBL = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_labels')
OUT = Path('/mnt/gstore/home/xiangyue/p3stats/a12'); OUT.mkdir(exist_ok=True); (OUT/'montage').mkdir(exist_ok=True)
TAUS = (0.25, 0.5, 0.75)
def tile_grid():
    tm = np.full((320,512), -1, np.int64)
    for r in range(5):
        y0 = r*62; y1 = (r+1)*62 if r<4 else 314
        for c in range(20):
            x0 = c*26; x1 = (c+1)*26 if c<19 else 520
            tm[3+y0:3+y1, max(x0-4,0):min(x1-4,512)] = r*20+c
    return torch.from_numpy(tm)
TM = tile_grid(); TILE_PIX = {t: (TM == t) for t in range(100)}
def load_mask(l):
    mk = json_to_mask(str(l), 314, 520); return cv2.copyMakeBorder(mk, 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516].astype(np.int64)
M = torch.from_numpy(np.stack([load_mask(LBL/f'label_{nm.split("_")[1]}.json') for nm in names]))
TT = torch.zeros(M.shape[0], 100, dtype=torch.long)
for t in range(100): TT[:, t] = M[:, TM == t].mode(dim=1).values
isn = (TT == 0); isd = (TT != 0); normal = (M == 0)
def bc(y):  # border/crop as in training and A8
    return cv2.copyMakeBorder(y.astype(np.float32), 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
CLEANS = {n: cv2.imread(str(CLEAN/f'image_{n:04d}.png'), cv2.IMREAD_GRAYSCALE).astype(np.float64)/255.0 for n in NUMS}
def gen_variant(cname, vname, spec):
    theta, pre = CFG[cname]; pipe = make_pipe(theta)
    imgs = []
    for n in NUMS:
        P = draw_params(n, random=(vname != 'V_F')); rng = noise_rng(n)
        y = compose(CLEANS[n], n, pipe, PRE_AB, spec, P, rng)
        imgs.append(bc(y))
    return imgs
def metrics_rows(cname, md, run, vname, pr):
    pos = (pr != 0)
    fpr = (pos & normal).flatten(1).sum(1).float() / normal.flatten(1).sum(1).clamp(min=1).float()
    frac = torch.zeros(pr.shape[0], 100); maj = torch.zeros(pr.shape[0], 100, dtype=torch.long)
    for t in range(100):
        sel = TILE_PIX[t]; pt = pr[:, sel]; frac[:, t] = (pt != 0).float().mean(1)
        cnt = torch.stack([(pt == k).sum(1) for k in (1,2,3)], 1); maj[:, t] = cnt.argmax(1) + 1
    rows = []
    for i in range(pr.shape[0]):
        vals = [f'{fpr[i]:.5f}']
        for tau in TAUS:
            det = frac[i] >= tau
            far = float((det & isn[i]).sum()) / max(1, int(isn[i].sum())); rec = float((det & isd[i]).sum()) / max(1, int(isd[i].sum()))
            recc = float((det & isd[i] & (maj[i] == TT[i])).sum()) / max(1, int(isd[i].sum()))
            vals += [f'{far:.4f}', f'{rec:.4f}', f'{recc:.4f}']
        rows.append([cname, md, run, vname, names[i]] + vals + [int(isn[i].sum()), int(isd[i].sum())])
    return rows
def main():
    CSV = OUT/'a12_per_image.csv'
    import hashlib, subprocess, PIL, scipy
    if not (OUT/'a12_manifest.json').exists():
        man = {'S': NUMS, 'variants': {c: list(variant_specs(CFG[c][1], CFG[c][0]['sigma_post'] > 0).keys()) for c in ORDER},
               'specs': {c: variant_specs(CFG[c][1], CFG[c][0]['sigma_post'] > 0) for c in ORDER},
               'theta': {c: CFG[c][0] for c in ORDER}, 'prenorm': {c: CFG[c][1] for c in ORDER},
               'checkpoints': {c: [{'path': str(ck), 'size': ck.stat().st_size, 'mtime': ck.stat().st_mtime} for ck in ckpts(CONFIGS[c][0])] for c in ORDER},
               'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'cv2': cv2.__version__, 'PIL': PIL.__version__, 'scipy': scipy.__version__},
               'input_hash_clean_S': hashlib.md5(b''.join(open(CLEAN/f'image_{n:04d}.png','rb').read() for n in NUMS)).hexdigest(),
               'git_head': subprocess.run(['git','rev-parse','HEAD'], cwd='/mnt/gstore/home/xiangyue/HBM_paper2', capture_output=True, text=True).stdout.strip()}
        json.dump(man, open(OUT/'a12_manifest.json', 'w'), indent=1)
        json.dump({str(n): draw_params(n) for n in NUMS}, open(OUT/'a12_params.json', 'w'), indent=0)
    done = set()
    if CSV.exists():
        with open(CSV) as f:
            rd = csv.reader(f); next(rd)
            cnt = {}
            for r in rd: cnt[(r[0], r[3], r[2])] = 1
        from collections import Counter
        c2 = Counter((k[0], k[1]) for k in cnt)
        done = {k for k, v in c2.items() if v >= 20}
        print('resuming; complete pairs:', len(done), flush=True)
    new = not CSV.exists()
    fh = open(CSV, 'a', newline=''); w = csv.writer(fh)
    if new: w.writerow(['config','mode','run','variant','image','normal_px_fpr'] + [f'{m}_t{int(t*100)}' for t in TAUS for m in ('normal_tile_far','defect_tile_recall','defect_tile_recall_cls')] + ['n_normal_tiles','n_defect_tiles'])
    # variant order: fidelity gate first, then recovery (L), then insertion (I); V_A_check for D_manual only
    VORDER = ['V_P', 'V_F', 'L_noise', 'L_blur', 'L_vig', 'L_post', 'L_con', 'L_ring', 'L_jpeg', 'L_res', 'L_pre', 'I_noise', 'I_blur', 'I_vig', 'I_post', 'I_con', 'I_ring', 'I_jpeg', 'I_res', 'I_pre', 'V_A_check']
    if len(sys.argv) > 1 and sys.argv[1] == 'am1': VORDER = ['L_noise_blur', 'I_noise_blur']   # amendment 1 (run after the main manifest)
    cache = {}   # configuration-independent variants: key (vname, pre_flag) -> imgs
    t_start = time.time()
    for vname in VORDER:
        for cname in ORDER:
            theta, pre = CFG[cname]; specs = {**variant_specs(pre, theta['sigma_post'] > 0), **amendment1_specs(pre, theta['sigma_post'] > 0)}
            if vname == 'V_A_check':
                if cname != 'D_manual': continue
                spec = dict(A_SPEC); spec['pre'] = pre
            else:
                if vname not in specs: continue
                spec = specs[vname]
            if (cname, vname) in done: continue
            indep = vname in ('V_P', 'V_F', 'L_jpeg', 'L_res') or vname == 'L_pre'
            key = (vname, pre if vname == 'L_pre' else None)
            t0 = time.time()
            if indep and key in cache: imgs = cache[key]
            else:
                imgs = gen_variant(cname, vname, spec)
                if indep: cache[key] = imgs
            if cname == 'D_manual' or (vname in ('L_pre', 'I_pre') and cname == 'AB-1') or (vname in ('L_post', 'I_post') and cname == 'D_sigma_p'):
                for i in range(4): cv2.imwrite(str(OUT/'montage'/f'{vname}__{cname}__{names[i]}.png'), np.clip(np.round(imgs[i]*255), 0, 255).astype(np.uint8))
            rd, mode = CONFIGS[cname]; md = 'zscore' if mode == 'zscore' else 'identity'
            pre_imgs = [prep(x, 'zscore').astype(np.float32) if md == 'zscore' else x for x in imgs]
            tg = time.time() - t0
            for ck in ckpts(rd):
                m = load_model(ck); pr = predict(m, pre_imgs, bs=32)
                for row in metrics_rows(cname, md, ck.parent.name, vname, pr): w.writerow(row)
                del m; torch.cuda.empty_cache()
            fh.flush()
            print(f'{vname:8s} {cname:10s} gen {tg:.0f}s total {time.time()-t0:.0f}s elapsed {(time.time()-t_start)/60:.1f} min', flush=True)
    print('A12_DONE', flush=True)

if __name__ == '__main__':
    main()

