# A7: pixel FPR on defect-free real images for all 8 configurations x all runs; sets: calib (0001-0050) and heldout (0251-2000)
from common import *; import csv, time
CONFIGS = {k: CONFIGS[k] for k in ('D_calib','AB-1','D_v5e','D_sigma_p')}; CONFIGS = {k: (v[0], 'none') for k, v in CONFIGS.items()}

REAL = sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2').glob('*.png'))
SETS = {'calib': REAL[0:50], 'heldout': REAL[250:2000]}
VALID = slice(3, 317); TOT = 314*512
OUT = Path(__file__).parent/'a7fix'; OUT.mkdir(exist_ok=True)
imgs = {s: [load_real(p) for p in L] for s, L in SETS.items()}
w = csv.writer(open(OUT/'a7_per_image.csv', 'w', newline='')); w.writerow(['config', 'run', 'set', 'image', 'pixel_fpr', 'bg_px'])
for cname, (rd, mode) in CONFIGS.items():
    pre = {s: [prep(x, mode).astype(np.float32) for x in v] for s, v in imgs.items()}
    for ck in ckpts(rd):
        m = load_model(ck); t0 = time.time()
        for s, L in pre.items():
            pr = predict(m, L, bs=32)[:, VALID, :]
            nb = (pr != 0).flatten(1).sum(1).numpy(); bg = (pr == 0).flatten(1).sum(1).numpy()
            for i, p in enumerate(SETS[s]): w.writerow([cname, ck.parent.name, s, p.stem, nb[i]/TOT, int(bg[i])])
        print(cname, ck.parent.name, f'{time.time()-t0:.0f}s', flush=True); del m; torch.cuda.empty_cache()
print('DONE')
# input intensity statistics of raw real images over valid rows
st = {s: (float(np.concatenate([x[VALID].ravel() for x in v]).mean()), float(np.concatenate([x[VALID].ravel() for x in v]).std())) for s, v in imgs.items()}
json.dump(st, open(OUT/'real_input_stats_raw.json', 'w')); print('input stats', st)
