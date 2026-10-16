# A8 paired evaluation, step 2: evaluate all 8 configurations x 20 runs on the A-degraded counterpart (a8_paired_gen.py) of R's held-out content,
# with the SAME tile grid, masks and per-image metrics as a7_labels_eval2.py (set name 'A_paired'). Inference prep: identity, except per-image z-score for
# the z-score configuration (mirrors FixedTestset / ZScoreFixedTestset). The R side is taken from a7_labels_v2_per_image.csv (set R_heldout, declared mode).
from common import *; import csv, time
from dataset import json_to_mask
TAUS = (0.25, 0.5, 0.75)
LBL_R = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_labels'); AIN = Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired')
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
names = [f'image_{n:04d}' for n in range(251, 2001)]
M = torch.from_numpy(np.stack([load_mask(LBL_R/f'label_{nm.split("_")[1]}.json') for nm in names]))
TT = torch.zeros(M.shape[0], 100, dtype=torch.long)
for t in range(100): TT[:, t] = M[:, TM == t].mode(dim=1).values
isn = (TT == 0); isd = (TT != 0); normal = (M == 0)
OUT = Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired')
w = csv.writer(open(OUT/'a8_paired_per_image.csv', 'w', newline='')); w.writerow(['config','mode','run','set','image','normal_px_fpr'] + [f'{m}_t{int(t*100)}' for t in TAUS for m in ('normal_tile_far','defect_tile_recall','defect_tile_recall_cls')] + ['n_normal_tiles','n_defect_tiles'])
for cname, (rd, mode) in CONFIGS.items():
    imgs = [np.load(AIN/cname/f'{nm}.npy').astype(np.float32) for nm in names]
    md = 'zscore' if mode == 'zscore' else 'identity'
    pre = [prep(x, 'zscore').astype(np.float32) if md == 'zscore' else x for x in imgs]
    for ck in ckpts(rd):
        m = load_model(ck); t0 = time.time(); pr = predict(m, pre, bs=32); pos = (pr != 0)
        fpr = (pos & normal).flatten(1).sum(1).float() / normal.flatten(1).sum(1).clamp(min=1).float()
        frac = torch.zeros(pr.shape[0], 100); maj = torch.zeros(pr.shape[0], 100, dtype=torch.long)
        for t in range(100):
            sel = TILE_PIX[t]; pt = pr[:, sel]; frac[:, t] = (pt != 0).float().mean(1)
            cnt = torch.stack([(pt == k).sum(1) for k in (1,2,3)], 1); maj[:, t] = cnt.argmax(1) + 1
        for i in range(pr.shape[0]):
            vals = [f'{fpr[i]:.5f}']
            for tau in TAUS:
                det = frac[i] >= tau
                far = float((det & isn[i]).sum()) / max(1, int(isn[i].sum())); rec = float((det & isd[i]).sum()) / max(1, int(isd[i].sum()))
                recc = float((det & isd[i] & (maj[i] == TT[i])).sum()) / max(1, int(isd[i].sum()))
                vals += [f'{far:.4f}', f'{rec:.4f}', f'{recc:.4f}']
            w.writerow([cname, md, ck.parent.name, 'A_paired', names[i]] + vals + [int(isn[i].sum()), int(isd[i].sum())])
        print(cname, md, ck.parent.name, f'{time.time()-t0:.0f}s', flush=True); del m; torch.cuda.empty_cache()
print('DONE')
