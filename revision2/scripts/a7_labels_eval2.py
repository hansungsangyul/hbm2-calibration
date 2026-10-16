# A7-labels v2 (2026-10-16, outline round 5): tile-level metrics with a PRE-SPECIFIED tile decision rule, same ROI definition for ID (test_id) and R (dirty2).
# Sets: ID = synthetic test_id (500 images, labels test_data/HBM_labels); R_calib = dirty2 0001-0050; R_heldout = dirty2 0251-2000 (labels download/HBM_labels).
# Per image: normal-tile pixel FPR (pred!=0 on normal-tile pixels); normal-tile false-alarm rate at tau (fraction of normal tiles whose predicted-positive
# pixel fraction >= tau); defect-tile recall at tau (fraction of defect tiles with predicted-positive fraction >= tau); class-correct defect-tile recall at tau
# (majority predicted class among positive pixels == true class). tau in {0.25, 0.5 (primary, fixed a priori), 0.75}. All configs x all runs; prenorm configs in both input modes.
from common import *; import csv, time
from dataset import json_to_mask
VALID = slice(3, 317); TAUS = (0.25, 0.5, 0.75)
R = sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2').glob('*.png')); LBL_R = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_labels')
TS = sorted((P4/'p42r_testsets'/'test_id').glob('image_*.npy')); LBL_ID = AUG/'test_data'/'HBM_labels'
def tile_grid():
    # tile index map on the padded/cropped frame (320x512), rows 3..317 valid; tiles 5x20 on the original 314x520 (tile 62x26, last tile absorbs remainder); columns 4..516 kept
    tm = np.full((320,512), -1, np.int64)
    for r in range(5):
        y0 = r*62; y1 = (r+1)*62 if r<4 else 314
        for c in range(20):
            x0 = c*26; x1 = (c+1)*26 if c<19 else 520
            xs0, xs1 = max(x0-4,0), min(x1-4,512)
            tm[3+y0:3+y1, xs0:xs1] = r*20+c
    return torch.from_numpy(tm)
TM = tile_grid()
def load_mask(lbl_path):
    mk = json_to_mask(str(lbl_path), 314, 520); return cv2.copyMakeBorder(mk, 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516].astype(np.int64)
sets = {'ID': ([np.load(p).astype(np.float32) for p in TS], [LBL_ID/f'label_{p.stem.split("_")[1]}.json' for p in TS], [p.stem for p in TS]),
        'R_calib': ([load_real(p) for p in R[0:50]], [LBL_R/f'label_{p.stem.split("_")[1]}.json' for p in R[0:50]], [p.stem for p in R[0:50]]),
        'R_heldout': ([load_real(p) for p in R[250:2000]], [LBL_R/f'label_{p.stem.split("_")[1]}.json' for p in R[250:2000]], [p.stem for p in R[250:2000]])}
masks = {s: torch.from_numpy(np.stack([load_mask(l) for l in v[1]])) for s, v in sets.items()}
# per-tile true class (tile-level labels => constant within tile): take the mask value at tile centre
def tile_true(M):  # M: (N,320,512) -> (N,100)
    out = torch.zeros(M.shape[0], 100, dtype=torch.long)
    for t in range(100):
        sel = (TM == t); out[:, t] = M[:, sel].mode(dim=1).values
    return out
TT = {s: tile_true(masks[s]) for s in sets}
TILE_PIX = {t: (TM == t) for t in range(100)}
OUT = Path('/mnt/gstore/home/xiangyue/p3stats/a7_labels'); OUT.mkdir(exist_ok=True)
w = csv.writer(open(OUT/'a7_labels_v2_per_image.csv', 'w', newline='')); w.writerow(['config','mode','run','set','image','normal_px_fpr'] + [f'{m}_t{int(t*100)}' for t in TAUS for m in ('normal_tile_far','defect_tile_recall','defect_tile_recall_cls')] + ['n_normal_tiles','n_defect_tiles'])
jobs = []
for cname, (rd, mode) in CONFIGS.items():
    for md in (['prenorm','none'] if mode == 'prenorm' else [mode]): jobs.append((cname, rd, md))
for cname, rd, md in jobs:
    pre = {s: [prep(x, md).astype(np.float32) for x in v[0]] for s, v in sets.items()}
    for ck in ckpts(rd):
        m = load_model(ck); t0 = time.time()
        for s, L in pre.items():
            pr = predict(m, L, bs=32); M = masks[s]; pos = (pr != 0)
            normal = (M == 0); fpr = (pos & normal).flatten(1).sum(1).float() / normal.flatten(1).sum(1).clamp(min=1).float()
            # per-tile positive fraction and majority predicted class
            frac = torch.zeros(pr.shape[0], 100); maj = torch.zeros(pr.shape[0], 100, dtype=torch.long)
            for t in range(100):
                sel = TILE_PIX[t]; pt = pr[:, sel]; frac[:, t] = (pt != 0).float().mean(1)
                cnt = torch.stack([(pt == k).sum(1) for k in (1,2,3)], 1); maj[:, t] = cnt.argmax(1) + 1
            tt = TT[s]; isn = (tt == 0); isd = (tt != 0)
            rows = []
            for i in range(pr.shape[0]):
                vals = [f'{fpr[i]:.5f}']
                for tau in TAUS:
                    det = frac[i] >= tau
                    far = float((det & isn[i]).sum()) / max(1, int(isn[i].sum())); rec = float((det & isd[i]).sum()) / max(1, int(isd[i].sum()))
                    recc = float((det & isd[i] & (maj[i] == tt[i])).sum()) / max(1, int(isd[i].sum()))
                    vals += [f'{far:.4f}', f'{rec:.4f}', f'{recc:.4f}']
                w.writerow([cname, md, ck.parent.name, s, sets[s][2][i]] + vals + [int(isn[i].sum()), int(isd[i].sum())])
        print(cname, md, ck.parent.name, f'{time.time()-t0:.0f}s', flush=True); del m; torch.cuda.empty_cache()
print('DONE')
