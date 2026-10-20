#!/usr/bin/env python3
"""A14 dataset-level TP/FP/FN counts on TEST_ID/TEST_OOD/TEST_CAL for the A14 runs.
Identical logic to analysis/gen_counts_csv.py (argmax over the full padded/cropped frame, counts pooled over all
images of a set). Historical D_manual and AB-1 counts are taken from f1_recompute_results/counts/per_image_counts.csv.
Output: a14/a14_counts.csv"""
import sys, csv, json
from pathlib import Path
import numpy as np, cv2, torch
sys.path.insert(0, '/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
from dataset import json_to_mask
from model import UNet
TESTSETS_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_testsets')
RUNS = Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a14/runs')
OUT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a14/a14_counts.csv')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TS = ['test_id', 'test_ood', 'test_cal']
def load_ts(t):
    d = TESTSETS_DIR/t; meta = json.load(open(d/'meta.json')); items = []
    for lf in sorted(Path(meta['label_dir']).glob('*.json')):
        n = lf.stem.replace('label_', ''); p = d/f'image_{n}.npy'
        if p.exists(): items.append((p, lf))
    return items
def mask_of(lf, img):
    m = json_to_mask(str(lf), img.shape[0]-6, 520)
    return cv2.copyMakeBorder(m, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516].astype(np.int64)
data = {t: load_ts(t) for t in TS}; cache = {}
for t, items in data.items():
    assert len(items) == 500, (t, len(items))
for cond in ['dman_r', 'ab1_r', 'c1', 'c2']:
    expected = [0] if cond in ['dman_r', 'ab1_r'] else list(range(20))
    actual = {p.name for p in (RUNS / cond).glob('run*') if p.is_dir()}
    assert actual == {f'run{i}' for i in expected}, (cond, actual)
    seed_base = {'dman_r': 760000, 'ab1_r': 770000, 'c1': 830000, 'c2': 840000}[cond]
    for i in expected:
        rd = RUNS / cond / f'run{i}'
        assert (rd / 'final.pth').is_file(), rd
        r = json.load(open(rd / 'results.json'))
        assert r['condition'] == cond and r['run_idx'] == i
        assert r['seed'] == seed_base + i
        assert r['epochs_completed'] == 41
model = UNet(in_channels=3, out_channels=4).to(DEV); model.eval()
rows = []
for cond in ['dman_r', 'ab1_r', 'c1', 'c2']:
    for rd in sorted((RUNS/cond).glob('run*')):
        pth = rd/'final.pth'
        if not pth.exists(): continue
        model.load_state_dict(torch.load(str(pth), map_location=DEV, weights_only=True))
        for t in TS:
            tp = np.zeros(4, np.int64); fp = np.zeros(4, np.int64); fn = np.zeros(4, np.int64)
            for p, lf in data[t]:
                img = np.load(str(p)); x = torch.from_numpy(np.stack([img]*3, 0).astype(np.float32)).unsqueeze(0).to(DEV)
                k = str(lf)
                if k not in cache: cache[k] = mask_of(lf, img)
                g = cache[k]
                with torch.no_grad(): pr = model(x).argmax(1).squeeze(0).cpu().numpy()
                for c in range(4):
                    pc = pr == c; gc = g == c
                    tp[c] += int((pc & gc).sum()); fp[c] += int((pc & ~gc).sum()); fn[c] += int((~pc & gc).sum())
            for c in range(4):
                rows.append(dict(config=cond, test_set=t.upper(), run_id=f'{cond}_{rd.name}', image_id='ALL', class_id=c,
                                 tp=int(tp[c]), fp=int(fp[c]), fn=int(fn[c])))
        print('done', cond, rd.name, flush=True)
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('saved', OUT, len(rows))
