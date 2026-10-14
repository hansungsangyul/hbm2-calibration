#!/usr/bin/env python3
"""
dataset-level TP/FP/FN 집계 → per_image_counts.csv
run cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
then python /mnt/gstore/home/xiangyue/HBM_paper2/gen_counts_csv.py
"""
import sys, csv, json
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, '/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
from dataset import json_to_mask
from model import UNet

TESTSETS_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_testsets')
OUT_CSV = Path('/mnt/gstore/home/xiangyue/HBM_paper2/per_image_counts.csv')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 4

CONFIGS = [
    ('D_calib',  Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_v2_runs/calib')),
    ('D_manual', Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_v2_runs/manual')),
    ('AB-1',     Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p43_runs/ab1')),
    ('AB-2+3',   Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p43_runs/ab2')),
    ('AB-2+3',   Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p43_runs/ab3')),
    ('D_v5e',    Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/exp3b_runs')),
    ('D_sigma_p',Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/exp3_runs')),
    ('z-score',  Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/exp4_runs')),
]

TESTSETS = ['test_id', 'test_ood', 'test_cal']

def load_testset(ts_name):
    ts_dir = TESTSETS_DIR / ts_name
    meta = json.load(open(ts_dir / 'meta.json'))
    lbl_dir = Path(meta['label_dir'])
    items = []
    for lf in sorted(lbl_dir.glob('*.json')):
        num = lf.stem.replace('label_', '')
        np_path = ts_dir / f'image_{num}.npy'
        if np_path.exists():
            items.append((num, np_path, lf))
    return items

def make_mask(lf, img_deg):
    H_orig = img_deg.shape[0] - 6
    W_orig = 520
    mask = json_to_mask(str(lf), H_orig, W_orig)
    mask = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
    return mask.astype(np.int64)

print('Loading testsets...')
testset_data = {ts: load_testset(ts) for ts in TESTSETS}
for ts, items in testset_data.items():
    print(f'  {ts}: {len(items)} images')

# 라벨 캐시 (동일 이미지 번호는 세 testset 모두 같은 label 파일)
label_cache = {}

model = UNet(in_channels=3, out_channels=N_CLASSES).to(DEVICE)
model.eval()

rows = []
total_runs = sum(len(list(run_root.glob('run*'))) for _, run_root in CONFIGS)
done = 0

for cfg_name, run_root in CONFIGS:
    run_dirs = sorted(run_root.glob('run*'))
    for run_dir in run_dirs:
        pth = run_dir / 'final.pth'
        if not pth.exists():
            print(f'  SKIP (no checkpoint): {pth}')
            continue
        run_id = f'{run_root.parent.name}_{run_root.name}_{run_dir.name}'
        state = torch.load(str(pth), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)

        for ts_name in TESTSETS:
            tp_acc = np.zeros(N_CLASSES, dtype=np.int64)
            fp_acc = np.zeros(N_CLASSES, dtype=np.int64)
            fn_acc = np.zeros(N_CLASSES, dtype=np.int64)

            for img_num, np_path, lf in testset_data[ts_name]:
                img_deg = np.load(str(np_path))
                img_3ch = np.stack([img_deg] * 3, axis=0).astype(np.float32)
                img_t = torch.from_numpy(img_3ch).unsqueeze(0).to(DEVICE)

                lf_key = str(lf)
                if lf_key not in label_cache:
                    label_cache[lf_key] = make_mask(lf, img_deg)
                mask = label_cache[lf_key]
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out = model(img_t)
                    pred = out.argmax(dim=1)  # (1,H,W)

                pred_np = pred.squeeze(0).cpu().numpy()
                for c in range(N_CLASSES):
                    p = pred_np == c
                    g = mask == c
                    tp_acc[c] += int(np.logical_and(p, g).sum())
                    fp_acc[c] += int(np.logical_and(p, ~g).sum())
                    fn_acc[c] += int(np.logical_and(~p, g).sum())

            for c in range(N_CLASSES):
                rows.append({
                    'config': cfg_name,
                    'test_set': ts_name.upper().replace('_', '_'),
                    'run_id': run_id,
                    'image_id': 'ALL',
                    'class_id': c,
                    'tp': int(tp_acc[c]),
                    'fp': int(fp_acc[c]),
                    'fn': int(fn_acc[c]),
                })

        done += 1
        if done % 10 == 0 or done == total_runs:
            print(f'  {done}/{total_runs} runs done')

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['config','test_set','run_id','image_id','class_id','tp','fp','fn'])
    w.writeheader()
    w.writerows(rows)

print(f'\nSaved: {OUT_CSV}  ({len(rows)} rows)')
