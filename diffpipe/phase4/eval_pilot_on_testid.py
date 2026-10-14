#!/usr/bin/env python3
"""
D-6: 파일럿 체크포인트 → TEST_ID 500장 평가 (HANDOFF v6.11 §D-6)
판정 기준: F1 >= 0.92 = 테스트셋 무죄
"""
import json, sys, time
from pathlib import Path

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from model import UNet
import config as cfg

CKPT_PATH = DIFFPIPE_DIR / 'phase5' / 'pilot_output' / 'pilot_checkpoint.pth'
TESTID_DIR = DIFFPIPE_DIR / 'phase4' / 'p42r_testsets' / 'test_id'
OUT_PATH   = DIFFPIPE_DIR / 'phase4' / 'd6_pilot_testid_result.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[D-6] device={DEVICE}")
print(f"  ckpt: {CKPT_PATH}")
print(f"  testset: {TESTID_DIR}")

class FixedTestset(Dataset):
    def __init__(self, ts_dir: Path):
        meta = json.load(open(ts_dir / 'meta.json'))
        lbl_dir = Path(meta['label_dir'])
        lbl_files = sorted(lbl_dir.glob('*.json'))
        self.images, self.label_files = [], []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            np_path = ts_dir / f'image_{num}.npy'
            if np_path.exists():
                self.images.append(np_path)
                self.label_files.append(lf)
        print(f"  FixedTestset: {len(self.images)} images")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        from dataset import json_to_mask
        img_deg = np.load(self.images[idx])
        img_3ch = np.stack([img_deg]*3, axis=0).astype(np.float32)
        lf = self.label_files[idx]
        H_orig = img_deg.shape[0] - 6
        W_orig = 520
        mask = json_to_mask(str(lf), H_orig, W_orig)
        mask = cv2.copyMakeBorder(mask, 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        return torch.from_numpy(img_3ch), torch.from_numpy(mask.astype(np.int64))

def compute_f1_per_class(preds, targets, n_classes=4):
    preds_flat = preds.argmax(dim=1).flatten()
    tgts_flat  = targets.flatten()
    f1s = []
    for c in range(n_classes):
        tp = ((preds_flat==c)&(tgts_flat==c)).sum().float()
        fp = ((preds_flat==c)&(tgts_flat!=c)).sum().float()
        fn = ((preds_flat!=c)&(tgts_flat==c)).sum().float()
        prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
        f1s.append((2*prec*rec/(prec+rec+1e-8)).item())
    return f1s

# Load model
model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
    pilot_epoch = ckpt.get('epoch', '?')
else:
    model.load_state_dict(ckpt)
    pilot_epoch = '?'
print(f"  loaded checkpoint (pilot epoch={pilot_epoch})")

ds = FixedTestset(TESTID_DIR)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

model.eval()
all_f1 = []
t0 = time.time()
with torch.no_grad():
    for imgs, msks in loader:
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
        outs = model(imgs)
        all_f1.append(compute_f1_per_class(outs, msks))

per_class = [float(np.mean([b[c] for b in all_f1])) for c in range(4)]
mean_f1 = float(np.mean(per_class))
wall = time.time() - t0

THRESHOLD = 0.92
verdict = "테스트셋 무죄 (H5 기각, 학습 측 문제)" if mean_f1 >= THRESHOLD else "H5 부상 (TEST_ID 재검 필요)"

print(f"\n[D-6 결과]")
print(f"  mean_F1 = {mean_f1:.4f}")
print(f"  per_class = {[f'{v:.3f}' for v in per_class]}")
print(f"  wall = {wall:.1f}s")
print(f"  판정 기준: F1 >= {THRESHOLD}")
print(f"  판정: {verdict}")

result = {
    'pilot_epoch': pilot_epoch,
    'testset': 'TEST_ID (500장)',
    'n_images': len(ds),
    'mean_f1': mean_f1,
    'f1_by_class': per_class,
    'threshold': THRESHOLD,
    'verdict': verdict,
    'wall_s': wall,
}
with open(OUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)
print(f"  저장: {OUT_PATH}")
