import sys, json, numpy as np, cv2, torch
from pathlib import Path
DP = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); P4 = DP/'phase4'; AUG = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
sys.path.insert(0, str(AUG)); sys.path.insert(0, str(DP/'src'))
from model import UNet; import config as cfg
A, B = 0.501834, 0.173320; DEV = 'cuda'
CONFIGS = {  # name: (run_dir, preprocessing)  -- preprocessing as in Table 7 / exp_config.USES_PRENORM
 'D_manual': (P4/'p42r_v2_runs'/'manual', 'none'), 'AB-1': (P4/'p43_runs'/'ab1', 'prenorm'),
 'AB-2': (P4/'p43_runs'/'ab2', 'none'), 'AB-3': (P4/'p43_runs'/'ab3', 'none'),
 'D_calib': (P4/'p42r_v2_runs'/'calib', 'prenorm'), 'D_v5e': (P4/'v77'/'exp3b_runs', 'prenorm'),
 'D_sigma_p': (P4/'v77'/'exp3_runs', 'prenorm'), 'z-score': (P4/'v77'/'exp4_runs', 'zscore')}
def prep(img, mode):
    if mode == 'prenorm': return A*img + B
    if mode == 'zscore': return (img - img.mean())/(img.std() + 1e-8)
    return img
def load_real(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    return cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
def load_model(ck):
    m = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEV); m.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)); m.eval(); return m
def ckpts(run_dir): return sorted(Path(run_dir).glob('run*/final.pth'), key=lambda p: int(p.parent.name[3:]))
@torch.no_grad()
def predict(model, imgs, bs=16, probs=False):
    out = []
    for i in range(0, len(imgs), bs):
        t = torch.from_numpy(np.stack([np.stack([x]*3) for x in imgs[i:i+bs]]).astype(np.float32)).to(DEV)
        lg = model(t); out.append((torch.softmax(lg, 1) if probs else lg.argmax(1)).cpu())
    return torch.cat(out)
