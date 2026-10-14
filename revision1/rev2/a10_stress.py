# A10: two alternative-simulator stress sets from the same 500 test originals at theta_manual, then frozen-model evaluation (all 8 configs x 20 runs), dataset-level macro F1.
from common import *; import time
sys.path.insert(0, str(AUG)); from dataset import json_to_mask
import torch.nn.functional as F
sys.path.insert(0, str(P4)); from generate_p42r_testsets import make_pipe, THETA_MANUAL
OUT = Path(__file__).parent/'a10'; OUT.mkdir(exist_ok=True)
SRC = AUG/'test_data'/'HBM_clean'; LBL = AUG/'test_data'/'HBM_labels'; files = sorted(SRC.glob('*.png')); assert len(files) == 500
def tail_kernel(fwhm, k=31, w_tail=0.2):
    x = torch.arange(k, dtype=torch.float32) - k//2; X, Y = torch.meshgrid(x, x, indexing='ij'); r2 = X**2 + Y**2
    g = torch.exp(-r2/(2*(fwhm/2.355)**2)); g /= g.sum(); gam = fwhm/2; l = 1/(1 + r2/gam**2)**1.5; l /= l.sum()   # 2-D Lorentzian-type (Cauchy) tail, same FWHM scale
    K = (1-w_tail)*g + w_tail*l; return (K/K.sum()).view(1, 1, k, k)
def gen(name, seed_base, mode):
    d = OUT/name; d.mkdir(exist_ok=True); th = dict(THETA_MANUAL); 
    if mode == 'post': th['sigma_post'] = 0.017
    pipe = make_pipe(th); K = tail_kernel(th['fwhm'])
    for i, f in enumerate(files):
        o = d/f"image_{f.stem.replace('image_', '')}.npy"
        if o.exists(): continue
        x = torch.from_numpy(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0); H, W = x.shape
        torch.manual_seed(seed_base + i); eps = {'eps1': torch.randn(H, W), 'eps2': torch.randn(H, W), 'eps3': torch.rand(H, W), 'eps_post': torch.randn(H, W)}
        with torch.no_grad():
            if mode == 'post': y = pipe(x, eps)
            else:
                y = pipe._stage3_salt_pepper(pipe._stage2_gaussian(pipe._stage1_poisson(x, eps['eps1']), eps['eps2']), eps['eps3'])
                y = F.conv2d(F.pad(y[None, None], [15]*4, mode='reflect'), K)[0, 0]
                y = pipe._stage7_ring(pipe._stage6_contrast(pipe._stage5_vignetting(y)))
        y = cv2.copyMakeBorder(y.numpy(), 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]; np.save(o, y.astype(np.float32))
    print('generated', name, flush=True)
gen('test_tailpsf', 530000, 'tail'); gen('test_postnoise', 540000, 'post')
masks = []
for f in files:
    n = f.stem.replace('image_', ''); m = json_to_mask(str(LBL/f'label_{n}.json'), 314, 520); masks.append(cv2.copyMakeBorder(m.astype(np.uint8), 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516])
M = torch.from_numpy(np.stack(masks).astype(np.int64))
rows = []
for ts in ['test_tailpsf', 'test_postnoise']:
    imgs = [np.load(OUT/ts/f"image_{f.stem.replace('image_', '')}.npy") for f in files]
    for cname, (rd, mode) in CONFIGS.items():
        L = [(x - x.mean())/(x.std() + 1e-8) if mode == 'zscore' else x for x in imgs]   # frozen-file convention of Section VI: no extra prenorm; z-score config normalizes per image
        for ck in ckpts(rd):
            m = load_model(ck); pr = predict(m, L, bs=32); cm = torch.bincount((M*4 + pr).flatten(), minlength=16).reshape(4, 4).double()
            tp = cm.diag(); f1 = (2*tp/(cm.sum(0) + cm.sum(1))).numpy()
            rows.append([ts, cname, ck.parent.name, *f1.tolist(), f1.mean(), f1[1:].mean()]); del m; torch.cuda.empty_cache()
        print(ts, cname, np.mean([r[7] for r in rows if r[0] == ts and r[1] == cname]).round(4), flush=True)
import csv; w = csv.writer(open(OUT/'a10_runs.csv', 'w', newline='')); w.writerow(['test_set', 'config', 'run', 'f1_bg', 'f1_crack', 'f1_hole', 'f1_open', 'macro4', 'macro_defect']); w.writerows(rows); print('DONE')
