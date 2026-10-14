import sys, json, gc, numpy as np, cv2, torch
from pathlib import Path
from scipy.ndimage import gaussian_filter
DP = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); AUG = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2'); TS = DP/'phase4'/'p42r_testsets'; OUT = Path(__file__).parent
sys.path.insert(0, str(AUG)); from model import UNet; import config as cfg
sys.path.insert(0, str(DP/'phase4')); 
DEV = 'cuda'; A, B = 0.501834, 0.173320; VALID = slice(3, 317); TOT = 314*512
def load_real(p): img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0; return cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
reals = [load_real(p) for p in sorted(REAL.glob('*.png'))[:50]]
CONF = {'D_calib': dict(runs=DP/'phase4'/'p42r_v2_runs'/'calib', ref='test_cal', pre=lambda x: A*x+B, fwhm=2.0956),
        'D_manual': dict(runs=DP/'phase4'/'p42r_v2_runs'/'manual', ref='test_id', pre=lambda x: x, fwhm=2.5)}
rng = np.random.default_rng(20260925)
def ref_stats(name):
    imgs = [np.load(TS/name/f'image_{i}.npy') for i in range(1501, 2001)]
    v = np.concatenate([im[VALID].ravel() for im in imgs]); return float(v.mean()), float(v.std()), np.quantile(v, np.linspace(0, 1, 257))
REF = {k: ref_stats(c['ref']) for k, c in CONF.items()}
def qmap(x, src_q, dst_q):  # global quantile mapping on valid rows, applied to whole image
    return np.interp(x, src_q, dst_q).astype(np.float32)
def interventions(cname, raws, gseed):
    c = CONF[cname]; mt, st, qt = REF[cname]; sig = c['fwhm']/2.355
    def prep(rs, blur=False, noise=False, mom=False, quant=False):
        out = []
        r_ = np.random.default_rng(gseed)
        for r in rs:
            x = r.copy()
            if blur: x = gaussian_filter(x, sig)
            if noise: x = np.clip(x + r_.normal(0, 0.041, x.shape).astype(np.float32), 0, 1)
            out.append(c['pre'](x))
        if mom or quant:
            v = np.concatenate([o[VALID].ravel() for o in out]); m, s = v.mean(), v.std(); qs = np.quantile(v, np.linspace(0, 1, 257))
            out = [((o-m)/s*st+mt if mom else qmap(o, qs, qt)) for o in out]
        return out
    return {'I0': prep(raws), 'I1': prep(raws, mom=True), 'I2': prep(raws, quant=True), 'I3': prep(raws, blur=True), 'I4': prep(raws, noise=True), 'I5': prep(raws, noise=True, quant=True), 'I6': prep(raws, blur=True, noise=True, quant=True)}
def fpr_of(pred):
    v = pred[VALID]; nb = (v != 0); return float(nb.sum())/TOT, int((v == 0).sum())
model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEV)
res = json.load(open(OUT/'a5_results_partial.json'))['results'] if (OUT/'a5_results_partial.json').exists() else {}
# synthetic control: D_calib on 50 test_cal images with I3/I4/I5, 4-class dataset-level macro F1, first 5 ckpts

from dataset import json_to_mask
meta = json.load(open(TS/'test_cal'/'meta.json')); lbl = Path(meta['label_dir'])
syn = []; 
for i in range(1501, 1551):
    im = np.load(TS/'test_cal'/f'image_{i}.npy'); m = json_to_mask(str(lbl/f'label_{i:04d}.json'), im.shape[0]-6, 520); m = cv2.copyMakeBorder(m, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]; syn.append((im, m))
mt, st, qt = REF['D_calib']; sig = 2.0956/2.355; r_ = np.random.default_rng(7)
def syn_iv(k, im):
    x = im.copy()
    if k in ('I3', 'I6'): x = gaussian_filter(x, sig)
    if k in ('I4', 'I5', 'I6'): x = np.clip(x + r_.normal(0, 0.041, x.shape).astype(np.float32), 0, 1)
    return x
ctrl = {}
for k in ('I0', 'I3', 'I4', 'I5'):
    cm = np.zeros((4, 4), dtype=np.int64)
    for ck in sorted((DP/'phase4'/'p42r_v2_runs'/'calib').glob('run*/final.pth'))[:5]:
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=False)); model.eval()
        with torch.no_grad():
            for im, m in syn:
                x = syn_iv(k, im); p = model(torch.from_numpy(np.stack([x]*3)).unsqueeze(0).to(DEV)).argmax(1)[0].cpu().numpy()
                cm += np.bincount((m[VALID].ravel()*4 + p[VALID].ravel()), minlength=16).reshape(4, 4)
    f1 = [2*cm[c, c]/max(cm[c, :].sum()+cm[:, c].sum(), 1) for c in range(4)]; ctrl[k] = dict(macro4=float(np.mean(f1)), per_class=[float(v) for v in f1])
    print('synthetic control', k, ctrl[k], flush=True)
json.dump(dict(results=res, synthetic_control=ctrl, ref={k: [v[0], v[1]] for k, v in REF.items()}), open(OUT/'a5_results.json', 'w'), indent=2); print('DONE')
