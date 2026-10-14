# Table 7 D_v5e row under the corrected (raw) convention: false components as in run_real_fpr_v5e_zscore.py (8-connectivity, >= 50 px, valid rows)
from common import *; from scipy.ndimage import label as nd_label
REAL = sorted(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2').glob('*.png'))[:5]; VALID = slice(3, 317)
imgs = [load_real(p) for p in REAL]; res = []
for ck in ckpts(CONFIGS['D_v5e'][0]):
    m = load_model(ck); pr = predict(m, imgs, bs=5)[:, VALID, :].numpy()
    for i in range(5):
        nb = (pr[i] != 0).astype(np.uint8); lab, n = nd_label(nb, structure=np.ones((3, 3), int)); res.append(sum(1 for c in range(1, n+1) if (lab == c).sum() >= 50))
    del m; torch.cuda.empty_cache()
print('median false components', float(np.median(res)), 'n', len(res))
