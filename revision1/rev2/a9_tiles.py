# A9: tile-level (microbump) scores on TEST_ID / TEST_OOD for D_calib, D_manual, AB-1 (all 20 runs). Scores = mean softmax over tile pixels.
from common import *; import time
sys.path.insert(0, str(AUG)); from dataset import json_to_mask
TS = P4/'p42r_testsets'; OUT = Path(__file__).parent/'a9'; OUT.mkdir(exist_ok=True)
IDX = list(range(1501, 2001)); LBL = AUG/'test_data'/'HBM_labels'
# tile map: original 314x520 grid, tile_h=62, tile_w=26 (json_to_mask geometry), padded/cropped like the images
H0, W0 = 314, 520; th, tw = H0//5, W0//20
tm = -np.ones((H0, W0), np.int64)
for r in range(5):
    for c in range(20): tm[r*th:(r+1)*th, c*tw:(c+1)*tw] = r*20 + c
tm = cv2.copyMakeBorder(tm.astype(np.float32), 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=-1)[:, 4:516].astype(np.int64)
tmt = torch.from_numpy(tm).to(DEV).flatten(); valid = tmt >= 0; ids = tmt[valid]; cnt = torch.bincount(ids, minlength=100).float()
# ground truth per tile + coverage
gt = np.zeros((500, 100), np.int8); cov = np.full((500, 100), np.nan, np.float32); cmap = {'normal': 0, 'crack': 1, 'hole': 2, 'open': 3}
for k, i in enumerate(IDX):
    for a in json.load(open(LBL/f'label_{i:04d}.json')):
        if isinstance(a, dict) and a.get('label', '').lower() in cmap:
            t = (a['row']-1)*20 + (a['col']-1); gt[k, t] = cmap[a['label'].lower()]; cov[k, t] = a.get('params', {}).get('coverage', np.nan)
np.save(OUT/'gt.npy', gt); np.save(OUT/'coverage.npy', cov)
CF = {'D_calib': CONFIGS['D_calib'][0], 'D_manual': CONFIGS['D_manual'][0], 'AB-1': CONFIGS['AB-1'][0]}
for ts in ['test_id', 'test_ood']:
    imgs = [np.load(TS/ts/f'image_{i}.npy').astype(np.float32) for i in IDX]   # frozen test files, no extra preprocessing (as in Section VI evaluation)
    for cname, rd in CF.items():
        S = np.zeros((20, 500, 100, 4), np.float16)
        for r, ck in enumerate(ckpts(rd)):
            m = load_model(ck)
            with torch.no_grad():
                for b in range(0, 500, 25):
                    t = torch.from_numpy(np.stack([np.stack([x]*3) for x in imgs[b:b+25]])).to(DEV)
                    p = torch.softmax(m(t), 1).flatten(2)[:, :, valid]          # [bs,4,Nvalid]
                    acc = torch.zeros(p.shape[0], 4, 100, device=DEV).index_add_(2, ids, p) / cnt
                    S[r, b:b+25] = acc.permute(0, 2, 1).cpu().numpy().astype(np.float16)
            print(ts, cname, ck.parent.name, flush=True); del m; torch.cuda.empty_cache()
        np.save(OUT/f'scores_{cname}_{ts}.npy', S)
print('DONE')
