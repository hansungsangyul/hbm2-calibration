# A13 (2026-10-17): label-assisted sensitivity analysis of the calibration statistics to the pixel support.
# Three frozen support conditions on the 50 calibration images (R 0001-0050 and their clean counterparts):
#   U = original support (as in CalibrationLoss / v5c), N = U ∩ normal-labelled tiles (per image), C = U ∩ label-unrestricted control tiles
#   (per image and tile row, the same number of tiles as labelled normal in that row, chosen by a fixed random permutation).
# The same per-image support is applied to the reference image i and to the synthetic image generated from its clean counterpart i.
# Spectral term: residual field computed before masking; power normalised by the effective window energy sum(W^2), W = Hann * support
# (for U this normalisation cancels in the log-ratio and reproduces the archived objective). Pre-registered in A13_SUPPORT_PREREG.md.
import sys, json, numpy as np, cv2, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe'); sys.path.insert(0, str(ROOT/'src'))
from losses import CalibrationLoss
LBL = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_labels')

def tile_id_map(H=314, W=520):
    tm = np.full((H, W), -1, np.int64)
    for r in range(5):
        y0 = r*62; y1 = (r+1)*62 if r < 4 else H
        for c in range(20):
            x0 = c*26; x1 = (c+1)*26 if c < 19 else W
            tm[y0:y1, x0:x1] = r*20 + c
    return tm

CLASS_MAP = {'normal': 0, 'crack': 1, 'hole': 2, 'open': 3}
def tile_classes(json_file):
    """100 tile classes (row-major, (row-1)*20+(col-1)) from a label JSON (list of {label,row,col}); same mapping as hbm_augmentation/dataset.json_to_mask."""
    tc = np.zeros(100, np.int64)
    for ann in json.load(open(json_file)):
        if isinstance(ann, dict) and ann.get('row') is not None and ann.get('col') is not None and ann.get('label', '').lower() in CLASS_MAP:
            tc[(ann['row'] - 1) * 20 + (ann['col'] - 1)] = CLASS_MAP[ann['label'].lower()]
    return tc

def support_masks(n_images, H, W, arm):
    """Per-image boolean pixel supports (H,W) for arm U/N/C; images 0001..n_images. Also returns per-image tile lists."""
    TM = tile_id_map(H, W); masks = []; tiles = []
    for i in range(1, n_images + 1):
        if arm == 'U':
            masks.append(np.ones((H, W), bool)); tiles.append(list(range(100))); continue
        tile_cls = tile_classes(LBL/f'label_{i:04d}.json')
        normal_tiles = [t for t in range(100) if tile_cls[t] == 0]
        if arm == 'N':
            sel = normal_tiles
        else:   # C: per row, same count as normal tiles in that row, fixed permutation independent of labels/intensities
            rs = np.random.RandomState(820000 + i); sel = []
            for r in range(5):
                k = sum(1 for t in normal_tiles if t // 20 == r)
                perm = rs.permutation(20); sel += [r*20 + int(c) for c in perm[:k]]
        m = np.isin(TM, sel); masks.append(m); tiles.append(sorted(sel))
    return masks, tiles

class SupportCalibrationLoss(CalibrationLoss):
    """CalibrationLoss with per-image pixel supports. forward(syn, img_idx) uses the support of image img_idx on the synthetic side."""
    def __init__(self, real_images, supports, **kw):
        super().__init__(real_images, **kw)          # builds the original flat-selection rules, bin edges, windows (frozen)
        self.supports = [torch.from_numpy(m) for m in supports]
        real_t = [torch.from_numpy(img.astype(np.float32)) for img in real_images]
        H, W, ps, B = self.H, self.W, self.patch_size, self.B
        nH, nW = self.nH, self.nW
        # ---- patch-level support: patch fully inside support ----
        self.patch_ok = []
        for m in self.supports:
            mc = m[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(nH*nW, ps*ps)
            self.patch_ok.append(mc.all(dim=1))                                  # [nH*nW] bool
        # original flat patch set and bin assignment (frozen from the full images)
        flat_idx_all = torch.cat([self.vm_patch_indices[b] for b in range(B)])
        bin_of = torch.full((nH*nW,), -1, dtype=torch.long)
        for b in range(B): bin_of[self.vm_patch_indices[b]] = b
        # per-image patch indices per bin (U-set ∩ support_i)
        self.vm_idx_img = []
        for ok in self.patch_ok:
            self.vm_idx_img.append([self.vm_patch_indices[b][ok[self.vm_patch_indices[b]]] for b in range(B)])
        # real-side vm curve: per patch position, mean over images where the position is supported (and flat)
        pm = torch.stack([img[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(nH*nW, ps*ps).mean(1) for img in real_t])   # [n_img, P]
        pv = torch.stack([img[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(nH*nW, ps*ps).var(1, unbiased=False) for img in real_t])
        okm = torch.stack(self.patch_ok).float()                                 # [n_img, P]
        cnt = okm.sum(0)                                                         # [P]
        avg_pv = (pv * okm).sum(0) / cnt.clamp(min=1)
        self.vm_real_log_norm_s = torch.zeros(B); self._vm_valid_bins_s = []
        logs = []
        for b in range(B):
            idx = self.vm_patch_indices[b]; idx = idx[cnt[idx] > 0]
            if len(idx) == 0: continue
            v = float(avg_pv[idx].mean())
            if v > 0: logs.append((b, np.log(v)))
        ml = np.mean([l for _, l in logs]) if logs else 0.0
        for b, l in logs: self.vm_real_log_norm_s[b] = l - ml; self._vm_valid_bins_s.append(b)
        self.vm_support_patch_counts = [int(ok.sum()) for ok in self.patch_ok]
        # ---- spectral term: residual before masking; W_i = Hann * flat_mask * support_i; normalise by sum(W_i^2) ----
        self.W_img = [self.hann_win * self.flat_mask * m.float() for m in self.supports]
        self.W_energy = torch.tensor([float((w ** 2).sum()) for w in self.W_img])
        P_acc = torch.zeros(self.nps_bins)
        for img, w, e in zip(real_t, self.W_img, self.W_energy):
            P_acc += self._nps_single_w(img, w, e)
        self.P_real_s = (P_acc / len(real_t)).clamp(min=1e-10)
        # ---- radial profile: per bin mean over supported pixels; average over images; normalised by its mean ----
        self.rad_idx_img = []
        prof_acc = torch.zeros(self.rad_bins); nvalid = torch.zeros(self.rad_bins)
        for img, m in zip(real_t, self.supports):
            flat_m = m.flatten(); img_flat = img.flatten(); per = []
            for b, idx_2d in enumerate(self.rad_bin_indices):
                fi = idx_2d[:, 0] * W + idx_2d[:, 1]; fi = fi[flat_m[fi]]
                per.append(fi)
                if len(fi) > 0: prof_acc[b] += img_flat[fi].mean(); nvalid[b] += 1
            self.rad_idx_img.append(per)
        prof_abs = prof_acc / nvalid.clamp(min=1)
        self.rad_valid_s = nvalid > 0
        self.prof_real_s = prof_abs / prof_abs[self.rad_valid_s].mean().clamp(min=1e-8)
        # ---- histogram over supported pixels ----
        h_acc = torch.zeros(self.hist_bins)
        for img, m in zip(real_t, self.supports):
            h_acc += self._soft_hist_np(img.flatten()[m.flatten()])
        self.hist_real_s = h_acc / len(real_t)
        self.support_pixel_counts = [int(m.sum()) for m in self.supports]
    def _nps_single_w(self, img, w, energy):
        img4 = img.unsqueeze(0).unsqueeze(0)
        blurred = F.conv2d(img4, self.gauss_kernel, padding=self.gauss_pad).squeeze(0).squeeze(0)
        res = img - blurred
        power = torch.fft.rfft2(res * w).abs() ** 2 / energy.clamp(min=1e-12)
        P = torch.zeros(self.nps_bins)
        for b in range(self.nps_bins):
            m = self.nps_bin_map == b
            if m.any(): P[b] = power[m].mean()
        return P
    # ---- differentiable losses with per-image support ----
    def loss_parts(self, syn, i):
        dev = syn.device; H, W, ps = self.H, self.W, self.patch_size
        # vm
        patches = syn[:self.nH*ps, :self.nW*ps].reshape(self.nH, ps, self.nW, ps).permute(0, 2, 1, 3).reshape(self.nH*self.nW, ps*ps)
        pvar = patches.var(dim=1, unbiased=False)
        parts, used = [], []
        for b in self._vm_valid_bins_s:
            idx = self.vm_idx_img[i][b]
            if len(idx) == 0: continue
            parts.append(torch.log(pvar[idx.to(dev)].mean().clamp(min=1e-12))); used.append(b)
        if parts:
            s = torch.stack(parts); s = s - s.mean()
            L_vm = ((s - self.vm_real_log_norm_s[used].to(dev)) ** 2).mean()
        else: L_vm = torch.zeros((), device=dev)
        # nps
        blurred = F.conv2d(syn.unsqueeze(0).unsqueeze(0), self.gauss_kernel.to(dev), padding=self.gauss_pad).squeeze(0).squeeze(0)
        res = syn - blurred
        power = torch.fft.rfft2(res * self.W_img[i].to(dev)).abs() ** 2 / self.W_energy[i].to(dev).clamp(min=1e-12)
        bm = self.nps_bin_map.to(dev)
        P = torch.stack([power[bm == b].mean() if (bm == b).any() else torch.zeros((), device=dev) for b in range(self.nps_bins)]).clamp(min=1e-10)
        L_nps = ((torch.log(P) - torch.log(self.P_real_s.to(dev))) ** 2).mean()
        # rad
        sf = syn.flatten(); prof = []
        for b, fi in enumerate(self.rad_idx_img[i]):
            prof.append(sf[fi.to(dev)].mean() if len(fi) > 0 else torch.zeros((), device=dev))
        prof = torch.stack(prof); valid = self.rad_valid_s.to(dev)
        pn = prof / prof[valid].mean().clamp(min=1e-8)
        L_rad = ((pn[valid] - self.prof_real_s.to(dev)[valid]) ** 2).mean()
        # hist
        x = sf[self.supports[i].flatten().to(dev)]
        diff = x.unsqueeze(1) - self.hist_centers.to(dev).unsqueeze(0)
        h = torch.exp(-diff ** 2 / (2 * self.hist_s ** 2)).mean(dim=0)
        L_hist = ((h - self.hist_real_s.to(dev)) ** 2).mean()
        return L_vm, L_nps, L_rad, L_hist
    def forward(self, syn, img_idx=None, lambda_weights=(1.0, 1.0, 1.0, 1.0)):
        if img_idx is None: return super().forward(syn, lambda_weights)
        syn = self._maybe_transpose(syn)
        L_vm, L_nps, L_rad, L_hist = self.loss_parts(syn, img_idx)
        lw = lambda_weights
        return lw[0]*L_vm + lw[1]*L_nps + lw[2]*L_rad + lw[3]*L_hist, {'L_vm': L_vm, 'L_nps': L_nps, 'L_rad': L_rad, 'L_hist': L_hist}
    def real_stats(self):
        return {'vm_log_norm': self.vm_real_log_norm_s.tolist(), 'vm_valid_bins': list(self._vm_valid_bins_s), 'P_real': self.P_real_s.tolist(),
                'prof_real': self.prof_real_s.tolist(), 'rad_valid': self.rad_valid_s.tolist(), 'hist_real': self.hist_real_s.tolist(),
                'support_pixel_counts': self.support_pixel_counts, 'vm_support_patch_counts': self.vm_support_patch_counts, 'W_energy': self.W_energy.tolist()}

