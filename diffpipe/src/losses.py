"""
CalibrationLoss: differentiable loss module for HBM nano-CT degradation parameter estimation.

Computes the difference between statistics of synthetic (degraded) images and
real CT images. All discrete operations (binning, masking) use the REAL image
to build fixed masks; those same masks are applied to synthetic images so that
gradients flow through synthetic pixel values only.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class CalibrationLoss(nn.Module):
    """
    Calibration loss module.

    __init__ precomputes all real-image statistics once (non-differentiable).
    forward() computes differentiable losses on synthetic pipeline output.
    """

    def __init__(
        self,
        real_images: list,
        device: str = 'cpu',
        B: int = 16,
        nps_bins: int = 32,
        rad_bins: int = 24,
        hist_bins: int = 64,
    ):
        """
        Args:
            real_images: list of numpy arrays (H,W) float32 in [0,1]
            device:      torch device string (unused here; move module with .to())
            B:           number of brightness bins for L_vm
            nps_bins:    radial frequency bins for L_nps
            rad_bins:    radial spatial bins for L_rad
            hist_bins:   histogram bins for L_hist
        """
        super().__init__()

        assert real_images, "Need at least one real image"

        self.B = B
        self.nps_bins = nps_bins
        self.rad_bins = rad_bins
        self.hist_bins = hist_bins
        self.patch_size = 8

        real_imgs_t = [torch.from_numpy(img.astype(np.float32)) for img in real_images]
        H, W = real_imgs_t[0].shape
        self.H, self.W = H, W
        nH, nW = H // self.patch_size, W // self.patch_size
        self.nH, self.nW = nH, nW

        # ------------------------------------------------------------------ #
        # 1. Gaussian blur kernel (sigma=3, fixed) for L_nps                 #
        # ------------------------------------------------------------------ #
        sigma_g = 3.0
        ks = int(6 * sigma_g + 1) | 1  # nearest odd number (~19)
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1 = torch.exp(-ax ** 2 / (2 * sigma_g ** 2))
        g1 = g1 / g1.sum()
        g2d = torch.outer(g1, g1)                            # [ks, ks]
        self.register_buffer('gauss_kernel', g2d.view(1, 1, ks, ks))
        self.gauss_pad = ks // 2

        # ------------------------------------------------------------------ #
        # 2. Hann window for L_nps                                           #
        # ------------------------------------------------------------------ #
        hann_h = torch.hann_window(H, periodic=False)
        hann_w = torch.hann_window(W, periodic=False)
        self.register_buffer('hann_win', torch.outer(hann_h, hann_w))  # [H, W]

        # ------------------------------------------------------------------ #
        # 3. NPS radial frequency bin map  [H, W//2+1]                       #
        # ------------------------------------------------------------------ #
        ky = torch.fft.fftfreq(H)        # [-0.5, 0.5)
        kx = torch.fft.rfftfreq(W)       # [0, 0.5]
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        K_rad = torch.sqrt(KY ** 2 + KX ** 2)
        k_max = float(K_rad.max()) * 1.001
        self._nps_k_max = k_max
        nps_edges = torch.linspace(0.0, k_max, nps_bins + 1)
        # bucketize with (nps_bins-1) internal boundaries → nps_bins buckets
        nps_bin_map = torch.bucketize(K_rad, nps_edges[1:-1]).clamp(0, nps_bins - 1)
        self.register_buffer('nps_bin_map', nps_bin_map)   # [H, W//2+1], long

        # ------------------------------------------------------------------ #
        # 4. Radial spatial bin indices for L_rad                            #
        # ------------------------------------------------------------------ #
        cy, cx = H / 2.0, W / 2.0
        yy = torch.arange(H, dtype=torch.float32) - cy
        xx = torch.arange(W, dtype=torch.float32) - cx
        YY, XX = torch.meshgrid(yy, xx, indexing='ij')
        R = torch.sqrt(YY ** 2 + XX ** 2)
        r_max = float(R.max()) * 1.001
        rad_edges = torch.linspace(0.0, r_max, rad_bins + 1)
        rad_bin_indices = []
        for b in range(rad_bins):
            mask_b = (R >= rad_edges[b]) & (R < rad_edges[b + 1])
            rad_bin_indices.append(mask_b.nonzero(as_tuple=False))  # [N, 2]
        # stored as plain list (not buffers) since they are just index tables
        self.rad_bin_indices = rad_bin_indices

        # ------------------------------------------------------------------ #
        # 5. Histogram centres for L_hist                                    #
        # ------------------------------------------------------------------ #
        self.register_buffer('hist_centers', torch.linspace(0.0, 1.0, hist_bins))
        self.hist_s = 1.0 / hist_bins   # bin width

        # ================================================================== #
        #                    Process real images                             #
        # ================================================================== #

        # ---- Patch-level gradient mask for L_vm flat-patch filter ----------- #
        # Only patches with low structural gradient are used, so that within-patch
        # variance reflects noise (Poisson+Gaussian) rather than image structure.
        ps = self.patch_size
        all_pg = []
        for img_t in real_imgs_t:
            img_np = img_t.numpy()
            gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
            gm_t = torch.from_numpy(np.sqrt(gx ** 2 + gy ** 2).astype(np.float32))
            crop_g = gm_t[: nH * ps, : nW * ps]
            gm_patches = crop_g.reshape(nH, ps, nW, ps).permute(0, 2, 1, 3)
            all_pg.append(gm_patches.mean(dim=(-1, -2)).flatten())   # [nH*nW]
        avg_pg = torch.stack(all_pg).mean(0)                          # [nH*nW]
        pg_thr = float(torch.quantile(avg_pg, 0.50))
        vm_flat_mask = avg_pg < pg_thr                                 # flat = low gradient

        # ---- L_vm: extract 8×8 patch means/vars (flat patches only) -------- #
        all_pm = []   # [N_real, nH*nW]
        all_pv = []
        for img_t in real_imgs_t:
            crop = img_t[: nH * ps, : nW * ps]
            patches = crop.reshape(nH, ps, nW, ps).permute(0, 2, 1, 3)  # [nH, nW, ps, ps]
            all_pm.append(patches.mean(dim=(-1, -2)).flatten())
            all_pv.append(patches.var(dim=(-1, -2), unbiased=False).flatten())
        all_pm = torch.stack(all_pm)   # [N_real, nH*nW]
        all_pv = torch.stack(all_pv)
        avg_pm_all = all_pm.mean(0)    # [nH*nW]
        avg_pv_all = all_pv.mean(0)

        # Filter to flat patches
        flat_linear_idx = vm_flat_mask.nonzero(as_tuple=False).flatten()  # [n_flat]
        flat_pm = avg_pm_all[vm_flat_mask]    # [n_flat] patch means
        flat_pv = avg_pv_all[vm_flat_mask]    # [n_flat] patch vars (noise-dominated)

        # Quantile bin edges from flat-patch means
        qs = torch.linspace(0.0, 1.0, B + 1)
        vm_edges = torch.quantile(flat_pm, qs)
        self.register_buffer('vm_bin_edges', vm_edges)  # [B+1]

        bin_asgn_flat = torch.bucketize(flat_pm, vm_edges[1:-1]).clamp(0, B - 1)

        vm_patch_indices = []   # list[B] of int64 tensors (original linear idx in nH*nW)
        vm_real_means = []
        vm_real_vars = []
        for b in range(B):
            in_b = (bin_asgn_flat == b).nonzero(as_tuple=False).flatten()
            if len(in_b) > 0:
                orig_idx = flat_linear_idx[in_b]   # [count_b] indices in [0, nH*nW)
                vm_patch_indices.append(orig_idx)
                vm_real_means.append(flat_pm[in_b].mean())
                vm_real_vars.append(flat_pv[in_b].mean())
            else:
                vm_patch_indices.append(torch.tensor([], dtype=torch.long))
                vm_real_means.append(torch.tensor(0.0))
                vm_real_vars.append(torch.tensor(0.0))

        self.vm_patch_indices = vm_patch_indices  # list, not buffer (irregular sizes)
        # [B, 2]: col-0 = mean_b, col-1 = var_b  (noise-dominated, flat regions only)
        self.register_buffer(
            'vm_real_curve',
            torch.stack([torch.stack(vm_real_means), torch.stack(vm_real_vars)], dim=1),
        )

        # Precompute log-normalized real variance curve for PSF-invariant L_vm.
        # Mean-subtracting log(variance) across bins cancels the PSF scale factor
        # (which is a multiplicative constant for all bins), leaving only the
        # SHAPE of the noise-mean relationship — sensitive to N and sigma_g.
        vm_real_log_vars = []
        vm_valid_bins = []
        for b in range(B):
            vr = float(vm_real_vars[b])
            if len(vm_patch_indices[b]) > 0 and vr > 0:
                vm_real_log_vars.append(math.log(vr))
                vm_valid_bins.append(b)
        self._vm_valid_bins = vm_valid_bins  # list attribute, not buffer

        mean_log_real = sum(vm_real_log_vars) / max(len(vm_real_log_vars), 1)
        vm_real_log_norm_t = torch.zeros(B)
        for i, b in enumerate(vm_valid_bins):
            vm_real_log_norm_t[b] = vm_real_log_vars[i] - mean_log_real
        self.register_buffer('vm_real_log_norm', vm_real_log_norm_t)  # [B]

        # ---- L_nps: flat mask (gradient mag < 30th pct of real images) --- #
        grad_mags = []
        for img_t in real_imgs_t:
            img_np = img_t.numpy()
            gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
            grad_mags.append(np.sqrt(gx ** 2 + gy ** 2))
        avg_gm = np.mean(grad_mags, axis=0)
        thr = float(np.percentile(avg_gm, 30))
        self.register_buffer('flat_mask', torch.from_numpy((avg_gm < thr).astype(np.float32)))

        # Accumulate P_real
        P_real_acc = torch.zeros(nps_bins)
        for img_t in real_imgs_t:
            P_real_acc += self._nps_single(img_t)
        self.register_buffer('P_real', (P_real_acc / len(real_imgs_t)).clamp(min=1e-10))

        # ---- L_rad: normalized radial mean profile from real images --------- #
        # Normalized by mean so the loss is insensitive to overall brightness
        # (contrast scale), sensitive only to the radial shape (vignetting).
        prof_acc = torch.zeros(rad_bins)
        for img_t in real_imgs_t:
            img_flat = img_t.flatten()
            for b, idx_2d in enumerate(self.rad_bin_indices):
                if len(idx_2d) > 0:
                    flat_idx = idx_2d[:, 0] * W + idx_2d[:, 1]
                    prof_acc[b] += img_flat[flat_idx].mean()
        prof_abs = prof_acc / len(real_imgs_t)                  # [rad_bins] absolute
        prof_norm = prof_abs / prof_abs.mean().clamp(min=1e-8)  # normalized shape
        self.register_buffer('prof_real', prof_norm)            # [rad_bins]

        # ---- L_hist: soft histogram from real images --------------------- #
        hist_acc = torch.zeros(hist_bins)
        for img_t in real_imgs_t:
            hist_acc += self._soft_hist_np(img_t.flatten())
        self.register_buffer('hist_real', hist_acc / len(real_imgs_t))  # [hist_bins]

    # ---------------------------------------------------------------------- #
    #  Private helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _nps_single(self, img: torch.Tensor) -> torch.Tensor:
        """Compute radially-averaged NPS for one (H,W) cpu tensor (precomp use)."""
        img4 = img.unsqueeze(0).unsqueeze(0)                                   # [1,1,H,W]
        blurred = F.conv2d(img4, self.gauss_kernel, padding=self.gauss_pad)
        blurred = blurred.squeeze(0).squeeze(0)                                # [H,W]
        res = img - blurred
        windowed = res * self.flat_mask * self.hann_win                        # [H,W]
        fft_out = torch.fft.rfft2(windowed)                                    # [H,W//2+1]
        power = fft_out.abs() ** 2

        P = torch.zeros(self.nps_bins)
        for b in range(self.nps_bins):
            m = self.nps_bin_map == b
            if m.any():
                P[b] = power[m].mean()
        return P

    def _soft_hist_np(self, x: torch.Tensor) -> torch.Tensor:
        """Soft histogram (cpu, no-grad) for a 1-D tensor x."""
        diff = x.unsqueeze(1) - self.hist_centers.unsqueeze(0)  # [N, hist_bins]
        return torch.exp(-diff ** 2 / (2 * self.hist_s ** 2)).mean(dim=0)     # [hist_bins]

    def _nps_bin_map_for(self, H: int, W: int, device) -> torch.Tensor:
        """Return nps_bin_map for the given (H, W); uses cached buffer if shapes match."""
        if H == self.H and W == self.W:
            return self.nps_bin_map.to(device)
        ky = torch.fft.fftfreq(H, device=device)
        kx = torch.fft.rfftfreq(W, device=device)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        K_rad = torch.sqrt(KY ** 2 + KX ** 2)
        k_max = float(K_rad.max()) * 1.001
        nps_edges = torch.linspace(0.0, k_max, self.nps_bins + 1, device=device)
        return torch.bucketize(K_rad, nps_edges[1:-1]).clamp(0, self.nps_bins - 1)

    def _rad_bins_for(self, H: int, W: int):
        """Return radial bin index list for the given (H, W); uses precomputed if shapes match."""
        if H == self.H and W == self.W:
            return self.rad_bin_indices, W
        cy, cx = H / 2.0, W / 2.0
        yy = torch.arange(H, dtype=torch.float32) - cy
        xx = torch.arange(W, dtype=torch.float32) - cx
        YY, XX = torch.meshgrid(yy, xx, indexing='ij')
        R = torch.sqrt(YY ** 2 + XX ** 2)
        r_max = float(R.max()) * 1.001
        rad_edges = torch.linspace(0.0, r_max, self.rad_bins + 1)
        indices = []
        for b in range(self.rad_bins):
            m = (R >= rad_edges[b]) & (R < rad_edges[b + 1])
            indices.append(m.nonzero(as_tuple=False))
        return indices, W

    # ---------------------------------------------------------------------- #
    #  Forward                                                                #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        syn: torch.Tensor,
        lambda_weights=(1.0, 1.0, 1.0, 1.0),
    ) -> tuple:
        """
        Args:
            syn: (H, W) or (B, H, W) float32 in [0, 1]
            lambda_weights: (w_vm, w_nps, w_rad, w_hist)
        Returns:
            (L_total, {'L_vm': ..., 'L_nps': ..., 'L_rad': ..., 'L_hist': ...})
        """
        if syn.dim() == 2:
            syn = self._maybe_transpose(syn)
            L_vm, L_nps, L_rad, L_hist = self._forward_single(syn)
        elif syn.dim() == 3:
            if syn.shape[1:] != (self.H, self.W):
                if (syn.shape[1], syn.shape[2]) == (self.W, self.H) and self.H != self.W:
                    syn = syn.permute(0, 2, 1)
            losses = [self._forward_single(s) for s in syn]
            L_vm   = torch.stack([l[0] for l in losses]).mean()
            L_nps  = torch.stack([l[1] for l in losses]).mean()
            L_rad  = torch.stack([l[2] for l in losses]).mean()
            L_hist = torch.stack([l[3] for l in losses]).mean()
        else:
            raise ValueError(f"syn must be 2-D or 3-D, got {syn.dim()}-D")

        lw = lambda_weights
        L_total = lw[0] * L_vm + lw[1] * L_nps + lw[2] * L_rad + lw[3] * L_hist
        parts = {'L_vm': L_vm, 'L_nps': L_nps, 'L_rad': L_rad, 'L_hist': L_hist}
        return L_total, parts

    def _maybe_transpose(self, syn: torch.Tensor) -> torch.Tensor:
        """Auto-transpose (W,H) → (H,W) when the input appears flipped."""
        if (
            syn.shape != (self.H, self.W)
            and syn.shape == (self.W, self.H)
            and self.H != self.W
        ):
            return syn.T   # view — gradient flows through
        return syn

    def _forward_single(self, syn: torch.Tensor) -> tuple:
        return (
            self._loss_vm(syn),
            self._loss_nps(syn),
            self._loss_rad(syn),
            self._loss_hist(syn),
        )

    # ---------------------------------------------------------------------- #
    #  Individual losses                                                      #
    # ---------------------------------------------------------------------- #

    def _loss_vm(self, syn: torch.Tensor) -> torch.Tensor:
        """L_vm — Log-normalized variance-mean curve (PSF-invariant, noise-sensitive).

        For each flat-region brightness bin b, computes log(mean patch variance_b).
        The mean is subtracted across bins before MSE.  PSF blur multiplies ALL
        variances by the same scale factor K_PSF, which appears as a constant in
        log space and is exactly cancelled by the mean subtraction — making L_vm
        sensitive to the SHAPE of the noise-mean relationship (N, sigma_g) but
        not to the blur width (fwhm).
        """
        H_s, W_s = syn.shape
        ps = self.patch_size
        nH_use = min(self.nH, H_s // ps)
        nW_use = min(self.nW, W_s // ps)
        N = nH_use * nW_use

        syn_crop = syn[: nH_use * ps, : nW_use * ps]
        patches = syn_crop.reshape(nH_use, ps, nW_use, ps).permute(0, 2, 1, 3)
        patches_flat = patches.reshape(N, ps * ps)  # [N, 64]
        patch_vars = patches_flat.var(dim=1, unbiased=False)  # [N] — differentiable

        syn_log_var_parts = []
        valid_b_used = []
        for b in self._vm_valid_bins:
            idx = self.vm_patch_indices[b]
            if len(idx) == 0:
                continue
            valid = idx[idx < N].to(syn.device)
            if len(valid) == 0:
                continue
            mean_var_b = patch_vars[valid].mean().clamp(min=1e-12)
            syn_log_var_parts.append(torch.log(mean_var_b))
            valid_b_used.append(b)

        if not syn_log_var_parts:
            return torch.zeros(1, device=syn.device, dtype=syn.dtype).squeeze()

        syn_log_vars = torch.stack(syn_log_var_parts)             # [n_valid]
        syn_log_norm = syn_log_vars - syn_log_vars.mean()         # mean-subtracted

        real_log_norm = self.vm_real_log_norm[valid_b_used].to(syn.device)  # [n_valid]
        return ((syn_log_norm - real_log_norm) ** 2).mean()

    def _loss_nps(self, syn: torch.Tensor) -> torch.Tensor:
        """L_nps — Noise Power Spectrum loss."""
        H_s, W_s = syn.shape
        dev = syn.device

        # Gaussian residual
        syn4 = syn.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        blurred = F.conv2d(
            syn4, self.gauss_kernel.to(dev), padding=self.gauss_pad
        ).squeeze(0).squeeze(0)                # [H,W]
        res = syn - blurred

        # Flat mask and Hann window (resize if shapes differ)
        flat_mask = self.flat_mask.to(dev)
        hann_win  = self.hann_win.to(dev)
        if flat_mask.shape != syn.shape:
            flat_mask = F.interpolate(
                flat_mask.unsqueeze(0).unsqueeze(0),
                size=syn.shape, mode='nearest',
            ).squeeze(0).squeeze(0)
            hann_win = F.interpolate(
                hann_win.unsqueeze(0).unsqueeze(0),
                size=syn.shape, mode='bilinear', align_corners=False,
            ).squeeze(0).squeeze(0)

        windowed = res * flat_mask * hann_win          # [H,W]
        fft_out  = torch.fft.rfft2(windowed)           # [H,W//2+1] complex
        power    = fft_out.abs() ** 2                   # [H,W//2+1]

        nps_bm = self._nps_bin_map_for(H_s, W_s, dev)  # [H,W//2+1]

        P_syn_parts = []
        for b in range(self.nps_bins):
            m = nps_bm == b
            if m.any():
                P_syn_parts.append(power[m].mean())
            else:
                P_syn_parts.append(
                    torch.zeros(1, device=dev, dtype=syn.dtype).squeeze()
                )

        P_syn  = torch.stack(P_syn_parts).clamp(min=1e-10)   # [nps_bins]
        P_real = self.P_real.to(dev).clamp(min=1e-10)
        return ((torch.log(P_syn) - torch.log(P_real)) ** 2).mean()

    def _loss_rad(self, syn: torch.Tensor) -> torch.Tensor:
        """L_rad — Normalized radial mean profile loss (contrast-invariant).

        Both synthetic and real profiles are normalized by their own mean before
        computing MSE, so the loss captures the *shape* of the radial falloff
        (vignetting) independently of overall brightness / contrast scale.
        """
        H_s, W_s = syn.shape
        dev = syn.device
        syn_flat = syn.flatten()   # [H*W]

        rad_indices, W_use = self._rad_bins_for(H_s, W_s)

        prof_parts = []
        for idx_2d in rad_indices:
            if len(idx_2d) > 0:
                idx_2d_d = idx_2d.to(dev)
                flat_idx = idx_2d_d[:, 0] * W_use + idx_2d_d[:, 1]
                prof_parts.append(syn_flat[flat_idx].mean())
            else:
                prof_parts.append(
                    torch.zeros(1, device=dev, dtype=syn.dtype).squeeze()
                )

        prof_syn = torch.stack(prof_parts)                             # [rad_bins] absolute
        prof_syn_norm = prof_syn / prof_syn.mean().clamp(min=1e-8)     # normalized shape
        return ((prof_syn_norm - self.prof_real.to(dev)) ** 2).mean()

    def _loss_hist(self, syn: torch.Tensor) -> torch.Tensor:
        """L_hist — Soft histogram loss."""
        x = syn.flatten()                                          # [N]
        centers = self.hist_centers.to(syn.device)
        diff = x.unsqueeze(1) - centers.unsqueeze(0)              # [N, hist_bins]
        h_syn = torch.exp(-diff ** 2 / (2 * self.hist_s ** 2)).mean(dim=0)  # [hist_bins]
        return ((h_syn - self.hist_real.to(syn.device)) ** 2).mean()
