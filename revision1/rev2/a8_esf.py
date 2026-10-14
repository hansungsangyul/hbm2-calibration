#!/usr/bin/env python3
"""
지시 23: ESF-based PSF FWHM Measurement (HANDOFF v6.1)
=======================================================
독립 Sobel-에지 기반 FWHM 측정 — fwhm=1.42 (v5) vs 2.50 (manual) 중재.

Pipeline:
  1. Self-validation (seeds 960000-960004): apply PSF fwhm=2.5, measure → check ∈ [2.25, 2.75]
  2. Real measurement (HBM_dirty2 img00-img04): Sobel edge → ESF fit → FWHM

ESF fit: A * erf((x - x0) / (√2 · σ)) + B
FWHM    = 2.3548 · σ

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python phase4/run_esf_psf.py

Output: phase4/esf_psf_measurement.json
        phase4/esf_plots/*.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.ndimage import label as nd_label, map_coordinates

# ── Paths ──────────────────────────────────────────────────────────────────────
DIFFPIPE_DIR = Path(__file__).parent.parent
PHASE4_DIR   = DIFFPIPE_DIR / 'phase4'
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
CLEAN_DIR    = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
PLOT_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a8/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
import torch
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────
PROFILE_HALF  = 8     # pixels each side of edge pixel for ESF profile (spec §3.2: ±8px)
MIN_VALID_EDGES = 20  # minimum valid edge fits per image
R2_THRESH     = 0.95  # minimum R² for a valid ESF fit (fallback 0.90 per spec §3.2.4)
COMP_LEN_MIN  = 20    # minimum connected-component pixel count
COMP_RMS_MAX  = 1.0   # max RMS deviation from straight line (pixels)
FWHM_SIGMA    = 2.3548 # conversion constant: FWHM = 2.3548 * σ


# ── Gaussian PSF kernel (matching pipeline.py logic exactly) ──────────────────
def make_psf_kernel_np(fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.355
    k = int(2 * round(3 * sigma) + 1)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    coords = np.arange(k, dtype=np.float64) - k // 2
    g1d = np.exp(-coords**2 / (2.0 * sigma**2))
    g1d /= g1d.sum()
    kernel = np.outer(g1d, g1d)
    return kernel.astype(np.float32)


# ── ESF model ─────────────────────────────────────────────────────────────────
def esf_model(x, A, x0, sigma, B):
    return A * erf((x - x0) / (np.sqrt(2.0) * sigma)) + B


def fit_esf_profile(profile: np.ndarray) -> tuple[float | None, float]:
    """Fit ESF to a 1-D intensity profile.
    Returns (sigma, R²) or (None, 0.0) on failure.
    """
    n = len(profile)
    xs = np.arange(n, dtype=np.float64) - n // 2

    # Coarse initial guess from profile slope
    A0   = (profile[-1] - profile[0]) / 2.0
    x0_0 = 0.0
    # Width guess from 10-90% rise
    lo, hi = profile.min(), profile.max()
    if abs(hi - lo) < 1e-6:
        return None, 0.0
    idx_lo = np.argmin(np.abs(profile - (lo + 0.1 * (hi - lo))))
    idx_hi = np.argmin(np.abs(profile - (lo + 0.9 * (hi - lo))))
    sigma0 = max(0.5, abs(int(idx_hi) - int(idx_lo)) / 2.5)
    B0 = (profile[0] + profile[-1]) / 2.0

    try:
        popt, _ = curve_fit(
            esf_model, xs, profile.astype(np.float64),
            p0=[A0, x0_0, sigma0, B0],
            bounds=([-2, -n/2, 0.1, -1], [2, n/2, n/2, 2]),
            maxfev=2000,
        )
        fitted = esf_model(xs, *popt)
        ss_res = np.sum((profile - fitted)**2)
        ss_tot = np.sum((profile - profile.mean())**2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        sigma = abs(popt[2])
        if sigma < 0.1 or sigma > PROFILE_HALF:
            return None, 0.0
        return float(sigma), float(r2)
    except (RuntimeError, ValueError):
        return None, 0.0


# ── Edge extraction + ESF fitting for one image ───────────────────────────────
def _collect_edge_sigmas(img, Gx, Gy, mag, labeled, n_comp, r2_thresh):
    """Extract valid σ estimates from all accepted edge components."""
    H, W = img.shape
    valid_sigmas = []
    for comp_id in range(1, n_comp + 1):
        ys, xs_px = np.where(labeled == comp_id)
        if len(ys) < COMP_LEN_MIN:
            continue
        if len(ys) >= 3:
            poly = np.polyfit(xs_px.astype(float), ys.astype(float), 1)
            y_pred = np.polyval(poly, xs_px.astype(float))
            rms = float(np.sqrt(np.mean((ys - y_pred)**2)))
        else:
            rms = 0.0
        if rms > COMP_RMS_MAX:
            continue
        for r, c in zip(ys, xs_px):
            gx_pix = Gx[r, c]
            gy_pix = Gy[r, c]
            gnorm = np.hypot(gx_pix, gy_pix) + 1e-12
            nx = gx_pix / gnorm
            ny = gy_pix / gnorm
            # Bilinear interpolation at exact subpixel positions (spec §3.2 "bilinear 보간")
            d_coords = np.arange(-PROFILE_HALF, PROFILE_HALF + 1, dtype=np.float64)
            row_coords = r + d_coords * ny
            col_coords = c + d_coords * nx
            if (row_coords.min() < 0 or row_coords.max() >= H or
                    col_coords.min() < 0 or col_coords.max() >= W):
                continue
            profile = map_coordinates(img.astype(np.float64), [row_coords, col_coords],
                                      order=1, mode='nearest')
            if len(profile) < PROFILE_HALF:
                continue
            sigma, r2 = fit_esf_profile(np.array(profile, dtype=np.float64))
            if sigma is not None and r2 >= r2_thresh:
                valid_sigmas.append(sigma)
    return valid_sigmas


def measure_image_fwhm(
    img: np.ndarray,
    label: str,
    plot: bool = False,
) -> dict:
    """
    img: float32 [0,1] grayscale (H,W)
    Returns dict with per-image FWHM stats.
    R² fallback: if n_valid<20 with R²≥0.95, retry with R²≥0.90 per spec §3.2.4.
    """
    H, W = img.shape

    # Sobel gradients
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    Gx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(Gx, Gy)

    # Top-5% threshold
    thresh = np.percentile(mag, 95.0)
    edge_bin = (mag >= thresh).astype(np.uint8)

    # Connected components
    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n_comp = nd_label(edge_bin, structure=struct)

    # Primary pass: R² ≥ 0.95
    r2_used = R2_THRESH
    valid_sigmas = _collect_edge_sigmas(img, Gx, Gy, mag, labeled, n_comp, r2_used)

    # Fallback: R² ≥ 0.90 if insufficient (spec §3.2.4)
    r2_relaxed = False
    if len(valid_sigmas) < MIN_VALID_EDGES:
        valid_sigmas = _collect_edge_sigmas(img, Gx, Gy, mag, labeled, n_comp, 0.90)
        r2_used = 0.90
        r2_relaxed = True

    n_valid = len(valid_sigmas)
    if n_valid == 0:
        median_fwhm = float('nan')
        mean_fwhm = float('nan')
        std_fwhm = float('nan')
    else:
        fwhms = [s * FWHM_SIGMA for s in valid_sigmas]
        median_fwhm = float(np.median(fwhms))
        mean_fwhm   = float(np.mean(fwhms))
        std_fwhm    = float(np.std(fwhms))

    sufficient = n_valid >= MIN_VALID_EDGES and not np.isnan(median_fwhm)
    result = {
        "label": label,
        "n_valid_edge_fits": n_valid,
        "n_components_checked": n_comp,
        "median_fwhm": median_fwhm,
        "mean_fwhm": mean_fwhm,
        "std_fwhm": std_fwhm,
        "sufficient": sufficient,
        "r2_threshold_used": r2_used,
        "r2_relaxed": r2_relaxed,
    }

    relax_note = " [R²→0.90]" if r2_relaxed else ""
    print(f"  [{label}] n_valid={n_valid}  median_FWHM={median_fwhm:.3f}  "
          f"mean={mean_fwhm:.3f}  std={std_fwhm:.3f}  "
          f"{'OK' if sufficient else 'INSUFFICIENT'}{relax_note}")

    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'{label}: image')
            axes[1].imshow(mag, cmap='hot')
            axes[1].set_title(f'{label}: Sobel mag (thresh={thresh:.1f})')
            plt.suptitle(f'{label}  median_FWHM={median_fwhm:.3f}  n_valid={n_valid}')
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f'{label}_edge.png', dpi=120)
            plt.close(fig)
        except ImportError:
            pass  # matplotlib optional

    return result


# ── Self-validation ────────────────────────────────────────────────────────────
print("=" * 65)
print("Phase 1: Self-validation (fwhm=2.5 synthetic blur, seeds 960000-960004)")
SELFVAL_FWHM = 2.5
SELFVAL_SEEDS = list(range(960000, 960005))
kernel = make_psf_kernel_np(SELFVAL_FWHM)
print(f"  PSF kernel size: {kernel.shape[0]}×{kernel.shape[1]}")
print(f"  σ = {SELFVAL_FWHM / 2.355:.4f} px  target FWHM = {SELFVAL_FWHM}")

clean_imgs = sorted(CLEAN_DIR.glob('*.png'))[:5]
if len(clean_imgs) < 5:
    print(f"  [WARN] Only {len(clean_imgs)} clean images found; need 5")

selfval_results = []
for i, (img_path, seed) in enumerate(zip(clean_imgs, SELFVAL_SEEDS)):
    rng = np.random.default_rng(seed)
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # Apply only Gaussian blur (replicating pipeline._stage4_psf_blur)
    pad = kernel.shape[0] // 2
    img_padded = cv2.copyMakeBorder(img_raw, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    img_blurred = cv2.filter2D(img_padded, -1, kernel)[pad:-pad, pad:-pad]
    res = measure_image_fwhm(img_blurred, label=f'selfval_{i:02d}', plot=True)
    selfval_results.append(res)

valid_sv = [r for r in selfval_results if r['sufficient']]
sv_fwhms = [r['median_fwhm'] for r in valid_sv]
sv_median = float(np.median(sv_fwhms)) if sv_fwhms else float('nan')
sv_pass = 2.25 <= sv_median <= 2.75

print(f"\nSelf-validation summary:")
print(f"  valid images: {len(valid_sv)}/{len(selfval_results)}")
print(f"  overall median FWHM: {sv_median:.3f}")
print(f"  target: [2.25, 2.75]  → {'PASS ✅' if sv_pass else 'FAIL ✗'}")

if not sv_pass:
    print("\n[STOP] §3.1 프로토콜: 자기 검증 실패 시 실측 진행 금지.")
    print("  sv_median={:.3f} ∉ [2.25, 2.75]. 원인: 정수 샘플링 편향 → bilinear 보정 시도중.".format(sv_median))
    _result_stub = {
        "_meta": {"attempt": 3, "script": "phase4/run_esf_psf.py"},
        "self_validation": {"overall_median_fwhm": sv_median, "pass": False},
        "real_measurement": None,
        "stopped_at": "self_validation_fail",
    }
    import json as _json
    (Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a8') / 'a8_esf_measurement.json').write_text(_json.dumps(_result_stub, indent=2))
    raise SystemExit(1)


# ── Real measurement ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Phase 2: Real measurement (HBM_dirty2 img_0001 to img_0005)")
real_imgs = sorted(REAL_DIR.glob('*.png'))[50:250]  # revision1 A8: reserved block 0051-0250
if len(real_imgs) < 5:
    print(f"  [WARN] Only {len(real_imgs)} dirty2 images found")

real_results = []
for i, img_path in enumerate(real_imgs):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    res = measure_image_fwhm(img, label=f'dirty2_img{i:02d}', plot=True)
    real_results.append(res)

valid_real = [r for r in real_results if r['sufficient']]
real_fwhms = [r['median_fwhm'] for r in valid_real]
real_median = float(np.median(real_fwhms)) if real_fwhms else float('nan')

print(f"\nReal measurement summary:")
print(f"  valid images: {len(valid_real)}/{len(real_results)}")
print(f"  individual medians: {[f'{f:.3f}' for f in real_fwhms]}")
print(f"  overall median FWHM: {real_median:.3f}")

# ── §6 판정 zone ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("§6 판정표 입력:")
if 1.1 <= real_median <= 1.7:
    zone = "Row3"
    zone_msg = f"ESF={real_median:.3f} ∈ [1.1,1.7] → Row 3 후보 (v5 fwhm=1.42 지지)"
elif 2.2 <= real_median <= 2.8:
    zone = "Row4"
    zone_msg = f"ESF={real_median:.3f} ∈ [2.2,2.8] → Row 4 후보 (manual fwhm=2.5 지지)"
else:
    zone = "Row5"
    zone_msg = f"ESF={real_median:.3f}: [1.1,1.7]·[2.2,2.8] 외부 → Row 5 (PENDING_CONSULTATION)"
print(f"  {zone_msg}")

# ── Save ───────────────────────────────────────────────────────────────────────
out = {
    "_meta": {
        "script": "phase4/run_esf_psf.py",
        "handoff": "HANDOFF_v6.1 지시 23",
        "attempt": 3,
        "method": "Sobel top-5% + connected_components(len≥20, RMS<1px) + ESF erf-fit (R²>0.95, fallback 0.90) + bilinear interpolation",
        "fwhm_formula": "FWHM = 2.3548 * sigma_esf",
        "profile_half": PROFILE_HALF,
        "change_from_attempt2": "Added bilinear interpolation (scipy.ndimage.map_coordinates order=1) for exact subpixel sampling along edge normal (fixes integer-rounding duplicate-sample bias that caused -16% underestimation)",
    },
    "self_validation": {
        "applied_fwhm": SELFVAL_FWHM,
        "target_range": [2.25, 2.75],
        "n_seeds": len(SELFVAL_SEEDS),
        "seeds": SELFVAL_SEEDS,
        "results": selfval_results,
        "overall_median_fwhm": sv_median,
        "pass": bool(sv_pass),
    },
    "real_measurement": {
        "source": str(REAL_DIR),
        "n_images": len(real_results),
        "results": real_results,
        "valid_count": len(valid_real),
        "per_image_medians": real_fwhms,
        "overall_median_fwhm": real_median,
    },
    "judgment_zone": {
        "zone": zone,
        "message": zone_msg,
        "esf_median": real_median,
    },
    "self_validation_pass": bool(sv_pass),
}

out_path = Path('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2/a8') / 'a8_esf_measurement.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Plots: {PLOT_DIR}/")
print("\n지시 23 ESF PSF measurement complete.")
print(f"  Self-val: {'PASS' if sv_pass else 'FAIL'}  sv_median={sv_median:.3f}")
print(f"  Real median FWHM: {real_median:.3f}  →  {zone}: {zone_msg}")
