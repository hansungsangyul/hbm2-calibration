# A12 (2026-10-17): Python port of the reference pipeline R = only_add_noise.m, level 2, local function add_realistic_noise_aggressive
# (MATLAB; HBM_paper/matlab/only_add_noise.m, saved 2025-10-09 12:53), plus the per-image parameter draws of that script, and stage-wise
# composition with model A's torch stages (DegradationPipeline). Pre-registered in A12_FACTORIAL_PREREG.md.
import numpy as np, cv2, io
from PIL import Image
from scipy.ndimage import convolve

MEANS = dict(g=0.04, d=0.015, L=6.0, phi=45.0, s=0.6, m=0.6, il=0.1, ih=0.9, ol=0.2, oh=0.8, q=60.0, c=1.04)

def draw_params(n, random=True):
    """Per-image parameters of only_add_noise.m level 2 (normrnd draws with the script's clipping), seeded by image number."""
    if not random:
        P = dict(MEANS); P['L'] = 6; P['q'] = 60; return P
    rs = np.random.RandomState(800000 + int(n))
    nr = lambda mu, sd: float(rs.normal(mu, sd))
    P = {}
    P['g'] = max(0.0, nr(0.04, 0.012))
    P['d'] = max(0.0, min(0.05, nr(0.015, 0.005)))
    P['L'] = int(max(1, int(np.floor(nr(6, 2) + 0.5))))          # MATLAB round (half away from zero; draws are positive here)
    P['phi'] = nr(45, 30)
    P['s'] = max(0.0, min(1.0, nr(0.6, 0.2)))
    P['m'] = max(0.3, min(1.0, nr(0.6, 0.1)))
    P['il'] = max(0.0, min(0.3, nr(0.1, 0.03)))
    P['ih'] = max(0.7, min(1.0, nr(0.9, 0.03)))
    P['ol'] = max(0.0, min(0.4, nr(0.2, 0.05)))
    P['oh'] = max(0.6, min(1.0, nr(0.8, 0.05)))
    P['q'] = int(max(40, min(95, int(np.floor(nr(60, 10) + 0.5)))))
    P['c'] = max(0.98, min(1.08, nr(1.04, 0.015)))
    return P

def noise_rng(n):
    """Noise draws for the R-port stages (independent of the parameter draws), seeded by image number."""
    return np.random.RandomState(810000 + int(n))

def fspecial_motion(length, theta):
    """MATLAB fspecial('motion', len, theta): anti-aliased line kernel (port of the MATLAB algorithm)."""
    length = max(1.0, float(length)); half = (length - 1.0) / 2.0
    phi = np.mod(theta, 180.0) / 180.0 * np.pi
    cosphi, sinphi = np.cos(phi), np.sin(phi)
    xsign = np.sign(cosphi) if cosphi != 0 else 1.0
    linewdt = 1.0; EPS = np.finfo(float).eps
    sx = np.fix(half * cosphi + linewdt * xsign - length * EPS)
    sy = np.fix(half * sinphi + linewdt - length * EPS)
    xs = np.arange(0, sx + xsign * 0.5, xsign) if xsign != 0 else np.array([0.0])   # 0:xsign:sx
    ys = np.arange(0, sy + 0.5, 1.0)                                                # 0:sy
    x, y = np.meshgrid(xs, ys)
    dist2line = y * cosphi - x * sinphi
    rad = np.sqrt(x ** 2 + y ** 2)
    lastpix = (rad >= half) & (np.abs(dist2line) <= linewdt)
    x2lastpix = half - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi) if cosphi != 0 else half - np.abs(y[lastpix])
    d = dist2line.copy()
    d[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
    d = linewdt + EPS - np.abs(d)
    d[d < 0] = 0.0
    h = np.rot90(d, 2)
    H = np.zeros((2 * d.shape[0] - 1, 2 * d.shape[1] - 1))
    H[:d.shape[0], :d.shape[1]] = h
    H[d.shape[0] - 1:, d.shape[1] - 1:] = d
    H = H / (H.sum() + EPS * length * length)
    if cosphi > 0: H = np.flipud(H)
    return H

# ---- R stages (float64 in [0,1]) --------------------------------------------------------------------------
def r_noise(x, P, rng):
    y = x + np.sqrt(1.5 * P['g']) * rng.standard_normal(x.shape)        # imnoise gaussian var 1.5g (clips)
    y = np.clip(y, 0, 1)
    u = rng.random_sample(x.shape); d = P['d']                           # imnoise salt & pepper: pepper u<d/2, salt d/2<=u<d
    y = np.where(u < d / 2, 0.0, np.where(u < d, 1.0, y))
    y = y + rng.standard_normal(x.shape) * P['g'] * 0.6                   # extra scanning-instability noise
    return np.clip(y, 0, 1)

def r_blur(x, P):
    k = fspecial_motion(P['L'], P['phi'])
    return convolve(x, k, mode='wrap')                                    # imfilter(..., 'conv', 'circular')

def r_vignette(x, P):
    rows, cols = x.shape
    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=float), np.arange(1, rows + 1, dtype=float))
    cx, cy = cols / 2.0, rows / 2.0
    v = 1 - P['s'] * ((X - cx) ** 2 + (Y - cy) ** 2) / (cx ** 2 + cy ** 2)
    return x * np.maximum(v, P['m'])

def r_imadjust(x, P):
    r = np.clip((x - P['il']) / (P['ih'] - P['il']), 0, 1)
    return r * (P['oh'] - P['ol']) + P['ol']

def to_uint8(x):   # MATLAB im2uint8: round(x*255) with clipping
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)

def r_jpeg(x, P):
    ok, enc = cv2.imencode('.jpg', to_uint8(x), [int(cv2.IMWRITE_JPEG_QUALITY), int(P['q'])])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)                        # uint8

def r_resample_u8(x8, P):
    """imresize(uint8, round(size*c)) then imresize back (bicubic, anti-aliased on shrink): PIL BICUBIC (a=-0.5, anti-aliased)."""
    h, w = x8.shape; h2, w2 = int(np.round(h * P['c'])), int(np.round(w * P['c']))
    im = Image.fromarray(x8).resize((w2, h2), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)

def r_resample_f(x, P):
    h, w = x.shape; h2, w2 = int(np.round(h * P['c'])), int(np.round(w * P['c']))
    im = Image.fromarray(x.astype(np.float32), mode='F').resize((w2, h2), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    return np.clip(np.asarray(im, dtype=np.float64), 0, 1)

def rport_full(x, P, rng):
    """Complete R-port: clean float [0,1] -> uint8-quantized float [0,1] (PNG storage), script order."""
    y = r_noise(x, P, rng); y = r_blur(y, P); y = r_vignette(y, P); y = r_imadjust(y, P)
    y8 = r_jpeg(y, P)
    if P['c'] != 1.0: y8 = r_resample_u8(y8, P)
    return y8.astype(np.float64) / 255.0

# ---- stage-wise composition with model A ------------------------------------------------------------------
import torch
def a_eps(n, H, W):
    torch.manual_seed(900000 + int(n))      # same seeds as a8_paired_gen.py
    return {'eps1': torch.randn(H, W), 'eps2': torch.randn(H, W), 'eps3': torch.rand(H, W), 'eps_post': torch.randn(H, W)}

def compose(x, n, pipe, prenorm_ab, spec, P, rng):
    """spec: dict(pre=bool, noise='A'|'R', blur='A'|'R', vig='A'|'R', post='A'|None, con='A'|'R', ring='A'|None, jpeg=bool, res=bool, quant=bool).
    x: clean float64 [0,1] (H,W) full size. Returns float64 [0,1]."""
    H, W = x.shape; eps = a_eps(n, H, W)
    y = x.copy()
    if spec['pre']: y = prenorm_ab[0] * y + prenorm_ab[1]
    def A(fn, *args):
        with torch.no_grad(): return fn(torch.from_numpy(y.astype(np.float32)), *args).numpy().astype(np.float64)
    # noise
    if spec['noise'] == 'A':
        y = A(pipe._stage1_poisson, eps['eps1']); y = A(pipe._stage2_gaussian, eps['eps2']); y = A(pipe._stage3_salt_pepper, eps['eps3'])
    else: y = r_noise(y, P, rng)
    # blur
    y = A(pipe._stage4_psf_blur) if spec['blur'] == 'A' else r_blur(y, P)
    # vignette
    y = A(pipe._stage5_vignetting) if spec['vig'] == 'A' else r_vignette(y, P)
    # post-noise (A only)
    if spec.get('post') == 'A': y = A(pipe._stage55_post_noise, eps['eps_post'])
    # contrast
    y = A(pipe._stage6_contrast) if spec['con'] == 'A' else r_imadjust(y, P)
    # ring (A only)
    if spec.get('ring') == 'A': y = A(pipe._stage7_ring)
    # jpeg / resample / quantisation
    if spec['jpeg']:
        y8 = r_jpeg(y, P)
        if spec['res'] and P['c'] != 1.0: y8 = r_resample_u8(y8, P)
        return y8.astype(np.float64) / 255.0
    if spec['res'] and P['c'] != 1.0:
        if spec['quant']: return r_resample_u8(to_uint8(y), P).astype(np.float64) / 255.0
        return r_resample_f(y, P)
    if spec['quant']: return to_uint8(y).astype(np.float64) / 255.0
    return np.clip(y, 0, 1)

A_SPEC = dict(pre=None, noise='A', blur='A', vig='A', post='A', con='A', ring='A', jpeg=False, res=False, quant=False)
R_SPEC = dict(pre=False, noise='R', blur='R', vig='R', post=None, con='R', ring=None, jpeg=True, res=True, quant=True)

def variant_specs(prenorm, has_post):
    """Ordered dict of variant name -> spec for one configuration. prenorm: trained with prenorm before degradation; has_post: sigma_post > 0."""
    V = {}
    V['V_P'] = dict(R_SPEC); V['V_F'] = dict(R_SPEC)           # V_F: nominal central parameters (handled by caller)
    base = dict(A_SPEC); base['pre'] = prenorm; base['post'] = 'A' if has_post else None
    for k in ['noise', 'blur', 'vig', 'con']:
        s = dict(base); s[k] = 'R'; V[f'I_{k}'] = s
    if has_post:
        s = dict(base); s['post'] = None; V['I_post'] = s
    s = dict(base); s['ring'] = None; V['I_ring'] = s
    s = dict(base); s['jpeg'] = True; V['I_jpeg'] = s
    s = dict(base); s['res'] = True; V['I_res'] = s
    if prenorm:
        s = dict(base); s['pre'] = False; V['I_pre'] = s
    for k in ['noise', 'blur', 'vig', 'con']:
        s = dict(R_SPEC); s[k] = 'A'; V[f'L_{k}'] = s
    if has_post:
        s = dict(R_SPEC); s['post'] = 'A'; V['L_post'] = s
    s = dict(R_SPEC); s['ring'] = 'A'; V['L_ring'] = s
    s = dict(R_SPEC); s['jpeg'] = False; V['L_jpeg'] = s
    s = dict(R_SPEC); s['res'] = False; V['L_res'] = s
    if prenorm:
        s = dict(R_SPEC); s['pre'] = True; V['L_pre'] = s
    return V

def amendment1_specs(prenorm, has_post):
    """A12 amendment 1 (registered 2026-10-17 after the single-stage results were inspected): two-factor substitutions in both directions."""
    V = {}
    s = dict(R_SPEC); s['noise'] = 'A'; s['blur'] = 'A'; V['L_noise_blur'] = s
    base = dict(A_SPEC); base['pre'] = prenorm; base['post'] = 'A' if has_post else None
    s = dict(base); s['noise'] = 'R'; s['blur'] = 'R'; V['I_noise_blur'] = s
    return V

