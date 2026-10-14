"""
HBM Canonical Dataset Generator — Clean Rebuild (Stage 1 engine)
================================================================
단일 정본 데이터셋 생성기. 모든 degradation 파라미터는 degradation_config.yaml
하나에서만 읽는다 (하드코딩 금지). 8단계 누적 열화 + 픽셀단위 GT 마스크 저장.

★ 적용 도메인: 열화는 OUTPUT 해상도(검출기 도메인, 314x520)에서 적용.
  - clean 그리드(canvas 3140x5200) → INTER_AREA 다운스케일 → 검출기 영상
  - 그 위에 8단계 열화 적용 (물리값 그대로, 육안 가시성 보장)
  - 마스크는 clean canvas에서 조립 후 INTER_NEAREST 다운스케일 (열화 무관)

7단계: 1 Poisson · 2 Gaussian · 3 S&P · 4 PSF · 5 Vignette · 6 Contrast · 7 Ring artifact
       (Beam-hardening은 재구성 도메인 현상이라 제외 — Stage 1b 판정)
그룹(누적): G1=0, G2=3, G3=4, G4=7
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../python (generateHBMDefects)

import cv2
import numpy as np
import yaml
import json
import shutil
import time
from pathlib import Path

from generateHBMDefects import (
    Config, load_image, analyze_structure,
    generate_crack, generate_void, generate_open,
)

DEFECT_TYPES = ['normal', 'crack', 'void', 'open']
LABEL_IDS = {'normal': 0, 'crack': 1, 'void': 2, 'open': 3}   # void == Hole


# ============================ Config 로딩 ============================

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _draw(spec, rng):
    """{mean,std,min,max} 분포에서 1개 샘플 (clamp)."""
    v = rng.normal(spec['mean'], spec['std'])
    return float(np.clip(v, spec['min'], spec['max']))


def sample_params(cfg, rng):
    """이미지별 열화 파라미터 1세트 샘플."""
    m = cfg['margin']
    g = cfg['gaussian']
    sigma = float(np.clip(g['sigma_bg'] * m + rng.normal(0, g['sigma_jitter']),
                          g['sigma_min'], g['sigma_max']))
    return dict(
        poisson_N=_draw(cfg['poisson']['N'], rng),
        gaussian_sigma=sigma,
        sp_density=_draw(cfg['salt_pepper']['density'], rng),
        psf_fwhm=_draw(cfg['psf']['fwhm_px'], rng),
        vignette_alpha=_draw(cfg['vignette']['alpha'], rng),
        vignette_min=cfg['vignette']['min_brightness'],
        contrast_in_low=_draw(cfg['contrast']['in_low'], rng),
        contrast_in_high=_draw(cfg['contrast']['in_high'], rng),
        contrast_out_low=_draw(cfg['contrast']['out_low'], rng),
        contrast_out_high=_draw(cfg['contrast']['out_high'], rng),
        ring_strength=_draw(cfg['ring']['strength'], rng),
        ring_n=_draw(cfg['ring']['n_rings'], rng),
    )


def mean_params(cfg):
    """대표 강도(분포 평균) 파라미터 — sample_check 시각화용."""
    m = cfg['margin']
    return dict(
        poisson_N=cfg['poisson']['N']['mean'],
        gaussian_sigma=cfg['gaussian']['sigma_bg'] * m,
        sp_density=cfg['salt_pepper']['density']['mean'],
        psf_fwhm=cfg['psf']['fwhm_px']['mean'],
        vignette_alpha=cfg['vignette']['alpha']['mean'],
        vignette_min=cfg['vignette']['min_brightness'],
        contrast_in_low=cfg['contrast']['in_low']['mean'],
        contrast_in_high=cfg['contrast']['in_high']['mean'],
        contrast_out_low=cfg['contrast']['out_low']['mean'],
        contrast_out_high=cfg['contrast']['out_high']['mean'],
        ring_strength=cfg['ring']['strength']['mean'],
        ring_n=cfg['ring']['n_rings']['mean'],
    )


# ===================== 개별 열화 단계 (출력 해상도) =====================
# 노이즈 계열은 결정성을 위해 rng(RandomState)를 받는다.

def stage1_poisson(img, N, rng):
    """광자 통계. 출력해상도 적용이라 증폭(amp) 불필요."""
    noisy = rng.poisson(np.maximum(img * N, 0)).astype(np.float64) / N
    return np.clip(noisy, 0, 1)


def stage2_gaussian(img, sigma, rng):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)


def stage3_salt_pepper(img, density, rng):
    out = img.copy()
    r = rng.random(img.shape)
    out[r < density / 2] = 1.0
    out[r > 1 - density / 2] = 0.0
    return out


def stage4_psf_blur(img, fwhm):
    sigma = fwhm / 2.355
    k = int(2 * round(3 * sigma) + 1)
    k = max(3, k if k % 2 == 1 else k + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def stage5_vignetting(img, alpha, min_brightness):
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    vig = 1 - alpha * d2 / (cx ** 2 + cy ** 2)
    vig = np.maximum(vig, min_brightness)
    return img * vig


def stage6_contrast(img, in_low, in_high, out_low, out_high):
    r = (img - in_low) / (in_high - in_low + 1e-8)
    return np.clip(r * (out_high - out_low) + out_low, 0, 1)


def stage7_ring(img, strength, n_rings):
    """Stage 7 — Ring artifact: 회전축(중심) 기준 동심원 변조 (검출기 비균일성)."""
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmax = r.max() + 1e-8
    mod = strength * np.sin(2 * np.pi * n_rings * r / rmax)
    return np.clip(img * (1 + mod), 0, 1)


def apply_degradation(img, p, max_stage, rng):
    """누적 열화 (출력해상도 img에 적용). max_stage: 0~7."""
    if max_stage == 0:
        return img.copy()
    r = img.copy()
    if max_stage >= 1: r = stage1_poisson(r, p['poisson_N'], rng)
    if max_stage >= 2: r = stage2_gaussian(r, p['gaussian_sigma'], rng)
    if max_stage >= 3: r = stage3_salt_pepper(r, p['sp_density'], rng)
    if max_stage >= 4: r = stage4_psf_blur(r, p['psf_fwhm'])
    if max_stage >= 5: r = stage5_vignetting(r, p['vignette_alpha'], p['vignette_min'])
    if max_stage >= 6: r = stage6_contrast(r, p['contrast_in_low'], p['contrast_in_high'],
                                           p['contrast_out_low'], p['contrast_out_high'])
    if max_stage >= 7: r = stage7_ring(r, p['ring_strength'], p['ring_n'])
    return r


STAGE_NAMES = ['Clean', '1.Poisson', '2.Gaussian', '3.S&P', '4.PSF',
               '5.Vignette', '6.Contrast', '7.Ring']


# ============================ 생성기 ============================

class HBMCanonicalGenerator:
    GROUP_KEYS = ['g1_clean', 'g2_noise', 'g3_artifact', 'g4_full']

    def __init__(self, img_folder, config_path,
                 grid_rows=5, grid_cols=20, type_probs=None):
        self.cfg_path = config_path
        self.cfg = load_config(config_path)
        self.scale = self.cfg['meta']['scale_factor']
        self.groups = self.cfg['groups']

        self.img_folder = Path(img_folder)
        self.grid_rows, self.grid_cols = grid_rows, grid_cols
        self.type_probs = type_probs or [0.25, 0.25, 0.25, 0.25]

        self.template_paths = sorted(self.img_folder.glob('*.png'))
        assert self.template_paths, f'PNG 없음: {self.img_folder}'
        self.ref_img = load_image(self.template_paths[0])
        self.tile_h, self.tile_w = self.ref_img.shape
        self.solder_mask, self.structure_mask, self.bulb_top, self.bulb_bottom = \
            analyze_structure(self.ref_img)

        self.canvas_h = self.tile_h * grid_rows
        self.canvas_w = self.tile_w * grid_cols
        self.out_h = int(self.canvas_h * self.scale)
        self.out_w = int(self.canvas_w * self.scale)
        self.defcfg = Config()

    # ---- 타일/그리드 (clean + 픽셀 클래스맵) ----
    def _make_tile(self, rng, forced_type=None):
        idx = rng.choice(len(self.template_paths))
        tpl = load_image(self.template_paths[idx])
        if tpl.shape != (self.tile_h, self.tile_w):
            tpl = cv2.resize(tpl, (self.tile_w, self.tile_h), interpolation=cv2.INTER_NEAREST)

        if forced_type is not None:
            dtype = forced_type
        else:
            probs = np.array(self.type_probs, float); probs /= probs.sum()
            dtype = rng.choice(DEFECT_TYPES, p=probs)

        if dtype == 'normal':
            return tpl, np.zeros((self.tile_h, self.tile_w), bool), 'normal', {}

        for _ in range(self.defcfg.max_retry):
            if dtype == 'crack':
                res, mask, prm = generate_crack(tpl, self.solder_mask, self.defcfg, rng)
            elif dtype == 'void':
                res, mask, prm = generate_void(tpl, self.solder_mask,
                                               self.bulb_top, self.bulb_bottom, self.defcfg, rng)
            else:
                res, mask, prm = generate_open(tpl, self.structure_mask,
                                               self.bulb_top, self.bulb_bottom, self.defcfg, rng)
            if np.sum(mask) > 10:
                return res, mask, dtype, prm
        return tpl, np.zeros((self.tile_h, self.tile_w), bool), 'normal', {'generation_failed': True}

    def make_clean_grid(self, rng, forced=None):
        """forced: dict {(row,col): type} 강제 결함 배치(없으면 확률 샘플).
        Returns: clean_canvas(float), class_mask_canvas(uint8 0/1/2/3), tile_labels.
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w), np.float64)
        cmask = np.zeros((self.canvas_h, self.canvas_w), np.uint8)
        labels = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                ftype = (forced or {}).get((r, c))
                tile, tmask, label, prm = self._make_tile(rng, forced_type=ftype)
                y0, x0 = r * self.tile_h, c * self.tile_w
                canvas[y0:y0+self.tile_h, x0:x0+self.tile_w] = tile
                lid = LABEL_IDS[label]
                if lid > 0:
                    cmask[y0:y0+self.tile_h, x0:x0+self.tile_w][tmask] = lid
                labels.append(dict(row=r, col=c, label=label, label_id=lid, params=prm))
        return canvas, cmask, labels

    def downscale_image(self, canvas):
        return cv2.resize(canvas, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)

    def downscale_mask(self, cmask):
        return cv2.resize(cmask, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)

    # ---- 본 생성 (Stage 1에서는 호출하지 않음; 승인 후 사용) ----
    def run(self, out_root, num_train, num_val, num_test, num_ood=0, ood_config_path=None):
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.cfg_path, out_root / 'degradation_config.yaml')
        # train/val = 그룹별(G1~G4) 전부 / test = 공통 g4_full만(확립된 ablation 프로토콜)
        splits = [('train', num_train, 0, True),
                  ('val', num_val, num_train, True),
                  ('test', num_test, num_train + num_val, False)]
        base = num_train + num_val + num_test
        for name, n, start, full in splits:
            self._gen_split(out_root / name, name, n, start, self.cfg, full=full)
        if num_ood and ood_config_path:
            ood_cfg = load_config(ood_config_path)
            ood_dir = out_root / 'test_ood'
            self._gen_split(ood_dir, 'test_ood', num_ood, base, ood_cfg, full=False)
            shutil.copy(ood_config_path, ood_dir / 'ood_config.yaml')  # dir 생성 후 스냅샷

    def _gen_split(self, sdir, name, n, start, cfg, full=True):
        keys = self.GROUP_KEYS if full else ['g4_full']
        for k in keys:
            (sdir / k / 'images').mkdir(parents=True, exist_ok=True)
            (sdir / k / 'masks').mkdir(parents=True, exist_ok=True)
        (sdir / 'labels').mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i in range(n):
            gidx = start + i
            seed = 42 + gidx
            # 재개(resume): 라벨 JSON이 이미 있으면 그 이미지는 완료된 것 → skip
            if (sdir / 'labels' / f'label_{gidx+1:05d}.json').exists():
                continue
            rng = np.random.RandomState(seed)
            canvas, cmask, labels = self.make_clean_grid(rng)
            clean_out = self.downscale_image(canvas)
            mask_out = self.downscale_mask(cmask)
            params = sample_params(cfg, np.random.RandomState(seed + 100000))
            fname = f'image_{gidx+1:05d}.png'
            mname = f'mask_{gidx+1:05d}.png'
            for k in keys:
                ms = self.groups[k]
                # 그룹마다 동일 노이즈 시드(seed+200000) → 같은 결함·노이즈에 열화수준만 차이(통제)
                deg = apply_degradation(clean_out, params, ms,
                                        np.random.RandomState(seed + 200000))
                cv2.imwrite(str(sdir / k / 'images' / fname), (deg * 255).astype(np.uint8))
                cv2.imwrite(str(sdir / k / 'masks' / mname), mask_out)  # 동일 클린 마스크
            with open(sdir / 'labels' / f'label_{gidx+1:05d}.json', 'w', encoding='utf-8') as f:
                json.dump(dict(filename=fname, mask=mname, seed=seed, split=name,
                               params=params, tiles=labels), f, indent=2)
            if (i + 1) % 50 == 0 or i == n - 1:
                print(f'  [{name}] {i+1}/{n}  ({time.time()-t0:.1f}s)')
