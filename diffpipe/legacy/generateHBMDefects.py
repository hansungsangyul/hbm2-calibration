"""
HBM Microbump Defect Generator (Python)
========================================
물리 기반 합성 결함 생성기 - inspectDefects2 테스트를 통해 확정된 파라미터

구조:
  - Cu pad (상부, 어두운 회색) → 결함 없음
  - Sn solder (중간, 밝은 회색) → Void, Crack, Open 생성
  - Cu pillar (하부, 어두운 회색) → 결함 없음

출력:
  images/  → image_00001.png, image_00002.png, ...
  masks/   → mask_00001.png,  mask_00002.png, ...
  labels.csv → filename, label, label_id, params...

label_id: 0=normal, 1=crack, 2=void, 3=open

사용법:
  python generateHBMDefects.py
  또는 import 해서 사용:
    from generateHBMDefects import HBMDefectGenerator
    gen = HBMDefectGenerator(config)
    gen.run()
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_closing, binary_opening
from pathlib import Path
import csv
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional
import json


# ========================= 파라미터 =========================

@dataclass
class ParamDist:
    mean: float
    std: float
    lo: float = 0.0
    hi: float = float('inf')

    def sample(self, rng: np.random.RandomState, as_int=False):
        val = rng.normal(self.mean, self.std)
        val = max(self.lo, min(self.hi, val))
        return int(round(val)) if as_int else val


@dataclass
class Config:
    # 경로
    img_folder: str = ''
    output_folder: str = ''

    # 그리드 (캔버스용, 개별 저장 시 무시)
    grid_R: int = 20
    grid_C: int = 20

    # 생성 설정
    num_images: int = 20000
    type_probs: list = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])  # [normal, crack, void, open]
    max_retry: int = 8
    noise_strength: float = 0.005

    # 품질 기준
    min_area_ratio: float = 0.002
    min_contrast: float = 0.01
    max_ssim: float = 1.0

    # Crack
    crack_width: ParamDist = field(default_factory=lambda: ParamDist(4, 1, 3, 6))
    crack_length: ParamDist = field(default_factory=lambda: ParamDist(120, 30, 80, 200))
    crack_stress_angle: ParamDist = field(default_factory=lambda: ParamDist(0, 25, -45, 45))
    crack_intensity: ParamDist = field(default_factory=lambda: ParamDist(0.05, 0.02, 0.01, 0.12))
    crack_noise_scale: ParamDist = field(default_factory=lambda: ParamDist(1.2, 0.3, 0.5, 1.8))
    crack_branch_prob: ParamDist = field(default_factory=lambda: ParamDist(0.12, 0.05, 0.0, 0.25))

    # Void (hole)
    void_size: ParamDist = field(default_factory=lambda: ParamDist(15, 6, 6, 35))
    void_max_count: ParamDist = field(default_factory=lambda: ParamDist(2, 2, 1, 8))
    void_intensity: ParamDist = field(default_factory=lambda: ParamDist(0.05, 0.02, 0.01, 0.12))
    void_row_ratio: ParamDist = field(default_factory=lambda: ParamDist(0.72, 0.10, 0.50, 0.90))
    void_band_width: ParamDist = field(default_factory=lambda: ParamDist(15, 5, 5, 30))

    # Open
    open_center_ratio: ParamDist = field(default_factory=lambda: ParamDist(0.55, 0.10, 0.35, 0.75))
    open_gap_ratio: ParamDist = field(default_factory=lambda: ParamDist(0.15, 0.05, 0.06, 0.35))
    open_intensity: ParamDist = field(default_factory=lambda: ParamDist(0.08, 0.03, 0.02, 0.18))
    open_wiggle: ParamDist = field(default_factory=lambda: ParamDist(8, 3, 3, 15))

    # 템플릿 분배
    decay_rate: float = 0.3


# ========================= 구조 분석 =========================

def analyze_structure(img: np.ndarray):
    """Cu pad/pillar와 Sn solder를 intensity 기반으로 분리
    
    Returns: solder_mask, structure_mask, bulb_top, bulb_bottom
    """
    h, w = img.shape

    # Otsu threshold
    _, thresh = cv2.threshold(
        (img * 255).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    structure_mask = thresh > 0

    # Morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    structure_uint8 = structure_mask.astype(np.uint8) * 255
    structure_uint8 = cv2.morphologyEx(structure_uint8, cv2.MORPH_CLOSE, k, iterations=2)
    structure_mask = structure_uint8 > 0

    # Row-wise width
    widths = np.sum(structure_mask, axis=1)
    max_w = np.max(widths[5:]) if len(widths) > 5 else np.max(widths)

    # Transition point
    transition_row = h // 4
    for r in range(20, h - 10):
        if r + 5 < h and widths[r] > max_w * 0.5 and widths[r + 5] < max_w * 0.3:
            transition_row = r
            break

    bulb_top = 5
    bulb_bottom = min(h, transition_row + 5)

    # Solder mask: bright pixels within bulb
    bulb_region = structure_mask[bulb_top:bulb_bottom, :]
    bulb_vals = img[bulb_top:bulb_bottom, :]
    bv = bulb_vals[bulb_region]

    if len(bv) > 0:
        solder_thresh = np.median(bv) - 0.04
    else:
        solder_thresh = 0.78

    solder_mask = np.zeros((h, w), dtype=bool)
    for r in range(bulb_top, bulb_bottom):
        for c in range(w):
            if structure_mask[r, c] and img[r, c] > solder_thresh:
                solder_mask[r, c] = True

    # Cleanup
    sm_uint8 = solder_mask.astype(np.uint8) * 255
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sm_uint8 = cv2.morphologyEx(sm_uint8, cv2.MORPH_CLOSE, k2)
    sm_uint8 = cv2.morphologyEx(sm_uint8, cv2.MORPH_OPEN, k3)
    solder_mask = sm_uint8 > 0

    return solder_mask, structure_mask, bulb_top, bulb_bottom


# ========================= Crack =========================

def generate_crack(img, solder_mask, cfg, rng):
    """어두운 선 (재료 부재). 솔더 전용. 1~2 시작점, 구불구불 경로, 분기."""
    h, w = img.shape
    result = img.copy()
    mask = np.zeros((h, w), dtype=bool)

    cwidth = cfg.crack_width.sample(rng, as_int=True)
    clength = cfg.crack_length.sample(rng, as_int=True)
    angle = cfg.crack_stress_angle.sample(rng)
    intensity = cfg.crack_intensity.sample(rng)
    nscale = cfg.crack_noise_scale.sample(rng)
    bprob = cfg.crack_branch_prob.sample(rng)

    params = dict(width=cwidth, length=clength, stress_angle=round(angle, 1),
                  intensity=round(intensity, 3), noise_scale=round(nscale, 2),
                  branch_prob=round(bprob, 3))

    # 솔더 내부 시작점
    interior = cv2.erode(solder_mask.astype(np.uint8) * 255,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    pts = np.where(interior > 0)
    if len(pts[0]) == 0:
        pts = np.where(solder_mask)
    if len(pts[0]) == 0:
        return result, mask, params

    num_seeds = rng.randint(1, 3)  # 1~2

    for _ in range(num_seeds):
        idx = rng.randint(0, len(pts[0]))
        sy, sx = int(pts[0][idx]), int(pts[1][idx])

        stress_rad = np.radians(angle)
        primary = np.array([np.cos(stress_rad), np.sin(stress_rad)])
        secondary = np.array([-np.sin(stress_rad), np.cos(stress_rad)])
        current_dir = primary + (rng.random() - 0.5) * 0.8 * secondary
        current_dir /= max(np.linalg.norm(current_dir), 1e-8)

        pos = np.array([float(sx), float(sy)])
        outside_count = 0

        for step in range(clength):
            ix, iy = int(round(pos[0])), int(round(pos[1]))
            if not (0 <= ix < w and 0 <= iy < h):
                break

            if not solder_mask[iy, ix]:
                outside_count += 1
                if outside_count > 5:
                    break
                pos += current_dir * 2
                continue
            outside_count = 0

            # 타원형 단면
            for dy in range(-cwidth, cwidth + 1):
                for dx in range(-cwidth, cwidth + 1):
                    yy, xx = iy + dy, ix + dx
                    if 0 <= yy < h and 0 <= xx < w and solder_mask[yy, xx]:
                        dist = np.sqrt((dx / max(cwidth, 1)) ** 2 + (dy / max(cwidth * 0.5, 1)) ** 2)
                        if dist <= 1:
                            mask[yy, xx] = True

            # 이동
            step_noise = (rng.random(2) - 0.5) * nscale * 1.5
            step_vec = current_dir + step_noise
            step_vec /= max(np.linalg.norm(step_vec), 1e-8)
            pos += step_vec * 2

            # 분기
            if rng.random() < bprob and step > 10:
                bdir = current_dir + (rng.random(2) - 0.5) * 0.8
                bdir /= max(np.linalg.norm(bdir), 1e-8)
                bpos = pos.copy()
                bcw = max(1, cwidth - 1)
                for _ in range(int(clength * 0.3)):
                    bx, by = int(round(bpos[0])), int(round(bpos[1]))
                    if not (0 <= bx < w and 0 <= by < h):
                        break
                    if not solder_mask[by, bx]:
                        break
                    for dy in range(-bcw, bcw + 1):
                        for dx in range(-bcw, bcw + 1):
                            yy, xx = by + dy, bx + dx
                            if 0 <= yy < h and 0 <= xx < w and solder_mask[yy, xx]:
                                if np.hypot(dx, dy) <= bcw:
                                    mask[yy, xx] = True
                    bpos += bdir * 1.5

            # 방향 회전
            dth = (rng.random() - 0.5) * 0.4
            Rm = np.array([[np.cos(dth), -np.sin(dth)],
                           [np.sin(dth), np.cos(dth)]])
            current_dir = Rm @ current_dir

    # Morphology cleanup
    mk = mask.astype(np.uint8) * 255
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = (mk > 0) & solder_mask
    result[mask] = intensity

    return result, mask, params


# ========================= Void =========================

def generate_void(img, solder_mask, bulb_top, bulb_bottom, cfg, rng):
    """IMC 계면 근처 어두운 점. Poisson disk 배치. 크기↔개수 반비례."""
    h, w = img.shape
    result = img.copy()
    mask = np.zeros((h, w), dtype=bool)

    vsize = cfg.void_size.sample(rng, as_int=True)
    max_count = cfg.void_max_count.sample(rng, as_int=True)
    intensity = cfg.void_intensity.sample(rng)
    row_ratio = cfg.void_row_ratio.sample(rng)
    band_w = cfg.void_band_width.sample(rng, as_int=True)

    # 크기↔개수 반비례
    if vsize >= 20:
        max_count = min(max_count, 2)
    elif vsize >= 12:
        max_count = min(max_count, 3)

    params = dict(size=vsize, max_count=max_count, actual_count=0,
                  intensity=round(intensity, 3), row_ratio=round(row_ratio, 3),
                  band_width=band_w)

    bulb_height = bulb_bottom - bulb_top
    interface_row = int(bulb_top + bulb_height * row_ratio)

    # 후보: 솔더 내 인터페이스 근처
    ys, xs = np.where(solder_mask)
    keep = np.abs(ys - interface_row) <= band_w
    coords = np.column_stack([xs[keep], ys[keep]])
    if len(coords) == 0:
        return result, mask, params

    # Poisson disk sampling
    min_dist = vsize * 2
    centers = poisson_disk_sampling(coords, min_dist, max_count, rng)
    params['actual_count'] = len(centers)

    for cx, cy in centers:
        cx, cy = int(cx), int(cy)
        hs = max(2, vsize + rng.randint(-2, 3))

        # 불규칙 폴리곤
        nv = rng.randint(7, 13)
        angles = np.linspace(0, 2 * np.pi, nv, endpoint=False) + (rng.random(nv) - 0.5) * 0.4
        radii = hs * (0.8 + 0.4 * rng.random(nv))
        # 스무딩
        radii_ext = np.concatenate([[radii[-1]], radii, [radii[0]]])
        radii = np.convolve(radii_ext, [0.2, 0.6, 0.2], mode='valid')

        for y in range(max(0, cy - hs * 2), min(h, cy + hs * 2 + 1)):
            for x in range(max(0, cx - hs * 2), min(w, cx + hs * 2 + 1)):
                if not solder_mask[y, x]:
                    continue
                dx, dy = x - cx, y - cy
                theta = np.arctan2(dy, dx)
                dist = np.hypot(dx, dy)
                a = np.abs(angles - theta)
                a = np.minimum(a, 2 * np.pi - a)
                mi = np.argmin(a)
                if dist <= radii[mi]:
                    mask[y, x] = True
                    ef = dist / radii[mi]
                    blend = 1.0 if ef < 0.8 else 1.0 - (ef - 0.8) / 0.2
                    result[y, x] = result[y, x] * (1 - blend) + intensity * blend

    return result, mask, params


# ========================= Open =========================

def generate_open(img, structure_mask, bulb_top, bulb_bottom, cfg, rng):
    """좌우 관통 gap. Multi-frequency wiggle 경계."""
    h, w = img.shape
    result = img.copy()
    mask = np.zeros((h, w), dtype=bool)

    center_ratio = cfg.open_center_ratio.sample(rng)
    gap_ratio = cfg.open_gap_ratio.sample(rng)
    intensity = cfg.open_intensity.sample(rng)
    wiggle_str = cfg.open_wiggle.sample(rng)

    params = dict(center_ratio=round(center_ratio, 3), gap_ratio=round(gap_ratio, 3),
                  intensity=round(intensity, 3), wiggle=round(wiggle_str, 1))

    bulb_height = bulb_bottom - bulb_top
    gap_center = int(bulb_top + bulb_height * center_ratio)
    gap_half = max(3, int(bulb_height * gap_ratio / 2))
    base_top = gap_center - gap_half
    base_bot = gap_center + gap_half

    top_curve = natural_boundary(w, wiggle_str, rng)
    bot_curve = natural_boundary(w, wiggle_str, rng)

    for col in range(w):
        top_y = int(round(base_top + top_curve[col]))
        bot_y = int(round(base_bot + bot_curve[col]))
        if top_y >= bot_y:
            top_y, bot_y = bot_y - 1, top_y + 1

        for row in range(max(0, top_y), min(h, bot_y)):
            if structure_mask[row, col]:
                mask[row, col] = True
                dist_to_edge = min(row - top_y, bot_y - row)
                blend = dist_to_edge / 2 if dist_to_edge < 2 else 1.0
                result[row, col] = result[row, col] * (1 - blend) + intensity * blend

    return result, mask, params


# ========================= 유틸리티 =========================

def natural_boundary(width, wiggle_strength, rng):
    t = np.linspace(0, 1, width)
    f1 = 1.5 + rng.random()
    f2 = 4 + rng.random() * 3
    f3 = 9 + rng.random() * 5

    wiggle = wiggle_strength * (
        np.sin(t * 2 * np.pi * f1 + rng.random() * 2 * np.pi) * 0.8 +
        np.sin(t * 2 * np.pi * f2 + rng.random() * 2 * np.pi) * 0.4 +
        np.sin(t * 2 * np.pi * f3 + rng.random() * 2 * np.pi) * 0.2 +
        (rng.random(width) - 0.5) * 0.5
    )
    transition = 0.5 + 0.5 * np.sin(t * np.pi) ** 1.5
    wiggle *= transition
    wiggle = np.convolve(wiggle, np.ones(3) / 3, mode='same')
    return wiggle


def poisson_disk_sampling(candidates, min_dist, max_samples, rng):
    if len(candidates) == 0 or max_samples == 0:
        return []
    idx = rng.permutation(len(candidates))
    candidates = candidates[idx]
    samples = [candidates[0]]
    remaining = list(candidates[1:])

    while len(samples) < max_samples and remaining:
        valid = []
        for i, p in enumerate(remaining):
            dists = np.sqrt(np.sum((np.array(samples) - p) ** 2, axis=1))
            if np.all(dists >= min_dist):
                valid.append(i)
        if not valid:
            break
        sel = valid[rng.randint(0, len(valid))]
        samples.append(remaining[sel])
        remaining.pop(sel)
    return samples


def evaluate_quality(original, result, defect_mask, solder_mask, cfg):
    ba = np.sum(solder_mask)
    dm = np.sum(defect_mask & solder_mask)
    if ba == 0:
        return False, {}
    area_ratio = dm / ba
    if dm > 0:
        contrast = np.mean(np.abs(result[defect_mask & solder_mask] -
                                   original[defect_mask & solder_mask]))
    else:
        contrast = 0
    mse = np.mean((original.astype(float) - result.astype(float)) ** 2)
    var_val = np.var(original) + 1e-6
    ssim_val = max(0, min(1, 1 - mse / var_val))

    passed = (area_ratio >= cfg.min_area_ratio and
              contrast >= cfg.min_contrast and
              ssim_val <= cfg.max_ssim)
    return passed, dict(area_ratio=area_ratio, contrast=contrast, ssim=ssim_val)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'이미지 로드 실패: {path}')
    return img.astype(np.float64) / 255.0


# ========================= 메인 생성기 =========================

class HBMDefectGenerator:
    DEFECT_TYPES = ['normal', 'crack', 'void', 'open']
    LABEL_IDS = {'normal': 0, 'crack': 1, 'void': 2, 'open': 3}

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # 템플릿 로드
        folder = Path(cfg.img_folder)
        self.template_paths = sorted(folder.glob('*.png'))
        assert len(self.template_paths) > 0, f'PNG 없음: {folder}'

        # 첫 번째로 구조 분석
        self.ref_img = load_image(self.template_paths[0])
        self.h, self.w = self.ref_img.shape
        self.solder_mask, self.structure_mask, self.bulb_top, self.bulb_bottom = \
            analyze_structure(self.ref_img)

        print(f'템플릿: {len(self.template_paths)}개, 크기: {self.w}x{self.h}')
        print(f'Bulb: rows {self.bulb_top}-{self.bulb_bottom}')
        print(f'솔더: {np.sum(self.solder_mask)}px')

        # 지수 감소 분배
        M = len(self.template_paths)
        weights = np.exp(-cfg.decay_rate * np.arange(M))
        self.template_weights = weights / weights.sum()

    def _choose_template(self, rng):
        idx = rng.choice(len(self.template_paths), p=self.template_weights)
        img = load_image(self.template_paths[idx])
        # 기준 크기와 다르면 리사이즈
        if img.shape != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return img

    def _choose_type(self, rng):
        probs = np.array(self.cfg.type_probs, dtype=float)
        probs /= probs.sum()
        return rng.choice(self.DEFECT_TYPES, p=probs)

    def generate_one(self, rng):
        """1장 생성 → (image, mask, label, params)"""
        cfg = self.cfg
        tpl = self._choose_template(rng)
        dtype = self._choose_type(rng)

        if dtype == 'normal':
            return tpl, np.zeros_like(tpl, dtype=bool), 'normal', {}

        for attempt in range(cfg.max_retry):
            if dtype == 'crack':
                result, mask, params = generate_crack(tpl, self.solder_mask, cfg, rng)
            elif dtype == 'void':
                result, mask, params = generate_void(
                    tpl, self.solder_mask, self.bulb_top, self.bulb_bottom, cfg, rng)
            elif dtype == 'open':
                result, mask, params = generate_open(
                    tpl, self.structure_mask, self.bulb_top, self.bulb_bottom, cfg, rng)
            else:
                return tpl, np.zeros_like(tpl, dtype=bool), 'normal', {}

            # 노이즈
            if cfg.noise_strength > 0:
                noise = (rng.random(result.shape) - 0.5) * cfg.noise_strength
                result = np.clip(result + noise, 0, 1)

            passed, _ = evaluate_quality(tpl, result, mask, self.solder_mask, cfg)
            if passed:
                return result, mask, dtype, params

        # 실패 시 normal
        return tpl, np.zeros_like(tpl, dtype=bool), 'normal', {'generation_failed': True}

    def run(self):
        """전체 데이터셋 생성"""
        cfg = self.cfg
        N = cfg.num_images
        out = Path(cfg.output_folder)
        img_dir = out / 'images'
        mask_dir = out / 'masks'
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # CSV 라벨
        csv_path = out / 'labels.csv'
        csvfile = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label', 'label_id', 'params'])

        counts = {t: 0 for t in self.DEFECT_TYPES}
        failed = 0
        t0 = time.time()

        for i in range(N):
            rng = np.random.RandomState()  # 매번 새 seed
            img, mask, label, params = self.generate_one(rng)

            fname = f'image_{i + 1:05d}.png'

            # 저장
            cv2.imwrite(str(img_dir / fname), (img * 255).astype(np.uint8))

            # 마스크: 0=배경, label_id=결함
            lid = self.LABEL_IDS[label]
            mask_save = np.zeros_like(mask, dtype=np.uint8)
            mask_save[mask] = lid if lid > 0 else 0
            cv2.imwrite(str(mask_dir / fname), mask_save)

            # CSV
            params_str = json.dumps(params) if params else ''
            writer.writerow([fname, label, lid, params_str])

            counts[label] += 1
            if params.get('generation_failed'):
                failed += 1

            if (i + 1) % 500 == 0 or i == N - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (N - i - 1) / rate if rate > 0 else 0
                print(f'  [{i + 1:5d}/{N}] {rate:.1f} img/s, ETA {eta / 60:.1f}min '
                      f'| N={counts["normal"]} C={counts["crack"]} '
                      f'V={counts["void"]} O={counts["open"]}')

        csvfile.close()

        elapsed = time.time() - t0
        print(f'\n✅ 완료! {N}장, {elapsed:.0f}초 ({elapsed / 60:.1f}분)')
        print(f'   Normal={counts["normal"]}, Crack={counts["crack"]}, '
              f'Void={counts["void"]}, Open={counts["open"]}')
        if failed:
            print(f'   ⚠️ 생성 실패 (normal 대체): {failed}')
        print(f'   저장: {out}')
        print(f'   labels.csv: {csv_path}')


# ========================= 실행 =========================

if __name__ == '__main__':
    cfg = Config(
        img_folder=r'D:\jonghyun\normal1_only',
        output_folder=r'D:\jonghyun\dataset_v1',
        num_images=20000,
        type_probs=[0.25, 0.25, 0.25, 0.25],
    )

    gen = HBMDefectGenerator(cfg)
    gen.run()
