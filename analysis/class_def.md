# class_def.md — Q1-1 ~ Q1-3, Q4-1 ~ Q4-2

생성: CC (Claude Sonnet 4.6), 2026-08-06  
원본: `dataset.py` (`hbm_augmentation`), `p42r_testsets/*/meta.json`

---

## Q1-1. 4클래스 분할 정의 (코드 직접 인용)

**파일**: `/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation/dataset.py:17-22`

```python
class_map = {
    'normal': 0,
    'crack':  1,
    'hole':   2,
    'open':   3
}
```

| 클래스 | 라벨 문자열 | 물리적 의미 |
|--------|-------------|-------------|
| **0** | `'normal'` | **정상 (배경)** — 결함 없음 |
| **1** | `'crack'`  | 균열 결함 |
| **2** | `'hole'`   | 구멍(홀) 결함 |
| **3** | `'open'`   | 오픈(단선) 결함 |

마스크 기본값 = 0 (`mask = np.zeros(..., dtype=np.int64)`).  
라벨 JSON에 명시된 영역에만 1·2·3이 채워지고, 나머지 전체가 0.

### 픽셀 위양성률 정의와의 일관성

```python
# run_exp2_real_fpr_v2.py:96 (Q7 검증 당시 인용)
non_bg_mask = (pred_class != 0)
pixel_fpr   = non_bg_mask.sum() / total_pixels
```

`pred_class != 0` = "클래스 0이 아닌 화소" = "결함으로 예측된 화소".  
**클래스 0 = 정상(배경)이어야만 이 정의가 성립한다.** ✓

---

## Q1-2. §V-A 캡션 오기 판정

> "클래스 0--3은 결함 심각도 수준이고..."

**오기입니다.** 클래스 0 = `'normal'` (정상, 배경). 결함 등급은 1·2·3 세 개입니다.

올바른 서술:
> "클래스 0은 정상(배경)이고, 클래스 1·2·3은 각각 균열·구멍·오픈 결함이다."

---

## Q1-3. 결함 등급 1·2·3의 물리적 구분 기준

| 등급 | 라벨 | 구분 기준 |
|------|------|-----------|
| 1 | `crack` | 균열 (선형 결함) |
| 2 | `hole`  | 구멍 (국소 면적 결함) |
| 3 | `open`  | 오픈/단선 (연결 불량) |

구분 기준은 **결함 형태·유형**이며, 크기·대비·깊이의 정량 임계값이 코드에 명시되어 있지 않습니다.  
라벨 JSON은 타일(tile) 단위(5행×20열 격자)로 부여되며, 타일 내 어느 유형의 결함이 존재하는지를 수동 라벨링한 것으로 추정됩니다.  
심사자 질문에 대한 답변이 원고에 필요한 부분입니다.

---

## Q4-1. TEST_OOD "1.5배 열화" — 정확한 파라미터

**파일**: `p42r_testsets/test_ood/meta.json`  
**desc**: "전작 R1-3 OOD ×1.5 (PSF 2.5→3.75, σ_g×1.5, ring 0.12→0.18), pre-norm 미적용"

기준 θ (TEST_ID = θ_manual):

| 파라미터 | TEST_ID (기준) | TEST_OOD (×1.5) | 비율 |
|----------|---------------|-----------------|------|
| `fwhm`           | 2.500  | **3.750**  | ×1.5 |
| `sigma_g`        | 0.0408 | **0.0612** | ×1.5 |
| `ring_strength`  | 0.12   | **0.18**   | ×1.5 |
| `N`              | 300.0  | 300.0      | ×1.0 (불변) |
| `p`              | 0.001  | 0.001      | ×1.0 |
| `alpha`          | 0.35   | 0.35       | ×1.0 |
| `in_low/in_high` | 0.05/0.9 | 0.05/0.9 | ×1.0 |
| `prenorm`        | False  | **False**  | (미적용) |

변경된 세 파라미터: **PSF 블러(fwhm)·포아송 노이즈(sigma_g)·링 아티팩트(ring_strength) = 전부 ×1.5**.

---

## Q4-2. 세 평가셋 θ 및 GT 클래스 비율

### θ 요약

| | TEST_ID | TEST_OOD | TEST_CAL |
|---|---|---|---|
| `fwhm` | 2.500 | 3.750 | 2.0956 |
| `sigma_g` | 0.0408 | 0.0612 | 0.0408 |
| `ring_strength` | 0.12 | 0.18 | 0.12 |
| `prenorm` | False | False | **True** |
| `eps_seed_base` | 500,000 | 510,000 | 520,000 |
| `n_images` | 500 | 500 | 500 |

TEST_CAL은 D_calib 파이프라인 파라미터(fwhm=2.0956)로 생성되며, prenorm=True가 적용됨.  
TEST_ID = D_manual 학습 분포와 일치.

### GT 클래스 비율

GT 라벨이 세 평가셋 모두 동일한 `test_data/HBM_labels`에서 옴 → **세 평가셋의 GT 클래스 비율은 동일합니다.**

(실측값은 `pred_distribution.csv` `gt_class*_pct` 열 참조 — test_ood의 GT는 어느 testset에서나 동일)

---

*참조 파일: `dataset.py:10-53`, `meta.json` (test_id/test_ood/test_cal)*
