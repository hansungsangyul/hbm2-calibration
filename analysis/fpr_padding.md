# fpr_padding.md — 요청 7: pixel FPR 분모의 패딩 문제

생성: CC (Claude Sonnet 4.6), 2026-08-05  
원본 데이터: `vf_table8_raw.csv`, `run_exp2_real_fpr_v2.py`, 실측 분석  

---

## Q7-1. pixel_fpr 분모가 패딩을 포함합니까?

**분모 = 163,840 (패딩 3,072픽셀 포함)** — 확정

### 코드 인용 (`run_exp2_real_fpr_v2.py`)

```python
def compute_fpr_metrics(pred_logits: torch.Tensor):
    pred_class = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    total_pixels = pred_class.size          # ← 320 × 512 = 163,840 (패딩 포함)
    non_bg_mask = (pred_class != 0).astype(np.uint8)
    pixel_fpr = float(non_bg_mask.sum()) / total_pixels   # run_exp2_real_fpr_v2.py:96
```

### 이미지 크기 추적

```
HBM_dirty2/*.png  원본: 314 × 520
cv2.copyMakeBorder(img, 3, 3, 0, 0, BORDER_CONSTANT, value=0)  →  320 × 520
[:, 4:516]                                                       →  320 × 512
```

| 항목 | 픽셀 수 |
|------|---------|
| 패딩 포함 전체 | 320 × 512 = **163,840** (현재 분모) |
| 원본 이미지 영역 | 314 × 512 = **160,768** (올바른 분모) |
| 패딩 (상단 3행 + 하단 3행) | 6 × 512 = **3,072** |
| 패딩 비율 | 3,072 / 163,840 = **0.01875 = 1.875%** |

---

## Q7-2. class_0 픽셀의 공간 위치

**class_0은 패딩 행(rows 0-2, 317-319)에만 집중됨** — 실측 확인

### 실측 결과 (D_calib run0, image_0001.png)

```
행별 class_0 픽셀 수 (512 = 한 행 전체):
  rows 0-2   (상단 패딩) : [512, 512, 512]    ← 전부 class_0
  rows 3-5   (실영상 상단): [  3,   0,   0]    ← 거의 없음
  rows 310-313(실영상 하단): [  0,   0,   0,  0] ← 없음
  rows 317-319(하단 패딩) : [512, 512, 512]    ← 전부 class_0
  
  총 class_0: 3,075픽셀 = 패딩 3,072 + 실영상 내 3픽셀
```

### 전 600 샘플 검증

| 검증 항목 | 결과 |
|----------|------|
| 모든 샘플에서 class_0_pixels ≥ 3,072 | **600/600** |
| class_0_pixels < 3,072인 샘플 수 | **0개** |
| 패딩 픽셀이 class_0으로 분류되지 않은 샘플 | **0개** |

→ **패딩 3,072픽셀은 전 조건·전 run에서 100% class_0으로 예측됨**  
→ 실영상 내 class_0은 조건별로 소수~수천 픽셀 다양

---

## Q7-3. 패딩 제외 기준 재계산

### 변환 공식

```
fpr_noPad = pixel_fpr × (163,840 / 160,768) = pixel_fpr / 0.98125

근거: 패딩 3,072픽셀은 모두 class_0 (background)
     → non_bg_pixels 합계는 패딩 포함/제외 동일
     → 분모만 163,840 → 160,768로 변경
```

### 패딩 제외 Table 8 (수정 후)

| condition | n_samples | **med_fpr_noPad** | min_fpr_noPad | med_fpr_full (이전) |
|-----------|-----------|-------------------|---------------|---------------------|
| calib | 100 | **1.0000** | 0.6935 | 0.9813 |
| manual | 100 | **1.0000** | 0.9986 | 0.9813 |
| ab1 | 100 | **1.0000** | 0.9571 | 0.9813 |
| ab2 | 100 | **1.0000** | 0.9985 | 0.9813 |
| ab3 | 100 | **1.0000** | 0.9824 | 0.9813 |
| exp3 | 100 | **1.0000** | 0.8668 | 0.9813 |

**전 조건 중앙값 FPR = 1.000 (100%)** — 패딩 제외 시 "실제 이미지 픽셀의 중앙적 예측은 100% non-background"

### 핵심 결론

- 종래 보고값 **0.981** = 패딩 비율(0.01875)이 결정하는 수학적 상수
  - 0.981 = 1 − 0.019 = 1 − (3,072/163,840) → 모델 특성이 아님
- 패딩 제외 실제 FPR = **1.000 (100%)** — real dirty2 이미지의 모든 실제 픽셀이 결함 클래스로 예측됨
- 조건 간 차이가 0.001 미만이었던 이유: **전부 동일한 패딩 상수**이기 때문

---

## Q7-4. 패딩 픽셀의 값

**패딩 = 0.0 (상수 zero)**

```python
# load_dirty2_raw (run_exp2_real_fpr_v2.py:78)
img = cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
```

**학습 시에도 동일 패딩 적용** (TrainDataset.__getitem__ L150, L153 in run_p42r_v2.py):
```python
img_deg = cv2.copyMakeBorder(img_deg, 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
mask    = cv2.copyMakeBorder(mask,    3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
```

→ 모델이 0-값 픽셀을 배경(class_0)으로 학습했으며, 추론 시에도 동일하게 예측

---

## Q7-5. 패딩 제외 FPR 분포 (5·25·50·75·95 백분위수)

### calib (n=100)

| 지표 | FPR_noPad |
|------|-----------|
| p5 | 0.9614 |
| p25 | 1.0000 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p95 | 1.0000 |
| mean actual_class0_in_image | 1,460 픽셀 |

### manual (n=100)

| 지표 | FPR_noPad |
|------|-----------|
| p5 | 0.9999 |
| p25 | 1.0000 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p95 | 1.0000 |
| mean actual_class0_in_image | 5 픽셀 |

### ab1 (n=100)

| 지표 | FPR_noPad |
|------|-----------|
| p5 | 0.9990 |
| p25 | 1.0000 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p95 | 1.0000 |
| mean actual_class0_in_image | 127 픽셀 |

### ab2 (n=100)

| 지표 | FPR_noPad |
|------|-----------|
| p5 | 0.9997 |
| p25 | 1.0000 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p95 | 1.0000 |
| mean actual_class0_in_image | 8 픽셀 |

### ab3 (n=100)

| 지표 | FPR_noPad |
|------|-----------|
| p5 | 0.9992 |
| p25 | 1.0000 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p95 | 1.0000 |
| mean actual_class0_in_image | 56 픽셀 |

### exp3 (n=100)

| 지표 | FPR_noPad |
|------|-----------|
| p5 | 0.9278 |
| p25 | 0.9917 |
| p50 | 1.0000 |
| p75 | 1.0000 |
| p95 | 1.0000 |
| mean actual_class0_in_image | 1,713 픽셀 |

### 해석

- **manual·ab2·ab3**: 실영상 내 배경 픽셀이 평균 5~56개 → FPR_noPad 거의 항상 1.000
- **calib·exp3**: 실영상 내 배경 픽셀 평균 1,460~1,713개 → FPR_noPad 분포가 다소 넓음
  - exp3 p5=0.928 — 일부 run에서는 FPR이 낮은 경우 존재
- **calib의 min_fpr_full=0.680** (→ min_fpr_noPad=0.693): calib 조건 일부 run에서 실제 배경 탐지 발생
- 중앙값 기준으로는 **전 조건 100%** — real dirty2 이미지의 실제 픽셀 기준 위양성률

---

## 원고 반영 권고

| 위치 | 현재 값 | 정정 값 | 근거 |
|------|---------|---------|------|
| 초록, §I, §V-F, §VII, §VIII | "0.98" | **1.00** | 패딩 제외 실제 이미지 기준 |
| Table 8 (`med_pixel_fpr`) | 0.9813 | **1.0000** | 전 조건 |
| Table 8 (`min_pixel_fpr`) | (조건별) | (별표 기재, 아래 참고) | |

min_fpr_noPad (전 조건): calib=0.69, manual=1.00, ab1=0.96, ab2=1.00, ab3=0.98, exp3=0.87

---

*raw data: `answers_round3_raw.json` (q7_3, q7_5 키)*  
*Table 8 원본: `vf_table8_raw.csv`*
