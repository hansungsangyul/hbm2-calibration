# normalization_audit.md
작성: CC (Claude Code) | 근거: HANDOFF_to_CC_20260730.md 요청 2 | 일자: 2026-07-30

---

## (a) 학습 데이터 prenorm 폐형 산출식

**파일 경로**: `diffpipe/phase4/run_phase4_v3.py`
**위치**: 파일 상단 docstring (L8, L27–28)

```python
# phase4/run_phase4_v3.py  L8, L27–28

#    x' = a·x + b, where a=0.501834, b=0.173320
#    Pre-computed (closed-form) from HBM_clean/dirty2 (mean, std) matching.

# Pre-normalization derivation:
#   HBM_clean:  mean=0.37551, std=0.25087
#   HBM_dirty2: mean=0.36176, std=0.12590
#   a = std_d / std_c = 0.12590/0.25087 = 0.501834
#   b = mean_d - a*mean_c = 0.36176 - 0.501834*0.37551 = 0.173320
```

**판정**: `a = σ_real / σ_template`, `b = μ_real − a·μ_template` 형태 — **1·2차 모멘트 정합 확인**. 최적화 루프 밖에서 폐형으로 1회 산출.

---

## (b) 추론 시점 정규화 (`calib+infer_pre`)

**파일 경로**: `diffpipe/phase4/v77/run_exp1_infer_renorm.py`
**함수**: `NormWrapperDataset.__getitem__` (L113–124)

```python
# phase4/v77/run_exp1_infer_renorm.py  L50, L69, L113–124

from exp_config import PRENORM_A, PRENORM_B   # 0.501834, 0.173320  (고정 상수)

MODES = ['orig', 'zscore', 'prenorm']

class NormWrapperDataset(Dataset):
    def __getitem__(self, idx):
        img_3ch, mask = self.base[idx]
        if self.mode == 'orig':
            pass
        elif self.mode == 'zscore':
            for c in range(img_3ch.shape[0]):
                ch = img_3ch[c]
                mu = ch.mean()
                sd = ch.std()
                img_3ch[c] = (ch - mu) / (sd + 1e-8)
        elif self.mode == 'prenorm':
            img_3ch = PRENORM_A * img_3ch + PRENORM_B   # ← 학습 때와 동일 상수 재사용
        return img_3ch, mask
```

**판정**: 추론 시점에 **학습 때 산출한 `(a, b) = (0.501834, 0.173320)` 고정 상수를 그대로 재사용**함 — 입력 영상마다 재계산하지 않음. 원고 §V-D "학습 때와 동일한 모멘트 정합" 서술 **정확**. z-score 모드는 별도 분기이며 per-channel 재계산.

---

## (c) `z-score norm` 행의 정규화 단위

**파일 경로**: `diffpipe/phase4/v77/run_exp4_zscore.py`
**함수**: 학습 루프 내 on-the-fly 적용 (L140–143) + `ZScoreFixedTestset.__getitem__` (L172–174)

```python
# phase4/v77/run_exp4_zscore.py  L140–144  (학습 데이터 on-the-fly)

img_deg = y.numpy()
if ZSCORE:
    img_deg = (img_deg - img_deg.mean()) / (img_deg.std() + 1e-8)
img_deg = cv2.copyMakeBorder(img_deg, 3,3,0,0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
img_3ch = np.stack([img_deg]*3, axis=0).astype(np.float32)

# phase4/v77/run_exp4_zscore.py  L172–175  (평가 데이터 ZScoreFixedTestset)

img_deg = np.load(self.images[idx])    # shape (H_pad, W_crop), grayscale
img_deg = (img_deg - img_deg.mean()) / (img_deg.std() + 1e-8)
img_3ch = np.stack([img_deg]*3, axis=0).astype(np.float32)
```

**판정**:
- **per-image** (이미지 단위, 배치 단위 아님): 각 img_deg 배열 전체에 `.mean()`·`.std()` 적용
- **1·2차 모멘트**: 평균 제거(1차) + 표준편차 정규화(2차) 동시 적용
- 입력이 grayscale이므로 per-image = per-channel과 동등

원고 "표준 per-image z-score" 서술 **정확**.

---

## 요약

| 확인 사항 | 원고 서술 | 코드 확인 | 판정 |
|---|---|---|---|
| (a) prenorm 폐형 산출식 | §III-B "1·2차 모멘트를 실영상에 맞추는 아핀 변환을 폐형으로 산출" | `a = std_d/std_c`, `b = mean_d − a·mean_c` | **일치 — 원고 유지 가능** |
| (b) 추론 시점 정규화 | §V-D "학습 때와 동일한 모멘트 정합" | 고정 상수 `(0.501834, 0.173320)` 재사용, 재계산 없음 | **일치 — 원고 유지 가능** |
| (c) z-score 단위 | "표준 per-image z-score" | per-image, 1·2차 모멘트 | **일치 — 원고 유지 가능** |
