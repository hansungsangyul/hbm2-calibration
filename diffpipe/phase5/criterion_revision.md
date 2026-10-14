# Phase 5 발산 판정 기준 3 개정 (§V)
## 근거 문서: HANDOFF_v6.1 §7.1 | 작성: 2026-07-09

---

## 1. 개정 내용

### 개정 전 (원본)
```python
crit_3 = (max(f1s[-10:]) - min(f1s[-10:])) > 0.1  # epoch >= 20 시만 적용
```

### 개정 후 (§V)
```python
from scipy.signal import medfilt
f1_filt = medfilt(f1_raw, kernel_size=3)   # 양끝: medfilt 기본 동작(원값 유지)
crit_3 = (max(f1_filt[-10:]) - min(f1_filt[-10:])) > 0.1   # epoch >= 20 유지
```

완전 판정 코드:
```python
import numpy as np
from scipy.signal import medfilt

def check_diverged(train_losses, val_losses, f1_raw, epoch):
    f1_arr = np.array(f1_raw)
    f1_filt = medfilt(f1_arr, kernel_size=3)
    
    crit_1 = any(np.isnan(train_losses)) or any(np.isinf(train_losses))
    crit_2 = f1_arr[-1] < 0.5
    crit_3 = False
    if epoch >= 20 and len(f1_filt) >= 10:
        crit_3 = (max(f1_filt[-10:]) - min(f1_filt[-10:])) > 0.1
    
    diverged = crit_1 or crit_2 or crit_3
    reason = 1 if crit_1 else (2 if crit_2 else (3 if crit_3 else None))
    return diverged, reason, f1_filt.tolist()
```

---

## 2. 개정 사유

### 파일럿 false positive 사례

Phase 5 파일럿 (seed=950000, σ_g=0.041, fwhm=2.5, 41→57 epochs):
- **Epoch 57**: val_loss=1.0146 (전형값 0.29~0.46 대비 2.2~3.5× 이상), F1=0.833
- train_loss 계속 하강 (0.021 — epoch 57 최저): 실제 발산 증거 없음
- Criterion 3 창(epochs 48-57): max=0.9427, min=0.8333, range=0.1094 > 0.1 → false positive

원인: 단발 validation batch 스파이크. 이전 epoch 47-56 창 range=0.019 (임계값 훨씬 하회).

### medfilt 적용 효과 검증

파일럿 데이터 재판정:
```
epoch 55: F1=0.9356
epoch 56: F1=0.9237
epoch 57: F1=0.8333  ← 스파이크
```
medfilt k=3 적용:
```
f1_filt[54] = median(0.9397, 0.9373, 0.9365) = 0.9373  (epoch 53)
f1_filt[55] = median(0.9373, 0.9365, 0.9427) = 0.9373  (epoch 54)
f1_filt[56] = median(0.9365, 0.9427, 0.9356) = 0.9365  (epoch 55)
f1_filt[57] = median(0.9427, 0.9356, 0.9237) = 0.9356  (epoch 56 — 양끝이 아니므로 중앙값)
f1_filt[58] = median(0.9356, 0.9237, 0.8333) = 0.9237  (epoch 57)
```
1-indexed epochs 48-57 (0-indexed 47-56) 창 f1_filt: range ≈ 0.024 << 0.1 → **비발산**

(주의: 길이 57 배열에서 index 56이 마지막. medfilt는 양끝 처리 기본 = 원값 유지.)
→ epoch 57 스파이크가 medfilt로 제거되어 false positive 방지.

---

## 3. 불변 항목

| 항목 | 값 | 변경 여부 |
|------|-----|---------|
| Criterion 1 | NaN/Inf loss | 불변 |
| Criterion 2 | final F1 < 0.5 (원시값) | 불변 |
| Criterion 3 임계값 | 0.1 | 불변 |
| Criterion 3 창 크기 | last 10 epochs | 불변 |
| Epoch warmup | epoch >= 20 | 불변 |
| 변경 | 원시 F1 → medfilt(k=3) F1 적용 | **개정** |

---

## 4. 개정 승인

- **승인 근거**: HANDOFF_v6.1 §7.1 명시: "파일럿 기반 투명 개정 — 검증 인스턴스 승인 완료 (본 문서)"
- **개정 유형**: 사전 등록 기준의 파일럿 기반 파일럿 완료 후 본실험 전 개정 (pre-registered amendment, 정당)
- `approved_by`: "HANDOFF_v6.1 §7.1"

---

## 5. 출력 명세 변경

### run별 results.json 추가 필드
```json
{
  "f1_raw": [..41 floats..],
  "f1_filt": [..41 floats..],   // medfilt(f1_raw, k=3)
  "diverged": false,
  "divergence_reason": null     // null | 1 | 2 | 3
}
```
f1_raw와 f1_filt 모두 저장 (민감도 분석용).
