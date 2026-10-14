# HANDOFF v7.8 §3 — 실험 6 진단 보고서

**생성일**: 2026-07-25 10:22
**셀**: (σ_g=0.041, fwhm=1.25)  epochs=60  모델=DeepLabV3+/ResNet50(ImageNet)
**GRV 산식**: UNet(Phase5) 동일 — Var(g)/E[|g|]²
**유능성 기준**: TEST_ID F1 ≥ 0.8

## 결과

| lr | seed | TEST_ID | TEST_OOD | diverged | COMPETENT |
|-----|------|---------|----------|----------|-----------|
| 1e-04 | 849000 | 0.6896 | 0.5284 | False | **False** |
| 1e-04 | 849001 | 0.6890 | 0.5687 | False | **False** |
| 5e-04 | 849100 | 0.2597 | 0.1531 | False | **False** |
| 5e-04 | 849101 | 0.8171 | 0.5516 | False | **True** |
| 1e-03 | 849200 | 0.2664 | 0.1645 | False | **False** |
| 1e-03 | 849201 | 0.8167 | 0.5765 | False | **True** |

## 판정

**COMPETENT_FOUND** — best_lr=0.0005
  → 본실행 (run_exp6_v2.py --lr 0.0005) 진행

*HANDOFF v7.8 §3 진단 완료*
