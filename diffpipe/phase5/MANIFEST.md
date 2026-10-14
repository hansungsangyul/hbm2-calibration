# Phase 5 MANIFEST — UNet Phase Diagram

## Purpose
σ_g × fwhm 그리드에서 UNet (HBM segmentation) 수렴/발산 위상도 생성.

## Grid (사전 등록 2026-07-08, configs/grid_v1.json)
- σ_g: 5 log-uniform pts [0.01025, 0.0205, 0.041, 0.082, 0.164] — center=0.041 (manual, v4 BOUNDARY_PINNED 분기)
- fwhm: 5 log-uniform pts [0.625, 1.25, 2.5, 5.0, 10.0]
- f=0 column: 5 cells (G2 조건, log 그리드 불포함)
- Total: 30 cells × 5 seeds = 150 runs
- **§Q 업데이트 견적: ~112h ≈ 4.7일** (파일럿 §Q 분석 기반, 2026-07-09)
  - **epoch 절단: 41** (plateau_epoch=21 + 20버퍼, cummax 20-epoch 개선 < 0.005 기준)
  - λ_max: 5회/run (epochs 5,10,20,30,40), 평균 117s; 일반: 36회 × 58.6s
  - 총 run당: ~45min; 150 runs: **112h** (1 GPU) / **56h** (2 GPU) — target ≤150h ✅
  - 이전 견적: 259h (100 epoch 기준) — §Q epoch 절단으로 달성

## Seeds
- Formula: seed = 950000 + cell_idx * 10 + rep_idx
- cell_idx: row-major order (σ_g-outer, fwhm-inner), f=0 column last

## Fixed Parameters
See configs/grid_v1.json fixed_params section.
- N=300 (manual, non-identifiable from v4 BOUNDARY_PINNED)
- alpha=0.525, in_low=0.003, in_high=0.854 (v3 consensus)
- ring_strength=0.009, ring_n=8.6

## Divergence Criterion (§5.2, immutable — §Q에서 "문구 그대로 유지" 지시)
1. loss NaN or Inf
2. Final F1 < 0.5
3. Last 10 epoch F1 variation > 0.1 — **epoch ≥ 20 이후에만 적용** (warmup: 초기 수렴 중 F1 0→0.9 상승은 false positive 발생)
   - 파일럿 4차 실행에서 확인: Epoch 10에서 false positive 발생 → 구현 버그 수정 (임계값/창 불변, warmup 추가)
   - **파일럿 최종 실행 false positive (epoch 57)**: val_loss 단발 스파이크 (1.0146, 전형 0.29-0.46). train_loss 지속 하강 (0.021). 창 47-56 범위=0.019 << 0.1. 真발산 아님. 기준 유지 (§Q 지시).

## Diagnostic Logging (§5.3)
- Gradient relative variance: every epoch
- λ_max (U-Net training loss Hessian max eigenvalue): **every 10 epochs** (§P 파일럿 실측 완화)
  - Method: **§P micro-batch HVP** — 32 images total, micro_bs=4, power_iter=10, early_stop=1%
  - seed 고정 950000 (전 run 동일 배치 — 비교 가능성)
  - Overhead: 97% → 10-epoch 주기 (pre-authorized). Amortized: ~9.7%/epoch
  - **batch=4 legacy DISCARDED**: §P 비교 결과 142.7% 편차 (4524.5 vs 1864.3) → DISCARD_BATCH4
- Frequency error decomposition: every 5 epochs

## λ_max Clarification (v3 report confusion fix)
λ_max = U-Net **training loss**의 파라미터 공간 Hessian 최대고유값.
캘리브레이션 Jacobian 아님. v3 CC 보고의 혼선이 HANDOFF v4에서 정정됨.

## Stage 2 Plan
1단계 완료 후: 발산율 0↔100% 전환 경계 인접 셀 8-12개 (5 seeds).
배치안 → 검증 인스턴스 승인 → 실행.

## Pilot Run (지시 15 §L) — **COMPLETE** (2026-07-09)
- Script: `run_phase5_pilot.py`
- Cell: σ_g=0.041, fwhm=2.5 (중앙 셀), seed=950000
- Output: `pilot_output/pilot_result_sg0.04100_fw2.500_s950000.json`
- Status: **COMPLETE** — 57 epochs run, criterion 3 false positive (epoch 57 val_loss spike)
- Peak F1: 0.9427 (epoch 54)
- GPU fix: `PYTORCH_CUDA_ALLOC_ALLOC_CONF=expandable_segments:True` (메모리 단편화 해결)

### §Q Plateau Analysis
- method: cummax rolling 20-epoch improvement < 0.005
- plateau_epoch: **21** (cummax[E41]-cummax[E21]=0.003 < 0.005)
- **new_epochs: 41** (plateau+20)
- 150-run estimate: **112h** (1 GPU) / **56h** (2 GPU)

### §P λ_max Comparison
| 방법 | λ_max | 시간 | 판정 |
|------|-------|------|------|
| batch=4 (legacy) | 4524.5 | 10.5s | **DISCARD** |
| micro-batch-32 (§P) | 1864.3 | 51.6s | canonical |
| 편차 | 142.7% | — | DISCARD_BATCH4 |

### λ_max batch 조정 이력
- Spec: 256 images
- 1차 실패: v4와 GPU 경쟁 (31.5 GiB 점유) → OOM (Epoch 1 training backward)
- 2차 실패: batch=64, create_graph=True → Epoch 5 λ_max 계산 시 OOM (HVP 연산 그래프 ~30 GiB)
- 3차: batch=4 (단기 임시) + adaptive OOM 복구
- **§P 최종**: micro-batch 32 (8×4) HVP 누적 — batch=4 DISCARDED (142.7% 편차)

## Cluster GPU Check (§Q 항목 4)
- GPU 목록: Tesla V100-PCIE-32GB, Quadro P620
- 훈련 가능: Tesla V100 (32GB) × 1
- Quadro P620: 메모리 부족 (~2GB, UNet 미지원)
- **병렬 훈련 불가**: 단일 V100 — 150 runs 직렬 (wall-clock = 112h)

## Config Hash
grid_v1.json registered 2026-07-08 before any Phase 5 execution.
§Q epoch 업데이트: 2026-07-09 (epochs 100→41, estimate 259h→112h)
(No git repo — MANIFEST serves as pre-registration record)

## Environment
- conda:unet (PyTorch 2.5.1, CUDA)
- Required env var: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (micro-batch HVP 메모리 단편화 방지)
- Script: `run_phase5_pilot.py` → `run_phase5_grid.py` (150 runs, §Q epoch=41 반영 필요)

## 최종 보고 추가 요건 (HANDOFF v6.8 §3 지시 37, 2026-07-12)

기존 요건(run별 results.json, 집계 CSV, 발산율/F1-std heatmap, G2·G4 마킹, A1–A3 사전 등록 분석)에 추가:

1. **연속 지표 heatmap 2종 추가**:
   - (a) 셀별 final_f1 std — 전작 G2 σ=0.068과 직접 비교 축
   - (b) 셀별 진동 진폭: medfilt-3 후 마지막 10 epoch max−min의 rep 평균
   - 근거: 발산 전건 crit 3(소프트 F1 진동)이므로 이진 발산율 단독으로는 정보 손실

2. **A1 미리보기 결과 반영**: 83 runs 탐색적 결과(r_pb=0.4544, GRV>60 혼동 행렬) 이미 interim_report에 포함. 본 분석은 150 runs 전체로 별도 수행, ROC 최적 임계 재산출.

3. **crit 3 전건 명시**: crit 3 전건·crit 1/2 0건 사실을 최종 보고에 명시 — "발산"의 조작적 정의가 학습 붕괴가 아닌 F1 진동임을 논문에 정확히 기술하기 위함.

## 체크포인트 저장 추가 (HANDOFF v6.6 §1 지시 32, 2026-07-12)
- `run_phase5_grid.py`에 `torch.save(model.state_dict(), out_dir/'final.pth')` 추가
- **적용 시작 run**: sg2_fw4/rep4 (cell_index=16, rep=4) — 이전 83 runs 미보존 (수용)
- 이미 완료된 83 runs: results.json만 존재, final.pth 없음 (Phase 5는 사후 모델 평가 불요 — 보험 목적)
- 이후 67 runs: results.json + final.pth 저장
