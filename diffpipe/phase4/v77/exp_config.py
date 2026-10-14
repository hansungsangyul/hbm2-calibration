"""
실험 전반 공유 설정값 — 모든 run_exp*.py, report_*.py는 여기서 import.
값을 바꿀 일이 생기면 이 파일 한 곳만 수정한다.
"""

# ── prenorm 아핀 파라미터 ─────────────────────────────────────────────────────
# 출처: Phase4 캘리브레이션 결과 (A=0.501834, B=0.173320)
PRENORM_A: float = 0.501834
PRENORM_B: float = 0.173320

# ── D_calib 기준 성능 (판정표 기준선) ──────────────────────────────────────────
# 출처: v7.7 §0 D_calib 기준 (report_immediate.md)
D_CALIB_ID:  float = 0.8757   # TEST_ID F1
D_CALIB_OOD: float = 0.787    # TEST_OOD F1

# ── exp3b OOD 결손 임계 ────────────────────────────────────────────────────────
# 출처: HANDOFF v7.7 §3b 판정표  결손 > 0.03 → INFERIOR
OOD_DEFICIT_THRESHOLD: float = 0.03

# ── exp6 유능성 기준 (사전 등록) ───────────────────────────────────────────────
# TEST_ID F1 ≥ 0.80 이어야 GRV 비교에 유효한 run으로 인정
COMPETENCE_THRESHOLD: float = 0.80

# ── exp6 셀 정의 ───────────────────────────────────────────────────────────────
# 두 버전이 seed_base가 다르므로 분리. 혼용 금지.

# v7.7 run_exp6_arch.py 용 (결과: exp6_runs/, exp6_results.csv)
CELLS_ARCH = [
    {'cell_id': 0, 'sigma_g': 0.0205, 'fwhm': 1.25, 'seed_base': 840000},
    {'cell_id': 1, 'sigma_g': 0.0205, 'fwhm': 5.0,  'seed_base': 841000},
    {'cell_id': 2, 'sigma_g': 0.041,  'fwhm': 1.25, 'seed_base': 842000},
    {'cell_id': 3, 'sigma_g': 0.041,  'fwhm': 5.0,  'seed_base': 843000},
]

# v7.8 run_exp6_v2.py 용 (결과: exp6_v2_runs/)
# HANDOFF v7.8에서 "v7과 다른 seed range"로 명시
CELLS_V2 = [
    {'cell_id': 0, 'sigma_g': 0.0205, 'fwhm': 1.25, 'seed_base': 850000},
    {'cell_id': 1, 'sigma_g': 0.0205, 'fwhm': 5.0,  'seed_base': 851000},
    {'cell_id': 2, 'sigma_g': 0.041,  'fwhm': 1.25, 'seed_base': 852000},
    {'cell_id': 3, 'sigma_g': 0.041,  'fwhm': 5.0,  'seed_base': 853000},
]

# ── exp2 조건별 prenorm 적용 여부 ──────────────────────────────────────────────
# True  → 추론 전 PRENORM_A * img + PRENORM_B 적용 (calib 모델 학습 조건 일치)
# False → 원본 픽셀 그대로 추론 (manual 모델 학습 조건 일치)
USES_PRENORM = {
    'calib':  True,    # D_calib 모델, prenorm 학습
    'manual': False,   # D_manual 모델, 원본 학습
    'ab1':    True,    # D_manual + prenorm 추론
    'ab2':    False,
    'ab3':    False,
    'exp3':   True,    # D_calib + σ_post, prenorm 학습
}

# ── exp6 Mann-Whitney 유의수준 ─────────────────────────────────────────────────
MW_ALPHA: float = 0.05

# ── ESF 허용 범위 (exp7) ───────────────────────────────────────────────────────
ESF_FWHM_RANGE = (1.9, 2.3)
