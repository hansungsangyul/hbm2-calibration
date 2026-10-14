#!/usr/bin/env python3
"""
HANDOFF v7.7 즉시군 통합 보고서 — 실험 1·2·5·7
===============================================
exp1_summary.json, exp2_summary.json, exp5_summary.json, exp7_summary.json을
읽어서 HANDOFF v7.7 판정표를 적용한 마크다운 보고서를 작성한다.

출력: V77_DIR / 'report_immediate.md'
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from datetime import datetime

V77_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77')
V77_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = V77_DIR / 'report_immediate.md'


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def load_json(path: Path):
    """파일이 없으면 None 반환 (PENDING 처리용)."""
    if not path.exists():
        print(f'  [PENDING] {path} not found')
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f'  [ERROR] Failed to load {path}: {e}')
        return None


def fmt(v, precision=4):
    if v is None or (isinstance(v, float) and (v != v)):  # nan
        return 'N/A'
    return f'{v:.{precision}f}'


# ── 각 실험 summary 로드 ──────────────────────────────────────────────────────
print('[report_immediate] 즉시군 summary 파일 로드...')
exp1 = load_json(V77_DIR / 'exp1_summary.json')
exp2 = load_json(V77_DIR / 'exp2_summary.json')
exp5 = load_json(V77_DIR / 'exp5_summary.json')
exp7 = load_json(V77_DIR / 'exp7_summary.json')

lines = []
lines.append(f'# HANDOFF v7.7 즉시군 통합 보고서')
lines.append(f'')
lines.append(f'**생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
lines.append(f'**포함 실험**: 실험 1(추론 정규화), 실험 2(실영상 FPR), '
             f'실험 5(ISO 프로파일), 실험 7(ESF 완화 민감도)')
lines.append(f'')
lines.append(f'---')
lines.append(f'')


# ── 실험 1: 추론 시 정규화 재평가 ─────────────────────────────────────────────
lines.append(f'## 실험 1 — 추론 시 정규화 재평가 (M4)')
lines.append(f'')
lines.append(f'**판정표 (HANDOFF v7.7 §1)**:')
lines.append(f'- gap = calib_ID − manual_ID (정규화 모드별)')
lines.append(f'- gap > −0.01  → **ID_RESOLVED** (ID 분포이동 문제 없음)')
lines.append(f'- −0.03 < gap ≤ −0.01 → **ID_PARTIAL** (부분 트레이드오프)')
lines.append(f'- gap ≤ −0.03 → **TRADEOFF_REAL** (실제 트레이드오프)')
lines.append(f'')

if exp1 is None:
    lines.append(f'> **STATUS: PENDING** — exp1_summary.json 없음')
else:
    verdicts = exp1.get('verdicts', {})
    per_cond = exp1.get('per_condition_mode', {})
    modes = ['orig', 'zscore', 'prenorm']
    lines.append(f'| 정규화 모드 | calib ID F1 | manual ID F1 | gap | 판정 |')
    lines.append(f'|------------|-------------|--------------|-----|------|')
    for mode in modes:
        v = verdicts.get(mode, {})
        lines.append(
            f'| {mode:8s} | {fmt(v.get("calib_mean_id"))} | '
            f'{fmt(v.get("manual_mean_id"))} | '
            f'{fmt(v.get("gap_calib_minus_manual"), 4)} | '
            f'**{v.get("verdict", "N/A")}** |'
        )
    lines.append(f'')
    # 핵심 판정: prenorm 모드
    pn_verdict = verdicts.get('prenorm', {}).get('verdict', 'N/A')
    lines.append(f'**핵심 결론**: prenorm 모드 판정 = **{pn_verdict}**')
    if pn_verdict == 'ID_RESOLVED':
        lines.append(f'  → "Self-Calibrating 프레임 유지: prenorm 추론 시 ID 분포이동 문제 없음"')
    elif pn_verdict == 'ID_PARTIAL':
        lines.append(f'  → "부분 트레이드오프 존재 — 추가 분석 필요"')
    elif pn_verdict == 'TRADEOFF_REAL':
        lines.append(f'  → "실제 트레이드오프 확인 — 재프레임 필요"')

lines.append(f'')
lines.append(f'---')
lines.append(f'')


# ── 실험 2: 실영상 무결함 위양성률 ────────────────────────────────────────────
lines.append(f'## 실험 2 — 실영상 무결함 위양성률 (M5)')
lines.append(f'')
lines.append(f'**판정표 (HANDOFF v7.7 §2)**:')
lines.append(f'- Mann-Whitney U, calib vs manual (n_false_components)')
lines.append(f'- calib < manual & p<0.05  → **CALIB_LOWER** (캘리브 모델이 FPR 낮음)')
lines.append(f'- p ≥ 0.05                 → **EQUIVALENT** (차이 없음)')
lines.append(f'- calib > manual & p<0.05  → **CALIB_HIGHER** (캘리브 모델이 FPR 높음)')
lines.append(f'')

if exp2 is None:
    lines.append(f'> **STATUS: PENDING** — exp2_summary.json 없음')
else:
    verdict_e2 = exp2.get('verdict', 'N/A')
    stats      = exp2.get('stats', {})
    lines.append(f'| 지표 | calib (mean) | manual (mean) | p-value | 판정 |')
    lines.append(f'|------|--------------|---------------|---------|------|')

    per_cond   = exp2.get('per_condition', {})
    mw         = exp2.get('mannwhitney_calib_vs_manual', {})
    fpr_calib  = per_cond.get('calib',  {}).get('median_pixel_fpr', None)
    fpr_manual = per_cond.get('manual', {}).get('median_pixel_fpr', None)
    fc_calib   = mw.get('calib_median_nfc',  None)
    fc_manual  = mw.get('manual_median_nfc', None)
    pval       = mw.get('p_value', None)

    lines.append(f'| pixel_fpr | {fmt(fpr_calib)} | {fmt(fpr_manual)} | — | — |')
    lines.append(f'| n_false_comp | {fmt(fc_calib, 2)} | {fmt(fc_manual, 2)} | '
                 f'{fmt(pval, 4) if pval is not None else "N/A"} | **{verdict_e2}** |')
    lines.append(f'')
    lines.append(f'**핵심 결론**: **{verdict_e2}**')

lines.append(f'')
lines.append(f'---')
lines.append(f'')


# ── 실험 5: ISO 프로파일 (iso_profile) ─────────────────────────────────────────
lines.append(f'## 실험 5 — ISO 프로파일 (M6)')
lines.append(f'')
lines.append(f'**판정표 (HANDOFF v7.7 §5)**:')
lines.append(f'- fwhm(σ_g=0.0205) < fwhm(σ_g=0.041) → **MONOTONE** (ISO 단조성 확인)')
lines.append(f'- 역전 또는 차이 없음 → **FLAT** (ISO 단조성 불확인)')
lines.append(f'')

if exp5 is None:
    lines.append(f'> **STATUS: PENDING** — exp5_summary.json 없음')
else:
    verdict_e5 = exp5.get('verdict', 'N/A')
    lines.append(f'| 항목 | 값 |')
    lines.append(f'|------|-----|')
    for k, v in exp5.items():
        if k not in ('verdict',):
            lines.append(f'| {k} | {v} |')
    lines.append(f'')
    lines.append(f'**핵심 결론**: **{verdict_e5}**')

lines.append(f'')
lines.append(f'---')
lines.append(f'')


# ── 실험 7: ESF 완화 기준 민감도 ──────────────────────────────────────────────
lines.append(f'## 실험 7 — ESF 완화 기준 민감도 (M9, CPU)')
lines.append(f'')
lines.append(f'**판정표 (HANDOFF v7.7 §7)**:')
lines.append(f'- R²·min_fits 완화 전 기준에서 f_ESF ∈ [1.9, 2.3] → **ROBUST**')
lines.append(f'- 이탈 기준 존재 → **SENSITIVE** + 이탈 조건 목록')
lines.append(f'')

if exp7 is None:
    lines.append(f'> **STATUS: PENDING** — exp7_summary.json 없음')
else:
    verdict_e7  = exp7.get('verdict', 'N/A')
    baseline_d  = exp7.get('baseline', {})
    baseline    = baseline_d.get('aggregate_fwhm_median', None)
    conditions  = exp7.get('conditions', [])
    n_cond_tot  = len(conditions)
    n_cond_ok   = sum(1 for c in conditions if c.get('in_range', False))
    esf_range   = exp7.get('esf_accept_range', [1.9, 2.3])
    outliers    = exp7.get('deviant_conditions', [])

    lines.append(f'| 항목 | 값 |')
    lines.append(f'|------|-----|')
    lines.append(f'| 기준 FWHM (R²=0.95, min_fits=20) | {fmt(baseline) if baseline is not None else "N/A"} |')
    lines.append(f'| 허용 범위 | [{esf_range[0]}, {esf_range[1]}] |')
    lines.append(f'| 범위 내 조건 수 | {n_cond_ok}/{n_cond_tot} |')
    lines.append(f'| 판정 | **{verdict_e7}** |')
    if outliers:
        lines.append(f'')
        lines.append(f'이탈 조건:')
        for o in outliers:
            r2  = o.get('r2_thresh', '?')
            mf  = o.get('min_fits', '?')
            fw  = o.get('agg_fwhm', None)
            lines.append(f'  - R²={r2}, min_fits={mf}: f_ESF={fmt(fw) if fw is not None else "N/A"}')
    lines.append(f'')
    lines.append(f'**핵심 결론**: **{verdict_e7}**')
    if verdict_e7 == 'ROBUST':
        lines.append(f'  → "ESF 측정값이 기준 완화에 robust — f≈2.1 신뢰도 유지"')
    else:
        lines.append(f'  → "ESF 민감도 존재 — 이탈 조건 검토 필요"')

lines.append(f'')
lines.append(f'---')
lines.append(f'')


# ── 종합 판정 요약 ─────────────────────────────────────────────────────────────
lines.append(f'## 즉시군 종합 판정 요약')
lines.append(f'')
lines.append(f'| 실험 | 상태 | 판정 | 의미 |')
lines.append(f'|------|------|------|------|')

def _get_verdict(summary, key='verdict'):
    if summary is None:
        return 'PENDING'
    return summary.get(key, 'N/A')

e1_main = 'PENDING' if exp1 is None else exp1.get('verdicts', {}).get('prenorm', {}).get('verdict', 'N/A')
e2_main = _get_verdict(exp2)
e5_main = _get_verdict(exp5)
e7_main = _get_verdict(exp7)

lines.append(f'| Exp1 (정규화 재평가) | {"완료" if exp1 else "대기"} | {e1_main} | prenorm 추론 시 ID 분포이동 |')
lines.append(f'| Exp2 (실영상 FPR)    | {"완료" if exp2 else "대기"} | {e2_main} | calib 모델 위양성 비교 |')
lines.append(f'| Exp5 (ISO 프로파일) | {"완료" if exp5 else "대기"} | {e5_main} | FWHM 단조성 확인 |')
lines.append(f'| Exp7 (ESF 민감도)   | {"완료" if exp7 else "대기"} | {e7_main} | ESF 측정 신뢰도 |')
lines.append(f'')
lines.append(f'---')
lines.append(f'*이 보고서는 자동 생성됩니다. PENDING 항목은 해당 실험 완료 후 재실행 하십시오.*')

# ── 저장 ──────────────────────────────────────────────────────────────────────
with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f'[report_immediate] 저장: {OUT_MD}')
print(f'  Exp1={e1_main}  Exp2={e2_main}  Exp5={e5_main}  Exp7={e7_main}')
