# -*- coding: utf-8 -*-
"""
Dataset-level F1 재계산 (리뷰 지적 2.1 대응) — 2026-08-20

입력 모드:
  (A) --counts-csv  : config,test_set,run_id,image_id,class_id,tp,fp,fn
  (B) --pred-dir    : 예측 argmax npy + --label-dir
  (C) --results-dir : 기존 results.json 디렉토리에서 per-run f1_by_class 읽기 (현재 사용)

출력 (out/ 디렉토리):
  f1_dataset_level_runs.csv    run 단위 per-class F1 + macro + defect macro
  f1_dataset_level_stats.csv   조건×testset 단위 mean/sd/n + 원고 대비
  f1_contrasts.csv             headline 대비 Welch t / p / 95%CI / Cohen's d
"""
import argparse, os, sys, glob, math, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = [0, 1, 2, 3]
DEFECTS = [1, 2, 3]

PAPER_VALUES = {
    ("D_calib",   "TEST_ID"):  0.876,  ("D_calib",   "TEST_OOD"): 0.787,  ("D_calib",   "TEST_CAL"): 0.917,
    ("D_manual",  "TEST_ID"):  0.927,  ("D_manual",  "TEST_OOD"): 0.596,  ("D_manual",  "TEST_CAL"): 0.585,
    ("AB-1",      "TEST_ID"):  0.735,  ("AB-1",      "TEST_OOD"): 0.815,  ("AB-1",      "TEST_CAL"): 0.828,
    ("AB-2+3",    "TEST_ID"):  0.904,  ("AB-2+3",    "TEST_OOD"): 0.492,  ("AB-2+3",    "TEST_CAL"): 0.544,
    ("D_v5e",     "TEST_ID"):  0.909,  ("D_v5e",     "TEST_OOD"): 0.682,  ("D_v5e",     "TEST_CAL"): 0.663,
    ("D_sigma_p", "TEST_ID"):  0.794,  ("D_sigma_p", "TEST_OOD"): 0.756,  ("D_sigma_p", "TEST_CAL"): 0.926,
    ("z-score",   "TEST_ID"):  0.939,  ("z-score",   "TEST_OOD"): 0.581,  ("z-score",   "TEST_CAL"): 0.603,
}

CONTRASTS = [
    ("D_calib",   "D_manual",  "TEST_ID"),
    ("D_calib",   "D_manual",  "TEST_OOD"),
    ("D_calib",   "D_manual",  "TEST_CAL"),
    ("AB-1",      "D_manual",  "TEST_OOD"),
    ("D_calib",   "AB-1",      "TEST_ID"),
    ("D_calib",   "AB-1",      "TEST_OOD"),
    ("AB-2+3",    "D_manual",  "TEST_OOD"),
    ("D_v5e",     "D_calib",   "TEST_OOD"),
    ("D_sigma_p", "D_calib",   "TEST_ID"),
]

PHASE4 = Path("/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4")

# results.json 경로와 config 이름 매핑
RESULTS_SOURCES = [
    ("D_calib",   PHASE4 / "p42r_v2_runs" / "calib"),
    ("D_manual",  PHASE4 / "p42r_v2_runs" / "manual"),
    ("AB-1",      PHASE4 / "p43_runs" / "ab1"),
    ("AB-2+3",    PHASE4 / "p43_runs" / "ab2"),
    ("AB-2+3",    PHASE4 / "p43_runs" / "ab3"),
    ("D_v5e",     PHASE4 / "v77" / "exp3b_runs"),
    ("D_sigma_p", PHASE4 / "v77" / "exp3_runs"),
    ("z-score",   PHASE4 / "v77" / "exp4_runs"),
]

TS_MAP = {"test_id": "TEST_ID", "test_ood": "TEST_OOD", "test_cal": "TEST_CAL"}


def f1_from_counts(tp, fp, fn):
    d = 2 * tp + fp + fn
    return float("nan") if d == 0 else 2 * tp / d


def load_from_results_json():
    """results.json의 f1_by_class 읽기 → rows"""
    rows = []
    for cfg_name, run_root in RESULTS_SOURCES:
        run_dirs = sorted(run_root.glob("run*")) if run_root.exists() else []
        if not run_dirs:
            print(f"  WARN: {run_root} — run 없음")
            continue
        for run_dir in run_dirs:
            rj = run_dir / "results.json"
            if not rj.exists():
                print(f"  WARN: {rj} 없음")
                continue
            d = json.load(open(rj))
            run_id = f"{run_root.name}_{run_dir.name}"
            for ts_raw, ts_up in TS_MAP.items():
                ts_data = d.get(ts_raw, {})
                f1bc = ts_data.get("f1_by_class")
                if f1bc is None:
                    print(f"  WARN: {cfg_name}/{run_id}/{ts_raw}: f1_by_class 없음")
                    continue
                f1s = {c: float(f1bc[c]) for c in CLASSES}
                rec0 = float("nan")
                rows.append(dict(
                    config=cfg_name, test_set=ts_up, run_id=run_id,
                    f1_c0=f1s[0], f1_c1=f1s[1], f1_c2=f1s[2], f1_c3=f1s[3],
                    macro4=float(np.nanmean([f1s[c] for c in CLASSES])),
                    defect_macro=float(np.nanmean([f1s[c] for c in DEFECTS])),
                    background_f1=f1s[0], background_recall=rec0,
                ))
    return rows


_CONFIG_REMAP = {"AB-2": "AB-2+3", "AB-3": "AB-2+3"}

def load_counts_csv(path):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cfg = _CONFIG_REMAP.get(row["config"], row["config"])
            k = (cfg, row["test_set"], row["run_id"])
            c = int(row["class_id"])
            a = acc[k][c]
            a[0] += int(row["tp"]); a[1] += int(row["fp"]); a[2] += int(row["fn"])
    rows = []
    for (cfg, ts, run), cc in sorted(acc.items()):
        f1s = {c: f1_from_counts(*cc[c]) for c in CLASSES}
        rec0 = cc[0][0] / (cc[0][0] + cc[0][2]) if (cc[0][0] + cc[0][2]) else float("nan")
        rows.append(dict(config=cfg, test_set=ts, run_id=run,
                         f1_c0=f1s[0], f1_c1=f1s[1], f1_c2=f1s[2], f1_c3=f1s[3],
                         macro4=float(np.nanmean([f1s[c] for c in CLASSES])),
                         defect_macro=float(np.nanmean([f1s[c] for c in DEFECTS])),
                         background_f1=f1s[0], background_recall=rec0))
    return rows


def load_pred_dirs(pred_dir, label_dir):
    def arr(p):
        a = np.load(p)
        return a["arr_0"] if hasattr(a, "files") else a
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for cfg in sorted(os.listdir(pred_dir)):
        for ts in sorted(os.listdir(os.path.join(pred_dir, cfg))):
            labels = {}
            for lp in glob.glob(os.path.join(label_dir, ts, "label_*.np*")):
                idx = os.path.basename(lp).split("_")[1].split(".")[0]
                labels[idx] = arr(lp).astype(np.int64)
            for run in sorted(os.listdir(os.path.join(pred_dir, cfg, ts))):
                k = (cfg, ts, run)
                for pp in glob.glob(os.path.join(pred_dir, cfg, ts, run, "pred_*.np*")):
                    idx = os.path.basename(pp).split("_")[1].split(".")[0]
                    if idx not in labels:
                        sys.exit(f"label missing for {pp}")
                    pred = arr(pp).astype(np.int64); lab = labels[idx]
                    for c in CLASSES:
                        p_c = pred == c; g = lab == c
                        a = acc[k][c]
                        a[0] += int(np.logical_and(p_c, g).sum())
                        a[1] += int(np.logical_and(p_c, ~g).sum())
                        a[2] += int(np.logical_and(~p_c, g).sum())
    return load_counts_from_acc(acc)


def load_counts_from_acc(acc):
    rows = []
    for (cfg, ts, run), cc in sorted(acc.items()):
        f1s = {c: f1_from_counts(*cc[c]) for c in CLASSES}
        rec0 = cc[0][0] / (cc[0][0] + cc[0][2]) if (cc[0][0] + cc[0][2]) else float("nan")
        rows.append(dict(config=cfg, test_set=ts, run_id=run,
                         f1_c0=f1s[0], f1_c1=f1s[1], f1_c2=f1s[2], f1_c3=f1s[3],
                         macro4=float(np.nanmean([f1s[c] for c in CLASSES])),
                         defect_macro=float(np.nanmean([f1s[c] for c in DEFECTS])),
                         background_f1=f1s[0], background_recall=rec0))
    return rows


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, float("nan"), float("nan"), float("nan"), (float("nan"), float("nan")), float("nan")
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
        ci = stats.t.ppf(0.975, df) * se
    except ImportError:
        from math import erfc
        p = erfc(abs(t) / math.sqrt(2))
        ci = 1.96 * se
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (ma - mb) / sp if sp > 0 else float("nan")
    return ma - mb, t, df, p, (ma - mb - ci, ma - mb + ci), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-csv")
    ap.add_argument("--pred-dir"); ap.add_argument("--label-dir")
    ap.add_argument("--use-existing", action="store_true",
                    help="results.json의 f1_by_class 사용 (기본값)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    if args.counts_csv:
        rows = load_counts_csv(args.counts_csv)
        mode = "counts-csv"
    elif args.pred_dir and args.label_dir:
        rows = load_pred_dirs(args.pred_dir, args.label_dir)
        mode = "pred-dir"
    else:
        rows = load_from_results_json()
        mode = "results-json (f1_by_class)"

    print(f"입력 모드: {mode}")
    print(f"총 {len(rows)}개 행 로드됨")

    if not rows:
        sys.exit("데이터 없음 — 경로 확인 필요")

    os.makedirs(args.out, exist_ok=True)

    # ── runs CSV ──────────────────────────────────────────────────────────────
    with open(os.path.join(args.out, "f1_dataset_level_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── stats CSV + 콘솔 출력 ─────────────────────────────────────────────────
    per = defaultdict(list)
    for r in rows:
        per[(r["config"], r["test_set"])].append(r)

    print("\n=== 조건별 macro4 (현재 vs 원고) ===")
    with open(os.path.join(args.out, "f1_dataset_level_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "test_set", "n", "macro4_mean", "macro4_sd",
                    "defect_mean", "defect_sd", "bg_mean", "bg_sd",
                    "paper_macro4", "delta_vs_paper"])
        for (cfg, ts), rs in sorted(per.items()):
            m4 = np.array([r["macro4"] for r in rs])
            df = np.array([r["defect_macro"] for r in rs])
            bg = np.array([r["background_f1"] for r in rs])
            pv = PAPER_VALUES.get((cfg, ts))
            delta = f"{m4.mean()-pv:+.4f}" if pv is not None else ""
            w.writerow([cfg, ts, len(rs),
                        f"{m4.mean():.4f}", f"{m4.std(ddof=1):.4f}",
                        f"{df.mean():.4f}", f"{df.std(ddof=1):.4f}",
                        f"{bg.mean():.4f}", f"{bg.std(ddof=1):.4f}",
                        pv if pv is not None else "", delta])
            if pv is not None:
                print(f"  {cfg:>10} {ts:<9} {m4.mean():.4f} ± {m4.std(ddof=1):.4f}"
                      f"  paper={pv:.3f}  Δ={m4.mean()-pv:+.4f}")

    # ── contrasts CSV ─────────────────────────────────────────────────────────
    print("\n=== headline 대비 ===")
    with open(os.path.join(args.out, "f1_contrasts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_a", "config_b", "test_set",
                    "mean_a", "mean_b", "delta",
                    "welch_t", "df", "p", "ci_lo", "ci_hi", "cohens_d"])
        for ca, cb, ts in CONTRASTS:
            A = [r["macro4"] for r in per.get((ca, ts), [])]
            B = [r["macro4"] for r in per.get((cb, ts), [])]
            if not A or not B:
                print(f"  SKIP {ca} vs {cb} @ {ts}: n_A={len(A)}, n_B={len(B)}")
                continue
            dlt, t, df, p, (lo, hi), d = welch(A, B)
            ma, mb = np.mean(A), np.mean(B)
            p_str = f"{p:.3e}" if not math.isnan(p) else "nan"
            print(f"  {ca:>10} - {cb:<10} @ {ts:<9}: Δ={dlt:+.4f}  t={t:.2f}  p={p_str}  d={d:.2f}")
            w.writerow([ca, cb, ts,
                        f"{ma:.4f}", f"{mb:.4f}", f"{dlt:+.4f}",
                        f"{t:.3f}" if not math.isnan(t) else "nan",
                        f"{df:.1f}" if not math.isnan(df) else "nan",
                        p_str,
                        f"{lo:+.4f}" if not math.isnan(lo) else "nan",
                        f"{hi:+.4f}" if not math.isnan(hi) else "nan",
                        f"{d:.3f}" if not math.isnan(d) else "nan"])

    print(f"\n출력: {args.out}/")
    print("  f1_dataset_level_runs.csv")
    print("  f1_dataset_level_stats.csv")
    print("  f1_contrasts.csv")


if __name__ == "__main__":
    main()
