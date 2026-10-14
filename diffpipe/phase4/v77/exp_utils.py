"""
실험 전반 공유 유틸리티 — run_exp*.py, report_*.py에서 import.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from exp_config import PRENORM_A, PRENORM_B


def apply_prenorm(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] 이미지에 캘리브레이션 prenorm 아핀 적용."""
    return PRENORM_A * img + PRENORM_B


def make_3ch_tensor(img: np.ndarray, prenorm: bool = False) -> torch.Tensor:
    """
    HxW float32 이미지 → (1,3,H,W) float32 tensor.
    prenorm=True이면 아핀 적용 후 변환.
    """
    if prenorm:
        img = apply_prenorm(img)
    img_3ch = np.stack([img] * 3, axis=0).astype(np.float32)
    return torch.from_numpy(img_3ch).unsqueeze(0)


def compute_grv(model: nn.Module) -> float:
    """
    GRV = Var(g) / E[|g|]²  (UNet Phase5와 동일 산식)

    model.backward() 호출 직후 grad가 채워진 상태에서 호출.
    모든 파라미터 기울기를 1D concat 후 요소별 분산 / 절댓값평균².

    주의: run_exp6_arch.py(v7.7)의 compute_grv는 산식이 다름 — 사용 금지.
         (std(‖∇‖₂)/mean(‖∇‖₂)는 배치별 노름의 변동계수이며 값 범위 0.08~0.11)
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().cpu().float().flatten())
    if not grads:
        return float('nan')
    g = torch.cat(grads)
    mean_abs = g.abs().mean().item()
    if mean_abs < 1e-12:
        return float('nan')
    return float(g.var().item() / (mean_abs ** 2))


def compute_f1_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = 4,
) -> list[float]:
    """(N,C,H,W) logits + (N,H,W) int targets → 클래스별 F1 리스트."""
    preds_flat = preds.argmax(dim=1).flatten()
    tgts_flat  = targets.flatten()
    f1s = []
    for c in range(n_classes):
        tp = ((preds_flat == c) & (tgts_flat == c)).sum().float()
        fp = ((preds_flat == c) & (tgts_flat != c)).sum().float()
        fn = ((preds_flat != c) & (tgts_flat == c)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append((2 * prec * rec / (prec + rec + 1e-8)).item())
    return f1s
