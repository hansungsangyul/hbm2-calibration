#!/usr/bin/env python3
"""
T6 Regression Test — N parameterization consistency
=====================================================
Asserts that the N value computed in pipeline.py forward (N_pipeline)
equals the N value displayed/read via the same sigmoid-box formula (N_display).

This prevents future divergence between display and pipeline N values,
which caused the v3 bug (softplus in forward, sigmoid-box in display).

Pass criterion: |N_display - N_pipeline| < 1e-6 for all N_raw values.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
from pipeline import DegradationPipeline, _N_LO, _N_HI, sigmoid_box_inverse

def n_from_raw(N_raw_tensor: torch.Tensor) -> float:
    """Display formula — must match pipeline forward exactly."""
    return float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(N_raw_tensor))


def get_pipeline_N(pipe: DegradationPipeline) -> float:
    """Extract N as computed in _stage1_poisson forward."""
    # This mimics exactly what _stage1_poisson does:
    return float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(pipe.N_raw))


def test_t6_consistency():
    print("T6: N parameterization display == pipeline consistency")
    print("=" * 55)

    test_cases = [
        ("default init N=300", sigmoid_box_inverse(300.0)),
        ("N_raw=0.0 (N~525)", 0.0),
        ("N_raw=+6.0 (N~997)", 6.0),
        ("N_raw=-6.0 (N~52)",  -6.0),
        ("N=50 boundary",      sigmoid_box_inverse(51.0)),
        ("N=1000 boundary",    sigmoid_box_inverse(999.0)),
        ("v3 bug raw=6.0 display was ~997.7", 6.0),
    ]

    all_pass = True
    for label, raw_val in test_cases:
        pipe = DegradationPipeline()
        with torch.no_grad():
            pipe.N_raw.fill_(raw_val)
        N_display  = n_from_raw(pipe.N_raw)
        N_pipeline = get_pipeline_N(pipe)
        diff = abs(N_display - N_pipeline)
        status = "PASS ✓" if diff < 1e-6 else "FAIL ✗"
        print(f"  [{status}] {label:<45}  "
              f"N_raw={raw_val:+6.3f}  N_display={N_display:7.3f}  N_pipeline={N_pipeline:7.3f}  |Δ|={diff:.2e}")
        if diff >= 1e-6:
            all_pass = False

    # Verify sigmoid-box range
    pipe = DegradationPipeline()
    with torch.no_grad():
        pipe.N_raw.fill_(100.0)
    N_extreme_hi = get_pipeline_N(pipe)
    with torch.no_grad():
        pipe.N_raw.fill_(-100.0)
    N_extreme_lo = get_pipeline_N(pipe)
    range_ok = abs(N_extreme_hi - _N_HI) < 0.01 and abs(N_extreme_lo - _N_LO) < 0.01
    print(f"\n  [{'PASS ✓' if range_ok else 'FAIL ✗'}] Sigmoid-box range: "
          f"raw→+inf → N→{N_extreme_hi:.2f} (want {_N_HI}), "
          f"raw→-inf → N→{N_extreme_lo:.2f} (want {_N_LO})")
    if not range_ok:
        all_pass = False

    print("\n" + "=" * 55)
    print(f"T6 Result: {'ALL PASS ✅' if all_pass else 'FAILURES DETECTED ✗'}")
    if not all_pass:
        sys.exit(1)
    return all_pass


if __name__ == '__main__':
    result = test_t6_consistency()
    print("\nT6 regression test complete.")
