"""Audit the temporal design matrix implied by STGC's HashGridT.

This is a CPU-only algebra check, NOT a trained-model or benchmark evaluation.
It examines one projection/level at a fixed spatial position before flow-based
neighbor aggregation. It does not measure the rank of the final rendered model.

Source inspected:
  hubigwaterdeep/STGC-NeRF-private, main commit
  154a200e21a8b7b96e599d79e56c8fc93712cf3e, model/hash_field.py

Run:
  python check_stgc_temporal_span.py
"""
from __future__ import annotations
import argparse
import json
import numpy as np


def temporal_design(times: np.ndarray, time_knots: int = 8, feature_bases: int = 4) -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
        raise ValueError('times must be a finite one-dimensional array within [0, 1]')
    if time_knots < 2 or feature_bases < 2:
        raise ValueError('both basis counts must be at least 2')
    hats = np.maximum(1.0 - np.abs((time_knots - 1) * t[:, None]
                                  - np.arange(time_knots)[None, :]), 0.0)
    knots = np.linspace(0.0, 1.0, feature_bases)
    lagrange = np.stack([
        np.prod(np.stack([(t - knots[m]) / (knots[j] - knots[m])
                          for m in range(feature_bases) if m != j]), axis=0)
        for j in range(feature_bases)
    ], axis=1)
    if not np.allclose(hats.sum(axis=1), 1.0):
        raise AssertionError('time interpolation partition of unity failed')
    if not np.allclose(lagrange.sum(axis=1), 1.0):
        raise AssertionError('Lagrange partition of unity failed')
    return (hats[:, :, None] * lagrange[:, None, :]).reshape(len(t), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=51)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    matrix = temporal_design(np.linspace(0.0, 1.0, args.samples))
    s = np.linalg.svd(matrix, compute_uv=False)
    tol = s.max() * max(matrix.shape) * np.finfo(np.float64).eps
    result = {
        'scope': 'illustrative uniform times, one projection/level, fixed position, pre-flow',
        'standard_shape': list(matrix.shape),
        'standard_numerical_rank': int(np.count_nonzero(s > tol)),
        'tolerance': float(tol),
        'standard_singular_values': s.tolist(),
        'tied_rank8_modal_hash_span_upper_bound_when_cap_inactive': 8,
        'independent_low8_high8_modal_hash_span_upper_bound_when_cap_inactive': 16,
        'caveats': [
            'No dataset, checkpoint, CUDA forward pass or training is used.',
            'The bound is conditional on a fixed learned temporal dictionary and an inactive query-dependent RMS cap.',
            'Flow-warped spatial queries, products, nonlinear decoders and other branches can enlarge the final function class.',
            'Replacing C(x) by MLP(z(x), x) alone does not enlarge its fixed temporal dictionary span.',
            'Exact rank is not an effective-rank or conditioning diagnosis; inspect the full spectrum with actual trained models.'
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
