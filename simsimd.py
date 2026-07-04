"""Compatibility shim for environments where the native simsimd wheel cannot load.

`albucore` 0.0.23/0.0.24 imports `simsimd` only for `wsum`. On ARM hosts,
the native wheel can fail to load `libgomp` with a static TLS error. This
module shadows the wheel on `sys.path` and provides the small subset of the
API needed by CanopyRS inference.
"""

from __future__ import annotations

import numpy as np

print("Using pure-Python simsimd shim", flush=True)


def _clip_for_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(values, info.min, info.max)
    return values


def wsum(a, b, *, alpha: float = 1.0, beta: float = 1.0) -> bytes:
    """Return a weighted sum encoded as a bytes buffer.

    The native `simsimd.wsum` return value is consumed by `numpy.frombuffer`
    in `albucore`, so the shim returns raw bytes with the input dtype.
    """

    left = np.asarray(a)
    right = np.asarray(b)

    if left.shape != right.shape:
        raise ValueError(f"wsum expects matching shapes, got {left.shape} and {right.shape}")

    dtype = left.dtype
    result = left.astype(np.float64, copy=False) * alpha + right.astype(np.float64, copy=False) * beta
    result = _clip_for_dtype(result, dtype)
    result = np.asarray(result, dtype=dtype, order="C")
    return result.tobytes()


__all__ = ["wsum"]