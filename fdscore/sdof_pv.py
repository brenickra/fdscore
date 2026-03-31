from __future__ import annotations

import numpy as np
from .sdof_transfer import h_baseacc_to_x, build_transfer_matrix


def build_pv_transfer_matrix(fs: float, n: int, f0: np.ndarray, zeta: float) -> np.ndarray:
    """Backwards-compatible PV transfer matrix builder."""
    return build_transfer_matrix(fs=float(fs), n=int(n), f0_hz=np.asarray(f0, dtype=float), zeta=float(zeta), metric="pv")
