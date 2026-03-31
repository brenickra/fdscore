"""fdscore public API.

fdscore provides numerical tools for FDS and PSD workflows.

Public contracts are defined in `CONTRACTS.md` at the repository root.
"""

from .types import (
    SNParams,
    SDOFParams,
    PSDParams,
    IterativeInversionParams,
    FDSResult,
    PSDResult,
    FDSTimePlan,
    PSDMetricsResult,
)

from .validate import ValidationError
from .grid import build_frequency_grid
from .fds_ops import scale_fds, sum_fds
from .fds_time import compute_fds_time, prepare_fds_time_plan
from .fds_spectral import compute_fds_spectral_psd, compute_fds_spectral_time
from .psd_welch import compute_psd_welch
from .inverse_closed_form import invert_fds_closed_form
from .inverse_iterative_spectral import invert_fds_iterative_spectral
from .inverse_iterative_time import invert_fds_iterative_time
from .synth_time import synthesize_time_from_psd
from .metrics import compute_psd_metrics

__all__ = [
    "SNParams",
    "SDOFParams",
    "PSDParams",
    "IterativeInversionParams",
    "FDSResult",
    "PSDResult",
    "FDSTimePlan",
    "PSDMetricsResult",
    "ValidationError",
    "build_frequency_grid",
    "compute_fds_time",
    "prepare_fds_time_plan",
    "compute_psd_welch",
    "compute_fds_spectral_psd",
    "compute_fds_spectral_time",
    "scale_fds",
    "sum_fds",
    "invert_fds_closed_form",
    "invert_fds_iterative_spectral",
    "invert_fds_iterative_time",
    "synthesize_time_from_psd",
    "compute_psd_metrics",
]
