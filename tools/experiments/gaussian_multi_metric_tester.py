"""Gaussian synthetic tester for metric and inversion cross-checks.

This script creates one synthetic Gaussian base-acceleration signal and evaluates, for
selected SDOF metrics:
1) Time-domain rainflow FDS
2) Spectral Dirlik FDS from time history (time -> Welch -> Dirlik)
3) Spectral Dirlik FDS from explicit PSD input

Inversion policy:
- Iterative spectral inversion: all metrics
- Closed-form inversion: PV only

The script is intentionally outside the public API and writes outputs under:
  tools/experiments/_outputs_gaussian_metric_check
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Avoid interactive windows by default for batch execution.
SHOW_PLOTS = os.getenv("SHOW_PLOTS", "0").strip().lower() in ("1", "true", "yes", "on")
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fdscore import (
    IterativeInversionParams,
    PSDParams,
    SDOFParams,
    SNParams,
    ValidationError,
    compute_fds_spectral_psd,
    compute_fds_spectral_time,
    compute_fds_time,
    compute_psd_metrics,
    compute_psd_welch,
    invert_fds_closed_form,
    invert_fds_iterative_spectral,
)
from fdscore.preprocess import preprocess_signal


OUT_DIR = Path(os.getenv("GAUSS_TEST_OUT_DIR", SCRIPT_DIR / "_outputs_gaussian_metric_check"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = int(os.getenv("RNG_SEED", "12345"))
FS_HZ = float(os.getenv("FS_HZ", "1000.0"))
DURATION_S = float(os.getenv("DURATION_S", "180.0"))
TARGET_RMS_G = float(os.getenv("TARGET_RMS_G", "0.65"))
PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "linear")

P_SCALE = float(os.getenv("P_SCALE", "6500.0"))
TEST_DURATION_S = float(os.getenv("TEST_DURATION_S", str(DURATION_S)))
DEFAULT_METRICS = ("pv", "disp", "vel", "acc")

ITER_ITERS = int(os.getenv("ITER_ITERS", "8"))
ITER_GAMMA = float(os.getenv("ITER_GAMMA", "0.80"))
ITER_GAIN_MIN = float(os.getenv("ITER_GAIN_MIN", "0.20"))
ITER_GAIN_MAX = float(os.getenv("ITER_GAIN_MAX", "5.00"))
ITER_SMOOTH_WIN = int(os.getenv("ITER_SMOOTH_WIN", "7"))

SN = SNParams(
    slope_k=10.0,
    ref_stress=120.0,
    ref_cycles=1_000_000.0,
    amplitude_from_range=True,
)
SDOF_BASE = SDOFParams(
    q=10.0,
    metric="pv",
    fmin=5.0,
    fmax=400.0,
    df=1.0,
)
PSD = PSDParams(
    method="welch",
    window="hann",
    nperseg=8192,
    noverlap=4096,
    detrend="constant",
    scaling="density",
    fmin=5.0,
    fmax=400.0,
)


def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace(":", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def save_csv(path: Path, header: str, cols: list[np.ndarray]) -> None:
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="")


def parse_metrics_env() -> list[str]:
    raw = os.getenv("TEST_METRICS", "").strip()
    metrics = list(DEFAULT_METRICS) if not raw else [m.strip().lower() for m in raw.split(",") if m.strip()]

    allowed = {"pv", "disp", "vel", "acc"}
    for metric in metrics:
        if metric not in allowed:
            raise RuntimeError(f"Invalid metric '{metric}'. Allowed: {sorted(allowed)}")

    dedup: list[str] = []
    seen = set()
    for metric in metrics:
        if metric not in seen:
            dedup.append(metric)
            seen.add(metric)
    return dedup


def build_sdof(metric: str) -> SDOFParams:
    return SDOFParams(
        q=float(SDOF_BASE.q),
        metric=metric,
        fmin=float(SDOF_BASE.fmin) if SDOF_BASE.fmin is not None else None,
        fmax=float(SDOF_BASE.fmax) if SDOF_BASE.fmax is not None else None,
        df=float(SDOF_BASE.df) if SDOF_BASE.df is not None else None,
        f=SDOF_BASE.f,
    )


def build_iter_params() -> IterativeInversionParams:
    return IterativeInversionParams(
        iters=int(ITER_ITERS),
        gamma=float(ITER_GAMMA),
        gain_min=float(ITER_GAIN_MIN),
        gain_max=float(ITER_GAIN_MAX),
        alpha_sharpness=1.0,
        floor=1e-30,
        smooth_enabled=(int(ITER_SMOOTH_WIN) > 1),
        smooth_window_bins=max(1, int(ITER_SMOOTH_WIN)),
        smooth_every_n_iters=1,
    )


def synthesize_gaussian_signal(*, fs_hz: float, duration_s: float, rms_target_g: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(float(fs_hz) * float(duration_s)))
    f = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))

    shape = (
        0.45 * np.exp(-0.5 * ((f - 35.0) / 18.0) ** 2)
        + 1.00 * np.exp(-0.5 * ((f - 95.0) / 28.0) ** 2)
        + 0.75 * np.exp(-0.5 * ((f - 210.0) / 52.0) ** 2)
        + 0.30 * np.exp(-0.5 * ((f - 320.0) / 45.0) ** 2)
        + 1e-6
    )

    white = rng.standard_normal(n)
    w_fft = np.fft.rfft(white)
    x = np.fft.irfft(w_fft * np.sqrt(shape), n=n)

    x = x - float(np.mean(x))
    x_std = float(np.std(x))
    if x_std <= 0.0:
        raise RuntimeError("Synthetic signal standard deviation is zero.")

    x = x * (float(rms_target_g) / x_std)
    return x.astype(float, copy=False)


def gaussianity_metrics(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sig = float(np.std(x))
    if sig <= 0.0:
        raise RuntimeError("Invalid signal std=0 for Gaussianity metrics.")
    z = (x - mu) / sig
    return {
        "mean": mu,
        "rms": float(np.sqrt(np.mean(x**2))),
        "std": sig,
        "skewness": float(np.mean(z**3)),
        "kurtosis_pearson": float(np.mean(z**4)),
    }


def median_abs_pct(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.clip(np.abs(b), 1e-30, None)
    return float(np.median(np.abs((a - b) / den)) * 100.0)


def median_abs_log10(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = (a > 0.0) & (b > 0.0)
    if not np.any(mask):
        return float("nan")
    return float(np.median(np.abs(np.log10(a[mask]) - np.log10(b[mask]))))


def interp_logsafe(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    y = np.interp(np.asarray(x_new, dtype=float), np.asarray(x_old, dtype=float), np.asarray(y_old, dtype=float))
    return np.clip(y, 1e-30, None)


def warmup_numba() -> None:
    """Warm up JIT paths before timing-heavy loops."""
    t = np.arange(0.0, 1.0, 1.0 / FS_HZ)
    x = 0.01 * np.sin(2.0 * np.pi * 20.0 * t)
    sdof = SDOFParams(q=10.0, metric="pv", fmin=20.0, fmax=40.0, df=10.0)
    _ = compute_fds_time(x=x, fs=FS_HZ, sn=SN, sdof=sdof, p_scale=P_SCALE, detrend="none")


def save_synth_time_history(x_raw: np.ndarray, x_proc: np.ndarray) -> None:
    t = np.arange(x_raw.size, dtype=float) / float(FS_HZ)
    save_csv(
        OUT_DIR / "TimeHistory_Sintetizado.csv",
        "time_s,acc_base_g_raw,acc_base_g_preprocessed",
        [t, np.asarray(x_raw, dtype=float), np.asarray(x_proc, dtype=float)],
    )


def save_fds_plot(metric_dir: Path, metric: str, f: np.ndarray, d_rf: np.ndarray, d_dt: np.ndarray, d_dp: np.ndarray) -> None:
    plt.figure(figsize=(10.0, 6.0))
    plt.loglog(f, d_rf, label=f"Rainflow ({metric})", linewidth=2.0)
    plt.loglog(f, d_dt, label=f"Dirlik time ({metric})", linewidth=1.8, alpha=0.9)
    plt.loglog(f, d_dp, label=f"Dirlik PSD ({metric})", linewidth=1.2, linestyle="--", alpha=0.95)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("FDS damage [-]")
    plt.title(f"FDS overlay | metric={metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_dir / "plot_FDS_overlay.png", dpi=140)


def save_psd_plot(
    metric_dir: Path,
    metric: str,
    f_seed: np.ndarray,
    p_seed: np.ndarray,
    f_iter: np.ndarray,
    p_iter: np.ndarray,
    f_closed: np.ndarray | None,
    p_closed: np.ndarray | None,
) -> None:
    plt.figure(figsize=(10.0, 6.0))
    plt.loglog(f_seed, np.clip(p_seed, 1e-30, None), label=f"Seed PSD ({metric})", linewidth=1.2, alpha=0.8)
    plt.loglog(f_iter, p_iter, label=f"Iterative PSD ({metric})", linewidth=2.0)
    if f_closed is not None and p_closed is not None:
        plt.loglog(f_closed, p_closed, label="Closed-form PSD (pv baseline)", linewidth=1.6, linestyle="--")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [(input-unit)^2/Hz]")
    plt.title(f"PSD inversion outputs | metric={metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_dir / "plot_PSD_inversion_outputs.png", dpi=140)


def summarize_psd(prefix: str, f: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = compute_psd_metrics(
        p,
        f_hz=f,
        duration_s=float(TEST_DURATION_S),
        acc_unit="g",
    )
    return {
        f"{prefix}_rms_acc_g": float(m.rms_acc_g),
        f"{prefix}_peak_acc_g": float(m.peak_acc_g),
        f"{prefix}_rms_vel_m_s": float(m.rms_vel_m_s),
        f"{prefix}_rms_disp_mm": float(m.rms_disp_mm),
    }


def run_metric_case(
    *,
    metric: str,
    x: np.ndarray,
    f_psd_seed: np.ndarray,
    psd_seed: np.ndarray,
    iter_params: IterativeInversionParams,
) -> dict[str, Any]:
    sdof = build_sdof(metric)
    metric_dir = OUT_DIR / f"metric_{_safe_name(metric)}"
    metric_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    fds_rainflow = compute_fds_time(x=x, fs=FS_HZ, sn=SN, sdof=sdof, p_scale=P_SCALE, detrend="none")
    t1 = time.perf_counter()

    fds_dirlik_time = compute_fds_spectral_time(
        x=x,
        fs=FS_HZ,
        sn=SN,
        sdof=sdof,
        psd=PSD,
        duration_s=DURATION_S,
        p_scale=P_SCALE,
    )
    t2 = time.perf_counter()

    fds_dirlik_psd = compute_fds_spectral_psd(
        f_psd_hz=f_psd_seed,
        psd_baseacc=psd_seed,
        duration_s=DURATION_S,
        sn=SN,
        sdof=sdof,
        p_scale=P_SCALE,
    )
    t3 = time.perf_counter()

    f_fds = np.asarray(fds_rainflow.f, dtype=float)
    d_rf = np.asarray(fds_rainflow.damage, dtype=float)
    d_dt = np.asarray(fds_dirlik_time.damage, dtype=float)
    d_dp = np.asarray(fds_dirlik_psd.damage, dtype=float)

    save_csv(
        metric_dir / "FDS_compare.csv",
        "freq_hz,fds_rainflow,fds_dirlik_time,fds_dirlik_psd",
        [f_fds, d_rf, d_dt, d_dp],
    )
    save_fds_plot(metric_dir, metric, f_fds, d_rf, d_dt, d_dp)

    fds_metrics = {
        "fds_med_abs_pct_dirlik_time_vs_rainflow": median_abs_pct(d_dt, d_rf),
        "fds_med_abs_pct_dirlik_psd_vs_rainflow": median_abs_pct(d_dp, d_rf),
        "fds_med_abs_log10_dirlik_time_vs_rainflow": median_abs_log10(d_dt, d_rf),
        "fds_med_abs_log10_dirlik_psd_vs_rainflow": median_abs_log10(d_dp, d_rf),
        "fds_med_abs_pct_dirlik_time_vs_psd": median_abs_pct(d_dt, d_dp),
        "fds_med_abs_log10_dirlik_time_vs_psd": median_abs_log10(d_dt, d_dp),
    }

    iter_psd = invert_fds_iterative_spectral(
        target=fds_rainflow,
        f_psd_hz=f_psd_seed,
        psd_seed=np.clip(psd_seed, 1e-30, None),
        duration_s=DURATION_S,
        sn=SN,
        sdof=sdof,
        p_scale=P_SCALE,
        params=iter_params,
    )
    t4 = time.perf_counter()

    f_iter = np.asarray(iter_psd.f, dtype=float)
    p_iter = np.asarray(iter_psd.psd, dtype=float)

    fds_iter_recon = compute_fds_spectral_psd(
        f_psd_hz=f_iter,
        psd_baseacc=p_iter,
        duration_s=DURATION_S,
        sn=SN,
        sdof=sdof,
        p_scale=P_SCALE,
    )
    t5 = time.perf_counter()

    d_iter_recon = np.asarray(fds_iter_recon.damage, dtype=float)

    save_csv(
        metric_dir / "FDS_recon_iterative_vs_target.csv",
        "freq_hz,target_fds_rainflow,fds_recon_from_iterative_psd",
        [f_fds, d_rf, d_iter_recon],
    )

    iter_metrics = {
        "iter_best_err_log10": float(iter_psd.meta.get("diagnostics", {}).get("best_err", float("nan"))),
        "iter_recon_fds_med_abs_pct_vs_target": median_abs_pct(d_iter_recon, d_rf),
        "iter_recon_fds_med_abs_log10_vs_target": median_abs_log10(d_iter_recon, d_rf),
    }

    f_closed = None
    p_closed = None
    closed_metrics = {
        "closed_form_enabled": metric == "pv",
        "closed_vs_iter_psd_med_abs_pct": float("nan"),
        "closed_vs_iter_psd_med_abs_log10": float("nan"),
        "closed_vs_seed_psd_med_abs_pct": float("nan"),
    }

    if metric == "pv":
        psd_closed = invert_fds_closed_form(fds_rainflow, test_duration_s=TEST_DURATION_S, strict_metric=True)
        f_closed = np.asarray(psd_closed.f, dtype=float)
        p_closed = np.asarray(psd_closed.psd, dtype=float)

        p_iter_on_closed = interp_logsafe(f_closed, f_iter, p_iter)
        p_seed_on_closed = interp_logsafe(f_closed, f_psd_seed, psd_seed)

        closed_metrics["closed_vs_iter_psd_med_abs_pct"] = median_abs_pct(p_iter_on_closed, p_closed)
        closed_metrics["closed_vs_iter_psd_med_abs_log10"] = median_abs_log10(p_iter_on_closed, p_closed)
        closed_metrics["closed_vs_seed_psd_med_abs_pct"] = median_abs_pct(p_seed_on_closed, p_closed)

        save_csv(
            metric_dir / "PSD_closed_form_baseline.csv",
            "freq_hz,psd_closed_form_from_rainflow_fds",
            [f_closed, p_closed],
        )

    t6 = time.perf_counter()

    if f_closed is not None and p_closed is not None:
        p_seed_on_closed = interp_logsafe(f_closed, f_psd_seed, psd_seed)
        p_iter_on_closed = interp_logsafe(f_closed, f_iter, p_iter)
        save_csv(
            metric_dir / "PSD_inversion_outputs.csv",
            "freq_hz,psd_seed_input,psd_iterative,psd_closed_form_pv_baseline",
            [f_closed, p_seed_on_closed, p_iter_on_closed, p_closed],
        )
        psd_summary = {
            **summarize_psd("seed", f_closed, p_seed_on_closed),
            **summarize_psd("iter", f_closed, p_iter_on_closed),
            **summarize_psd("closed", f_closed, p_closed),
        }
    else:
        p_seed_on_iter = interp_logsafe(f_iter, f_psd_seed, psd_seed)
        save_csv(
            metric_dir / "PSD_inversion_outputs.csv",
            "freq_hz,psd_seed_input,psd_iterative",
            [f_iter, p_seed_on_iter, p_iter],
        )
        psd_summary = {
            **summarize_psd("seed", f_iter, p_seed_on_iter),
            **summarize_psd("iter", f_iter, p_iter),
        }

    save_psd_plot(metric_dir, metric, f_psd_seed, psd_seed, f_iter, p_iter, f_closed, p_closed)

    return {
        "metric": metric,
        **fds_metrics,
        **iter_metrics,
        **closed_metrics,
        **psd_summary,
        "time_fds_rainflow_s": t1 - t0,
        "time_fds_dirlik_time_s": t2 - t1,
        "time_fds_dirlik_psd_s": t3 - t2,
        "time_iterative_inversion_s": t4 - t3,
        "time_iterative_recon_s": t5 - t4,
        "time_closed_form_s": t6 - t5,
        "time_total_metric_s": t6 - t0,
    }


def save_combined_psd_plot(metrics: list[str]) -> None:
    metric_colors = {
        "pv": "tab:blue",
        "disp": "tab:orange",
        "vel": "tab:green",
        "acc": "tab:red",
    }

    plt.figure(figsize=(12.0, 7.0))
    pv_closed_added = False

    for metric in metrics:
        p_csv = OUT_DIR / f"metric_{_safe_name(metric)}" / "PSD_inversion_outputs.csv"
        if not p_csv.exists():
            continue

        arr = np.loadtxt(p_csv, delimiter=",", skiprows=1)
        if arr.ndim != 2:
            continue

        color = metric_colors.get(metric)
        if arr.shape[1] >= 3:
            plt.loglog(arr[:, 0], arr[:, 2], color=color, linewidth=1.9, alpha=0.9, label=f"Iterative {metric}")
        if metric == "pv" and arr.shape[1] >= 4 and not pv_closed_added:
            plt.loglog(arr[:, 0], arr[:, 3], color="black", linewidth=2.0, linestyle="--", label="Closed-form pv")
            pv_closed_added = True

    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [(input-unit)^2/Hz]")
    plt.title("PSD inversion outputs | all metrics")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_PSD_inversion_all_metrics.png", dpi=140)


def save_cross_metric_iterative_psd_csv(metrics: list[str]) -> Path | None:
    """Compare iterative PSD outputs against PV baseline on a common frequency grid."""
    if "pv" not in metrics:
        return None

    p_ref = OUT_DIR / "metric_pv" / "PSD_inversion_outputs.csv"
    if not p_ref.exists():
        return None

    ref = np.loadtxt(p_ref, delimiter=",", skiprows=1)
    if ref.ndim != 2 or ref.shape[1] < 3:
        return None

    f_ref = ref[:, 0]
    p_ref_iter = np.clip(ref[:, 2], 1e-30, None)

    rows: list[tuple[str, float, float, float, float]] = []
    for metric in metrics:
        p_csv = OUT_DIR / f"metric_{_safe_name(metric)}" / "PSD_inversion_outputs.csv"
        if not p_csv.exists():
            continue
        arr = np.loadtxt(p_csv, delimiter=",", skiprows=1)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        p_metric = interp_logsafe(f_ref, arr[:, 0], arr[:, 2])
        abs_rel = np.abs((p_metric - p_ref_iter) / np.clip(p_ref_iter, 1e-30, None))
        med_pct = float(np.median(abs_rel) * 100.0)
        q90_pct = float(np.quantile(abs_rel, 0.90) * 100.0)
        q99_pct = float(np.quantile(abs_rel, 0.99) * 100.0)
        med_log = median_abs_log10(p_metric, p_ref_iter)
        rows.append((metric, med_pct, q90_pct, q99_pct, med_log))

    out = OUT_DIR / "PSD_iterative_cross_metric_vs_pv.csv"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("metric,med_abs_pct_vs_pv,q90_abs_pct_vs_pv,q99_abs_pct_vs_pv,med_abs_log10_vs_pv\n")
        for metric, med_pct, q90_pct, q99_pct, med_log in rows:
            fh.write(f"{metric},{med_pct},{q90_pct},{q99_pct},{med_log}\n")
    return out


def save_summary_csv(rows: list[dict[str, Any]]) -> Path:
    summary_csv = OUT_DIR / "metrics_comparison_summary.csv"
    if not rows:
        return summary_csv

    keys = list(rows[0].keys())
    with summary_csv.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            vals: list[str] = []
            for key in keys:
                value = row.get(key)
                if isinstance(value, bool):
                    vals.append("true" if value else "false")
                else:
                    vals.append(str(value))
            fh.write(",".join(vals) + "\n")

    return summary_csv


def save_summary_text(rows: list[dict[str, Any]], metrics: list[str], gm: dict[str, float]) -> None:
    lines = [
        "Gaussian synthetic tester: FDS + iterative inversion for all metrics",
        f"seed={RNG_SEED}",
        f"metrics={metrics}",
        f"fs_hz={FS_HZ}",
        f"duration_s={DURATION_S}",
        f"target_rms_g={TARGET_RMS_G}",
        f"preprocess_mode={PREPROCESS_MODE}",
        f"p_scale={P_SCALE}",
        f"test_duration_s={TEST_DURATION_S}",
        f"ITER_ITERS={ITER_ITERS}",
        f"ITER_GAMMA={ITER_GAMMA}",
        f"ITER_GAIN_MIN={ITER_GAIN_MIN}",
        f"ITER_GAIN_MAX={ITER_GAIN_MAX}",
        f"ITER_SMOOTH_WIN={ITER_SMOOTH_WIN}",
        f"SN={asdict(SN)}",
        f"SDOF_BASE={asdict(SDOF_BASE)}",
        f"PSD={asdict(PSD)}",
        "",
        "Signal Gaussianity:",
        f"  mean={gm['mean']:.6g}",
        f"  rms={gm['rms']:.6g}",
        f"  std={gm['std']:.6g}",
        f"  skewness={gm['skewness']:.6g}",
        f"  kurtosis_pearson={gm['kurtosis_pearson']:.6g}",
        "",
    ]

    for row in rows:
        metric = row["metric"]
        lines.append(f"Metric={metric}")
        lines.append(
            "  FDS med abs pct | "
            f"Dirlik-time vs Rainflow={float(row['fds_med_abs_pct_dirlik_time_vs_rainflow']):.6f}, "
            f"Dirlik-PSD vs Rainflow={float(row['fds_med_abs_pct_dirlik_psd_vs_rainflow']):.6f}, "
            f"Dirlik-time vs Dirlik-PSD={float(row['fds_med_abs_pct_dirlik_time_vs_psd']):.12f}"
        )
        lines.append(
            "  FDS med abs log10 | "
            f"Dirlik-time vs Rainflow={float(row['fds_med_abs_log10_dirlik_time_vs_rainflow']):.6e}, "
            f"Dirlik-PSD vs Rainflow={float(row['fds_med_abs_log10_dirlik_psd_vs_rainflow']):.6e}, "
            f"Dirlik-time vs Dirlik-PSD={float(row['fds_med_abs_log10_dirlik_time_vs_psd']):.6e}"
        )
        lines.append(
            "  Iterative recon (spectral predictor) | "
            f"best_err_log10={float(row['iter_best_err_log10']):.6e}, "
            f"med_abs_pct_vs_target={float(row['iter_recon_fds_med_abs_pct_vs_target']):.6f}, "
            f"med_abs_log10_vs_target={float(row['iter_recon_fds_med_abs_log10_vs_target']):.6e}"
        )
        if bool(row["closed_form_enabled"]):
            lines.append(
                "  PV closed baseline vs iterative PSD | "
                f"med_abs_pct={float(row['closed_vs_iter_psd_med_abs_pct']):.6f}, "
                f"med_abs_log10={float(row['closed_vs_iter_psd_med_abs_log10']):.6e}"
            )
        else:
            lines.append("  Closed-form inversion: not applied (PV-only policy).")
        lines.append(
            "  Timings [s] | "
            f"rainflow={float(row['time_fds_rainflow_s']):.3f}, "
            f"dirlik_time={float(row['time_fds_dirlik_time_s']):.3f}, "
            f"dirlik_psd={float(row['time_fds_dirlik_psd_s']):.3f}, "
            f"iter={float(row['time_iterative_inversion_s']):.3f}, "
            f"recon={float(row['time_iterative_recon_s']):.3f}, "
            f"total={float(row['time_total_metric_s']):.3f}"
        )
        lines.append("")

    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_compat_aliases() -> None:
    pv_dir = OUT_DIR / "metric_pv"
    if not pv_dir.exists():
        return

    fds_csv = pv_dir / "FDS_compare.csv"
    if fds_csv.exists():
        (OUT_DIR / "FDS_compare_rainflow_vs_dirlik.csv").write_text(fds_csv.read_text(encoding="utf-8"), encoding="utf-8")

    closed_csv = pv_dir / "PSD_closed_form_baseline.csv"
    if closed_csv.exists():
        (OUT_DIR / "PSD_compare_from_closed_form_inversion.csv").write_text(closed_csv.read_text(encoding="utf-8"), encoding="utf-8")

    fds_png = pv_dir / "plot_FDS_overlay.png"
    if fds_png.exists():
        (OUT_DIR / "plot_FDS_overlay.png").write_bytes(fds_png.read_bytes())


def main() -> None:
    metrics = parse_metrics_env()
    warmup_numba()

    x_raw = synthesize_gaussian_signal(
        fs_hz=FS_HZ,
        duration_s=DURATION_S,
        rms_target_g=TARGET_RMS_G,
        seed=RNG_SEED,
    )
    x = preprocess_signal(x_raw, mode=PREPROCESS_MODE)
    gm = gaussianity_metrics(x)
    save_synth_time_history(x_raw, x)

    f_psd_seed, psd_seed = compute_psd_welch(x, fs=FS_HZ, psd=PSD)
    save_csv(
        OUT_DIR / "PSD_input_signal_welch.csv",
        "freq_hz,psd_input_signal",
        [f_psd_seed, psd_seed],
    )

    iter_params = build_iter_params()

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        print(f"Running metric={metric} ...")
        try:
            rows.append(
                run_metric_case(
                    metric=metric,
                    x=x,
                    f_psd_seed=f_psd_seed,
                    psd_seed=psd_seed,
                    iter_params=iter_params,
                )
            )
        except ValidationError as exc:
            print(f"Metric={metric} failed with ValidationError: {exc}")
            raise

    save_combined_psd_plot(metrics)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")

    summary_csv = save_summary_csv(rows)
    cross_metric_csv = save_cross_metric_iterative_psd_csv(metrics)
    save_summary_text(rows, metrics, gm)
    write_compat_aliases()

    print("\nGaussian multi-metric tester completed.")
    print(f"Outputs: {OUT_DIR}")
    print(f"Summary CSV: {summary_csv}")
    if cross_metric_csv is not None:
        print(f"Cross-metric PSD CSV: {cross_metric_csv}")


if __name__ == "__main__":
    main()
