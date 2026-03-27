from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import shutil

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Import from sibling module ────────────────────────────────────────
# We import only the dataclass — no circular dependency
from src.pipeline.data_ingestion import IngestionResult, setup_logger

# W&B tracking
try:
    from wandb_tracker import init_run, finish_run, log_preprocessing_summary, is_enabled
except ImportError:
    def init_run(*a, **kw):    return None
    def finish_run(*a, **kw):  pass
    def log_preprocessing_summary(*a, **kw): pass
    def is_enabled(*a, **kw):  return False


# ──────────────────────────────────────────────────────────────────────
# TYPED RESULT DATACLASSES
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SplitStats:
    """Row counts and shapes for one train/val/test split."""
    n_windows: int
    X_shape:   Tuple[int, int]
    y_shape:   Tuple[int, int]


@dataclass
class PreprocessingReport:
    """Audit record for one metric's preprocessing step."""
    metric_name:       str
    processed_at:      str
    normalization:     str
    scaler_min:        float
    scaler_max:        float
    past_window:       int
    future_window:     int
    total_timesteps:   int
    total_windows:     int
    train:             dict
    val:               dict
    test:              dict
    output_files:      List[str] = field(default_factory=list)
    plot_files:        List[str] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    """Everything the model training script needs for one metric."""
    metric_name:    str
    train_X:        np.ndarray   # (N_train, past_window)
    train_y:        np.ndarray   # (N_train, future_window)
    val_X:          np.ndarray
    val_y:          np.ndarray
    test_X:         np.ndarray
    test_y:         np.ndarray
    scaler:         MinMaxScaler
    timestamps:     pd.DatetimeIndex
    scaled_full:    np.ndarray   # full normalized series (for anomaly scoring)
    report:         PreprocessingReport

def _atomic_npy_save(path: Path, array: np.ndarray):
    """
    Save a numpy array atomically.

    WHY the special suffix handling:
      np.save("myfile.tmp", array) silently writes "myfile.tmp.npy" —
      numpy always appends .npy unless the name already ends in .npy.
      So we give the temp file a name ending in .npy, which stops numpy
      adding a second extension, then move it to the final destination.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # suffix ends in .npy so numpy does NOT append a second .npy
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npy")
    try:
        os.close(fd)
        np.save(tmp, array)          # writes exactly to `tmp`
        shutil.move(tmp, str(path))  # atomic rename to final path
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_pkl_save(path: Path, obj):
    """Save pickle object atomically via tempfile in same directory."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def normalize_series(series: pd.Series,
                     train_end_idx: int,
                     metric_name: str,
                     logger: logging.Logger) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on the training portion ONLY, then transform the
    full series with that scaler.

    """
    values = series.values.reshape(-1, 1).astype(np.float64)

    # Fit ONLY on training slice
    train_values = values[:train_end_idx]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values)

    # Transform full series
    scaled_full = scaler.transform(values).flatten().astype(np.float32)

    logger.info(
        f"[{metric_name}] Normalized | "
        f"scaler range fitted on train: [{scaler.data_min_[0]:.4f}, {scaler.data_max_[0]:.4f}] "
        f"| scaled range: [{scaled_full.min():.3f}, {scaled_full.max():.3f}]"
    )
    return scaled_full, scaler


def create_windows(scaled: np.ndarray,
                   past: int,
                   future: int,
                   metric_name: str,
                   logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:

    total   = past + future
    n_win   = len(scaled) - total + 1

    if n_win <= 0:
        raise ValueError(
            f"[{metric_name}] Series too short ({len(scaled)}) for "
            f"window size past={past} + future={future}={total}. "
            f"Need at least {total + 1} timesteps."
        )

    # numpy stride trick — no data copy, just a view
    shape   = (n_win, total)
    strides = (scaled.strides[0], scaled.strides[0])
    windows = np.lib.stride_tricks.as_strided(scaled, shape=shape, strides=strides)

    X = windows[:, :past].copy()            # copy so arrays are contiguous
    y = windows[:, past:past + future].copy()

    logger.info(
        f"[{metric_name}] Windowed | "
        f"total_steps={len(scaled)} | windows={n_win} | "
        f"X={X.shape} y={y.shape}"
    )
    return X, y

def split_windows(X: np.ndarray,
                  y: np.ndarray,
                  train_r: float,
                  val_r: float,
                  metric_name: str,
                  logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    Chronological split — NEVER shuffle time series.

    In production: training data is always the oldest data. The model
    predicts FORWARD in time, so we evaluate it the same way.

    Split timeline:
      |──────── 70% train ─────────|── 15% val ──|── 15% test ──|
      t=0                         t1            t2             t_end
    ────────────────────────────────────────────────────────────────────

    Returns dict with keys: train_X, train_y, val_X, val_y, test_X, test_y
    """
    n  = len(X)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))

    splits = {
        "train_X": X[:t1],    "train_y": y[:t1],
        "val_X":   X[t1:t2],  "val_y":   y[t1:t2],
        "test_X":  X[t2:],    "test_y":  y[t2:],
    }

    logger.info(
        f"[{metric_name}] Split | "
        f"train={t1} | val={t2-t1} | test={n-t2} | total={n}"
    )
    return splits, t1   # return t1 so caller can fit scaler on same boundary


def plot_signal_qa(series: pd.Series,
                   scaled: np.ndarray,
                   splits_at: Tuple[int, int],
                   metric_name: str,
                   out_dir: Path,
                   logger: logging.Logger) -> str:
    """
    4-panel QA plot:
      Panel 1: Raw signal over time with train/val/test boundaries
      Panel 2: Normalized signal [0,1]
      Panel 3: Value distribution (histogram)
      Panel 4: Rolling mean + std (stationarity check)
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle(
        f"{metric_name} — Preprocessing QA",
        fontsize=13, fontweight="bold", y=1.01
    )

    t1_ts = series.index[splits_at[0]] if splits_at[0] < len(series) else series.index[-1]
    t2_ts = series.index[splits_at[1]] if splits_at[1] < len(series) else series.index[-1]
    ax = axes[0]
    ax.plot(series.index, series.values, color="steelblue", linewidth=0.7, label="raw")
    ax.axvline(t1_ts, color="orange",  linestyle="--", linewidth=1.2, label="train/val")
    ax.axvline(t2_ts, color="crimson", linestyle="--", linewidth=1.2, label="val/test")
    ax.set_title("Raw Signal (original units)", fontsize=10)
    ax.set_ylabel("value")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()

    ax = axes[1]
    ax.plot(series.index, scaled, color="seagreen", linewidth=0.7, label="normalized")
    ax.axvline(t1_ts, color="orange",  linestyle="--", linewidth=1.2)
    ax.axvline(t2_ts, color="crimson", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax.fill_between(series.index,
                    scaled,
                    alpha=0.15, color="seagreen")
    ax.set_title("Normalized Signal [0, 1]", fontsize=10)
    ax.set_ylabel("scaled value")
    ax.set_ylim(-0.1, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))

    ax = axes[2]
    ax.hist(series.values, bins=60, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(series.mean(), color="red",    linestyle="--", linewidth=1.2, label=f"mean={series.mean():.2f}")
    ax.axvline(series.median(), color="gold", linestyle="--", linewidth=1.2, label=f"median={series.median():.2f}")
    ax.set_title("Value Distribution", fontsize=10)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    window = min(288, len(series) // 4)    # 288 = 1 day at 5-min cadence
    roll   = series.rolling(window)
    ax = axes[3]
    ax.plot(series.index, roll.mean(), color="purple", linewidth=1.0, label=f"rolling mean (w={window})")
    ax.fill_between(
        series.index,
        roll.mean() - roll.std(),
        roll.mean() + roll.std(),
        alpha=0.25, color="purple", label="±1 std"
    )
    ax.set_title("Rolling Mean ± Std (stationarity check)", fontsize=10)
    ax.set_ylabel("value")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()

    plt.tight_layout()
    out_path = out_dir / f"{metric_name}_qa.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"[{metric_name}] QA plot → {out_path}")
    return str(out_path)


def plot_window_samples(X: np.ndarray,
                        y: np.ndarray,
                        metric_name: str,
                        past: int,
                        future: int,
                        out_dir: Path,
                        logger: logging.Logger,
                        n_samples: int = 6) -> str:
    """
    Show n_samples sliding windows evenly spaced across the dataset.
    Blue = input (past), Red dashed = target (future).
    Helps you visually confirm the windowing logic is correct.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"{metric_name} — Sliding Window Samples", fontsize=12, fontweight="bold")

    indices = np.linspace(0, len(X) - 1, n_samples, dtype=int)
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        past_steps   = np.arange(past)
        future_steps = np.arange(past, past + future)

        ax.plot(past_steps,   X[idx], color="steelblue", linewidth=1.5, label="input")
        ax.plot(future_steps, y[idx],
                color="tomato", linewidth=1.5, linestyle="--",
                marker="o", markersize=4, label="target")
        ax.axvline(past - 0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"window #{idx:,}", fontsize=9)
        ax.set_xlabel("timestep")
        if i == 0:
            ax.set_ylabel("normalized value")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    out_path = out_dir / f"{metric_name}_windows.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"[{metric_name}] Window samples → {out_path}")
    return str(out_path)


def preprocess_metric(ingestion_result: IngestionResult,
                      config: dict,
                      logger: logging.Logger) -> PreprocessingResult:
    """
    Full preprocessing pipeline for one metric.
    Accepts an IngestionResult — tightly coupled to ingestion output.
    """
    name    = ingestion_result.metric_name
    series  = ingestion_result.series
    pre_cfg = config["preprocessing"]
    art_cfg = config["artifacts"]

    past   = pre_cfg["past_window"]
    future = pre_cfg["future_window"]
    t_r    = pre_cfg["train_ratio"]
    v_r    = pre_cfg["val_ratio"]

    logger.info(f"[{name}] -- Starting preprocessing ---------------")

    out_dir     = Path(art_cfg["output_dir"])
    reports_dir = Path(art_cfg["reports_dir"])
    scaler_dir  = Path(art_cfg["scaler_dir"])
    plots_dir   = reports_dir / "plots"

    for d in [out_dir, reports_dir, scaler_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n          = len(series)
    train_end  = int(n * t_r)
    val_end    = int(n * (t_r + v_r))

    scaled_full, scaler = normalize_series(series, train_end, name, logger)


    X, y = create_windows(scaled_full, past, future, name, logger)


    splits, _ = split_windows(X, y, t_r, v_r, name, logger)


    saved_files = []
    for key, arr in splits.items():
        path = out_dir / f"{name}_{key}.npy"
        _atomic_npy_save(path, arr)
        saved_files.append(str(path))
        logger.debug(f"[{name}] Saved {path.name}  shape={arr.shape}")

    # Save full scaled series (used by anomaly engine during inference)
    full_path = out_dir / f"{name}_scaled_full.npy"
    _atomic_npy_save(full_path, scaled_full)
    saved_files.append(str(full_path))

    # Save timestamps as ISO strings (for anomaly result alignment)
    ts_path = out_dir / f"{name}_timestamps.npy"
    _atomic_npy_save(ts_path, np.array(series.index.astype(str)))
    saved_files.append(str(ts_path))

    scaler_path = scaler_dir / f"{name}_scaler.pkl"
    _atomic_pkl_save(scaler_path, scaler)
    logger.info(f"[{name}] Scaler saved → {scaler_path}")

    plot_files = []
    if art_cfg.get("save_plots", True):
        p1 = plot_signal_qa(
            series, scaled_full,
            splits_at=(train_end, val_end),
            metric_name=name,
            out_dir=plots_dir,
            logger=logger,
        )
        p2 = plot_window_samples(
            splits["train_X"], splits["train_y"],
            metric_name=name,
            past=past, future=future,
            out_dir=plots_dir,
            logger=logger,
        )
        plot_files = [p1, p2]

    n_splits = {k: v for k, v in {
        "train": len(splits["train_X"]),
        "val":   len(splits["val_X"]),
        "test":  len(splits["test_X"]),
    }.items()}

    report = PreprocessingReport(
        metric_name     = name,
        processed_at    = datetime.now(timezone.utc).isoformat(),
        normalization   = pre_cfg["normalization"],
        scaler_min      = float(scaler.data_min_[0]),
        scaler_max      = float(scaler.data_max_[0]),
        past_window     = past,
        future_window   = future,
        total_timesteps = len(series),
        total_windows   = len(X),
        train           = {"windows": n_splits["train"], "X_shape": list(splits["train_X"].shape)},
        val             = {"windows": n_splits["val"],   "X_shape": list(splits["val_X"].shape)},
        test            = {"windows": n_splits["test"],  "X_shape": list(splits["test_X"].shape)},
        output_files    = saved_files,
        plot_files      = plot_files,
    )

    report_path = reports_dir / f"{name}_preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"[{name}] Preprocessing report → {report_path}")

    logger.info(
        f"[{name}] Done | "
        f"train={n_splits['train']} | val={n_splits['val']} | test={n_splits['test']} windows"
    )

    return PreprocessingResult(
        metric_name  = name,
        train_X      = splits["train_X"],
        train_y      = splits["train_y"],
        val_X        = splits["val_X"],
        val_y        = splits["val_y"],
        test_X       = splits["test_X"],
        test_y       = splits["test_y"],
        scaler       = scaler,
        timestamps   = series.index,
        scaled_full  = scaled_full,
        report       = report,
    )


def run_preprocessing(ingestion_results: Dict[str, IngestionResult],
                      config: dict,
                      logger: logging.Logger) -> Dict[str, PreprocessingResult]:
    """
    Preprocess all ingested metrics. Returns dict metric_name → PreprocessingResult.
    """
    logger.info("=" * 65)
    logger.info("  DATA PREPROCESSING  START")
    logger.info("=" * 65)

    init_run(config, stage="preprocessing", logger=logger)

    results = {}
    for name, ing_result in ingestion_results.items():
        result = preprocess_metric(ing_result, config, logger)
        results[name] = result

    #  Summary 
    logger.info("=" * 65)
    logger.info("  PREPROCESSING SUMMARY")
    logger.info("=" * 65)
    for name, res in results.items():
        logger.info(
            f"  {name:15s} | "
            f"timesteps={res.report.total_timesteps:>5} | "
            f"windows={res.report.total_windows:>5} | "
            f"train={len(res.train_X):>5} | "
            f"val={len(res.val_X):>4} | "
            f"test={len(res.test_X):>4}"
        )
    logger.info("=" * 65)

    # Log all preprocessing reports to W&B
    if is_enabled(config):
        from dataclasses import asdict as _asdict
        log_preprocessing_summary(
            [_asdict(res.report) for res in results.values()],
            config, logger
        )

    finish_run(logger)
    return results


if __name__ == "__main__":
    import argparse
    from data_ingestion import run_ingestion, resolve_config_paths

    _script_dir  = Path(__file__).resolve().parent
    _default_cfg = _script_dir / "config" / "pipeline_config.json"
    if not _default_cfg.exists():
        _default_cfg = Path("config") / "pipeline_config.json"

    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--config", default=str(_default_cfg),
        help="Absolute or relative path to pipeline_config.json"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path.resolve()}")
        print("  Either run from the project root or pass --config <path>")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    config = resolve_config_paths(config, config_path)
    logger = setup_logger("preprocessing", config)

    ingestion_results = run_ingestion(config, logger)
    results = run_preprocessing(ingestion_results, config, logger)
    print(f"Preprocessing complete. {len(results)} metrics ready for model training.")