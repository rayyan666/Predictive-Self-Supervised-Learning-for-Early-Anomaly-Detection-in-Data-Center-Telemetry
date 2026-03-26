from __future__ import annotations

import json
import logging
import pickle
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import model architecture — needed to rebuild model from checkpoint
# We import SSLPredictor directly so anomaly_engine has no training dependency
try:
    from model import SSLPredictor, resolve_config_paths, setup_logger
except ImportError:
    from src.pipeline.model import SSLPredictor, resolve_config_paths, setup_logger

# W&B tracking
try:
    from wandb_tracker import init_run, finish_run, log_anomaly_results, is_enabled
except ImportError:
    def init_run(*a, **kw):          return None
    def finish_run(*a, **kw):        pass
    def log_anomaly_results(*a, **kw): pass
    def is_enabled(*a, **kw):        return False

@dataclass
class AnomalyReport:
    """Full audit record for one metric's anomaly detection run."""
    metric_name:       str
    scored_at:         str
    total_timesteps:   int
    scored_timesteps:  int       # total - past_window (first window has no score)
    train_timesteps:   int
    threshold_k:       float     # sigma multiplier
    threshold_value:   float     # actual threshold in normalised units
    score_mean:        float
    score_std:         float
    score_p95:         float
    score_p99:         float
    anomaly_count:     int
    anomaly_rate_pct:  float
    anomaly_csv_path:  str = ""
    plot_path:         str = ""
    anomaly_timestamps: List[str] = field(default_factory=list)


def load_model(metric_name: str,
               config: dict,
               device: torch.device) -> Tuple[SSLPredictor, dict]:
    """
    Rebuild the SSLPredictor from the saved checkpoint.

    WHY rebuild instead of just torch.load(model)?
      torch.load(model) pickles the entire class definition — fragile
      across Python versions and refactors.
      The checkpoint only stores model_state_dict (just the weights).
      We rebuild the architecture from config, then load weights.
      This is more robust and the standard production pattern.
    """
    art_cfg   = config["artifacts"]
    mdl_cfg   = config["model"]
    pre_cfg   = config["preprocessing"]

    ckpt_path = Path(art_cfg["checkpoint_dir"]) / f"{metric_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[{metric_name}] Checkpoint not found: {ckpt_path}\n"
            f"  Run model.py first to train the model."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SSLPredictor(
        input_size    = 1,
        hidden_size   = mdl_cfg["hidden_size"],
        num_layers    = mdl_cfg["num_layers"],
        dropout       = mdl_cfg["dropout"],
        future_window = pre_cfg["future_window"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()   # CRITICAL: sets dropout to 0, batchnorm to eval mode

    return model, ckpt


def score_full_series(model: SSLPredictor,
                      scaled_full: np.ndarray,
                      past_window: int,
                      future_window: int,
                      batch_size: int,
                      device: torch.device,
                      logger: logging.Logger,
                      metric_name: str) -> np.ndarray:
    model.eval()
    N      = len(scaled_full)
    total  = past_window + future_window
    scores = np.full(N, np.nan, dtype=np.float32)

    # Build all windows at once using stride trick (fast, no Python loop)
    n_windows = N - total + 1
    shape     = (n_windows, total)
    strides   = (scaled_full.strides[0], scaled_full.strides[0])
    windows   = np.lib.stride_tricks.as_strided(scaled_full, shape=shape, strides=strides)

    X_all = windows[:, :past_window].astype(np.float32)          # (n_win, past)
    y_all = windows[:, past_window:past_window+future_window].astype(np.float32)  # (n_win, fut)

    # Batch inference — faster than one sample at a time
    X_t = torch.from_numpy(X_all).unsqueeze(-1)   # (n_win, past, 1)
    y_t = torch.from_numpy(y_all)                  # (n_win, future)

    ds     = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_mse = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(X_batch)                                   # (B, future)
            # MSE per sample = mean over future steps
            mse     = ((preds - y_batch) ** 2).mean(dim=1).cpu().numpy()
            all_mse.append(mse)

    mse_array = np.concatenate(all_mse)   # (n_windows,)

    # Assign each score to the timestep at the START of the predicted window
    # Score at index i corresponds to predicting from context [i-past : i]
    for i, score in enumerate(mse_array):
        scores[i + past_window] = score

    logger.info(
        f"[{metric_name}] Scored {n_windows:,} windows | "
        f"score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}] | "
        f"mean={np.nanmean(scores):.4f}"
    )
    return scores


def compute_threshold(scores: np.ndarray,
                      train_size: int,
                      past_window: int,
                      k: float,
                      logger: logging.Logger,
                      metric_name: str) -> Tuple[float, dict]:
    # Extract valid (non-NaN) scores from the training window only
    train_scores = scores[past_window : past_window + train_size]
    train_scores = train_scores[~np.isnan(train_scores)]

    mean_s = float(np.mean(train_scores))
    std_s  = float(np.std(train_scores))
    threshold = mean_s + k * std_s

    all_valid = scores[~np.isnan(scores)]

    stats = {
        "train_score_mean": round(mean_s, 6),
        "train_score_std":  round(std_s, 6),
        "threshold_k":      k,
        "threshold":        round(threshold, 6),
        "score_mean":       round(float(np.mean(all_valid)), 6),
        "score_std":        round(float(np.std(all_valid)), 6),
        "score_p95":        round(float(np.percentile(all_valid, 95)), 6),
        "score_p99":        round(float(np.percentile(all_valid, 99)), 6),
    }

    logger.info(
        f"[{metric_name}] Threshold | "
        f"train_mean={mean_s:.4f} | train_std={std_s:.4f} | "
        f"k={k} | threshold={threshold:.4f}"
    )
    return threshold, stats

def flag_anomalies(scores: np.ndarray,
                   timestamps: np.ndarray,
                   scaled_full: np.ndarray,
                   threshold: float,
                   scaler,
                   min_gap_steps: int,
                   metric_name: str,
                   logger: logging.Logger) -> pd.DataFrame:
    df = pd.DataFrame({
        "timestamp":     pd.to_datetime(timestamps),
        "scaled_value":  scaled_full.astype(np.float64),
        "anomaly_score": scores.astype(np.float64),
    })

    # Inverse transform to original units
    df["raw_value"] = scaler.inverse_transform(
        df["scaled_value"].values.reshape(-1, 1)
    ).flatten()

    # Flag raw anomalies
    df["is_anomaly"] = (df["anomaly_score"] > threshold) & (~df["anomaly_score"].isna())

    # ── Deduplicate into events ───────────────────────────────
    # Assign an event ID to consecutive anomaly clusters
    df["anomaly_event_id"] = 0
    in_event  = False
    event_id  = 0
    last_anom = -min_gap_steps - 1

    for i, row in df.iterrows():
        if row["is_anomaly"]:
            if (i - last_anom) > min_gap_steps:
                event_id += 1
                in_event = True
            df.at[i, "anomaly_event_id"] = event_id
            last_anom = i
        else:
            if in_event and (i - last_anom) > min_gap_steps:
                in_event = False

    n_raw    = int(df["is_anomaly"].sum())
    n_events = int(df["anomaly_event_id"].max())

    logger.info(
        f"[{metric_name}] Anomalies | "
        f"flagged={n_raw} timesteps | "
        f"events={n_events} | "
        f"rate={n_raw/max(len(df),1)*100:.2f}%"
    )
    return df

def plot_anomalies(df: pd.DataFrame,
                   threshold: float,
                   metric_name: str,
                   unit: str,
                   train_size: int,
                   past_window: int,
                   out_dir: Path,
                   logger: logging.Logger) -> str:
    fig, axes = plt.subplots(4, 1, figsize=(18, 16))
    fig.suptitle(
        f"{metric_name} — Anomaly Detection Results",
        fontsize=13, fontweight="bold", y=1.01
    )

    valid_df   = df.dropna(subset=["anomaly_score"])
    anom_df    = df[df["is_anomaly"]]
    normal_df  = df[~df["is_anomaly"]]

    # Mark train/test boundary
    boundary_idx = past_window + train_size
    if boundary_idx < len(df):
        boundary_ts = df["timestamp"].iloc[boundary_idx]
    else:
        boundary_ts = df["timestamp"].iloc[-1]

    ax = axes[0]
    ax.plot(df["timestamp"], df["raw_value"],
            color="steelblue", linewidth=0.7, alpha=0.9, label="signal")
    if len(anom_df):
        ax.scatter(anom_df["timestamp"], anom_df["raw_value"],
                   color="red", s=12, zorder=5, label=f"anomaly ({len(anom_df)})")
    ax.axvline(boundary_ts, color="orange", linestyle="--",
               linewidth=1.2, label="train/test boundary")
    ax.set_title(f"Raw Signal with Anomalies — {unit}", fontsize=10)
    ax.set_ylabel(unit)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()

    ax = axes[1]
    ax.plot(valid_df["timestamp"], valid_df["anomaly_score"],
            color="purple", linewidth=0.6, alpha=0.8, label="anomaly score (MSE)")
    ax.axhline(threshold, color="red", linestyle="--",
               linewidth=1.5, label=f"threshold={threshold:.4f}")
    ax.fill_between(
        valid_df["timestamp"], valid_df["anomaly_score"], threshold,
        where=(valid_df["anomaly_score"] > threshold),
        color="red", alpha=0.25, label="anomalous region"
    )
    ax.axvline(boundary_ts, color="orange", linestyle="--", linewidth=1.2)
    ax.set_title("Anomaly Score (MSE) over Time", fontsize=10)
    ax.set_ylabel("MSE score")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate()
    ax = axes[2]
    train_scores = valid_df[valid_df["timestamp"] < boundary_ts]["anomaly_score"]
    test_scores  = valid_df[valid_df["timestamp"] >= boundary_ts]["anomaly_score"]

    ax.hist(train_scores, bins=60, color="steelblue", alpha=0.6,
            label=f"train scores (n={len(train_scores)})", density=True)
    ax.hist(test_scores, bins=60, color="tomato", alpha=0.5,
            label=f"test scores (n={len(test_scores)})", density=True)
    ax.axvline(threshold, color="red", linestyle="--",
               linewidth=1.5, label=f"threshold={threshold:.4f}")
    ax.set_title("Score Distribution — Train vs Test", fontsize=10)
    ax.set_xlabel("MSE score")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)

    ax = axes[3]
    if len(anom_df) > 0:
        # Find the event with the highest peak score
        worst_event_id = (
            df[df["anomaly_event_id"] > 0]
            .groupby("anomaly_event_id")["anomaly_score"]
            .max()
            .idxmax()
        )
        event_ts = df[df["anomaly_event_id"] == worst_event_id]["timestamp"]
        t_start  = event_ts.min() - pd.Timedelta(hours=2)
        t_end    = event_ts.max() + pd.Timedelta(hours=2)

        zoom_df   = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)]
        zoom_anom = zoom_df[zoom_df["is_anomaly"]]

        ax.plot(zoom_df["timestamp"], zoom_df["raw_value"],
                color="steelblue", linewidth=1.0, label="signal")
        ax.scatter(zoom_anom["timestamp"], zoom_anom["raw_value"],
                   color="red", s=25, zorder=5, label="anomaly")
        ax2 = ax.twinx()
        ax2.plot(zoom_df["timestamp"], zoom_df["anomaly_score"],
                 color="purple", linewidth=0.8, alpha=0.6, linestyle="--")
        ax2.axhline(threshold, color="red", linestyle=":", linewidth=1.0)
        ax2.set_ylabel("MSE score", color="purple", fontsize=8)
        ax.set_title(f"Zoom — Worst Anomaly Event (event #{worst_event_id})", fontsize=10)
        ax.set_ylabel(unit)
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, "No anomalies detected",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Zoom — No Anomalies", fontsize=10)

    plt.tight_layout()
    out_path = str(out_dir / f"{metric_name}_anomaly_results.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"[{metric_name}] Plot -> {out_path}")
    return out_path


def run_anomaly_detection(metric_name: str,
                          config: dict,
                          logger: logging.Logger) -> AnomalyReport:
    mdl_cfg   = config["model"]
    pre_cfg   = config["preprocessing"]
    ing_cfg   = config["ingestion"]
    art_cfg   = config["artifacts"]
    anom_cfg  = config["anomaly"]

    past_window   = pre_cfg["past_window"]
    future_window = pre_cfg["future_window"]
    k             = anom_cfg["threshold_k"]
    min_gap       = anom_cfg["min_gap_steps"]
    batch_size    = anom_cfg.get("score_batch_size", 512)

    processed_dir = Path(art_cfg["output_dir"])
    scaler_dir    = Path(art_cfg["scaler_dir"])
    reports_dir   = Path(art_cfg["reports_dir"])
    anomaly_dir   = Path(anom_cfg["anomaly_output_dir"])
    plots_dir     = reports_dir / "plots"

    for d in [anomaly_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit   = ing_cfg["metrics"][metric_name]["unit"]

    logger.info(f"[{metric_name}] -- Starting anomaly detection ---------------")
    logger.info(f"[{metric_name}] Device={device} | k={k} | min_gap={min_gap}")

    model, ckpt = load_model(metric_name, config, device)
    logger.info(f"[{metric_name}] Model loaded from epoch {ckpt['epoch']} | val_loss={ckpt['val_loss']:.6f}")

    scaled_full = np.load(processed_dir / f"{metric_name}_scaled_full.npy")
    timestamps  = np.load(processed_dir / f"{metric_name}_timestamps.npy", allow_pickle=True)
    scaler      = pickle.load(open(scaler_dir / f"{metric_name}_scaler.pkl", "rb"))
    train_size  = len(np.load(processed_dir / f"{metric_name}_train_X.npy"))

    logger.info(
        f"[{metric_name}] Data | "
        f"total={len(scaled_full)} | train={train_size} | "
        f"range=[{scaled_full.min():.3f}, {scaled_full.max():.3f}]"
    )
    scores = score_full_series(
        model, scaled_full, past_window, future_window,
        batch_size, device, logger, metric_name
    )

    threshold, score_stats = compute_threshold(
        scores, train_size, past_window, k, logger, metric_name
    )

    df = flag_anomalies(
        scores, timestamps, scaled_full,
        threshold, scaler, min_gap, metric_name, logger
    )

    # Full series CSV — every timestep with its score + flag
    csv_path = anomaly_dir / f"{metric_name}_anomaly_scores.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"[{metric_name}] Full scores CSV -> {csv_path}")

    flagged_only = df[df["is_anomaly"]].copy()
    flagged_csv  = anomaly_dir / f"{metric_name}_anomalies_only.csv"
    flagged_only.to_csv(flagged_csv, index=False)
    logger.info(f"[{metric_name}] Anomalies-only CSV ({len(flagged_only)} rows) -> {flagged_csv}")

    plot_path = plot_anomalies(
        df, threshold, metric_name, unit,
        train_size, past_window, plots_dir, logger
    )
    n_anom = int(df["is_anomaly"].sum())
    report = AnomalyReport(
        metric_name       = metric_name,
        scored_at         = datetime.now(timezone.utc).isoformat(),
        total_timesteps   = len(df),
        scored_timesteps  = int((~df["anomaly_score"].isna()).sum()),
        train_timesteps   = train_size,
        threshold_k       = k,
        threshold_value   = float(threshold),
        score_mean        = score_stats["score_mean"],
        score_std         = score_stats["score_std"],
        score_p95         = score_stats["score_p95"],
        score_p99         = score_stats["score_p99"],
        anomaly_count     = n_anom,
        anomaly_rate_pct  = round(n_anom / max(len(df), 1) * 100, 3),
        anomaly_csv_path  = str(csv_path),
        plot_path         = plot_path,
        anomaly_timestamps= list(df[df["is_anomaly"]]["timestamp"].astype(str)),
    )

    # Save JSON report
    report_path = reports_dir / f"{metric_name}_anomaly_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info(f"[{metric_name}] Anomaly report -> {report_path}")

    log_anomaly_results(
        metric_name     = metric_name,
        anomaly_summary = {
            "threshold":        report.threshold_value,
            "anomaly_count":    report.anomaly_count,
            "anomaly_rate_pct": report.anomaly_rate_pct,
            "score_mean":       report.score_mean,
            "score_p95":        report.score_p95,
            "score_p99":        report.score_p99,
        },
        score_plot_path = plot_path,
        config          = config,
        logger          = logger,
    )

    logger.info(
        f"[{metric_name}] Done | "
        f"anomalies={n_anom} ({report.anomaly_rate_pct:.2f}%) | "
        f"threshold={threshold:.4f}"
    )
    return report

def run_all(config: dict,
            logger: logging.Logger,
            metrics: Optional[List[str]] = None):
    """Run anomaly detection for all metrics (or a subset)."""

    all_metrics = list(config["ingestion"]["metrics"].keys())
    targets     = metrics if metrics else all_metrics

    logger.info("=" * 65)
    logger.info("  ANOMALY ENGINE  START")
    logger.info(f"  Metrics: {targets}")
    logger.info("=" * 65)

    init_run(config, stage="anomaly_detection", logger=logger)

    reports = {}
    for metric_name in targets:
        logger.info(f"\n[{metric_name}] ========================================")
        report = run_anomaly_detection(metric_name, config, logger)
        reports[metric_name] = report

    finish_run(logger)

    # ── Final summary ─────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  ANOMALY DETECTION SUMMARY")
    logger.info("=" * 65)
    logger.info(
        f"  {'Metric':15s} | {'Threshold':>10} | "
        f"{'Anomalies':>10} | {'Rate %':>8} | {'Events':>7}"
    )
    logger.info("-" * 65)
    for name, r in reports.items():
        n_events = len(set(
            pd.read_csv(r.anomaly_csv_path)
            .query("anomaly_event_id > 0")["anomaly_event_id"]
            .tolist()
        )) if r.anomaly_count > 0 else 0
        logger.info(
            f"  {name:15s} | {r.threshold_value:10.4f} | "
            f"{r.anomaly_count:10d} | {r.anomaly_rate_pct:8.2f} | {n_events:>7}"
        )
    logger.info("=" * 65)
    return reports

if __name__ == "__main__":
    import argparse

    _script_dir  = Path(__file__).resolve().parent
    _default_cfg = _script_dir / "config" / "pipeline_config.json"
    if not _default_cfg.exists():
        _default_cfg = Path("config") / "pipeline_config.json"

    parser = argparse.ArgumentParser(description="Run anomaly engine")
    parser.add_argument("--config",  default=str(_default_cfg))
    parser.add_argument("--metric",  default=None,
                        help="Single metric (e.g. --metric rds_cpu). Default: all.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path.resolve()}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    config  = resolve_config_paths(config, config_path)
    logger  = setup_logger("anomaly", config)
    metrics = [args.metric] if args.metric else None

    run_all(config, logger, metrics)
