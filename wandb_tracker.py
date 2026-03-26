"""
╔══════════════════════════════════════════════════════════════════════╗
║                      wandb_tracker.py                                ║
║                                                                      ║
║  Centralized Weights & Biases integration for the SSL pipeline.      ║
║                                                                      ║
║  WHY A SEPARATE MODULE?                                              ║
║    Every pipeline stage (ingestion, preprocessing, model) imports    ║
║    this module. That means:                                          ║
║      1. W&B logic lives in ONE place — easy to update               ║
║      2. If W&B is disabled in config, all stages silently skip it    ║
║         without any code changes in the other files                  ║
║      3. All 3 stages log to the SAME W&B run → one unified view      ║
║         of the full pipeline on the W&B dashboard                    ║
║                                                                      ║
║  WHAT EACH STAGE LOGS:                                               ║
║                                                                      ║
║  Ingestion:                                                          ║
║    - Per-metric quality summary (rows, nulls, gaps, checksums)       ║
║    - Quality gate pass/fail status                                   ║
║    - W&B Table with full quality report per metric                   ║
║                                                                      ║
║  Preprocessing:                                                      ║
║    - Window counts, split sizes, scaler bounds                       ║
║    - QA plots (raw vs normalized, window samples) as W&B Images      ║
║                                                                      ║
║  Model training (per epoch):                                         ║
║    - train_loss, val_loss (live — plots update as training runs)     ║
║    - learning_rate (shows LR decay from ReduceLROnPlateau)           ║
║    - gradient_norm (detects exploding/vanishing gradients)           ║
║    - best_val_loss, best_epoch (summary metrics)                     ║
║                                                                      ║
║  Test evaluation:                                                    ║
║    - test_mse, test_mae, test_rmse per metric                        ║
║    - Loss curve image as W&B Image                                   ║
║    - Model checkpoint as W&B Artifact (versioned, downloadable)      ║
║                                                                      ║
║  GRACEFUL DEGRADATION:                                               ║
║    If wandb is not installed or disabled in config, every function   ║
║    in this module becomes a no-op. The rest of the pipeline never    ║
║    needs to check — it just calls tracker functions unconditionally. ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# SAFE WANDB IMPORT
# If wandb is not installed, _WANDB_AVAILABLE = False and every
# function becomes a silent no-op. Pipeline never crashes.
# ──────────────────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _is_active(config: dict) -> bool:
    """Return True if wandb is installed AND enabled in config."""
    return (
        _WANDB_AVAILABLE
        and config.get("wandb", {}).get("enabled", False)
        and wandb.run is not None   # a run is currently active
    )


# ──────────────────────────────────────────────────────────────────────
# RUN LIFECYCLE
# ──────────────────────────────────────────────────────────────────────

def init_run(config: dict,
             stage: str,
             metric_name: Optional[str] = None,
             logger: Optional[logging.Logger] = None) -> Optional[Any]:
    """
    Initialize a W&B run for a pipeline stage.

    HOW RUNS MAP TO STAGES:
      Each call to `python -m src.pipeline.model` creates ONE W&B run.
      Within that run, all metrics are logged with metric-specific
      prefixes (e.g. "rds_cpu/train_loss", "ec2_disk/train_loss").
      This gives you side-by-side comparison of all metrics in one run.

    If wandb.enabled = false in config → returns None silently.

    Args:
      config       – full pipeline config dict
      stage        – "ingestion" | "preprocessing" | "training"
      metric_name  – optional, used to label the run name
      logger       – optional, for logging the W&B run URL
    """
    wcfg = config.get("wandb", {})
    if not _WANDB_AVAILABLE or not wcfg.get("enabled", False):
        return None

    # Build run name: e.g. "training-rds_cpu" or just "ingestion"
    run_name = wcfg.get("run_name") or (
        f"{stage}-{metric_name}" if metric_name else stage
    )

    run = wandb.init(
        project   = wcfg.get("project", "ssl-anomaly-detector"),
        entity    = wcfg.get("entity") or None,
        name      = run_name,
        tags      = wcfg.get("tags", []) + [stage],
        notes     = wcfg.get("notes", ""),
        config    = _flatten_config(config),    # log all hyperparams
        reinit    = True,                       # allow multiple runs per process
    )

    if logger and run:
        logger.info(f"[wandb] Run initialized | {run.name} | {run.url}")

    return run


def finish_run(logger: Optional[logging.Logger] = None):
    """
    Finish the current W&B run. Call at the end of each stage.
    Safe to call even if wandb is not active.
    """
    if _WANDB_AVAILABLE and wandb.run is not None:
        url = wandb.run.url
        wandb.finish()
        if logger:
            logger.info(f"[wandb] Run finished | {url}")


# ──────────────────────────────────────────────────────────────────────
# INGESTION LOGGING
# ──────────────────────────────────────────────────────────────────────

def log_ingestion_quality(quality_reports: List[dict],
                          config: dict,
                          logger: Optional[logging.Logger] = None):
    """
    Log quality reports from all ingested metrics to W&B.

    Logs two things:
      1. Summary metrics (flat key-value) — appear in the Run overview
         e.g.  ingestion/rds_cpu/rows = 4033
               ingestion/rds_cpu/quality_gate = PASS

      2. A W&B Table — one row per metric, all quality fields as columns.
         Visible as a searchable table in the W&B UI.
    """
    if not _is_active(config):
        return

    # ── 1. Summary metrics ────────────────────────────────────
    summary = {}
    for report in quality_reports:
        name = report["metric_name"]
        prefix = f"ingestion/{name}"
        summary.update({
            f"{prefix}/rows":               report["final_row_count"],
            f"{prefix}/null_pct":           report["null_pct"],
            f"{prefix}/duplicate_count":    report["duplicate_count"],
            f"{prefix}/gaps_filled":        report["gaps_filled"],
            f"{prefix}/outliers_clipped":   report["outliers_clipped_hi"] + report["outliers_clipped_lo"],
            f"{prefix}/max_gap_minutes":    report["max_gap_minutes"],
            f"{prefix}/value_min":          report["value_min"],
            f"{prefix}/value_max":          report["value_max"],
            f"{prefix}/quality_gate":       "PASS" if report["passed_quality_gate"] else "FAIL",
            f"{prefix}/checksum_md5":       report["checksum_md5"][:8],
        })

    wandb.log(summary, step=0)

    # ── 2. Quality table ──────────────────────────────────────
    columns = [
        "metric", "rows", "null_pct", "duplicates",
        "gaps_filled", "outliers", "max_gap_min",
        "val_min", "val_max", "quality_gate", "checksum"
    ]
    table = wandb.Table(columns=columns)
    for r in quality_reports:
        table.add_data(
            r["metric_name"],
            r["final_row_count"],
            r["null_pct"],
            r["duplicate_count"],
            r["gaps_filled"],
            r["outliers_clipped_hi"] + r["outliers_clipped_lo"],
            r["max_gap_minutes"],
            round(r["value_min"], 4),
            round(r["value_max"], 4),
            "PASS" if r["passed_quality_gate"] else "FAIL",
            r["checksum_md5"][:8],
        )
    wandb.log({"ingestion/quality_table": table}, step=0)

    if logger:
        logger.info(f"[wandb] Ingestion quality logged for {len(quality_reports)} metrics")


# ──────────────────────────────────────────────────────────────────────
# PREPROCESSING LOGGING
# ──────────────────────────────────────────────────────────────────────

def log_preprocessing_summary(preprocessing_reports: List[dict],
                               config: dict,
                               logger: Optional[logging.Logger] = None):
    """
    Log preprocessing results to W&B.

    Logs:
      - Window counts, split sizes, scaler bounds per metric
      - QA plots as W&B Images (raw vs normalized, window samples)
    """
    if not _is_active(config):
        return

    summary = {}
    images  = {}

    for report in preprocessing_reports:
        name   = report["metric_name"]
        prefix = f"preprocessing/{name}"

        summary.update({
            f"{prefix}/total_timesteps":  report["total_timesteps"],
            f"{prefix}/total_windows":    report["total_windows"],
            f"{prefix}/train_windows":    report["train"]["windows"],
            f"{prefix}/val_windows":      report["val"]["windows"],
            f"{prefix}/test_windows":     report["test"]["windows"],
            f"{prefix}/scaler_min":       report["scaler_min"],
            f"{prefix}/scaler_max":       report["scaler_max"],
            f"{prefix}/past_window":      report["past_window"],
            f"{prefix}/future_window":    report["future_window"],
        })

        # Log QA plots as W&B Images
        for plot_path in report.get("plot_files", []):
            p = Path(plot_path)
            if p.exists():
                plot_key = f"preprocessing/{name}/{p.stem}"
                images[plot_key] = wandb.Image(str(p), caption=f"{name} — {p.stem}")

    wandb.log({**summary, **images}, step=0)

    if logger:
        logger.info(f"[wandb] Preprocessing summary logged for {len(preprocessing_reports)} metrics")


# ──────────────────────────────────────────────────────────────────────
# MODEL TRAINING LOGGING
# ──────────────────────────────────────────────────────────────────────

def log_epoch(metric_name: str,
              epoch: int,
              train_loss: float,
              val_loss: float,
              learning_rate: float,
              grad_norm: Optional[float],
              config: dict):
    """
    Log per-epoch training metrics to W&B.

    Called inside the training loop every epoch.
    W&B plots these in real-time — you see the loss curve building
    live on the dashboard while training is still running.

    Metrics logged with metric-name prefix so all metrics appear
    as separate lines on the same chart:
      rds_cpu/train_loss    ─┐
      ec2_disk/train_loss   ─┤  all on one chart, different colors
      elb_requests/train_loss─┘
    """
    if not _is_active(config):
        return

    payload = {
        f"{metric_name}/train_loss":    train_loss,
        f"{metric_name}/val_loss":      val_loss,
        f"{metric_name}/learning_rate": learning_rate,
    }
    if grad_norm is not None:
        payload[f"{metric_name}/grad_norm"] = grad_norm

    wandb.log(payload, step=epoch)


def log_training_complete(metric_name: str,
                          report_dict: dict,
                          test_results: dict,
                          loss_plot_path: str,
                          checkpoint_path: str,
                          config: dict,
                          logger: Optional[logging.Logger] = None):
    """
    Log end-of-training summary metrics, loss plot image, and
    optionally the model checkpoint as a versioned W&B Artifact.

    W&B Artifacts:
      A W&B Artifact is a versioned, tracked file stored in W&B.
      Every time you retrain, a new artifact version is created.
      You can download any previous version by its version number.
      This is how production ML teams manage model lineage.

      artifact name: "ssl-{metric_name}-model"
      type:          "model"
      metadata:      best_val_loss, epochs, test_mse, etc.
    """
    if not _is_active(config):
        return

    wcfg   = config.get("wandb", {})
    prefix = metric_name

    # ── Summary metrics ───────────────────────────────────────
    wandb.summary.update({
        f"{prefix}/best_val_loss":    report_dict["best_val_loss"],
        f"{prefix}/best_epoch":       report_dict["best_epoch"],
        f"{prefix}/epochs_trained":   report_dict["epochs_trained"],
        f"{prefix}/total_params":     report_dict["total_params"],
        f"{prefix}/training_secs":    report_dict["training_secs"],
        f"{prefix}/test_mse":         test_results["test_mse"],
        f"{prefix}/test_mae":         test_results["test_mae"],
        f"{prefix}/test_rmse":        test_results["test_rmse"],
    })

    # ── Loss curve image ──────────────────────────────────────
    if Path(loss_plot_path).exists():
        wandb.log({
            f"{prefix}/loss_curve": wandb.Image(
                loss_plot_path,
                caption=f"{metric_name} training loss"
            )
        })

    # ── Model artifact ────────────────────────────────────────
    if wcfg.get("log_model_artifact", True) and Path(checkpoint_path).exists():
        artifact = wandb.Artifact(
            name     = f"ssl-{metric_name}-model",
            type     = "model",
            metadata = {
                "metric_name":   metric_name,
                "best_val_loss": report_dict["best_val_loss"],
                "best_epoch":    report_dict["best_epoch"],
                "test_mse":      test_results["test_mse"],
                "hidden_size":   report_dict["hidden_size"],
                "num_layers":    report_dict["num_layers"],
                "past_window":   report_dict["past_window"],
                "future_window": report_dict["future_window"],
            }
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        if logger:
            logger.info(f"[wandb] Model artifact logged: ssl-{metric_name}-model")


# ──────────────────────────────────────────────────────────────────────
# ANOMALY ENGINE LOGGING  (used in next step)
# ──────────────────────────────────────────────────────────────────────

def log_anomaly_results(metric_name: str,
                        anomaly_summary: dict,
                        score_plot_path: Optional[str],
                        config: dict,
                        logger: Optional[logging.Logger] = None):
    """
    Log anomaly detection results to W&B.
    Called by anomaly_engine.py after scoring the full series.

    Logs:
      - threshold value, anomaly count, anomaly rate
      - Anomaly score plot as W&B Image
      - W&B Table of flagged anomaly timestamps + scores
    """
    if not _is_active(config):
        return

    prefix = f"anomaly/{metric_name}"

    wandb.log({
        f"{prefix}/threshold":       anomaly_summary.get("threshold", 0),
        f"{prefix}/anomaly_count":   anomaly_summary.get("anomaly_count", 0),
        f"{prefix}/anomaly_rate_pct": anomaly_summary.get("anomaly_rate_pct", 0),
        f"{prefix}/score_mean":      anomaly_summary.get("score_mean", 0),
        f"{prefix}/score_p95":       anomaly_summary.get("score_p95", 0),
        f"{prefix}/score_p99":       anomaly_summary.get("score_p99", 0),
    })

    if score_plot_path and Path(score_plot_path).exists():
        wandb.log({
            f"{prefix}/score_plot": wandb.Image(
                score_plot_path,
                caption=f"{metric_name} anomaly scores"
            )
        })

    if logger:
        logger.info(f"[wandb] Anomaly results logged for {metric_name}")


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def _flatten_config(config: dict, prefix: str = "", sep: str = "/") -> dict:
    """
    Flatten a nested config dict into W&B-compatible flat key-value pairs.

    Example:
      {"model": {"hidden_size": 64, "num_layers": 2}}
      → {"model/hidden_size": 64, "model/num_layers": 2}

    W&B stores this as hyperparameters — you can filter and compare
    runs by any of these values in the W&B UI.
    """
    flat = {}
    for key, val in config.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if key.startswith("_"):          # skip comment fields
            continue
        if isinstance(val, dict):
            flat.update(_flatten_config(val, full_key, sep))
        elif isinstance(val, (str, int, float, bool, list)) or val is None:
            flat[full_key] = val
    return flat


def is_enabled(config: dict) -> bool:
    """Convenience check — used in pipeline stages to log conditionally."""
    return _WANDB_AVAILABLE and config.get("wandb", {}).get("enabled", False)
