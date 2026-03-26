from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# W&B tracking
try:
    from wandb_tracker import (
        init_run, finish_run,
        log_epoch, log_training_complete,
        is_enabled,
    )
except ImportError:
    def init_run(*a, **kw):              return None
    def finish_run(*a, **kw):            pass
    def log_epoch(*a, **kw):             pass
    def log_training_complete(*a, **kw): pass
    def is_enabled(*a, **kw):            return False

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (slight perf cost, worth it for prod)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def setup_logger(name: str, config: dict) -> logging.Logger:
    log_cfg = config["logging"]
    log_dir = Path(log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    level  = getattr(logging, log_cfg.get("log_level", "INFO"))
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    if log_cfg.get("log_to_console", True):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_cfg.get("log_to_file", True):
        log_file = log_dir / f"model_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# RESULT DATACLASS

@dataclass
class TrainingReport:
    """Full audit record of one model's training run."""
    metric_name:      str
    trained_at:       str
    device:           str
    seed:             int
    # architecture
    input_size:       int
    hidden_size:      int
    num_layers:       int
    dropout:          float
    # training config
    past_window:      int
    future_window:    int
    epochs_planned:   int
    epochs_trained:   int        # may be less if early stopping triggers
    batch_size:       int
    learning_rate:    float
    weight_decay:     float
    early_stop_patience: int
    # results
    best_val_loss:    float
    final_train_loss: float
    best_epoch:       int
    train_losses:     List[float] = field(default_factory=list)
    val_losses:       List[float] = field(default_factory=list)
    # artifacts
    checkpoint_path:  str = ""
    plot_path:        str = ""
    total_params:     int = 0
    training_secs:    float = 0.0


# MODEL ARCHITECTURE


class SSLPredictor(nn.Module):
    """
    LSTM encoder + MLP prediction head.

    WHY LSTM for this task?
    ────────────────────────────────────────────────────────────────────
    AWS telemetry is a sequence — each value depends on what came before.
    LSTMs have internal gating mechanisms that learn:

      Forget gate  — discard irrelevant history
                     (e.g. old CPU spike that was a one-off)
      Input gate   — absorb new relevant information
                     (e.g. rising request count predicts CPU rise soon)
      Output gate  — decide what to expose as the hidden state

    This makes LSTMs far better at capturing temporal patterns than
    a simple feedforward net or a fixed rolling average.

    WHY 2 layers?
    ────────────────────────────────────────────────────────────────────
    Layer 1 learns short-range patterns (minute-to-minute fluctuations).
    Layer 2 learns patterns in the patterns (hourly trends, periodicity).
    More than 2 layers rarely helps on ~4000-row datasets and risks
    overfitting.

    WHY a prediction head (not just use LSTM output directly)?
    ────────────────────────────────────────────────────────────────────
    The LSTM hidden state is a 64-dim latent vector — a compressed
    summary of the past window. The MLP head maps this to 6 future
    values. The ReLU + Dropout in the head adds non-linearity and
    regularization before the final linear projection.
    """

    def __init__(self,
                 input_size:   int = 1,
                 hidden_size:  int = 64,
                 num_layers:   int = 2,
                 dropout:      float = 0.2,
                 future_window: int = 6):
        super().__init__()

        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.future_window = future_window

        # LSTM Encoder 
        # batch_first=True → input shape is (batch, seq_len, features)
        # dropout only applied between LSTM layers (not after last layer)
        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Prediction Head 
        # Takes the final hidden state → predicts future_window values
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, future_window),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (batch, past_window, input_size)  — normalized past values

        Returns:
          pred: (batch, future_window)  — predicted future values
        """
        # Run through LSTM
        # output: (batch, seq_len, hidden_size) — all hidden states
        # h_n:    (num_layers, batch, hidden_size) — final hidden states
        _, (h_n, _) = self.encoder(x)

        # Take the last layer's hidden state as the sequence summary
        # h_n[-1] shape: (batch, hidden_size)
        context = h_n[-1]

        # Project to future values
        return self.head(context)          # (batch, future_window)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# DATA LOADING

def load_splits(metric_name: str,
                processed_dir: Path,
                device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load the .npy files produced by data_preprocessing.py and
    convert them to PyTorch tensors on the correct device.

    Shape transformations:
      .npy loaded:  (N, past_window)   float32
      unsqueeze(-1): (N, past_window, 1)  — adds the feature dimension
                                            LSTM expects (batch, seq, features)
    """
    splits = {}
    for split in ["train", "val", "test"]:
        X = np.load(processed_dir / f"{metric_name}_{split}_X.npy")
        y = np.load(processed_dir / f"{metric_name}_{split}_y.npy")

        # (N, past_window) → (N, past_window, 1)
        # The "1" is the feature dimension — we have 1 feature (the metric value).
        # This is the shape LSTM's input_size=1 expects.
        X_t = torch.from_numpy(X).unsqueeze(-1).to(device)   # (N, 24, 1)
        y_t = torch.from_numpy(y).to(device)                  # (N, 6)

        splits[f"{split}_X"] = X_t
        splits[f"{split}_y"] = y_t

    return splits

# TRAINING LOOP

def train_one_metric(metric_name: str,
                     config: dict,
                     logger: logging.Logger) -> TrainingReport:
    """
    Full training pipeline for one metric. Returns a TrainingReport.

    Steps:
      1. Load .npy splits → tensors
      2. Build DataLoader (batched, shuffled for train)
      3. Instantiate model, optimizer, scheduler
      4. Train loop: forward → loss → backward → step
      5. Validate each epoch → track best val loss
      6. Early stopping if val loss stagnates
      7. Save best checkpoint
      8. Plot loss curves
    """
    # Config 
    mdl_cfg  = config["model"]
    pre_cfg  = config["preprocessing"]
    art_cfg  = config["artifacts"]

    hidden_size      = mdl_cfg["hidden_size"]
    num_layers       = mdl_cfg["num_layers"]
    dropout          = mdl_cfg["dropout"]
    epochs           = mdl_cfg["epochs"]
    batch_size       = mdl_cfg["batch_size"]
    lr               = mdl_cfg["learning_rate"]
    weight_decay     = mdl_cfg["weight_decay"]
    patience         = mdl_cfg["early_stop_patience"]
    seed             = mdl_cfg["seed"]
    future_window    = pre_cfg["future_window"]

    processed_dir    = Path(art_cfg["output_dir"])
    checkpoint_dir   = Path(art_cfg["checkpoint_dir"])
    reports_dir      = Path(art_cfg["reports_dir"])
    plots_dir        = reports_dir / "plots"

    for d in [checkpoint_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    #  Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{metric_name}] Device: {device}")

    # Reproducibility 
    set_seed(seed)

    #  Data 
    logger.info(f"[{metric_name}] Loading splits from {processed_dir}")
    splits   = load_splits(metric_name, processed_dir, device)

    train_ds = TensorDataset(splits["train_X"], splits["train_y"])
    val_ds   = TensorDataset(splits["val_X"],   splits["val_y"])

    # shuffle=True for training — breaks temporal order within batches
    # which is fine because each window is already self-contained.
    # Never shuffle val/test — we want to evaluate in chronological order.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    logger.info(
        f"[{metric_name}] Data loaded | "
        f"train={len(train_ds)} | val={len(val_ds)} | "
        f"batches/epoch={len(train_loader)}"
    )

    #  Model 
    model = SSLPredictor(
        input_size    = 1,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        dropout       = dropout,
        future_window = future_window,
    ).to(device)

    n_params = model.count_parameters()
    logger.info(f"[{metric_name}] Model | params={n_params:,} | architecture=LSTM({hidden_size}x{num_layers})")

    # Optimizer + Scheduler 
    # Adam with weight decay (= AdamW behaviour).
    # Weight decay adds L2 regularization to prevent overfitting on small datasets.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ReduceLROnPlateau: if val loss doesn't improve for 5 epochs,
    # halve the learning rate. This lets training converge more precisely
    # in the later stages without manually tuning a schedule.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    criterion = nn.MSELoss()

    # Training state 
    best_val_loss   = float("inf")
    best_epoch      = 0
    epochs_no_improve = 0
    train_losses    = []
    val_losses      = []
    checkpoint_path = checkpoint_dir / f"{metric_name}_best.pt"

    # Initialize W&B run for this metric's training
    init_run(config, stage="training", metric_name=metric_name, logger=logger)

    logger.info(f"[{metric_name}] Starting training | epochs={epochs} | batch={batch_size} | lr={lr}")
    logger.info("=" * 65)

    t_start = time.time()

    for epoch in range(1, epochs + 1):

        # Train phase 
        model.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)             # forward pass
            loss = criterion(pred, y_batch)   # MSE vs actual future
            loss.backward()                   # compute gradients

            # Gradient clipping: prevent exploding gradients in LSTM.
            # Clips the norm of all parameter gradients to max 1.0.
            # Common best practice for RNNs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)

        avg_train = train_loss_sum / len(train_ds)

        #  Validation phase 
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(X_batch)

        avg_val = val_loss_sum / len(val_ds)

        train_losses.append(round(avg_train, 8))
        val_losses.append(round(avg_val, 8))

        # LR scheduler step 
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        #  Compute gradient norm (W&B monitoring) 
        # Grad norm spike = exploding gradients. Near zero = vanishing.
        grad_norm = None
        if is_enabled(config):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.detach().data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        # Logging 
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"[{metric_name}] epoch {epoch:3d}/{epochs} | "
                f"train={avg_train:.6f} | val={avg_val:.6f} | "
                f"lr={current_lr:.2e}"
            )

        #  W&B: log every epoch for live loss curve 
        log_epoch(
            metric_name   = metric_name,
            epoch         = epoch,
            train_loss    = avg_train,
            val_loss      = avg_val,
            learning_rate = current_lr,
            grad_norm     = grad_norm,
            config        = config,
        )

        #  Checkpoint: save best model 
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch    = epoch
            epochs_no_improve = 0

            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":     avg_val,
                "train_loss":   avg_train,
                "config":       mdl_cfg,
                "metric_name":  metric_name,
                "future_window": future_window,
            }, checkpoint_path)

        else:
            epochs_no_improve += 1

        # Early stopping 
        # If val loss hasn't improved for `patience` consecutive epochs,
        # stop training. This prevents wasting time and overfitting.
        # The best checkpoint is already saved so we don't lose progress.
        if epochs_no_improve >= patience:
            logger.info(
                f"[{metric_name}] Early stopping at epoch {epoch} | "
                f"no improvement for {patience} epochs | "
                f"best val={best_val_loss:.6f} at epoch {best_epoch}"
            )
            break

    training_secs = time.time() - t_start
    epochs_trained = len(train_losses)

    logger.info("=" * 65)
    logger.info(
        f"[{metric_name}] Training complete | "
        f"epochs={epochs_trained} | best_val={best_val_loss:.6f} | "
        f"best_epoch={best_epoch} | time={training_secs:.1f}s"
    )
    logger.info(f"[{metric_name}] Checkpoint saved -> {checkpoint_path}")

    # Loss curve plot 
    plot_path = _plot_loss_curves(
        metric_name, train_losses, val_losses,
        best_epoch, plots_dir, logger
    )

    #  Training report 
    report = TrainingReport(
        metric_name         = metric_name,
        trained_at          = datetime.now(timezone.utc).isoformat(),
        device              = str(device),
        seed                = seed,
        input_size          = 1,
        hidden_size         = hidden_size,
        num_layers          = num_layers,
        dropout             = dropout,
        past_window         = pre_cfg["past_window"],
        future_window       = future_window,
        epochs_planned      = epochs,
        epochs_trained      = epochs_trained,
        batch_size          = batch_size,
        learning_rate       = lr,
        weight_decay        = weight_decay,
        early_stop_patience = patience,
        best_val_loss       = round(best_val_loss, 8),
        final_train_loss    = round(train_losses[-1], 8),
        best_epoch          = best_epoch,
        train_losses        = train_losses,
        val_losses          = val_losses,
        checkpoint_path     = str(checkpoint_path),
        plot_path           = plot_path,
        total_params        = n_params,
        training_secs       = round(training_secs, 2),
    )

    report_path = reports_dir / f"{metric_name}_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"[{metric_name}] Training report -> {report_path}")

    return report

# LOSS CURVE PLOT

def _plot_loss_curves(metric_name: str,
                      train_losses: List[float],
                      val_losses: List[float],
                      best_epoch: int,
                      out_dir: Path,
                      logger: logging.Logger) -> str:
    """
    Plot training and validation loss curves.
    Marks the best epoch with a vertical dashed line.
    Includes a zoomed inset panel for the last 30% of training
    so you can see fine-grained convergence behaviour.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{metric_name} — Training Loss Curves", fontsize=12, fontweight="bold")

    epochs  = list(range(1, len(train_losses) + 1))
    colors  = {"train": "steelblue", "val": "tomato", "best": "seagreen"}

    for ax, log_scale in zip(axes, [False, True]):
        ax.plot(epochs, train_losses, color=colors["train"],
                linewidth=1.2, label="train loss")
        ax.plot(epochs, val_losses,   color=colors["val"],
                linewidth=1.2, label="val loss")
        ax.axvline(best_epoch, color=colors["best"],
                   linestyle="--", linewidth=1.5,
                   label=f"best epoch={best_epoch}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss" + (" (log)" if log_scale else ""))
        ax.set_title("Loss over epochs" + (" — log scale" if log_scale else " — linear scale"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if log_scale and min(val_losses) > 0:
            ax.set_yscale("log")

    plt.tight_layout()
    out_path = str(out_dir / f"{metric_name}_training_loss.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"[{metric_name}] Loss plot -> {out_path}")
    return out_path


# EVALUATE ON TEST SET

def evaluate_on_test(metric_name: str,
                     config: dict,
                     logger: logging.Logger) -> dict:
    """
    Load the best saved checkpoint and evaluate it on the held-out
    test set. Returns a dict with test MSE and MAE.

    This is called AFTER training completes — the test set is never
    touched during training or early-stopping decisions.
    """
    mdl_cfg  = config["model"]
    pre_cfg  = config["preprocessing"]
    art_cfg  = config["artifacts"]

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(art_cfg["checkpoint_dir"])
    processed_dir  = Path(art_cfg["output_dir"])
    future_window  = pre_cfg["future_window"]

    # Load checkpoint 
    ckpt_path = checkpoint_dir / f"{metric_name}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)

    model = SSLPredictor(
        input_size    = 1,
        hidden_size   = mdl_cfg["hidden_size"],
        num_layers    = mdl_cfg["num_layers"],
        dropout       = mdl_cfg["dropout"],
        future_window = future_window,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load test split 
    test_X = np.load(processed_dir / f"{metric_name}_test_X.npy")
    test_y = np.load(processed_dir / f"{metric_name}_test_y.npy")

    X_t = torch.from_numpy(test_X).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(test_y).to(device)

    test_ds     = TensorDataset(X_t, y_t)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Inference 
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    test_mse = float(np.mean((preds - targets) ** 2))
    test_mae = float(np.mean(np.abs(preds - targets)))

    results = {
        "metric_name": metric_name,
        "test_mse":    round(test_mse, 8),
        "test_mae":    round(test_mae, 8),
        "test_rmse":   round(float(np.sqrt(test_mse)), 8),
        "n_test":      len(test_y),
    }

    logger.info(
        f"[{metric_name}] Test results | "
        f"MSE={test_mse:.6f} | MAE={test_mae:.6f} | RMSE={np.sqrt(test_mse):.6f}"
    )
    return results


# RUN ALL METRICS

def run_training(config: dict,
                 logger: logging.Logger,
                 metrics: Optional[List[str]] = None) -> Dict[str, TrainingReport]:
    """
    Train one model per metric. If `metrics` is None, train all metrics
    defined in config["ingestion"]["metrics"].
    """
    all_metrics = list(config["ingestion"]["metrics"].keys())
    targets     = metrics if metrics else all_metrics

    logger.info("=" * 65)
    logger.info("  MODEL TRAINING  START")
    logger.info(f"  Metrics: {targets}")
    logger.info("=" * 65)

    reports      = {}
    test_results = {}

    for metric_name in targets:
        logger.info(f"\n[{metric_name}] ========================================")
        report = train_one_metric(metric_name, config, logger)
        reports[metric_name] = report

        test_res = evaluate_on_test(metric_name, config, logger)
        test_results[metric_name] = test_res

        # Log full training summary + model artifact to W&B, then close run
        log_training_complete(
            metric_name     = metric_name,
            report_dict     = asdict(report),
            test_results    = test_res,
            loss_plot_path  = report.plot_path,
            checkpoint_path = report.checkpoint_path,
            config          = config,
            logger          = logger,
        )
        finish_run(logger)

    # Final summary 
    logger.info("=" * 65)
    logger.info("  TRAINING SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  {'Metric':15s} | {'Epochs':>6} | {'Best Val MSE':>12} | {'Test MSE':>10} | {'Test MAE':>10}")
    logger.info("-" * 65)
    for name in targets:
        r  = reports[name]
        tr = test_results[name]
        logger.info(
            f"  {name:15s} | {r.epochs_trained:6d} | "
            f"{r.best_val_loss:12.6f} | {tr['test_mse']:10.6f} | {tr['test_mae']:10.6f}"
        )
    logger.info("=" * 65)

    return reports


# PATH RESOLUTION  (same helper used by ingestion/preprocessing)

def resolve_config_paths(config: dict, config_path: Path) -> dict:
    cwd     = Path.cwd()
    cfg_dir = config_path.resolve().parent

    def to_abs(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        cwd_candidate = cwd / path
        if cwd_candidate.exists():
            return str(cwd_candidate)
        cfg_candidate = cfg_dir / path
        if cfg_candidate.exists():
            return str(cfg_candidate)
        return str(cwd_candidate)   # default to cwd-relative for new output dirs

    config["ingestion"]["source_dir"]  = to_abs(config["ingestion"]["source_dir"])
    config["artifacts"]["output_dir"]  = to_abs(config["artifacts"]["output_dir"])
    config["artifacts"]["reports_dir"] = to_abs(config["artifacts"]["reports_dir"])
    config["artifacts"]["scaler_dir"]  = to_abs(config["artifacts"]["scaler_dir"])
    config["artifacts"]["checkpoint_dir"] = to_abs(config["artifacts"]["checkpoint_dir"])
    config["logging"]["log_dir"]       = to_abs(config["logging"]["log_dir"])
    return config

# STANDALONE ENTRY POINT

if __name__ == "__main__":
    import argparse

    _script_dir  = Path(__file__).resolve().parent
    _default_cfg = _script_dir / "config" / "pipeline_config.json"
    if not _default_cfg.exists():
        _default_cfg = Path("config") / "pipeline_config.json"

    parser = argparse.ArgumentParser(description="Train SSL anomaly detection models")
    parser.add_argument("--config", default=str(_default_cfg))
    parser.add_argument(
        "--metric", default=None,
        help="Train a single metric only (e.g. --metric rds_cpu). Default: train all."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path.resolve()}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    config  = resolve_config_paths(config, config_path)
    logger  = setup_logger("model", config)
    metrics = [args.metric] if args.metric else None

    run_training(config, logger, metrics)