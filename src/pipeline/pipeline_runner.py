"""
╔══════════════════════════════════════════════════════════════════════╗
║                      pipeline_runner.py                              ║
║                                                                      ║
║  Production orchestrator for the SSL Anomaly Detection pipeline.     ║
║                                                                      ║
║  WHAT THIS FILE DOES:                                                ║
║    Chains all 4 stages in the correct order with:                    ║
║      • Smart stage skipping  — don't re-run stages whose outputs     ║
║        are still fresh and whose inputs haven't changed              ║
║      • Data-change detection — re-ingests + retrains automatically   ║
║        when raw CSV checksums change (new data arrived)              ║
║      • Fail-fast with context — if stage N fails, stages N+1..4     ║
║        are skipped with a clear explanation, not a cryptic traceback ║
║      • Run modes — full / ingest-only / train-only / score-only      ║
║      • Structured run summary — one JSON written per execution       ║
║      • W&B pipeline-level run tracking                               ║
║                                                                      ║
║  STAGE DEPENDENCY GRAPH:                                             ║
║                                                                      ║
║   [1] ingestion  ──► produces: quality_reports, checksums            ║
║         │                                                            ║
║         ▼                                                            ║
║   [2] preprocessing ──► produces: .npy splits, scalers              ║
║         │                                                            ║
║         ▼  (skipped if checkpoint is fresh AND data unchanged)       ║
║   [3] model training ──► produces: checkpoints, training reports     ║
║         │                                                            ║
║         ▼                                                            ║
║   [4] anomaly engine ──► produces: anomaly CSVs, score plots         ║
║                                                                      ║
║  SMART SKIP LOGIC:                                                   ║
║                                                                      ║
║   Stage is SKIPPED when ALL of the following are true:               ║
║     1. All expected output artifacts exist on disk                   ║
║     2. Raw data checksums match the last run (data hasn't changed)   ║
║     3. --force flag is NOT set                                       ║
║     4. Stage is not explicitly listed in --stages                    ║
║                                                                      ║
║   Stage is FORCED when:                                              ║
║     • --force flag passed                                            ║
║     • Raw CSV checksum differs from last run  (data changed)         ║
║     • Upstream stage re-ran (outputs are newer than this stage)      ║
║     • --stages explicitly includes this stage                        ║
║                                                                      ║
║  USAGE:                                                              ║
║    # Full pipeline (smart skip)                                      ║
║    python -m src.pipeline.pipeline_runner                            ║
║                                                                      ║
║    # Force all stages to re-run                                      ║
║    python -m src.pipeline.pipeline_runner --force                    ║
║                                                                      ║
║    # Run specific stages only                                        ║
║    python -m src.pipeline.pipeline_runner --stages ingest preprocess ║
║    python -m src.pipeline.pipeline_runner --stages train score       ║
║                                                                      ║
║    # Single metric                                                   ║
║    python -m src.pipeline.pipeline_runner --metric rds_cpu           ║
║                                                                      ║
║    # Dry run — show what WOULD run without executing                 ║
║    python -m src.pipeline.pipeline_runner --dry-run                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# ── Stage imports ─────────────────────────────────────────────────────
try:
    from data_ingestion import (
        run_ingestion, resolve_config_paths, setup_logger
    )
    from data_preprocessing import run_preprocessing
    from model import run_training
    from anomaly_engine import run_all as run_anomaly
except ImportError:
    from src.pipeline.data_ingestion import (
        run_ingestion, resolve_config_paths, setup_logger
    )
    from src.pipeline.data_preprocessing import run_preprocessing
    from src.pipeline.model import run_training
    from src.pipeline.anomaly_engine import run_all as run_anomaly

# ── W&B ───────────────────────────────────────────────────────────────
try:
    from wandb_tracker import init_run, finish_run, is_enabled
except ImportError:
    from src.pipeline.wandb_tracker import init_run, finish_run, is_enabled


# ──────────────────────────────────────────────────────────────────────
# STAGE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

class Stage(str, Enum):
    INGEST     = "ingest"
    PREPROCESS = "preprocess"
    TRAIN      = "train"
    SCORE      = "score"

    @classmethod
    def all(cls) -> List["Stage"]:
        return [cls.INGEST, cls.PREPROCESS, cls.TRAIN, cls.SCORE]


# ──────────────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ──────────────────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    SKIPPED = "SKIPPED"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"


@dataclass
class StageResult:
    stage:       str
    status:      str
    skip_reason: str   = ""
    error:       str   = ""
    duration_s:  float = 0.0
    artifacts:   List[str] = field(default_factory=list)


@dataclass
class RunSummary:
    run_id:       str
    started_at:   str
    finished_at:  str
    duration_s:   float
    mode:         str           # "full" | "partial"
    metrics:      List[str]
    force:        bool
    stages_requested: List[str]
    stages:       List[StageResult] = field(default_factory=list)
    overall:      str = "SUCCESS"  # "SUCCESS" | "PARTIAL" | "FAILED"
    config_path:  str = ""


# ──────────────────────────────────────────────────────────────────────
# FRESHNESS CHECKS
# ──────────────────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _all_exist(paths: List[Path]) -> bool:
    """Return True only if every path in the list exists."""
    return all(p.exists() for p in paths)


def _data_changed(config: dict) -> bool:
    """
    Compare current raw CSV checksums against the last recorded checksums.

    If the checksums differ → raw data has changed → all downstream
    stages must re-run to stay consistent.

    Returns True if ANY metric's checksum has changed or is missing.
    """
    checksums_path = Path(config["artifacts"]["reports_dir"]) / "checksums.json"
    if not checksums_path.exists():
        return True   # no prior run — treat as changed

    with open(checksums_path) as f:
        recorded = json.load(f)

    ing_cfg = config["ingestion"]
    for metric_name, meta in ing_cfg["metrics"].items():
        raw_path = Path(ing_cfg["source_dir"]) / meta["filename"]
        if not raw_path.exists():
            return True
        current_md5 = _md5(raw_path)
        if recorded.get(metric_name, {}).get("md5") != current_md5:
            return True
    return False


def _checkpoint_exists(metric_name: str, config: dict) -> bool:
    ckpt = Path(config["artifacts"]["checkpoint_dir"]) / f"{metric_name}_best.pt"
    return ckpt.exists()


def _preprocessing_outputs_exist(metrics: List[str], config: dict) -> bool:
    processed = Path(config["artifacts"]["output_dir"])
    required  = []
    for m in metrics:
        for split in ["train_X", "train_y", "val_X", "val_y", "test_X", "test_y"]:
            required.append(processed / f"{m}_{split}.npy")
        required.append(Path(config["artifacts"]["scaler_dir"]) / f"{m}_scaler.pkl")
    return _all_exist(required)


def _anomaly_outputs_exist(metrics: List[str], config: dict) -> bool:
    anomaly_dir = Path(config["anomaly"]["anomaly_output_dir"])
    return _all_exist([anomaly_dir / f"{m}_anomaly_scores.csv" for m in metrics])


# ──────────────────────────────────────────────────────────────────────
# STAGE SKIP DECISION ENGINE
# This is the heart of the runner's intelligence.
# ──────────────────────────────────────────────────────────────────────

def decide_stages(config: dict,
                  metrics: List[str],
                  force: bool,
                  requested: List[Stage],
                  logger) -> Dict[Stage, tuple[bool, str]]:
    """
    For each stage, decide: (should_run: bool, reason: str).

    Decision logic per stage:
    ─────────────────────────────────────────────────────────────────
    INGEST:
      Run if:  forced | data changed | quality reports missing

    PREPROCESS:
      Run if:  forced | ingest ran | .npy splits missing

    TRAIN:
      Run if:  forced | preprocess ran | any checkpoint missing

    SCORE:
      Run if:  forced | train ran | anomaly CSVs missing
    ─────────────────────────────────────────────────────────────────

    The `upstream_ran` cascade ensures that if stage N re-runs,
    stages N+1..4 always re-run too — even if their outputs exist.
    This prevents stale outputs from a previous run being mixed
    with fresh outputs from the current run.
    """
    # Respect explicit --stages flag
    if requested:
        return {s: (s in requested, f"explicitly requested") for s in Stage.all()}

    data_changed = _data_changed(config)
    decisions    = {}
    upstream_ran = False    # becomes True as soon as any stage runs

    # ── Stage 1: INGEST ───────────────────────────────────────
    quality_reports = [
        Path(config["artifacts"]["reports_dir"]) / f"{m}_quality_report.json"
        for m in metrics
    ]
    if force:
        run, reason = True, "--force flag"
    elif data_changed:
        run, reason = True, "raw data checksums changed"
    elif not _all_exist(quality_reports):
        run, reason = True, "quality reports missing"
    else:
        run, reason = False, "quality reports fresh + data unchanged"

    decisions[Stage.INGEST] = (run, reason)
    if run:
        upstream_ran = True

    # ── Stage 2: PREPROCESS ───────────────────────────────────
    if force:
        run, reason = True, "--force flag"
    elif upstream_ran:
        run, reason = True, "upstream stage re-ran"
    elif not _preprocessing_outputs_exist(metrics, config):
        run, reason = True, ".npy splits or scalers missing"
    else:
        run, reason = False, "preprocessed artifacts fresh"

    decisions[Stage.PREPROCESS] = (run, reason)
    if run:
        upstream_ran = True

    # ── Stage 3: TRAIN ────────────────────────────────────────
    missing_ckpts = [m for m in metrics if not _checkpoint_exists(m, config)]
    if force:
        run, reason = True, "--force flag"
    elif upstream_ran:
        run, reason = True, "upstream stage re-ran"
    elif missing_ckpts:
        run, reason = True, f"missing checkpoints: {missing_ckpts}"
    else:
        run, reason = False, "all checkpoints exist + data unchanged"

    decisions[Stage.TRAIN] = (run, reason)
    if run:
        upstream_ran = True

    # ── Stage 4: SCORE ────────────────────────────────────────
    if force:
        run, reason = True, "--force flag"
    elif upstream_ran:
        run, reason = True, "upstream stage re-ran"
    elif not _anomaly_outputs_exist(metrics, config):
        run, reason = True, "anomaly CSVs missing"
    else:
        run, reason = False, "anomaly results fresh"

    decisions[Stage.SCORE] = (run, reason)

    return decisions


# ──────────────────────────────────────────────────────────────────────
# INDIVIDUAL STAGE RUNNERS  (wrap each stage with timing + error catch)
# ──────────────────────────────────────────────────────────────────────

def _run_ingest(config: dict, metrics: List[str], logger) -> StageResult:
    t = time.time()
    try:
        results = run_ingestion(config, logger)
        artifacts = [
            str(Path(config["artifacts"]["reports_dir"]) / f"{m}_quality_report.json")
            for m in results
        ]
        return StageResult(
            stage     = Stage.INGEST,
            status    = StageStatus.SUCCESS,
            duration_s= round(time.time() - t, 2),
            artifacts = artifacts,
        )
    except Exception as e:
        return StageResult(
            stage     = Stage.INGEST,
            status    = StageStatus.FAILED,
            error     = traceback.format_exc(),
            duration_s= round(time.time() - t, 2),
        )


def _run_preprocess(config: dict,
                    ingestion_results,
                    metrics: List[str],
                    logger) -> StageResult:
    t = time.time()
    try:
        # If ingestion didn't run this cycle, re-run ingestion silently
        # just to get the IngestionResult objects preprocessing needs.
        # This is safe — it reads from existing CSVs without re-writing reports.
        if ingestion_results is None:
            logger.info("[runner] Re-loading ingestion results for preprocessing input...")
            ingestion_results = run_ingestion(config, logger)

        results = run_preprocessing(ingestion_results, config, logger)
        artifacts = [
            str(Path(config["artifacts"]["output_dir"]) / f"{m}_train_X.npy")
            for m in results
        ]
        return StageResult(
            stage     = Stage.PREPROCESS,
            status    = StageStatus.SUCCESS,
            duration_s= round(time.time() - t, 2),
            artifacts = artifacts,
        )
    except Exception as e:
        return StageResult(
            stage     = Stage.PREPROCESS,
            status    = StageStatus.FAILED,
            error     = traceback.format_exc(),
            duration_s= round(time.time() - t, 2),
        )


def _run_train(config: dict, metrics: List[str], logger) -> StageResult:
    t = time.time()
    try:
        reports = run_training(config, logger, metrics if metrics else None)
        artifacts = [
            str(Path(config["artifacts"]["checkpoint_dir"]) / f"{m}_best.pt")
            for m in reports
        ]
        return StageResult(
            stage     = Stage.TRAIN,
            status    = StageStatus.SUCCESS,
            duration_s= round(time.time() - t, 2),
            artifacts = artifacts,
        )
    except Exception as e:
        return StageResult(
            stage     = Stage.TRAIN,
            status    = StageStatus.FAILED,
            error     = traceback.format_exc(),
            duration_s= round(time.time() - t, 2),
        )


def _run_score(config: dict, metrics: List[str], logger) -> StageResult:
    t = time.time()
    try:
        reports = run_anomaly(config, logger, metrics if metrics else None)
        artifacts = [
            str(Path(config["anomaly"]["anomaly_output_dir"]) / f"{m}_anomaly_scores.csv")
            for m in reports
        ]
        return StageResult(
            stage     = Stage.SCORE,
            status    = StageStatus.SUCCESS,
            duration_s= round(time.time() - t, 2),
            artifacts = artifacts,
        )
    except Exception as e:
        return StageResult(
            stage     = Stage.SCORE,
            status    = StageStatus.FAILED,
            error     = traceback.format_exc(),
            duration_s= round(time.time() - t, 2),
        )


# ──────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(config: dict,
                 config_path: Path,
                 metrics: Optional[List[str]] = None,
                 force: bool = False,
                 requested_stages: Optional[List[Stage]] = None,
                 dry_run: bool = False,
                 logger = None) -> RunSummary:
    """
    Orchestrate all 4 pipeline stages in order.

    Args:
      config           – loaded + resolved config dict
      config_path      – path to config file (for run summary)
      metrics          – subset of metrics to process (None = all)
      force            – if True, re-run all stages regardless of freshness
      requested_stages – if set, only run these specific stages
      dry_run          – print plan but don't execute anything
      logger           – shared logger instance
    """
    run_id    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    all_metrics = list(config["ingestion"]["metrics"].keys())
    targets     = metrics if metrics else all_metrics

    logger.info("=" * 65)
    logger.info("  SSL ANOMALY DETECTION PIPELINE  START")
    logger.info(f"  Run ID   : {run_id}")
    logger.info(f"  Metrics  : {targets}")
    logger.info(f"  Force    : {force}")
    logger.info(f"  Dry run  : {dry_run}")
    logger.info("=" * 65)

    # ── Build execution plan ──────────────────────────────────
    decisions = decide_stages(config, targets, force, requested_stages or [], logger)

    # ── Print plan ────────────────────────────────────────────
    logger.info("\n  EXECUTION PLAN:")
    logger.info(f"  {'Stage':12s}  {'Action':8s}  Reason")
    logger.info("  " + "-" * 55)
    for stage in Stage.all():
        will_run, reason = decisions[stage]
        action = "RUN   " if will_run else "SKIP  "
        icon   = ">>>" if will_run else "---"
        logger.info(f"  {icon} {stage.value:12s}  {action}  {reason}")
    logger.info("")

    if dry_run:
        logger.info("  [DRY RUN] No stages executed.")
        return RunSummary(
            run_id    = run_id,
            started_at= started_at,
            finished_at= datetime.now(timezone.utc).isoformat(),
            duration_s= round(time.time() - t_start, 2),
            mode      = "dry-run",
            metrics   = targets,
            force     = force,
            stages_requested = [s.value for s in (requested_stages or [])],
            overall   = "SKIPPED",
        )

    # ── W&B pipeline run ──────────────────────────────────────
    init_run(config, stage="pipeline", logger=logger)

    # ── Execute stages in order ───────────────────────────────
    stage_results: List[StageResult] = []
    ingestion_results = None   # passed to preprocessing if ingest ran
    pipeline_failed   = False

    # -- Stage 1: INGEST --
    will_run, reason = decisions[Stage.INGEST]
    if not will_run:
        logger.info(f"[runner] SKIP ingest  — {reason}")
        stage_results.append(StageResult(
            stage=Stage.INGEST, status=StageStatus.SKIPPED, skip_reason=reason
        ))
    else:
        logger.info(f"[runner] RUN  ingest  — {reason}")
        result = _run_ingest(config, targets, logger)
        stage_results.append(result)

        if result.status == StageStatus.FAILED:
            logger.error(f"[runner] INGEST FAILED\n{result.error}")
            pipeline_failed = True
        else:
            logger.info(f"[runner] ingest OK  ({result.duration_s}s)")
            # Keep ingestion results in memory — pass to preprocessing
            try:
                ingestion_results = run_ingestion(config, logger)
            except Exception:
                pass   # already ran above, this is just to get the result objects

    # -- Stage 2: PREPROCESS --
    will_run, reason = decisions[Stage.PREPROCESS]
    if pipeline_failed:
        reason = "skipped — upstream stage failed"
        will_run = False

    if not will_run:
        logger.info(f"[runner] SKIP preprocess — {reason}")
        stage_results.append(StageResult(
            stage=Stage.PREPROCESS, status=StageStatus.SKIPPED, skip_reason=reason
        ))
    else:
        logger.info(f"[runner] RUN  preprocess — {reason}")
        result = _run_preprocess(config, ingestion_results, targets, logger)
        stage_results.append(result)

        if result.status == StageStatus.FAILED:
            logger.error(f"[runner] PREPROCESS FAILED\n{result.error}")
            pipeline_failed = True
        else:
            logger.info(f"[runner] preprocess OK  ({result.duration_s}s)")

    # -- Stage 3: TRAIN --
    will_run, reason = decisions[Stage.TRAIN]
    if pipeline_failed:
        reason = "skipped — upstream stage failed"
        will_run = False

    if not will_run:
        logger.info(f"[runner] SKIP train  — {reason}")
        stage_results.append(StageResult(
            stage=Stage.TRAIN, status=StageStatus.SKIPPED, skip_reason=reason
        ))
    else:
        logger.info(f"[runner] RUN  train  — {reason}")
        result = _run_train(config, targets, logger)
        stage_results.append(result)

        if result.status == StageStatus.FAILED:
            logger.error(f"[runner] TRAIN FAILED\n{result.error}")
            pipeline_failed = True
        else:
            logger.info(f"[runner] train OK  ({result.duration_s}s)")

    # -- Stage 4: SCORE --
    will_run, reason = decisions[Stage.SCORE]
    if pipeline_failed:
        reason = "skipped — upstream stage failed"
        will_run = False

    if not will_run:
        logger.info(f"[runner] SKIP score  — {reason}")
        stage_results.append(StageResult(
            stage=Stage.SCORE, status=StageStatus.SKIPPED, skip_reason=reason
        ))
    else:
        logger.info(f"[runner] RUN  score  — {reason}")
        result = _run_score(config, targets, logger)
        stage_results.append(result)

        if result.status == StageStatus.FAILED:
            logger.error(f"[runner] SCORE FAILED\n{result.error}")
            pipeline_failed = True
        else:
            logger.info(f"[runner] score OK  ({result.duration_s}s)")

    # ── Determine overall status ──────────────────────────────
    statuses = [r.status for r in stage_results]
    if StageStatus.FAILED in statuses:
        overall = "FAILED"
    elif all(s == StageStatus.SKIPPED for s in statuses):
        overall = "ALL_SKIPPED"
    elif StageStatus.SKIPPED in statuses:
        overall = "PARTIAL"
    else:
        overall = "SUCCESS"

    finish_run(logger)

    # ── Build run summary ─────────────────────────────────────
    finished_at = datetime.now(timezone.utc).isoformat()
    total_secs  = round(time.time() - t_start, 2)

    summary = RunSummary(
        run_id           = run_id,
        started_at       = started_at,
        finished_at      = finished_at,
        duration_s       = total_secs,
        mode             = "full" if not requested_stages else "partial",
        metrics          = targets,
        force            = force,
        stages_requested = [s.value for s in (requested_stages or [])],
        stages           = stage_results,
        overall          = overall,
        config_path      = str(config_path),
    )

    # ── Save run summary JSON ─────────────────────────────────
    runs_dir = Path(config["artifacts"]["reports_dir"]) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / f"run_{run_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    # Also overwrite latest.json so tooling can always find the last run
    latest_path = runs_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    # ── Print final summary table ─────────────────────────────
    logger.info("=" * 65)
    logger.info("  PIPELINE RUN SUMMARY")
    logger.info("=" * 65)
    logger.info(f"  Run ID  : {run_id}")
    logger.info(f"  Overall : {overall}")
    logger.info(f"  Total   : {total_secs}s")
    logger.info("")
    logger.info(f"  {'Stage':12s}  {'Status':10s}  {'Time':>7s}  Detail")
    logger.info("  " + "-" * 55)

    for r in stage_results:
        time_str = f"{r.duration_s:>6.1f}s" if r.duration_s else "      -"
        detail   = r.skip_reason if r.status == StageStatus.SKIPPED else (
                   "OK" if r.status == StageStatus.SUCCESS else
                   r.error.split("\n")[-2] if r.error else "error"
        )
        icon = {"SUCCESS": "OK ", "SKIPPED": "-- ", "FAILED": "!! "}[r.status]
        logger.info(f"  {icon} {r.stage:12s}  {r.status:10s}  {time_str}  {detail[:40]}")

    logger.info("")
    logger.info(f"  Run summary -> {summary_path}")
    logger.info("=" * 65)

    if overall == "ALL_SKIPPED":
        logger.info("  All stages skipped — outputs are fresh.")
        logger.info("  Use --force to re-run everything.")

    return summary


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _script_dir  = Path(__file__).resolve().parent
    _default_cfg = _script_dir / "config" / "pipeline_config.json"
    if not _default_cfg.exists():
        _default_cfg = Path("config") / "pipeline_config.json"

    parser = argparse.ArgumentParser(
        description="SSL Anomaly Detection — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.pipeline_runner
  python -m src.pipeline.pipeline_runner --force
  python -m src.pipeline.pipeline_runner --stages train score
  python -m src.pipeline.pipeline_runner --metric rds_cpu --force
  python -m src.pipeline.pipeline_runner --dry-run
        """
    )
    parser.add_argument(
        "--config", default=str(_default_cfg),
        help="Path to pipeline_config.json"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force all stages to re-run, ignoring freshness checks"
    )
    parser.add_argument(
        "--stages", nargs="+",
        choices=[s.value for s in Stage.all()],
        default=None,
        help="Run only these specific stages (e.g. --stages train score)"
    )
    parser.add_argument(
        "--metric", default=None,
        help="Process a single metric only (e.g. --metric rds_cpu)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan without running anything"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path.resolve()}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    config = resolve_config_paths(config, config_path)
    logger = setup_logger("runner", config)

    requested = [Stage(s) for s in args.stages] if args.stages else None
    metrics   = [args.metric] if args.metric else None

    summary = run_pipeline(
        config           = config,
        config_path      = config_path,
        metrics          = metrics,
        force            = args.force,
        requested_stages = requested,
        dry_run          = args.dry_run,
        logger           = logger,
    )

    # Exit with non-zero code if pipeline failed — important for CI/CD
    if summary.overall == "FAILED":
        sys.exit(1)
