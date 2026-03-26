from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# W&B tracking — import our centralized tracker (no-op if disabled)
try:
    from wandb_tracker import init_run, finish_run, log_ingestion_quality, is_enabled
except ImportError:
    # Fallback stubs so ingestion works even without wandb_tracker on path
    def init_run(*a, **kw):    return None
    def finish_run(*a, **kw):  pass
    def log_ingestion_quality(*a, **kw): pass
    def is_enabled(*a, **kw):  return False
def setup_logger(name: str, config: dict) -> logging.Logger:
    log_cfg  = config["logging"]
    log_dir  = Path(log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger   = logging.getLogger(name)
    level    = getattr(logging, log_cfg.get("log_level", "INFO"))
    logger.setLevel(level)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    if log_cfg.get("log_to_console", True):
        # Force UTF-8 on Windows — cp1252 console cannot encode box-drawing chars
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_cfg.get("log_to_file", True):
        log_file = log_dir / f"ingestion_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


@dataclass
class QualityReport:
    """Per-metric data quality summary. Written to disk after ingestion."""
    metric_name:         str
    source_file:         str
    ingested_at:         str
    raw_row_count:       int
    final_row_count:     int
    null_count:          int
    null_pct:            float
    duplicate_count:     int
    duplicate_pct:       float
    max_gap_minutes:     float
    gaps_filled:         int
    outliers_clipped_hi: int
    outliers_clipped_lo: int
    value_min:           float
    value_max:           float
    value_mean:          float
    value_std:           float
    date_start:          str
    date_end:            str
    expected_min:        float
    expected_max:        float
    range_violations:    int
    passed_quality_gate: bool
    failure_reasons:     list = field(default_factory=list)
    checksum_md5:        str  = ""


@dataclass
class IngestionResult:
    metric_name:    str
    series:         pd.Series          # time-indexed, cleaned, ready for preprocessing
    quality_report: QualityReport
    checksum:       str
    passed:         bool               # False = quality gate failed, pipeline should stop



def _md5_checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(val) -> float:
    """Convert numpy types to plain Python float for JSON serialization."""
    try:
        return float(val)
    except Exception:
        return 0.0


def validate_schema(df: pd.DataFrame,
                    metric_name: str,
                    ing_cfg: dict,
                    logger: logging.Logger) -> pd.DataFrame:
    ts_col  = ing_cfg["timestamp_col"]
    val_col = ing_cfg["value_col"]
    missing = [c for c in [ts_col, val_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{metric_name}] Schema violation: missing columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
    except Exception as e:
        raise ValueError(
            f"[{metric_name}] Cannot parse '{ts_col}' as datetime: {e}"
        )
    if not pd.api.types.is_numeric_dtype(df[val_col]):
        coerced = pd.to_numeric(df[val_col], errors="coerce")
        n_failed = coerced.isna().sum() - df[val_col].isna().sum()
        if n_failed > 0:
            raise ValueError(
                f"[{metric_name}] '{val_col}' has {n_failed} non-numeric values"
            )
        df[val_col] = coerced
        logger.warning(f"[{metric_name}] value column coerced to numeric")

    logger.debug(f"[{metric_name}] Schema validation passed")
    return df


def clean_series(raw: pd.Series,
                 metric_name: str,
                 ing_cfg: dict,
                 qual_cfg: dict,
                 report: QualityReport,
                 logger: logging.Logger) -> pd.Series:
    resample_freq    = ing_cfg["resample_freq"]
    max_gap_steps    = ing_cfg["max_gap_fill_steps"]
    lower_pct        = qual_cfg["outlier_lower_pct"]
    upper_pct        = qual_cfg["outlier_upper_pct"]

    n_dups = raw.index.duplicated().sum()
    if n_dups > 0:
        raw = raw.groupby(level=0).mean()
        logger.info(f"[{metric_name}] Deduplicated {n_dups} duplicate timestamps")
    report.duplicate_count = int(n_dups)
    report.duplicate_pct   = round(n_dups / max(report.raw_row_count, 1) * 100, 3)

    gaps = raw.index.to_series().diff().dt.total_seconds().div(60).dropna()
    report.max_gap_minutes = _safe_float(gaps.max()) if len(gaps) else 0.0

    raw = raw.resample(resample_freq).mean()
    n_missing = int(raw.isna().sum())

    if n_missing > 0:
        raw = raw.ffill(limit=max_gap_steps).bfill(limit=max_gap_steps)
        still_missing = int(raw.isna().sum())
        filled = n_missing - still_missing
        report.gaps_filled = filled

        if still_missing > 0:
            logger.warning(
                f"[{metric_name}] {still_missing} NaNs remain after gap-fill "
                f"(gap too large to fill — dropping these rows)"
            )
            raw = raw.dropna()
        else:
            logger.info(f"[{metric_name}] Filled {filled} missing slots via forward-fill")
    else:
        report.gaps_filled = 0

    p_lo = raw.quantile(lower_pct / 100)
    p_hi = raw.quantile(upper_pct / 100)
    n_hi = int((raw > p_hi).sum())
    n_lo = int((raw < p_lo).sum())
    raw  = raw.clip(lower=p_lo, upper=p_hi)
    report.outliers_clipped_hi = n_hi
    report.outliers_clipped_lo = n_lo

    if n_hi + n_lo > 0:
        logger.info(
            f"[{metric_name}] Clipped {n_hi} high / {n_lo} low outliers "
            f"[p{lower_pct:.0f}={p_lo:.2f}, p{upper_pct:.0f}={p_hi:.2f}]"
        )

    exp_min = report.expected_min
    exp_max = report.expected_max
    violations = int(((raw < exp_min) | (raw > exp_max)).sum())
    report.range_violations = violations
    if violations > 0:
        logger.warning(
            f"[{metric_name}] {violations} values outside expected range "
            f"[{exp_min}, {exp_max}]"
        )

    report.final_row_count = len(raw)
    report.value_min       = _safe_float(raw.min())
    report.value_max       = _safe_float(raw.max())
    report.value_mean      = _safe_float(raw.mean())
    report.value_std       = _safe_float(raw.std())
    report.date_start      = str(raw.index[0])
    report.date_end        = str(raw.index[-1])

    logger.debug(
        f"[{metric_name}] Clean complete — {report.final_row_count} rows, "
        f"range [{report.value_min:.2f}, {report.value_max:.2f}]"
    )
    return raw

def run_quality_gate(report: QualityReport,
                     qual_cfg: dict,
                     logger: logging.Logger) -> bool:
    failures = []

    if report.final_row_count < qual_cfg["min_rows_required"]:
        failures.append(
            f"Too few rows: {report.final_row_count} < {qual_cfg['min_rows_required']}"
        )

    if report.null_pct > qual_cfg["max_null_pct"]:
        failures.append(
            f"Null % too high: {report.null_pct:.2f}% > {qual_cfg['max_null_pct']}%"
        )

    if report.duplicate_pct > qual_cfg["max_duplicate_pct"]:
        failures.append(
            f"Duplicate % too high: {report.duplicate_pct:.2f}% > {qual_cfg['max_duplicate_pct']}%"
        )

    report.failure_reasons    = failures
    report.passed_quality_gate = len(failures) == 0

    if failures:
        for reason in failures:
            logger.error(f"[{report.metric_name}] QUALITY GATE FAILED: {reason}")
    else:
        logger.info(f"[{report.metric_name}] Quality gate PASSED [OK]")

    return report.passed_quality_gate


def ingest_metric(metric_name: str,
                  config: dict,
                  logger: logging.Logger) -> IngestionResult:
    """
    Full ingestion pipeline for one metric.

    Returns an IngestionResult with .passed = False if quality gate
    fails — caller decides whether to abort or continue.
    """
    ing_cfg  = config["ingestion"]
    qual_cfg = config["quality"]
    art_cfg  = config["artifacts"]

    metric_meta = ing_cfg["metrics"][metric_name]
    source_file = metric_meta["filename"]
    source_path = Path(ing_cfg["source_dir"]) / source_file

    logger.info(f"[{metric_name}] -- Starting ingestion ------------------")
    logger.info(f"[{metric_name}] Source: {source_path}")

    if not source_path.exists():
        raise FileNotFoundError(
            f"[{metric_name}] Raw data file not found: {source_path}\n"
            f"  Expected location: {source_path.resolve()}\n"
            f"  Make sure CSVs are placed in {ing_cfg['source_dir']}/"
        )

    checksum = _md5_checksum(source_path)
    logger.info(f"[{metric_name}] MD5 checksum: {checksum}")

    try:
        df = pd.read_csv(source_path)
    except Exception as e:
        raise IOError(f"[{metric_name}] Failed to read CSV: {e}")

    raw_rows = len(df)
    logger.info(f"[{metric_name}] Loaded {raw_rows:,} raw rows")

    df = validate_schema(df, metric_name, ing_cfg, logger)

    ts_col  = ing_cfg["timestamp_col"]
    val_col = ing_cfg["value_col"]
    df = df.sort_values(ts_col).set_index(ts_col)
    series = df[val_col].rename(metric_name)

    null_count = int(series.isna().sum())

    report = QualityReport(
        metric_name         = metric_name,
        source_file         = source_file,
        ingested_at         = datetime.now(timezone.utc).isoformat(),
        raw_row_count       = raw_rows,
        final_row_count     = raw_rows,
        null_count          = null_count,
        null_pct            = round(null_count / max(raw_rows, 1) * 100, 3),
        duplicate_count     = 0,
        duplicate_pct       = 0.0,
        max_gap_minutes     = 0.0,
        gaps_filled         = 0,
        outliers_clipped_hi = 0,
        outliers_clipped_lo = 0,
        value_min           = _safe_float(series.min()),
        value_max           = _safe_float(series.max()),
        value_mean          = _safe_float(series.mean()),
        value_std           = _safe_float(series.std()),
        date_start          = str(series.index.min()),
        date_end            = str(series.index.max()),
        expected_min        = metric_meta["expected_min"],
        expected_max        = metric_meta["expected_max"],
        range_violations    = 0,
        passed_quality_gate = False,
        checksum_md5        = checksum,
    )

    series = clean_series(series, metric_name, ing_cfg, qual_cfg, report, logger)

    passed = run_quality_gate(report, qual_cfg, logger)

    reports_dir = Path(art_cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{metric_name}_quality_report.json"

    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info(f"[{metric_name}] Quality report → {report_path}")

    if art_cfg.get("save_checksums", True):
        manifest_path = Path(art_cfg["reports_dir"]) / "checksums.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        manifest[metric_name] = {
            "file":        source_file,
            "md5":         checksum,
            "ingested_at": report.ingested_at,
            "rows":        report.final_row_count,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info(
        f"[{metric_name}] Ingestion complete — "
        f"{report.final_row_count:,} rows | gate={'PASS' if passed else 'FAIL'}"
    )

    return IngestionResult(
        metric_name    = metric_name,
        series         = series,
        quality_report = report,
        checksum       = checksum,
        passed         = passed,
    )



def run_ingestion(config: dict,
                  logger: logging.Logger) -> Dict[str, IngestionResult]:
    """
    Ingest all metrics defined in config["ingestion"]["metrics"].

    Returns a dict of metric_name → IngestionResult.
    Raises RuntimeError if ANY metric fails its quality gate,
    because training on partial bad data is worse than not training.
    """
    logger.info("=" * 65)
    logger.info("  DATA INGESTION  START")
    logger.info("=" * 65)

    init_run(config, stage="ingestion", logger=logger)

    results  = {}
    failures = []

    for metric_name in config["ingestion"]["metrics"]:
        try:
            result = ingest_metric(metric_name, config, logger)
            results[metric_name] = result
            if not result.passed:
                failures.append(metric_name)
        except (FileNotFoundError, IOError, ValueError) as e:
            logger.error(f"[{metric_name}] INGESTION ERROR: {e}")
            failures.append(metric_name)

    logger.info("=" * 65)
    logger.info(f"  INGESTION SUMMARY: {len(results)}/{len(config['ingestion']['metrics'])} metrics ingested")
    for name, res in results.items():
        status = "[PASS]" if res.passed else "[FAIL]"
        logger.info(
            f"  {status} | {name:15s} | "
            f"{res.quality_report.final_row_count:>5} rows | "
            f"md5={res.checksum[:8]}..."
        )

    if failures:
        finish_run(logger)
        raise RuntimeError(
            f"Ingestion quality gate failed for: {failures}. "
            f"Check quality reports in {config['artifacts']['reports_dir']}/"
        )

    # Log all quality reports to W&B as a table + summary metrics
    if is_enabled(config):
        quality_reports = [results[n].quality_report for n in results
                           if hasattr(results[n], "quality_report")]
        # Convert dataclasses to dicts
        from dataclasses import asdict as _asdict
        log_ingestion_quality(
            [_asdict(r) for r in quality_reports],
            config, logger
        )

    finish_run(logger)
    logger.info("=" * 65)
    return results


def resolve_config_paths(config: dict, config_path: Path) -> dict:
    cwd      = Path.cwd()
    cfg_dir  = config_path.resolve().parent

    def to_abs(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        # Prefer cwd-relative (standard: run from project root)
        cwd_candidate = cwd / path
        if cwd_candidate.exists():
            return str(cwd_candidate)
        # Fallback: config-dir-relative
        cfg_candidate = cfg_dir / path
        if cfg_candidate.exists():
            return str(cfg_candidate)
        # Neither exists yet (output dirs don't exist before first run)
        # Default to cwd-relative so they get created in the right place
        return str(cwd_candidate)

    config["ingestion"]["source_dir"]  = to_abs(config["ingestion"]["source_dir"])
    config["artifacts"]["output_dir"]  = to_abs(config["artifacts"]["output_dir"])
    config["artifacts"]["reports_dir"] = to_abs(config["artifacts"]["reports_dir"])
    config["artifacts"]["scaler_dir"]  = to_abs(config["artifacts"]["scaler_dir"])
    config["logging"]["log_dir"]       = to_abs(config["logging"]["log_dir"])
    return config


if __name__ == "__main__":
    import argparse

    # Look for config next to this script file first, then fall back to cwd
    _script_dir  = Path(__file__).resolve().parent
    _default_cfg = _script_dir / "config" / "pipeline_config.json"
    if not _default_cfg.exists():
        _default_cfg = Path("config") / "pipeline_config.json"

    parser = argparse.ArgumentParser(description="Run data ingestion pipeline")
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
    logger = setup_logger("ingestion", config)
    results = run_ingestion(config, logger)
    print(f"Ingestion complete. {len(results)} metrics ready for preprocessing.")