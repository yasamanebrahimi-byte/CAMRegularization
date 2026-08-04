#!/usr/bin/env python3
"""Generate across-seed validation summaries for the CAM cutout experiments.

This script reads run-level ``config.json`` and ``metrics.csv`` files under
``runs/`` and writes only under ``runs/summary/``. It does not run training or
modify source experiment artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


SUMMARY_DIR = Path(__file__).resolve().parent
RUNS_ROOT = SUMMARY_DIR.parent
REPO_ROOT = RUNS_ROOT.parent
TABLES_DIR = SUMMARY_DIR / "tables"
PLOTS_DIR = SUMMARY_DIR / "plots"

EXPECTED_DATASETS = ("cifar100", "drive_zip")
DATASET_LABELS = {"cifar100": "CIFAR-100", "drive_zip": "RawMal-TF"}
DATASET_SLUGS = {"cifar100": "cifar100", "drive_zip": "rawmal_tf"}
EXPECTED_ARCHITECTURE = "resnet18"
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_EPOCHS = 100
EXPECTED_AREAS = (0.05, 0.10, 0.20, 0.30)
EXPECTED_MS = (4, 8)
CONDITIONS = ("none", "random", "cam_low", "cam_high")
AUGMENTED_CONDITIONS = ("random", "cam_low", "cam_high")
REQUIRED_METRIC_COLUMNS = ("epoch", "train_loss", "train_acc1", "eval_loss", "eval_acc1")
COLLAPSE_THRESHOLD = 0.05

PLOT_DIRS = (
    "best_accuracy_by_area",
    "final_accuracy_by_area",
    "paired_cam_effects",
    "aulc_by_area",
    "learning_curves",
    "stability",
    "m8_minus_m4",
    "mean_heatmaps",
    "variability_heatmaps",
    "mean_vs_variability",
    "condition_comparisons",
)

CONDITION_LABELS = {
    "none": "No cutout",
    "random": "Random cutout",
    "cam_low": "Low-saliency cutout",
    "cam_high": "High-saliency cutout",
}
CONDITION_SHORT_LABELS = {
    "none": "No cutout",
    "random": "Random",
    "cam_low": "Low-saliency",
    "cam_high": "High-saliency",
}
CONDITION_COLORS = {
    "none": "#4d4d4d",
    "random": "#0072B2",
    "cam_low": "#009E73",
    "cam_high": "#D55E00",
}
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}
M_MARKERS = {4: "o", 8: "s", None: "D"}

PER_RUN_METRICS = (
    "best_validation_accuracy",
    "best_validation_epoch",
    "final_validation_accuracy",
    "best_to_final_degradation",
    "validation_aulc",
    "training_accuracy_at_best_validation_epoch",
    "train_validation_gap_at_best_epoch",
    "final_train_validation_gap",
    "final20_validation_accuracy_mean",
    "final20_validation_accuracy_std",
    "maximum_validation_drawdown",
    "collapse_event_count",
)
PAIRED_EFFECT_METRICS = (
    "best_validation_accuracy",
    "final_validation_accuracy",
    "validation_aulc",
    "best_to_final_degradation",
    "maximum_validation_drawdown",
)
PERCENT_LIKE_METRICS = {
    "best_validation_accuracy",
    "final_validation_accuracy",
    "best_to_final_degradation",
    "validation_aulc",
    "training_accuracy_at_best_validation_epoch",
    "train_validation_gap_at_best_epoch",
    "final_train_validation_gap",
    "final20_validation_accuracy_mean",
    "final20_validation_accuracy_std",
    "maximum_validation_drawdown",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_summary_dir() -> None:
    """Remove stale generated artifacts while leaving this script in place."""
    summary = SUMMARY_DIR.resolve()
    script = Path(__file__).resolve()
    for child in list(SUMMARY_DIR.iterdir()):
        resolved = child.resolve()
        if resolved == script:
            continue
        if summary not in resolved.parents:
            raise RuntimeError(f"Refusing to remove path outside summary: {child}")
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in PLOT_DIRS:
        (PLOTS_DIR / dirname).mkdir(parents=True, exist_ok=True)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def display_dataset(dataset_id: Any) -> str:
    return DATASET_LABELS.get(str(dataset_id), str(dataset_id or ""))


def dataset_slug_from_label(dataset_label: str) -> str:
    for dataset_id, label in DATASET_LABELS.items():
        if dataset_label == label:
            return DATASET_SLUGS[dataset_id]
    return dataset_label.lower().replace(" ", "_").replace("-", "_")


def safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "" or pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_area(value: Any) -> float | None:
    number = safe_float(value)
    if number is None:
        return None
    for expected in EXPECTED_AREAS:
        if abs(number - expected) < 1e-9:
            return expected
    return number


def area_label(area: Any) -> str:
    number = safe_float(area)
    if number is None:
        return ""
    return f"{number:.2f}"


def area_slug(area: float) -> str:
    return f"area{area:.2f}".replace(".", "p")


def metric_label(metric: str) -> str:
    labels = {
        "best_validation_accuracy": "Best validation accuracy",
        "best_validation_epoch": "Earliest best-validation epoch",
        "final_validation_accuracy": "Final validation accuracy",
        "best_to_final_degradation": "Best-to-final degradation",
        "validation_aulc": "Normalized validation AULC",
        "training_accuracy_at_best_validation_epoch": "Training accuracy at best-validation epoch",
        "train_validation_gap_at_best_epoch": "Train-validation gap at best epoch",
        "final_train_validation_gap": "Final train-validation gap",
        "final20_validation_accuracy_mean": "Final-20 validation accuracy mean",
        "final20_validation_accuracy_std": "Final-20 validation accuracy standard deviation",
        "maximum_validation_drawdown": "Maximum validation drawdown",
        "collapse_event_count": "Collapse-event count",
    }
    return labels.get(metric, metric.replace("_", " "))


def normalize_baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.pop("out_dir", None)
    return normalized


def read_metrics(metrics_path: Path) -> tuple[pd.DataFrame | None, list[str]]:
    issues: list[str] = []
    if not metrics_path.exists():
        return None, ["missing_metrics_csv"]
    try:
        frame = pd.read_csv(metrics_path)
    except Exception as exc:  # noqa: BLE001 - the integrity table needs parser detail
        return None, [f"metrics_csv_parse_error:{exc}"]

    missing_columns = [column for column in REQUIRED_METRIC_COLUMNS if column not in frame.columns]
    if missing_columns:
        issues.append("missing_metric_columns:" + ",".join(missing_columns))

    if len(frame) != EXPECTED_EPOCHS:
        issues.append(f"wrong_epoch_count:{len(frame)}")

    if "epoch" in frame.columns:
        epochs = pd.to_numeric(frame["epoch"], errors="coerce")
        if len(frame) == EXPECTED_EPOCHS:
            expected_epochs = np.arange(1, EXPECTED_EPOCHS + 1, dtype=float)
            if epochs.isna().any() or not np.array_equal(epochs.to_numpy(dtype=float), expected_epochs):
                issues.append("epoch_sequence_not_1_to_100")
        frame["epoch"] = epochs

    for column in REQUIRED_METRIC_COLUMNS:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            issues.append(f"non_numeric_metric_column:{column}")
        frame[column] = numeric

    if "eval_split" in frame.columns:
        split_values = {str(value).strip().lower() for value in frame["eval_split"].dropna().unique()}
        if split_values and split_values != {"val"}:
            issues.append("eval_split_not_validation:" + ",".join(sorted(split_values)))

    return frame, issues


def validate_path_metadata(config_path: Path, config: dict[str, Any], mode: str, area: float | None) -> list[str]:
    issues: list[str] = []
    try:
        parts = config_path.resolve().relative_to(RUNS_ROOT.resolve()).parts
    except ValueError:
        return ["config_outside_runs_root"]

    if len(parts) < 6:
        return ["unexpected_or_old_flat_run_path"]

    path_dataset, path_architecture, path_seed, path_area, run_folder = parts[:5]
    config_dataset = config.get("dataset")
    config_architecture = config.get("model", config.get("architecture"))
    config_seed = config.get("seed")
    if config_dataset is not None and path_dataset != str(config_dataset):
        issues.append(f"path_dataset_mismatch:{path_dataset}")
    if config_architecture is not None and path_architecture != str(config_architecture):
        issues.append(f"path_architecture_mismatch:{path_architecture}")
    if config_seed is not None and path_seed != str(config_seed):
        issues.append(f"path_seed_mismatch:{path_seed}")
    if mode == "none":
        if canonical_area(path_area) not in EXPECTED_AREAS:
            issues.append(f"baseline_path_area_unexpected:{path_area}")
    elif area is not None and canonical_area(path_area) != area:
        issues.append(f"path_cutout_area_mismatch:{path_area}")

    run_name = str(config.get("run_name", ""))
    if run_name and run_folder != run_name:
        issues.append(f"run_name_folder_mismatch:{run_folder}")
    return issues


def discover_runs() -> tuple[list[dict[str, Any]], dict[int, pd.DataFrame]]:
    records: list[dict[str, Any]] = []
    metric_frames: dict[int, pd.DataFrame] = {}
    config_paths = sorted(
        path
        for path in RUNS_ROOT.rglob("config.json")
        if SUMMARY_DIR.resolve() not in path.resolve().parents
    )

    for run_id, config_path in enumerate(config_paths):
        metrics_path = config_path.with_name("metrics.csv")
        record: dict[str, Any] = {
            "run_id": run_id,
            "source_path": relative_path(config_path.parent),
            "config_path": relative_path(config_path),
            "metrics_path": relative_path(metrics_path),
            "dataset_id": None,
            "dataset": "",
            "architecture": "",
            "seed": None,
            "condition": "",
            "mode": "",
            "M": None,
            "cutout_area": None,
            "config_epochs": None,
            "metrics_epoch_count": None,
            "required_columns_ok": False,
            "epoch_sequence_ok": False,
            "metrics_sha256": "",
            "config_sha256": "",
            "normalized_baseline_config_sha256": "",
            "status": "invalid",
            "analysis_valid": False,
            "issues": [],
        }

        try:
            config_text = config_path.read_text(encoding="utf-8")
            config = json.loads(config_text)
            record["config_sha256"] = sha256_text(config_text)
        except Exception as exc:  # noqa: BLE001 - recorded for the integrity report
            record["issues"].append(f"config_parse_error:{exc}")
            records.append(record)
            continue

        dataset_id = config.get("dataset")
        architecture = config.get("model", config.get("architecture"))
        seed = safe_int(config.get("seed"))
        mode = str(config.get("cutout_mode", ""))
        m_value = safe_int(config.get("cutout_m"))
        cutout_area = canonical_area(config.get("cutout_area"))
        config_epochs = safe_int(config.get("epochs"))

        record.update(
            {
                "dataset_id": dataset_id,
                "dataset": display_dataset(dataset_id),
                "architecture": str(architecture or ""),
                "seed": seed,
                "mode": mode,
                "condition": CONDITION_LABELS.get(mode, str(mode or "")),
                "M": None if mode == "none" else m_value,
                "cutout_area": None if mode == "none" else cutout_area,
                "config_epochs": config_epochs,
            }
        )
        if mode == "none":
            record["normalized_baseline_config_sha256"] = sha256_text(canonical_json(normalize_baseline_config(config)))

        if dataset_id not in EXPECTED_DATASETS:
            record["issues"].append(f"unexpected_dataset:{dataset_id}")
        if architecture != EXPECTED_ARCHITECTURE:
            record["issues"].append(f"unexpected_architecture:{architecture}")
        if seed not in EXPECTED_SEEDS:
            record["issues"].append(f"unexpected_seed:{seed}")
        if config_epochs != EXPECTED_EPOCHS:
            record["issues"].append(f"unexpected_config_epoch_count:{config_epochs}")
        if mode not in CONDITIONS:
            record["issues"].append(f"unexpected_cutout_mode:{mode}")
        elif mode == "none":
            if m_value not in (0, None):
                record["issues"].append(f"baseline_cutout_m_not_zero:{m_value}")
            if cutout_area is not None:
                record["issues"].append(f"baseline_cutout_area_not_null:{cutout_area}")
        else:
            if m_value not in EXPECTED_MS:
                record["issues"].append(f"unexpected_cutout_m:{m_value}")
            if cutout_area not in EXPECTED_AREAS:
                record["issues"].append(f"unexpected_cutout_area:{cutout_area}")

        record["issues"].extend(validate_path_metadata(config_path, config, mode, cutout_area))

        frame, metric_issues = read_metrics(metrics_path)
        record["issues"].extend(metric_issues)
        if metrics_path.exists():
            record["metrics_sha256"] = sha256_file(metrics_path)
        if frame is not None:
            record["metrics_epoch_count"] = len(frame)
            record["required_columns_ok"] = not any(issue.startswith("missing_metric_columns") for issue in metric_issues)
            record["epoch_sequence_ok"] = "epoch_sequence_not_1_to_100" not in metric_issues and len(frame) == EXPECTED_EPOCHS
            metric_frames[run_id] = frame

        if record["issues"]:
            unexpected_prefixes = (
                "unexpected_dataset",
                "unexpected_architecture",
                "unexpected_seed",
                "unexpected_cutout_mode",
                "unexpected_cutout_m",
                "unexpected_cutout_area",
                "unexpected_or_old_flat_run_path",
            )
            record["status"] = (
                "ignored_unexpected"
                if any(str(issue).startswith(unexpected_prefixes) for issue in record["issues"])
                else "invalid"
            )
        else:
            record["status"] = "valid_candidate"
        records.append(record)

    return records, metric_frames


def select_analysis_runs(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    candidates = [record for record in records if record["status"] == "valid_candidate"]
    baselines: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    augmented: dict[tuple[str, int, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    duplicate_rows: list[dict[str, Any]] = []

    for record in candidates:
        dataset_id = str(record["dataset_id"])
        seed = int(record["seed"])
        mode = str(record["mode"])
        if mode == "none":
            baselines[(dataset_id, seed)].append(record)
        else:
            augmented[(dataset_id, seed, mode, int(record["M"]), float(record["cutout_area"]))].append(record)

    for (dataset_id, seed), group in baselines.items():
        metric_counts = Counter(record["metrics_sha256"] for record in group)
        normalized_config_counts = Counter(record["normalized_baseline_config_sha256"] for record in group)
        raw_config_counts = Counter(record["config_sha256"] for record in group)
        selected_metric_hash = sorted(metric_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        selected_config_hash = sorted(normalized_config_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        selected_pool = [
            record
            for record in group
            if record["metrics_sha256"] == selected_metric_hash
            and record["normalized_baseline_config_sha256"] == selected_config_hash
        ]
        selected = sorted(selected_pool, key=lambda record: record["source_path"])[0]
        copies_differ = len(metric_counts) > 1 or len(normalized_config_counts) > 1
        raw_config_path_only = len(raw_config_counts) > 1 and len(normalized_config_counts) == 1

        for record in sorted(group, key=lambda item: item["source_path"]):
            same_as_selected = (
                record["metrics_sha256"] == selected_metric_hash
                and record["normalized_baseline_config_sha256"] == selected_config_hash
            )
            if record is selected:
                record["analysis_valid"] = True
                record["status"] = "valid_selected_baseline"
                if copies_differ:
                    record["issues"].append("selected_from_baseline_group_with_differences")
            elif same_as_selected:
                record["analysis_valid"] = False
                record["status"] = "duplicate_baseline_exact"
                record["issues"].append("deduplicated_exact_no_cutout_copy")
            else:
                record["analysis_valid"] = False
                record["status"] = "duplicate_baseline_inconsistent"
                record["issues"].append("excluded_no_cutout_copy_differs_from_selected")

            duplicate_rows.append(
                {
                    "dataset": display_dataset(dataset_id),
                    "seed": seed,
                    "source_path": record["source_path"],
                    "selected_for_analysis": record is selected,
                    "baseline_group_size": len(group),
                    "baseline_copies_differ": copies_differ,
                    "metrics_match_selected": record["metrics_sha256"] == selected_metric_hash,
                    "normalized_config_matches_selected": record["normalized_baseline_config_sha256"] == selected_config_hash,
                    "raw_config_path_only_differences_in_group": raw_config_path_only,
                    "metrics_sha256": record["metrics_sha256"],
                    "metrics_hash_count": metric_counts[record["metrics_sha256"]],
                    "normalized_baseline_config_sha256": record["normalized_baseline_config_sha256"],
                    "normalized_config_hash_count": normalized_config_counts[record["normalized_baseline_config_sha256"]],
                    "raw_config_sha256": record["config_sha256"],
                    "raw_config_hash_count": raw_config_counts[record["config_sha256"]],
                    "status": record["status"],
                    "issues": ";".join(record["issues"]),
                }
            )

    for _signature, group in augmented.items():
        if len(group) == 1:
            group[0]["analysis_valid"] = True
            group[0]["status"] = "valid"
            continue
        for record in group:
            record["analysis_valid"] = False
            record["status"] = "duplicate_run"
            record["issues"].append("duplicate_nonbaseline_signature")

    analysis_runs = [record for record in records if record["analysis_valid"]]
    duplicate_table = pd.DataFrame(
        duplicate_rows,
        columns=[
            "dataset",
            "seed",
            "source_path",
            "selected_for_analysis",
            "baseline_group_size",
            "baseline_copies_differ",
            "metrics_match_selected",
            "normalized_config_matches_selected",
            "raw_config_path_only_differences_in_group",
            "metrics_sha256",
            "metrics_hash_count",
            "normalized_baseline_config_sha256",
            "normalized_config_hash_count",
            "raw_config_sha256",
            "raw_config_hash_count",
            "status",
            "issues",
        ],
    )
    return analysis_runs, duplicate_table


def compute_drawdown_and_collapses(values: np.ndarray) -> tuple[float, int]:
    running_best = float(values[0])
    max_drawdown = 0.0
    collapse_count = 0
    in_collapse = False

    for value in values[1:]:
        value = float(value)
        drop = running_best - value
        max_drawdown = max(max_drawdown, drop)
        if drop >= COLLAPSE_THRESHOLD:
            if not in_collapse:
                collapse_count += 1
                in_collapse = True
        else:
            in_collapse = False
        if value > running_best:
            running_best = value
            in_collapse = False

    return max_drawdown, collapse_count


def trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x=x))
    return float(np.trapz(values, x=x))


def compute_per_run_metrics(analysis_runs: list[dict[str, Any]], metric_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in sorted(
        analysis_runs,
        key=lambda item: (
            str(item["dataset"]),
            int(item["seed"]),
            str(item["mode"]),
            -1 if item["M"] is None else int(item["M"]),
            -1.0 if item["cutout_area"] is None else float(item["cutout_area"]),
            item["source_path"],
        ),
    ):
        frame = metric_frames[record["run_id"]]
        epochs = frame["epoch"].to_numpy(dtype=float)
        val_acc = frame["eval_acc1"].to_numpy(dtype=float)
        train_acc = frame["train_acc1"].to_numpy(dtype=float)
        val_loss = frame["eval_loss"].to_numpy(dtype=float)
        train_loss = frame["train_loss"].to_numpy(dtype=float)

        best_value = float(val_acc.max())
        best_index = int(np.flatnonzero(val_acc == best_value)[0])
        final_value = float(val_acc[-1])
        final20 = val_acc[-20:]
        max_drawdown, collapse_count = compute_drawdown_and_collapses(val_acc)
        m_value = None if record["mode"] == "none" else int(record["M"])
        cutout_area = None if record["mode"] == "none" else float(record["cutout_area"])

        rows.append(
            {
                "dataset": record["dataset"],
                "architecture": record["architecture"],
                "seed": int(record["seed"]),
                "condition": record["condition"],
                "mode": record["mode"],
                "M": m_value,
                "cutout_area": cutout_area,
                "epochs": EXPECTED_EPOCHS,
                "source_path": record["source_path"],
                "best_validation_accuracy": best_value,
                "best_validation_epoch": int(epochs[best_index]),
                "final_validation_accuracy": final_value,
                "best_to_final_degradation": best_value - final_value,
                "validation_aulc": trapezoid(val_acc, epochs) / float(epochs[-1] - epochs[0]),
                "training_accuracy_at_best_validation_epoch": float(train_acc[best_index]),
                "train_validation_gap_at_best_epoch": float(train_acc[best_index] - best_value),
                "final_train_validation_gap": float(train_acc[-1] - final_value),
                "final20_validation_accuracy_mean": float(final20.mean()),
                "final20_validation_accuracy_std": float(final20.std(ddof=1)),
                "maximum_validation_drawdown": max_drawdown,
                "collapse_event_count": collapse_count,
                "validation_loss_at_best_validation_epoch": float(val_loss[best_index]),
                "final_validation_loss": float(val_loss[-1]),
                "final_training_loss": float(train_loss[-1]),
            }
        )
    return pd.DataFrame(rows)


def t_critical_975(df: int) -> float:
    table = {
        1: 12.706204736432095,
        2: 4.302652729911275,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636305,
        6: 2.4469118487916806,
        7: 2.3646242515927844,
        8: 2.306004135204166,
        9: 2.2621571627409915,
        10: 2.2281388519649385,
    }
    return table.get(df, 1.959963984540054)


def stats_for_values(values: list[Any]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    n = len(clean)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sample_variance": np.nan,
            "sample_std": np.nan,
            "standard_error": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    mean = float(np.mean(clean))
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sample_variance": np.nan,
            "sample_std": np.nan,
            "standard_error": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "min": float(np.min(clean)),
            "max": float(np.max(clean)),
        }
    sample_variance = float(np.var(clean, ddof=1))
    sample_std = float(np.std(clean, ddof=1))
    standard_error = sample_std / math.sqrt(n)
    half_width = t_critical_975(n - 1) * standard_error
    return {
        "n": n,
        "mean": mean,
        "sample_variance": sample_variance,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def expected_parameter_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in EXPECTED_DATASETS:
        rows.append(
            {
                "dataset": display_dataset(dataset_id),
                "condition": CONDITION_LABELS["none"],
                "mode": "none",
                "M": None,
                "cutout_area": None,
            }
        )
        for mode in AUGMENTED_CONDITIONS:
            for m_value in EXPECTED_MS:
                for area in EXPECTED_AREAS:
                    rows.append(
                        {
                            "dataset": display_dataset(dataset_id),
                            "condition": CONDITION_LABELS[mode],
                            "mode": mode,
                            "M": m_value,
                            "cutout_area": area,
                        }
                    )
    return rows


def subset_for(per_run: pd.DataFrame, dataset: str, mode: str, m_value: int | None, area: float | None) -> pd.DataFrame:
    subset = per_run[(per_run["dataset"] == dataset) & (per_run["mode"] == mode)]
    if mode == "none":
        return subset[subset["M"].isna() & subset["cutout_area"].isna()]
    return subset[(subset["M"] == m_value) & (np.isclose(subset["cutout_area"].astype(float), float(area)))]


def build_aggregate_table(per_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for grid_row in expected_parameter_grid():
        subset = subset_for(
            per_run,
            grid_row["dataset"],
            grid_row["mode"],
            grid_row["M"],
            grid_row["cutout_area"],
        )
        seeds_present = sorted(int(seed) for seed in subset["seed"].dropna().unique())
        seeds_missing = [seed for seed in EXPECTED_SEEDS if seed not in seeds_present]
        row: dict[str, Any] = {
            "dataset": grid_row["dataset"],
            "architecture": EXPECTED_ARCHITECTURE,
            "condition": grid_row["condition"],
            "mode": grid_row["mode"],
            "M": grid_row["M"],
            "cutout_area": grid_row["cutout_area"],
            "valid_seed_count": len(seeds_present),
            "seeds_present": ";".join(str(seed) for seed in seeds_present),
            "seeds_missing": ";".join(str(seed) for seed in seeds_missing),
        }
        for metric in PER_RUN_METRICS:
            values = [float(value) for value in subset.sort_values("seed")[metric].dropna().tolist()]
            stats_row = stats_for_values(values)
            for key, value in stats_row.items():
                suffix = "valid_seed_count" if key == "n" else key
                row[f"{metric}_{suffix}"] = value
        rows.append(row)

    table = pd.DataFrame(rows)
    mode_order = {"none": 0, "random": 1, "cam_low": 2, "cam_high": 3}
    table["_dataset_order"] = table["dataset"].map({"CIFAR-100": 0, "RawMal-TF": 1})
    table["_mode_order"] = table["mode"].map(mode_order)
    table["_m_order"] = table["M"].fillna(-1).astype(float)
    table["_area_order"] = table["cutout_area"].fillna(-1).astype(float)
    table = table.sort_values(["_dataset_order", "_mode_order", "_m_order", "_area_order"]).drop(
        columns=["_dataset_order", "_mode_order", "_m_order", "_area_order"]
    )
    return table


def row_lookup(per_run: pd.DataFrame) -> dict[tuple[str, int, str, int | None, float | None], dict[str, Any]]:
    lookup: dict[tuple[str, int, str, int | None, float | None], dict[str, Any]] = {}
    for _, row in per_run.iterrows():
        m_value = None if pd.isna(row["M"]) else int(row["M"])
        area = None if pd.isna(row["cutout_area"]) else float(row["cutout_area"])
        lookup[(str(row["dataset"]), int(row["seed"]), str(row["mode"]), m_value, area)] = row.to_dict()
    return lookup


def build_seed_level_paired_effects(per_run: pd.DataFrame) -> pd.DataFrame:
    lookup = row_lookup(per_run)
    rows: list[dict[str, Any]] = []

    def append_rows(
        dataset: str,
        seed: int,
        comparison_key: str,
        comparison: str,
        condition: str,
        mode: str,
        area: float | None,
        m_value: int | None,
        condition_a: str,
        condition_b: str,
        m_a: int | None,
        m_b: int | None,
        row_a: dict[str, Any],
        row_b: dict[str, Any],
    ) -> None:
        for metric in PAIRED_EFFECT_METRICS:
            value_a = float(row_a[metric])
            value_b = float(row_b[metric])
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": EXPECTED_ARCHITECTURE,
                    "seed": seed,
                    "comparison_key": comparison_key,
                    "comparison": comparison,
                    "condition": condition,
                    "mode": mode,
                    "M": m_value,
                    "cutout_area": area,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "M_a": m_a,
                    "M_b": m_b,
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "value_a": value_a,
                    "value_b": value_b,
                    "difference": value_a - value_b,
                    "difference_interpretation": f"positive means {condition_a} is higher than {condition_b}",
                }
            )

    for dataset_id in EXPECTED_DATASETS:
        dataset = display_dataset(dataset_id)
        for seed in EXPECTED_SEEDS:
            baseline = lookup.get((dataset, seed, "none", None, None))
            for area in EXPECTED_AREAS:
                for m_value in EXPECTED_MS:
                    random_row = lookup.get((dataset, seed, "random", m_value, area))
                    low_row = lookup.get((dataset, seed, "cam_low", m_value, area))
                    high_row = lookup.get((dataset, seed, "cam_high", m_value, area))
                    if low_row is not None and random_row is not None:
                        append_rows(
                            dataset,
                            seed,
                            "cam_low_minus_random",
                            "Low-saliency cutout minus random cutout",
                            CONDITION_LABELS["cam_low"],
                            "cam_low",
                            area,
                            m_value,
                            f"{CONDITION_LABELS['cam_low']} M{m_value}",
                            f"{CONDITION_LABELS['random']} M{m_value}",
                            m_value,
                            m_value,
                            low_row,
                            random_row,
                        )
                    if high_row is not None and random_row is not None:
                        append_rows(
                            dataset,
                            seed,
                            "cam_high_minus_random",
                            "High-saliency cutout minus random cutout",
                            CONDITION_LABELS["cam_high"],
                            "cam_high",
                            area,
                            m_value,
                            f"{CONDITION_LABELS['cam_high']} M{m_value}",
                            f"{CONDITION_LABELS['random']} M{m_value}",
                            m_value,
                            m_value,
                            high_row,
                            random_row,
                        )
                    if random_row is not None and baseline is not None:
                        append_rows(
                            dataset,
                            seed,
                            "random_minus_no_cutout",
                            "Random cutout minus no cutout",
                            CONDITION_LABELS["random"],
                            "random",
                            area,
                            m_value,
                            f"{CONDITION_LABELS['random']} M{m_value}",
                            CONDITION_LABELS["none"],
                            m_value,
                            None,
                            random_row,
                            baseline,
                        )
                for mode in AUGMENTED_CONDITIONS:
                    m4 = lookup.get((dataset, seed, mode, 4, area))
                    m8 = lookup.get((dataset, seed, mode, 8, area))
                    if m8 is not None and m4 is not None:
                        append_rows(
                            dataset,
                            seed,
                            "m8_minus_m4",
                            "M8 minus M4",
                            CONDITION_LABELS[mode],
                            mode,
                            area,
                            None,
                            f"{CONDITION_LABELS[mode]} M8",
                            f"{CONDITION_LABELS[mode]} M4",
                            8,
                            4,
                            m8,
                            m4,
                        )

    return pd.DataFrame(rows)


def build_paired_effects_table(seed_effects: pd.DataFrame) -> pd.DataFrame:
    if seed_effects.empty:
        return pd.DataFrame()
    group_cols = [
        "dataset",
        "architecture",
        "comparison_key",
        "comparison",
        "condition",
        "mode",
        "M",
        "cutout_area",
        "condition_a",
        "condition_b",
        "M_a",
        "M_b",
        "metric",
        "metric_label",
        "difference_interpretation",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in seed_effects.groupby(group_cols, dropna=False, sort=True):
        row = {column: value for column, value in zip(group_cols, keys)}
        seed_values = {int(seed): float(value) for seed, value in zip(group["seed"], group["difference"])}
        stats_row = stats_for_values(list(seed_values.values()))
        row.update(
            {
                "n_paired_seeds": stats_row["n"],
                "seeds_paired": ";".join(str(seed) for seed in sorted(seed_values)),
                "seeds_missing": ";".join(str(seed) for seed in EXPECTED_SEEDS if seed not in seed_values),
                "mean_paired_difference": stats_row["mean"],
                "sample_variance": stats_row["sample_variance"],
                "sample_std": stats_row["sample_std"],
                "standard_error": stats_row["standard_error"],
                "ci95_low": stats_row["ci95_low"],
                "ci95_high": stats_row["ci95_high"],
                "min": stats_row["min"],
                "max": stats_row["max"],
                "seed_42_difference": seed_values.get(42, np.nan),
                "seed_43_difference": seed_values.get(43, np.nan),
                "seed_44_difference": seed_values.get(44, np.nan),
                "seed_level_differences": ";".join(f"{seed}:{seed_values[seed]:.10f}" for seed in sorted(seed_values)),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {
        "cam_low_minus_random": 0,
        "cam_high_minus_random": 1,
        "random_minus_no_cutout": 2,
        "m8_minus_m4": 3,
    }
    table["_dataset_order"] = table["dataset"].map({"CIFAR-100": 0, "RawMal-TF": 1})
    table["_comparison_order"] = table["comparison_key"].map(order)
    table["_m_order"] = table["M"].fillna(-1).astype(float)
    table["_area_order"] = table["cutout_area"].fillna(-1).astype(float)
    table = table.sort_values(
        ["_dataset_order", "_comparison_order", "mode", "_m_order", "_area_order", "metric"]
    ).drop(columns=["_dataset_order", "_comparison_order", "_m_order", "_area_order"])
    return table


def build_inventory_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "dataset",
        "architecture",
        "seed",
        "condition",
        "mode",
        "M",
        "cutout_area",
        "config_epochs",
        "metrics_epoch_count",
        "required_columns_ok",
        "epoch_sequence_ok",
        "status",
        "analysis_valid",
        "issue_count",
        "issues",
        "source_path",
        "config_path",
        "metrics_path",
        "config_sha256",
        "metrics_sha256",
    ]
    rows: list[dict[str, Any]] = []
    status_order = {
        "valid": 0,
        "valid_selected_baseline": 1,
        "duplicate_baseline_exact": 2,
        "duplicate_baseline_inconsistent": 3,
        "duplicate_run": 4,
        "invalid": 5,
        "ignored_unexpected": 6,
        "valid_candidate": 7,
    }
    for record in records:
        row = {column: record.get(column, "") for column in columns}
        row["issue_count"] = len(record.get("issues", []))
        row["issues"] = ";".join(record.get("issues", []))
        row["_status_order"] = status_order.get(record["status"], 99)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=columns)
    table["_dataset_order"] = table["dataset"].map({"CIFAR-100": 0, "RawMal-TF": 1}).fillna(99)
    table["_seed_order"] = pd.to_numeric(table["seed"], errors="coerce").fillna(999)
    table["_m_order"] = pd.to_numeric(table["M"], errors="coerce").fillna(-1)
    table["_area_order"] = pd.to_numeric(table["cutout_area"], errors="coerce").fillna(-1)
    table = table.sort_values(
        ["_dataset_order", "_seed_order", "_status_order", "mode", "_m_order", "_area_order", "source_path"]
    )
    return table[columns]


def build_missing_or_invalid_table(records: list[dict[str, Any]], per_run: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = row_lookup(per_run)

    for dataset_id in EXPECTED_DATASETS:
        dataset = display_dataset(dataset_id)
        for seed in EXPECTED_SEEDS:
            if (dataset, seed, "none", None, None) not in lookup:
                rows.append(
                    {
                        "issue_type": "missing_expected_baseline",
                        "dataset": dataset,
                        "architecture": EXPECTED_ARCHITECTURE,
                        "seed": seed,
                        "condition": CONDITION_LABELS["none"],
                        "mode": "none",
                        "M": None,
                        "cutout_area": None,
                        "details": "No selected valid no-cutout baseline for this dataset and seed.",
                        "source_path": "",
                    }
                )
            for mode in AUGMENTED_CONDITIONS:
                for m_value in EXPECTED_MS:
                    for area in EXPECTED_AREAS:
                        if (dataset, seed, mode, m_value, area) not in lookup:
                            rows.append(
                                {
                                    "issue_type": "missing_expected_run",
                                    "dataset": dataset,
                                    "architecture": EXPECTED_ARCHITECTURE,
                                    "seed": seed,
                                    "condition": CONDITION_LABELS[mode],
                                    "mode": mode,
                                    "M": m_value,
                                    "cutout_area": area,
                                    "details": "Expected current-grid run is absent or excluded from analysis.",
                                    "source_path": "",
                                }
                            )

    reported_statuses = {
        "invalid",
        "ignored_unexpected",
        "duplicate_run",
        "duplicate_baseline_inconsistent",
    }
    for record in records:
        if record["status"] in reported_statuses:
            rows.append(
                {
                    "issue_type": record["status"],
                    "dataset": record["dataset"],
                    "architecture": record["architecture"],
                    "seed": record["seed"],
                    "condition": record["condition"],
                    "mode": record["mode"],
                    "M": record["M"],
                    "cutout_area": record["cutout_area"],
                    "details": ";".join(record["issues"]),
                    "source_path": record["source_path"],
                }
            )

    columns = [
        "issue_type",
        "dataset",
        "architecture",
        "seed",
        "condition",
        "mode",
        "M",
        "cutout_area",
        "details",
        "source_path",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def save_figure(fig: plt.Figure, path_without_extension: Path) -> list[Path]:
    path_without_extension.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for extension in ("png", "pdf"):
        output_path = path_without_extension.with_suffix(f".{extension}")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def pct_formatter(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.0f}"))


def pp_formatter(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.1f}"))


def set_xticks_with_labels(ax: plt.Axes, ticks: Any, labels: list[str], **kwargs: Any) -> None:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, **kwargs)


def set_yticks_with_labels(ax: plt.Axes, ticks: Any, labels: list[str], **kwargs: Any) -> None:
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, **kwargs)


def finish_plot(fig: plt.Figure, caption: str, rect_top: float = 0.93) -> None:
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, rect_top))


def plot_metric_by_area(per_run: pd.DataFrame, metric: str, folder: str, suffix: str, y_label: str) -> list[Path]:
    outputs: list[Path] = []
    offsets = {"random": -0.006, "cam_low": 0.0, "cam_high": 0.006}
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        for m_value in EXPECTED_MS:
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for mode in AUGMENTED_CONDITIONS:
                means: list[float] = []
                stds: list[float] = []
                for area in EXPECTED_AREAS:
                    subset = subset_for(per_run, dataset, mode, m_value, area).sort_values("seed")
                    values = [float(value) for value in subset[metric].dropna().tolist()]
                    stats_row = stats_for_values(values)
                    mean = float(stats_row["mean"]) * 100 if not pd.isna(stats_row["mean"]) else np.nan
                    std = float(stats_row["sample_std"]) * 100 if not pd.isna(stats_row["sample_std"]) else 0.0
                    means.append(mean)
                    stds.append(std)
                    for _, seed_row in subset.iterrows():
                        ax.scatter(
                            area + offsets[mode],
                            float(seed_row[metric]) * 100,
                            color=CONDITION_COLORS[mode],
                            marker=SEED_MARKERS.get(int(seed_row["seed"]), "o"),
                            s=30,
                            edgecolor="white",
                            linewidth=0.5,
                            alpha=0.85,
                            zorder=3,
                        )
                ax.errorbar(
                    EXPECTED_AREAS,
                    means,
                    yerr=stds,
                    color=CONDITION_COLORS[mode],
                    marker="o",
                    capsize=4,
                    label=CONDITION_LABELS[mode],
                )

            baseline = subset_for(per_run, dataset, "none", None, None)
            if not baseline.empty:
                base_values = [float(value) for value in baseline[metric].dropna().tolist()]
                base_stats = stats_for_values(base_values)
                base_mean = float(base_stats["mean"]) * 100
                base_std = 0.0 if pd.isna(base_stats["sample_std"]) else float(base_stats["sample_std"]) * 100
                if base_std > 0:
                    ax.axhspan(
                        base_mean - base_std,
                        base_mean + base_std,
                        color=CONDITION_COLORS["none"],
                        alpha=0.10,
                        label="No cutout +/- SD",
                    )
                ax.axhline(
                    base_mean,
                    color=CONDITION_COLORS["none"],
                    linestyle="--",
                    linewidth=1.5,
                    label="No cutout mean",
                )

            ax.set_title(f"{dataset}, M{m_value}: {metric_label(metric)} versus cutout area")
            ax.set_xlabel("Cutout area")
            ax.set_ylabel(y_label)
            set_xticks_with_labels(ax, EXPECTED_AREAS, [area_label(area) for area in EXPECTED_AREAS])
            pct_formatter(ax)
            ax.legend(frameon=False, ncol=2)
            finish_plot(fig, "Mean across seeds; error bars and the baseline band show +/-1 sample SD across seeds.")
            outputs.extend(save_figure(fig, PLOTS_DIR / folder / f"{slug}_M{m_value}_{suffix}_by_area"))
    return outputs


def plot_paired_cam_effects(paired: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    metric = "best_validation_accuracy"
    comparisons = (
        ("cam_low_minus_random", "Low-saliency - random", "cam_low", -0.006),
        ("cam_high_minus_random", "High-saliency - random", "cam_high", 0.006),
    )
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        for m_value in EXPECTED_MS:
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for comparison_key, label, mode, offset in comparisons:
                means: list[float] = []
                stds: list[float] = []
                for area in EXPECTED_AREAS:
                    row = paired[
                        (paired["dataset"] == dataset)
                        & (paired["comparison_key"] == comparison_key)
                        & (paired["M"] == m_value)
                        & (np.isclose(paired["cutout_area"].astype(float), area))
                        & (paired["metric"] == metric)
                    ]
                    if row.empty:
                        means.append(np.nan)
                        stds.append(0.0)
                        continue
                    effect = row.iloc[0]
                    means.append(float(effect["mean_paired_difference"]) * 100)
                    stds.append(0.0 if pd.isna(effect["sample_std"]) else float(effect["sample_std"]) * 100)
                    for seed in EXPECTED_SEEDS:
                        value = effect.get(f"seed_{seed}_difference", np.nan)
                        if not pd.isna(value):
                            ax.scatter(
                                area + offset,
                                float(value) * 100,
                                color=CONDITION_COLORS[mode],
                                marker=SEED_MARKERS.get(seed, "o"),
                                s=30,
                                edgecolor="white",
                                linewidth=0.5,
                                alpha=0.85,
                                zorder=3,
                            )
                ax.errorbar(
                    EXPECTED_AREAS,
                    means,
                    yerr=stds,
                    color=CONDITION_COLORS[mode],
                    marker="o",
                    capsize=4,
                    label=label,
                )
            ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0)
            ax.set_title(f"{dataset}, M{m_value}: paired CAM effect versus random cutout")
            ax.set_xlabel("Cutout area")
            ax.set_ylabel("Paired best validation accuracy difference (percentage points)")
            set_xticks_with_labels(ax, EXPECTED_AREAS, [area_label(area) for area in EXPECTED_AREAS])
            pp_formatter(ax)
            ax.legend(frameon=False)
            finish_plot(fig, "Differences are computed within seed first; error bars show sample SD across paired seeds.")
            outputs.extend(save_figure(fig, PLOTS_DIR / "paired_cam_effects" / f"{slug}_M{m_value}_paired_cam_effect_vs_random"))
    return outputs


def curves_for(
    per_run: pd.DataFrame,
    metric_frames: dict[int, pd.DataFrame],
    run_records_by_source: dict[str, dict[str, Any]],
    dataset: str,
    mode: str,
    m_value: int,
    area: float,
    metric_column: str,
) -> np.ndarray:
    if mode == "none":
        subset = subset_for(per_run, dataset, "none", None, None)
    else:
        subset = subset_for(per_run, dataset, mode, m_value, area)
    curves: list[np.ndarray] = []
    for _, row in subset.sort_values("seed").iterrows():
        source_path = str(row["source_path"])
        record = run_records_by_source[source_path]
        curves.append(metric_frames[int(record["run_id"])][metric_column].to_numpy(dtype=float))
    if not curves:
        return np.empty((0, EXPECTED_EPOCHS))
    return np.vstack(curves)


def plot_learning_curves(
    per_run: pd.DataFrame,
    metric_frames: dict[int, pd.DataFrame],
    analysis_runs: list[dict[str, Any]],
) -> list[Path]:
    outputs: list[Path] = []
    run_records_by_source = {record["source_path"]: record for record in analysis_runs}
    specs = (
        ("eval_acc1", "Validation accuracy (%)", "validation_accuracy", True),
        ("eval_loss", "Validation loss", "validation_loss", False),
    )
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        for m_value in EXPECTED_MS:
            for area in EXPECTED_AREAS:
                for column, y_label, suffix, as_percent in specs:
                    fig, ax = plt.subplots(figsize=(7.6, 4.8))
                    for mode in CONDITIONS:
                        curves = curves_for(per_run, metric_frames, run_records_by_source, dataset, mode, m_value, area, column)
                        if curves.size == 0:
                            continue
                        mean = curves.mean(axis=0)
                        std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros(curves.shape[1])
                        if as_percent:
                            mean = mean * 100
                            std = std * 100
                        epochs = np.arange(1, curves.shape[1] + 1)
                        ax.plot(epochs, mean, color=CONDITION_COLORS[mode], label=CONDITION_LABELS[mode])
                        ax.fill_between(epochs, mean - std, mean + std, color=CONDITION_COLORS[mode], alpha=0.14, linewidth=0)
                    metric_name = "validation accuracy" if as_percent else "validation loss"
                    ax.set_title(f"{dataset}, M{m_value}, area {area_label(area)}: mean {metric_name} curve")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel(y_label)
                    ax.set_xlim(1, EXPECTED_EPOCHS)
                    if as_percent:
                        pct_formatter(ax)
                    ax.legend(frameon=False, ncol=2)
                    finish_plot(fig, "Lines are epoch-wise means across seeds; shaded bands show +/-1 sample SD; no smoothing.")
                    outputs.extend(
                        save_figure(
                            fig,
                            PLOTS_DIR / "learning_curves" / f"{slug}_M{m_value}_{area_slug(area)}_{suffix}_curve",
                        )
                    )
    return outputs


def plot_stability(per_run: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    specs = (
        ("best_to_final_degradation", "Best-to-final degradation (percentage points)", True),
        ("maximum_validation_drawdown", "Maximum drawdown (percentage points)", True),
        ("collapse_event_count", "Collapse-event count", False),
        ("final20_validation_accuracy_std", "Final-20 validation accuracy SD (percentage points)", True),
    )
    offsets = {"random": -0.006, "cam_low": 0.0, "cam_high": 0.006}
    for metric, y_label, as_percent in specs:
        for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
            slug = dataset_slug_from_label(dataset)
            for m_value in EXPECTED_MS:
                fig, ax = plt.subplots(figsize=(7.4, 4.8))
                for mode in AUGMENTED_CONDITIONS:
                    means: list[float] = []
                    stds: list[float] = []
                    for area in EXPECTED_AREAS:
                        subset = subset_for(per_run, dataset, mode, m_value, area).sort_values("seed")
                        values = [float(value) for value in subset[metric].dropna().tolist()]
                        stats_row = stats_for_values(values)
                        scale = 100 if as_percent else 1
                        means.append(float(stats_row["mean"]) * scale if not pd.isna(stats_row["mean"]) else np.nan)
                        stds.append(float(stats_row["sample_std"]) * scale if not pd.isna(stats_row["sample_std"]) else 0.0)
                        for _, seed_row in subset.iterrows():
                            ax.scatter(
                                area + offsets[mode],
                                float(seed_row[metric]) * scale,
                                color=CONDITION_COLORS[mode],
                                marker=SEED_MARKERS.get(int(seed_row["seed"]), "o"),
                                s=28,
                                edgecolor="white",
                                linewidth=0.5,
                                alpha=0.85,
                                zorder=3,
                            )
                    ax.errorbar(
                        EXPECTED_AREAS,
                        means,
                        yerr=stds,
                        color=CONDITION_COLORS[mode],
                        marker="o",
                        capsize=4,
                        label=CONDITION_LABELS[mode],
                    )
                baseline = subset_for(per_run, dataset, "none", None, None)
                if not baseline.empty:
                    base_values = [float(value) for value in baseline[metric].dropna().tolist()]
                    base_stats = stats_for_values(base_values)
                    scale = 100 if as_percent else 1
                    base_mean = float(base_stats["mean"]) * scale
                    base_std = 0.0 if pd.isna(base_stats["sample_std"]) else float(base_stats["sample_std"]) * scale
                    if base_std > 0:
                        ax.axhspan(
                            base_mean - base_std,
                            base_mean + base_std,
                            color=CONDITION_COLORS["none"],
                            alpha=0.10,
                            label="No cutout +/- SD",
                        )
                    ax.axhline(
                        base_mean,
                        color=CONDITION_COLORS["none"],
                        linestyle="--",
                        linewidth=1.4,
                        label="No cutout mean",
                    )
                ax.set_title(f"{dataset}, M{m_value}: {metric_label(metric)}")
                ax.set_xlabel("Cutout area")
                ax.set_ylabel(y_label)
                set_xticks_with_labels(ax, EXPECTED_AREAS, [area_label(area) for area in EXPECTED_AREAS])
                if as_percent:
                    pp_formatter(ax)
                ax.legend(frameon=False, ncol=2)
                finish_plot(fig, "Mean across seeds; error bars and the baseline band show +/-1 sample SD across seeds.")
                outputs.extend(save_figure(fig, PLOTS_DIR / "stability" / f"{slug}_M{m_value}_{metric}_by_area"))
    return outputs


def plot_m8_minus_m4(paired: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    metric = "best_validation_accuracy"
    offsets = {"random": -0.006, "cam_low": 0.0, "cam_high": 0.006}
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for mode in AUGMENTED_CONDITIONS:
            means: list[float] = []
            stds: list[float] = []
            for area in EXPECTED_AREAS:
                row = paired[
                    (paired["dataset"] == dataset)
                    & (paired["comparison_key"] == "m8_minus_m4")
                    & (paired["mode"] == mode)
                    & (np.isclose(paired["cutout_area"].astype(float), area))
                    & (paired["metric"] == metric)
                ]
                if row.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                    continue
                effect = row.iloc[0]
                means.append(float(effect["mean_paired_difference"]) * 100)
                stds.append(0.0 if pd.isna(effect["sample_std"]) else float(effect["sample_std"]) * 100)
                for seed in EXPECTED_SEEDS:
                    value = effect.get(f"seed_{seed}_difference", np.nan)
                    if not pd.isna(value):
                        ax.scatter(
                            area + offsets[mode],
                            float(value) * 100,
                            color=CONDITION_COLORS[mode],
                            marker=SEED_MARKERS.get(seed, "o"),
                            s=30,
                            edgecolor="white",
                            linewidth=0.5,
                            alpha=0.85,
                            zorder=3,
                        )
            ax.errorbar(
                EXPECTED_AREAS,
                means,
                yerr=stds,
                color=CONDITION_COLORS[mode],
                marker="o",
                capsize=4,
                label=CONDITION_LABELS[mode],
            )
        ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_title(f"{dataset}: paired M8-minus-M4 effect")
        ax.set_xlabel("Cutout area")
        ax.set_ylabel("Best validation accuracy difference (percentage points)")
        set_xticks_with_labels(ax, EXPECTED_AREAS, [area_label(area) for area in EXPECTED_AREAS])
        pp_formatter(ax)
        ax.legend(frameon=False)
        finish_plot(
            fig,
            "Differences are computed within seed; error bars show sample SD. Descriptive because M changes augmented samples and optimizer updates.",
        )
        outputs.extend(save_figure(fig, PLOTS_DIR / "m8_minus_m4" / f"{slug}_paired_m8_minus_m4_best_validation_accuracy"))
    return outputs


def aggregate_cell(aggregate: pd.DataFrame, dataset: str, mode: str, m_value: int | None, area: float | None) -> pd.Series | None:
    subset = aggregate[(aggregate["dataset"] == dataset) & (aggregate["mode"] == mode)]
    if mode == "none":
        subset = subset[subset["M"].isna() & subset["cutout_area"].isna()]
    else:
        subset = subset[(subset["M"] == m_value) & (np.isclose(subset["cutout_area"].astype(float), float(area)))]
    if subset.empty:
        return None
    return subset.iloc[0]


def plot_heatmaps(aggregate: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    row_specs = [(mode, m_value) for mode in AUGMENTED_CONDITIONS for m_value in EXPECTED_MS]
    row_labels = [f"{CONDITION_SHORT_LABELS[mode]} M{m_value}" for mode, m_value in row_specs]
    all_means = aggregate[aggregate["mode"] != "none"]["best_validation_accuracy_mean"].dropna().to_numpy(dtype=float) * 100
    all_stds = aggregate[aggregate["mode"] != "none"]["best_validation_accuracy_sample_std"].dropna().to_numpy(dtype=float) * 100
    mean_vmin, mean_vmax = float(np.nanmin(all_means)), float(np.nanmax(all_means))
    std_vmin, std_vmax = 0.0, float(np.nanmax(all_stds))

    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        baseline = aggregate_cell(aggregate, dataset, "none", None, None)
        baseline_text = "No cutout baseline unavailable"
        if baseline is not None and not pd.isna(baseline["best_validation_accuracy_mean"]):
            baseline_text = (
                f"No cutout mean: {float(baseline['best_validation_accuracy_mean']) * 100:.2f}%; "
                f"SD: {float(baseline['best_validation_accuracy_sample_std']) * 100:.2f} pp"
            )
        for metric_column, title_metric, folder, suffix, vmin, vmax, colorbar_label in (
            (
                "best_validation_accuracy_mean",
                "Mean best validation accuracy",
                "mean_heatmaps",
                "mean_best_validation_accuracy_heatmap",
                mean_vmin,
                mean_vmax,
                "Mean best validation accuracy (%)",
            ),
            (
                "best_validation_accuracy_sample_std",
                "Best-validation accuracy sample SD",
                "variability_heatmaps",
                "std_best_validation_accuracy_heatmap",
                std_vmin,
                std_vmax,
                "Sample SD across seeds (percentage points)",
            ),
        ):
            matrix = np.full((len(row_specs), len(EXPECTED_AREAS)), np.nan)
            for row_index, (mode, m_value) in enumerate(row_specs):
                for col_index, area in enumerate(EXPECTED_AREAS):
                    cell = aggregate_cell(aggregate, dataset, mode, m_value, area)
                    if cell is not None and not pd.isna(cell[metric_column]):
                        matrix[row_index, col_index] = float(cell[metric_column]) * 100
            fig, ax = plt.subplots(figsize=(7.6, 4.9))
            image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            set_xticks_with_labels(ax, np.arange(len(EXPECTED_AREAS)), [area_label(area) for area in EXPECTED_AREAS])
            set_yticks_with_labels(ax, np.arange(len(row_specs)), row_labels)
            ax.set_xlabel("Cutout area")
            ax.set_title(f"{dataset}: {title_metric}\n{baseline_text}")
            midpoint = (vmin + vmax) / 2.0
            for row_index in range(matrix.shape[0]):
                for col_index in range(matrix.shape[1]):
                    value = matrix[row_index, col_index]
                    text = "NA" if np.isnan(value) else f"{value:.2f}"
                    color = "white" if not np.isnan(value) and value < midpoint else "black"
                    ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=8.5)
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label(colorbar_label)
            finish_plot(fig, "Cells are across-seed summaries; the no-cutout baseline is shown separately.", rect_top=0.90)
            outputs.extend(save_figure(fig, PLOTS_DIR / folder / f"{slug}_{suffix}"))
    return outputs


def plot_mean_vs_variability(aggregate: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        subset = aggregate[aggregate["dataset"] == dataset].copy()
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for mode in CONDITIONS:
            mode_subset = subset[subset["mode"] == mode]
            if mode == "none":
                label = CONDITION_LABELS[mode]
                marker = M_MARKERS[None]
                color = CONDITION_COLORS[mode]
                for _, row in mode_subset.iterrows():
                    ax.scatter(
                        float(row["best_validation_accuracy_mean"]) * 100,
                        float(row["best_validation_accuracy_sample_std"]) * 100,
                        color=color,
                        marker=marker,
                        s=70,
                        edgecolor="white",
                        linewidth=0.6,
                        label=label,
                    )
            else:
                for m_value in EXPECTED_MS:
                    m_subset = mode_subset[mode_subset["M"] == m_value]
                    ax.scatter(
                        m_subset["best_validation_accuracy_mean"].astype(float) * 100,
                        m_subset["best_validation_accuracy_sample_std"].astype(float) * 100,
                        color=CONDITION_COLORS[mode],
                        marker=M_MARKERS[m_value],
                        s=54,
                        edgecolor="white",
                        linewidth=0.5,
                        alpha=0.9,
                        label=f"{CONDITION_LABELS[mode]} M{m_value}",
                    )
        ax.set_title(f"{dataset}: mean performance versus across-seed variability")
        ax.set_xlabel("Mean best validation accuracy (%)")
        ax.set_ylabel("Sample SD across seeds (percentage points)")
        pct_formatter(ax)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        finish_plot(fig, "Each point is one condition/M/area combination; no cutout appears once.")
        outputs.extend(save_figure(fig, PLOTS_DIR / "mean_vs_variability" / f"{slug}_mean_vs_variability_best_validation_accuracy"))
    return outputs


def plot_condition_comparisons(per_run: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    bar_specs = [("none", None), ("random", 4), ("random", 8), ("cam_low", 4), ("cam_low", 8), ("cam_high", 4), ("cam_high", 8)]
    labels = [
        "No cutout",
        "Random M4",
        "Random M8",
        "Low-saliency M4",
        "Low-saliency M8",
        "High-saliency M4",
        "High-saliency M8",
    ]
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        slug = dataset_slug_from_label(dataset)
        for area in EXPECTED_AREAS:
            means: list[float] = []
            stds: list[float] = []
            colors: list[str] = []
            seed_points: list[tuple[int, int, float, str]] = []
            for index, (mode, m_value) in enumerate(bar_specs):
                subset = subset_for(per_run, dataset, mode, m_value, None if mode == "none" else area).sort_values("seed")
                values = [float(value) for value in subset["best_validation_accuracy"].dropna().tolist()]
                stats_row = stats_for_values(values)
                means.append(float(stats_row["mean"]) * 100 if not pd.isna(stats_row["mean"]) else np.nan)
                stds.append(float(stats_row["sample_std"]) * 100 if not pd.isna(stats_row["sample_std"]) else 0.0)
                colors.append(CONDITION_COLORS[mode])
                for _, row in subset.iterrows():
                    seed_points.append((index, int(row["seed"]), float(row["best_validation_accuracy"]) * 100, mode))
            fig, ax = plt.subplots(figsize=(9.2, 4.9))
            x = np.arange(len(bar_specs))
            ax.bar(x, means, yerr=stds, color=colors, alpha=0.75, capsize=4, edgecolor="#333333", linewidth=0.4)
            for index, seed, value, mode in seed_points:
                jitter = {42: -0.08, 43: 0.0, 44: 0.08}.get(seed, 0.0)
                ax.scatter(
                    index + jitter,
                    value,
                    color=CONDITION_COLORS[mode],
                    marker=SEED_MARKERS.get(seed, "o"),
                    s=31,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )
            ax.set_title(f"{dataset}, area {area_label(area)}: condition comparison")
            ax.set_ylabel("Mean best validation accuracy (%)")
            set_xticks_with_labels(ax, x, labels, rotation=30, ha="right")
            pct_formatter(ax)
            legend_handles = [Patch(facecolor=CONDITION_COLORS[mode], label=CONDITION_LABELS[mode]) for mode in CONDITIONS]
            ax.legend(handles=legend_handles, frameon=False, ncol=4, fontsize=8)
            finish_plot(fig, "Bars are means across seeds; error bars show sample SD; points show individual seeds.")
            outputs.extend(save_figure(fig, PLOTS_DIR / "condition_comparisons" / f"{slug}_{area_slug(area)}_condition_comparison"))
    return outputs


def create_all_plots(
    per_run: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    metric_frames: dict[int, pd.DataFrame],
    analysis_runs: list[dict[str, Any]],
) -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(
        plot_metric_by_area(
            per_run,
            "best_validation_accuracy",
            "best_accuracy_by_area",
            "mean_best_validation_accuracy",
            "Mean best validation accuracy (%)",
        )
    )
    outputs.extend(
        plot_metric_by_area(
            per_run,
            "final_validation_accuracy",
            "final_accuracy_by_area",
            "mean_final_validation_accuracy",
            "Mean final validation accuracy (%)",
        )
    )
    outputs.extend(plot_paired_cam_effects(paired))
    outputs.extend(
        plot_metric_by_area(
            per_run,
            "validation_aulc",
            "aulc_by_area",
            "mean_validation_aulc",
            "Mean normalized validation AULC (%)",
        )
    )
    outputs.extend(plot_learning_curves(per_run, metric_frames, analysis_runs))
    outputs.extend(plot_stability(per_run))
    outputs.extend(plot_m8_minus_m4(paired))
    outputs.extend(plot_heatmaps(aggregate))
    outputs.extend(plot_mean_vs_variability(aggregate))
    outputs.extend(plot_condition_comparisons(per_run))
    return outputs


def fmt_fraction_as_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_difference_pp(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:+.{digits}f} pp"


def fmt_unsigned_pp(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f} pp"


def aggregate_best_rows(aggregate: pd.DataFrame) -> pd.DataFrame:
    return aggregate[
        [
            "dataset",
            "condition",
            "mode",
            "M",
            "cutout_area",
            "valid_seed_count",
            "best_validation_accuracy_mean",
            "best_validation_accuracy_sample_variance",
            "best_validation_accuracy_sample_std",
        ]
    ].copy()


def describe_top_performance(aggregate: pd.DataFrame) -> str:
    rows: list[str] = []
    best = aggregate_best_rows(aggregate)
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        subset = best[(best["dataset"] == dataset) & best["best_validation_accuracy_mean"].notna()]
        if subset.empty:
            rows.append(f"- {dataset}: no aggregate rows were available.")
            continue
        subset = subset.sort_values("best_validation_accuracy_mean", ascending=False).head(3)
        pieces = []
        for _, row in subset.iterrows():
            m_text = "" if pd.isna(row["M"]) else f", M{int(row['M'])}"
            area_text = "" if pd.isna(row["cutout_area"]) else f", area {area_label(row['cutout_area'])}"
            pieces.append(f"{row['condition']}{m_text}{area_text}: {fmt_fraction_as_pct(row['best_validation_accuracy_mean'])}")
        rows.append(f"- {dataset}: " + "; ".join(pieces) + ".")
    return "\n".join(rows)


def describe_lowest_variance(aggregate: pd.DataFrame) -> str:
    rows: list[str] = []
    best = aggregate_best_rows(aggregate)
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        subset = best[(best["dataset"] == dataset) & best["best_validation_accuracy_sample_variance"].notna()]
        if subset.empty:
            rows.append(f"- {dataset}: no sample variances were available.")
            continue
        subset = subset.sort_values(["best_validation_accuracy_sample_variance", "best_validation_accuracy_mean"], ascending=[True, False]).head(3)
        pieces = []
        for _, row in subset.iterrows():
            m_text = "" if pd.isna(row["M"]) else f", M{int(row['M'])}"
            area_text = "" if pd.isna(row["cutout_area"]) else f", area {area_label(row['cutout_area'])}"
            pieces.append(
                f"{row['condition']}{m_text}{area_text}: variance {float(row['best_validation_accuracy_sample_variance']):.8f}, "
                f"SD {fmt_unsigned_pp(row['best_validation_accuracy_sample_std'])}"
            )
        rows.append(f"- {dataset}: " + "; ".join(pieces) + ".")
    return "\n".join(rows)


def describe_cam_consistency(paired: pd.DataFrame) -> str:
    rows: list[str] = []
    subset = paired[
        (paired["metric"] == "best_validation_accuracy")
        & (paired["comparison_key"].isin(["cam_low_minus_random", "cam_high_minus_random"]))
    ]
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        dataset_subset = subset[subset["dataset"] == dataset]
        for key, label in (
            ("cam_low_minus_random", "Low-saliency cutout"),
            ("cam_high_minus_random", "High-saliency cutout"),
        ):
            key_subset = dataset_subset[dataset_subset["comparison_key"] == key]
            if key_subset.empty:
                rows.append(f"- {dataset}: {label} has no paired comparisons against random cutout.")
                continue
            complete_positive = 0
            complete_negative = 0
            mixed = 0
            for _, row in key_subset.iterrows():
                diffs = [row[f"seed_{seed}_difference"] for seed in EXPECTED_SEEDS if not pd.isna(row[f"seed_{seed}_difference"])]
                if diffs and all(float(value) > 0 for value in diffs):
                    complete_positive += 1
                elif diffs and all(float(value) < 0 for value in diffs):
                    complete_negative += 1
                else:
                    mixed += 1
            cell_means = key_subset["mean_paired_difference"].dropna().astype(float)
            range_text = "NA"
            if not cell_means.empty:
                range_text = f"{fmt_difference_pp(cell_means.min())} to {fmt_difference_pp(cell_means.max())}"
            rows.append(
                f"- {dataset}: {label} versus random has {complete_positive} area/M cells positive for all available seeds, "
                f"{complete_negative} negative for all available seeds, and {mixed} mixed. Cell mean paired effects range from {range_text}."
            )
    return "\n".join(rows)


def describe_area_m_effects(paired: pd.DataFrame) -> str:
    subset = paired[
        (paired["metric"] == "best_validation_accuracy")
        & (paired["comparison_key"].isin(["cam_low_minus_random", "cam_high_minus_random"]))
    ].copy()
    if subset.empty:
        return "No CAM-versus-random paired effects were available."
    rows: list[str] = []
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        dataset_subset = subset[subset["dataset"] == dataset]
        if dataset_subset.empty:
            continue
        top = dataset_subset.sort_values("mean_paired_difference", ascending=False).iloc[0]
        bottom = dataset_subset.sort_values("mean_paired_difference", ascending=True).iloc[0]
        rows.append(
            f"- {dataset}: largest mean CAM advantage is {top['comparison']} at M{int(top['M'])}, area {area_label(top['cutout_area'])} "
            f"({fmt_difference_pp(top['mean_paired_difference'])}); largest mean CAM deficit is {bottom['comparison']} at M{int(bottom['M'])}, "
            f"area {area_label(bottom['cutout_area'])} ({fmt_difference_pp(bottom['mean_paired_difference'])})."
        )
    return "\n".join(rows)


def describe_dataset_differences(aggregate: pd.DataFrame, paired: pd.DataFrame) -> str:
    lines: list[str] = []
    best = aggregate_best_rows(aggregate)
    for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
        subset = best[(best["dataset"] == dataset) & best["best_validation_accuracy_mean"].notna()]
        if subset.empty:
            continue
        top = subset.sort_values("best_validation_accuracy_mean", ascending=False).iloc[0]
        baseline = subset[subset["mode"] == "none"].iloc[0]
        lines.append(
            f"- {dataset}: the highest aggregate mean best validation accuracy is {fmt_fraction_as_pct(top['best_validation_accuracy_mean'])}; "
            f"the no-cutout mean is {fmt_fraction_as_pct(baseline['best_validation_accuracy_mean'])}."
        )
    cam = paired[(paired["metric"] == "best_validation_accuracy") & (paired["comparison_key"].isin(["cam_low_minus_random", "cam_high_minus_random"]))]
    if not cam.empty:
        for key, label in (
            ("cam_low_minus_random", "low-saliency minus random"),
            ("cam_high_minus_random", "high-saliency minus random"),
        ):
            ranges = {}
            for dataset in (display_dataset(dataset_id) for dataset_id in EXPECTED_DATASETS):
                values = (
                    cam[(cam["dataset"] == dataset) & (cam["comparison_key"] == key)]["mean_paired_difference"]
                    .dropna()
                    .astype(float)
                )
                if not values.empty:
                    ranges[dataset] = (float(values.min()), float(values.max()))
            if len(ranges) == 2:
                lines.append(
                    f"- For {label}, CIFAR-100 cell mean paired effects range from {fmt_difference_pp(ranges['CIFAR-100'][0])} "
                    f"to {fmt_difference_pp(ranges['CIFAR-100'][1])}; RawMal-TF ranges from {fmt_difference_pp(ranges['RawMal-TF'][0])} "
                    f"to {fmt_difference_pp(ranges['RawMal-TF'][1])}."
                )
    return "\n".join(lines)


def report_valid_inventory(records: list[dict[str, Any]], per_run: pd.DataFrame, missing_invalid: pd.DataFrame) -> str:
    status_counts = Counter(record["status"] for record in records)
    missing_count = (
        int(missing_invalid["issue_type"].astype(str).str.startswith("missing_expected").sum())
        if not missing_invalid.empty
        else 0
    )
    invalid_count = (
        int((~missing_invalid["issue_type"].astype(str).str.startswith("missing_expected")).sum())
        if not missing_invalid.empty
        else 0
    )
    return (
        f"Discovered {len(records)} run folders with configs and metrics. After deduplicating no-cutout baselines, "
        f"{len(per_run)} runs are analysis-valid: {status_counts.get('valid', 0)} augmented runs and "
        f"{status_counts.get('valid_selected_baseline', 0)} selected no-cutout baselines. "
        f"The missing-or-invalid table contains {missing_count} missing expected combinations and {invalid_count} invalid or excluded run records."
    )


def write_summary_report(
    records: list[dict[str, Any]],
    per_run: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    duplicate_baselines: pd.DataFrame,
    missing_invalid: pd.DataFrame,
    plot_outputs: list[Path],
) -> None:
    status_counts = Counter(record["status"] for record in records)
    exact_duplicates = status_counts.get("duplicate_baseline_exact", 0)
    inconsistent_duplicates = status_counts.get("duplicate_baseline_inconsistent", 0)
    baseline_groups = duplicate_baselines.groupby(["dataset", "seed"]).ngroups if not duplicate_baselines.empty else 0
    inconsistent_groups = 0
    if not duplicate_baselines.empty:
        inconsistent_groups = int(
            duplicate_baselines.groupby(["dataset", "seed"])["baseline_copies_differ"]
            .first()
            .astype(bool)
            .sum()
        )
    exact_word = "copy" if exact_duplicates == 1 else "copies"
    inconsistent_word = "copy" if inconsistent_duplicates == 1 else "copies"
    group_word = "group" if inconsistent_groups == 1 else "groups"
    combination_count = int(len(aggregate))

    lines = [
        "# CAM Cutout Validation Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Research Question",
        "",
        "This project asks whether saliency-guided cutout improves validation performance or stability relative to standard random cutout. The four conditions are no cutout, random cutout, low-saliency cutout, and high-saliency cutout. The primary estimates are means across seeds 42, 43, and 44 for each matched dataset, M, cutout-area, and condition combination.",
        "",
        "The CSV `eval_*` metrics are validation metrics, not held-out test results.",
        "",
        "## Valid Run Inventory",
        "",
        report_valid_inventory(records, per_run, missing_invalid),
        "",
        f"The aggregate table has {combination_count} dataset/condition/M/area rows: one no-cutout row per dataset plus separate rows for every M and area for random, low-saliency, and high-saliency cutout. No aggregate row averages across different areas, M values, or conditions.",
        "",
        "## No-Cutout Duplicate Handling",
        "",
        "No-cutout runs appear under multiple area directories even though area does not apply. The generator compares metric hashes and normalized config hashes, then selects exactly one no-cutout observation per dataset and seed. Repeated baseline copies are excluded from seed counts and paired comparisons.",
        "",
        f"Baseline duplicate findings: {baseline_groups} dataset/seed baseline groups, {exact_duplicates} exact duplicate {exact_word} excluded, {inconsistent_duplicates} differing {inconsistent_word} excluded, and {inconsistent_groups} {group_word} with any differing baseline content. Details are in `tables/duplicate_baselines.csv`.",
        "",
        "## Highest Mean Performance",
        "",
        describe_top_performance(aggregate),
        "",
        "## Lowest Across-Seed Variance",
        "",
        describe_lowest_variance(aggregate),
        "",
        "## CAM Versus Random Cutout",
        "",
        describe_cam_consistency(paired),
        "",
        "## Effects by Area and M",
        "",
        describe_area_m_effects(paired),
        "",
        "## CIFAR-100 Versus RawMal-TF",
        "",
        describe_dataset_differences(aggregate, paired),
        "",
        "## Seed Consistency",
        "",
        "The paired-effect table computes each comparison within seed before aggregating. Apparent improvements are strongest when all three seed-level differences in the matched area/M cell have the same sign; mixed-sign cells should be read as seed-sensitive rather than reliable treatment wins.",
        "",
        "## M8 Minus M4",
        "",
        "The M8-minus-M4 paired effects are descriptive because M changes the number of augmented samples and optimizer updates. The generator computes M8 minus M4 within the same dataset, seed, area, and cutout condition before reporting means and variability.",
        "",
        "## Plot Notes",
        "",
        f"The plot directory contains {len(plot_outputs) // 2} figures, each saved as high-resolution PNG and PDF. Plotted accuracies are percentages, plotted differences are percentage points, and every error bar or shaded band represents sample standard deviation across seeds.",
        "",
        "## Limitations",
        "",
        "Only three seeds are available, so variance and t-based 95% confidence intervals are exploratory. The files support validation accuracy, validation loss, and stability summaries, but they do not support:",
        "",
        "- held-out test accuracy;",
        "- macro-F1;",
        "- per-family metrics;",
        "- confusion matrices;",
        "- calibration;",
        "- sample-level predictions;",
        "- saliency-faithfulness measurements;",
        "- zero-padding overlap;",
        "- wall-clock or GPU-efficiency analysis.",
        "",
        "No unavailable result is fabricated here.",
    ]
    (SUMMARY_DIR / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integrity_report(
    records: list[dict[str, Any]],
    per_run: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    duplicate_baselines: pd.DataFrame,
    missing_invalid: pd.DataFrame,
    plot_outputs: list[Path],
) -> None:
    status_counts = Counter(record["status"] for record in records)
    missing_count = (
        int(missing_invalid["issue_type"].astype(str).str.startswith("missing_expected").sum())
        if not missing_invalid.empty
        else 0
    )
    invalid_count = (
        int((~missing_invalid["issue_type"].astype(str).str.startswith("missing_expected")).sum())
        if not missing_invalid.empty
        else 0
    )
    duplicate_groups: list[dict[str, Any]] = []
    if not duplicate_baselines.empty:
        for (dataset, seed), group in duplicate_baselines.groupby(["dataset", "seed"], sort=True):
            selected = group[group["selected_for_analysis"] == True]
            duplicate_groups.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "group_size": int(len(group)),
                    "selected_source_path": "" if selected.empty else str(selected.iloc[0]["source_path"]),
                    "baseline_copies_differ": bool(group["baseline_copies_differ"].iloc[0]),
                    "metrics_all_match_selected": bool(group["metrics_match_selected"].all()),
                    "normalized_configs_all_match_selected": bool(group["normalized_config_matches_selected"].all()),
                    "excluded_sources": group[group["selected_for_analysis"] == False]["source_path"].astype(str).tolist(),
                }
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "datasets": ["CIFAR-100", "RawMal-TF"],
            "architecture": EXPECTED_ARCHITECTURE,
            "seeds": list(EXPECTED_SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "cutout_areas": list(EXPECTED_AREAS),
            "M_values": list(EXPECTED_MS),
            "conditions": [CONDITION_LABELS[condition] for condition in CONDITIONS],
            "metric_source": "CSV eval columns are validation metrics, not held-out test metrics.",
        },
        "counts": {
            "discovered_run_count": int(len(records)),
            "valid_run_count": int(len(per_run)),
            "invalid_or_missing_run_count": int(missing_count + invalid_count),
            "missing_expected_combination_count": int(missing_count),
            "invalid_or_excluded_run_count": int(invalid_count),
            "unique_across_seed_parameter_combinations": int(len(aggregate)),
            "aggregate_rows": int(len(aggregate)),
            "paired_effect_rows": int(len(paired)),
            "plot_files": int(len(plot_outputs)),
            "figure_count": int(len(plot_outputs) // 2),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "calculations": {
            "primary_values": "Aggregate tables and plots use means across seeds.",
            "variance_and_standard_deviation": "Sample calculations use ddof=1.",
            "standard_error": "sample_std / sqrt(valid seed count).",
            "confidence_interval": "Exploratory two-sided 95% t-based interval.",
            "paired_effects": "Differences are computed within seed before aggregation.",
            "maximum_drawdown": "Largest drop from the previous running-best validation accuracy.",
            "collapse_event": "A drop of at least 0.05 from the previous running-best validation accuracy; consecutive below-threshold epochs count as one event until recovery.",
            "plot_error_bars": "Sample standard deviation across seeds.",
        },
        "duplicate_baselines": duplicate_groups,
        "generated_files": [relative_path(path) for path in sorted(SUMMARY_DIR.rglob("*")) if path.is_file()],
        "missing_or_invalid_rows": (
            missing_invalid.where(pd.notna(missing_invalid), None).to_dict(orient="records")
            if not missing_invalid.empty
            else []
        ),
        "unsupported_outputs": [
            "held-out test accuracy",
            "macro-F1",
            "per-family metrics",
            "confusion matrices",
            "calibration",
            "sample-level predictions",
            "saliency-faithfulness measurements",
            "zero-padding overlap",
            "wall-clock or GPU-efficiency analysis",
        ],
    }
    (SUMMARY_DIR / "integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def verify_outputs(plot_outputs: list[Path]) -> None:
    required_files = [
        SUMMARY_DIR / "generate_summary.py",
        SUMMARY_DIR / "summary_report.md",
        SUMMARY_DIR / "integrity_report.json",
        TABLES_DIR / "run_inventory.csv",
        TABLES_DIR / "per_run_validation_metrics.csv",
        TABLES_DIR / "aggregate_validation_metrics.csv",
        TABLES_DIR / "paired_effects_across_seeds.csv",
        TABLES_DIR / "missing_or_invalid_runs.csv",
        TABLES_DIR / "duplicate_baselines.csv",
    ]
    missing = [relative_path(path) for path in required_files if not path.exists()]
    for dirname in PLOT_DIRS:
        directory = PLOTS_DIR / dirname
        if not directory.exists() or not any(directory.glob("*.png")) or not any(directory.glob("*.pdf")):
            missing.append(relative_path(directory))
    if missing:
        raise RuntimeError("Missing expected outputs: " + ", ".join(missing))

    png_count = len(list(PLOTS_DIR.rglob("*.png")))
    pdf_count = len(list(PLOTS_DIR.rglob("*.pdf")))
    if png_count != pdf_count:
        raise RuntimeError(f"PNG/PDF plot count mismatch: {png_count} PNG and {pdf_count} PDF")

    forbidden_in_filenames = ("drive_zip", "malimg", "densenet", "densenet121")
    filenames = "\n".join(path.name.lower() for path in SUMMARY_DIR.rglob("*") if path.is_file())
    hits = [term for term in forbidden_in_filenames if term in filenames]
    if hits:
        raise RuntimeError("Forbidden term found in generated filenames: " + ", ".join(hits))

    report_text = (SUMMARY_DIR / "summary_report.md").read_text(encoding="utf-8").lower()
    forbidden_report_terms = ("drive_zip", "malimg", "densenet", "densenet121")
    hits = [term for term in forbidden_report_terms if term in report_text]
    if hits:
        raise RuntimeError("Forbidden term found in summary_report.md: " + ", ".join(hits))


def print_completion(
    records: list[dict[str, Any]],
    per_run: pd.DataFrame,
    aggregate: pd.DataFrame,
    duplicate_baselines: pd.DataFrame,
    missing_invalid: pd.DataFrame,
    plot_outputs: list[Path],
) -> None:
    status_counts = Counter(record["status"] for record in records)
    missing_count = (
        int(missing_invalid["issue_type"].astype(str).str.startswith("missing_expected").sum())
        if not missing_invalid.empty
        else 0
    )
    invalid_count = (
        int((~missing_invalid["issue_type"].astype(str).str.startswith("missing_expected")).sum())
        if not missing_invalid.empty
        else 0
    )
    exact_duplicates = status_counts.get("duplicate_baseline_exact", 0)
    inconsistent_duplicates = status_counts.get("duplicate_baseline_inconsistent", 0)
    baseline_findings = (
        f"{exact_duplicates} exact duplicate no-cutout copies excluded; "
        f"{inconsistent_duplicates} differing no-cutout copies excluded."
    )
    generated_files = [relative_path(path) for path in sorted(SUMMARY_DIR.rglob("*")) if path.is_file()]

    print(f"valid_run_count: {len(per_run)}")
    print(f"invalid_or_missing_run_count: {missing_count + invalid_count}")
    print(f"unique_across_seed_parameter_combinations: {len(aggregate)}")
    print(f"baseline_duplicate_findings: {baseline_findings}")
    print("generated_file_list:")
    for path in generated_files:
        print(f"- {path}")
    print("outside_runs_summary_changed: false")
    print(f"plot_files: {len(plot_outputs)}")


def main() -> None:
    configure_matplotlib()
    clean_summary_dir()
    records, metric_frames = discover_runs()
    analysis_runs, duplicate_baselines = select_analysis_runs(records)
    per_run = compute_per_run_metrics(analysis_runs, metric_frames)
    aggregate = build_aggregate_table(per_run)
    seed_effects = build_seed_level_paired_effects(per_run)
    paired = build_paired_effects_table(seed_effects)
    missing_invalid = build_missing_or_invalid_table(records, per_run)

    write_csv(build_inventory_table(records), TABLES_DIR / "run_inventory.csv")
    write_csv(per_run, TABLES_DIR / "per_run_validation_metrics.csv")
    write_csv(aggregate, TABLES_DIR / "aggregate_validation_metrics.csv")
    write_csv(paired, TABLES_DIR / "paired_effects_across_seeds.csv")
    write_csv(missing_invalid, TABLES_DIR / "missing_or_invalid_runs.csv")
    write_csv(duplicate_baselines, TABLES_DIR / "duplicate_baselines.csv")

    plot_outputs = create_all_plots(per_run, aggregate, paired, metric_frames, analysis_runs)
    write_summary_report(records, per_run, aggregate, paired, duplicate_baselines, missing_invalid, plot_outputs)
    write_integrity_report(records, per_run, aggregate, paired, duplicate_baselines, missing_invalid, plot_outputs)
    verify_outputs(plot_outputs)
    print_completion(records, per_run, aggregate, duplicate_baselines, missing_invalid, plot_outputs)


if __name__ == "__main__":
    main()
