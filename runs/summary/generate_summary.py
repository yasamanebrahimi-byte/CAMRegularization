#!/usr/bin/env python3
"""Generate publication-oriented summaries from existing run artifacts.

This script is intentionally self-contained and writes only under runs/summary.
It does not retrain models and does not modify any existing run artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
SUMMARY_DIR = RUNS_ROOT / "summary"
PLOTS_DIR = SUMMARY_DIR / "plots"

REQUESTED_DATASET_DIRS = ["cifar100", "malimg", "rawmaltf"]
EXTRA_DATASET_DIRS = ["drive_zip"]
RAWMALTF_FOLDERS = {"rawmaltf", "drive_zip"}

CONDITION_ORDER = [
    "none",
    "random_M4",
    "random_M8",
    "cam_low_M4",
    "cam_high_M4",
    "cam_low_M8",
    "cam_high_M8",
]

CONDITION_LABELS = {
    "none": "none",
    "random_M4": "random M4",
    "random_M8": "random M8",
    "cam_low_M4": "cam_low M4",
    "cam_high_M4": "cam_high M4",
    "cam_low_M8": "cam_low M8",
    "cam_high_M8": "cam_high M8",
}

MODE_ORDER = {"none": 0, "random": 1, "cam_low": 2, "cam_high": 3}
STRING_METRIC_COLUMNS = {
    "split",
    "evalsplit",
    "validationsplitname",
    "testsplitname",
    "dataset",
    "mode",
    "cutoutmode",
}

CREATED_FILES: set[Path] = set()
PLOTS_CREATED = 0


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return ""
        return float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    if isinstance(value, bool):
        return value
    return value


def clean_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        cleaned.append({key: clean_scalar(value) for key, value in record.items()})
    return cleaned


def write_csv(path: Path, records: Iterable[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    records = clean_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for record in records:
            for key in record:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    CREATED_FILES.add(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    CREATED_FILES.add(path)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(data), indent=2, sort_keys=True), encoding="utf-8")
    CREATED_FILES.add(path)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return rel(value)
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def parse_run_name(run_name: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "model": "",
        "seed": "",
        "cutout_mode": "",
        "cutout_m": "",
        "cutout_area": "",
    }
    seed_match = re.search(r"_seed(\d+)", run_name)
    if seed_match:
        parsed["seed"] = int(seed_match.group(1))
        parsed["model"] = run_name[: seed_match.start()]

    if "_cam_low" in run_name:
        parsed["cutout_mode"] = "cam_low"
    elif "_cam_high" in run_name:
        parsed["cutout_mode"] = "cam_high"
    elif "_random" in run_name:
        parsed["cutout_mode"] = "random"
    elif run_name.endswith("_none") or "_none" in run_name:
        parsed["cutout_mode"] = "none"

    m_match = re.search(r"_M(\d+)", run_name)
    if m_match:
        parsed["cutout_m"] = int(m_match.group(1))

    area_match = re.search(r"_area([0-9.]+)", run_name)
    if area_match:
        area_text = area_match.group(1).rstrip(".")
        try:
            parsed["cutout_area"] = float(area_text)
        except ValueError:
            parsed["cutout_area"] = area_text
    return parsed


def comparable_number(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f):
        return ""
    if abs(f - round(f)) < 1e-12:
        return int(round(f))
    return round(f, 12)


def dataset_display_name(dataset_folder: str, config_dataset: str = "") -> str:
    if dataset_folder in RAWMALTF_FOLDERS or str(config_dataset).lower() in RAWMALTF_FOLDERS:
        return "RawMal-TF (drive_zip)" if dataset_folder == "drive_zip" else "RawMal-TF"
    if dataset_folder == "cifar100":
        return "CIFAR100"
    if dataset_folder == "malimg":
        return "MalImg"
    return dataset_folder or config_dataset


def condition_key(mode: Any, m_value: Any) -> str:
    mode = str(mode or "").strip()
    if mode == "none":
        return "none"
    m_value = comparable_number(m_value)
    if m_value == "":
        return mode
    return f"{mode}_M{m_value}"


def condition_label(mode: Any, m_value: Any) -> str:
    key = condition_key(mode, m_value)
    return CONDITION_LABELS.get(key, key.replace("_", " "))


def condition_sort_key(key_or_label: str) -> Tuple[int, str]:
    if key_or_label in CONDITION_ORDER:
        return (CONDITION_ORDER.index(key_or_label), key_or_label)
    normalized = key_or_label.replace(" ", "_")
    if normalized in CONDITION_ORDER:
        return (CONDITION_ORDER.index(normalized), normalized)
    return (len(CONDITION_ORDER), str(key_or_label))


def pick_column(columns: Sequence[str], candidates: Sequence[str], exclude_prefixes: Sequence[str] = ()) -> str:
    normalized_to_original = {normalize_name(col): col for col in columns}
    excluded = tuple(normalize_name(prefix) for prefix in exclude_prefixes)
    for candidate in candidates:
        norm = normalize_name(candidate)
        if norm in normalized_to_original:
            original = normalized_to_original[norm]
            original_norm = normalize_name(original)
            if not any(original_norm.startswith(prefix) for prefix in excluded):
                return original

    candidate_norms = [normalize_name(candidate) for candidate in candidates]
    scored: List[Tuple[int, str]] = []
    for col in columns:
        norm_col = normalize_name(col)
        if any(norm_col.startswith(prefix) for prefix in excluded):
            continue
        for idx, norm_candidate in enumerate(candidate_norms):
            if norm_candidate and norm_candidate in norm_col:
                scored.append((idx, col))
                break
    if scored:
        scored.sort()
        return scored[0][1]
    return ""


def metric_columns(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if df is None:
        return {"epoch": "", "train_acc": "", "eval_acc": "", "train_loss": "", "eval_loss": ""}
    columns = list(df.columns)
    epoch = pick_column(columns, ["epoch", "epochs"])
    train_acc = pick_column(
        columns,
        [
            "train_acc",
            "train_acc1",
            "train_accuracy",
            "train_top1",
            "top1_train",
            "train_acc_top1",
        ],
    )
    eval_acc = pick_column(
        columns,
        [
            "val_acc",
            "val_acc1",
            "validation_acc",
            "validation_accuracy",
            "eval_acc",
            "eval_acc1",
            "eval_accuracy",
            "test_acc",
            "test_acc1",
            "test_accuracy",
            "accuracy",
            "acc",
            "acc1",
        ],
        exclude_prefixes=("train",),
    )
    train_loss = pick_column(columns, ["train_loss", "training_loss", "loss_train"])
    eval_loss = pick_column(
        columns,
        ["val_loss", "validation_loss", "eval_loss", "test_loss", "loss_val", "loss_eval", "loss_test"],
        exclude_prefixes=("train",),
    )
    return {
        "epoch": epoch,
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if not column or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def finite_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f) or math.isinf(f):
        return ""
    return f


def value_at_index(series: pd.Series, idx: Any) -> Any:
    if series.empty or idx is None or idx not in series.index:
        return ""
    return finite_or_blank(series.loc[idx])


def last_valid_value(series: pd.Series) -> Any:
    if series.empty:
        return ""
    valid = series.dropna()
    if valid.empty:
        return ""
    return finite_or_blank(valid.iloc[-1])


def best_max(series: pd.Series) -> Tuple[Any, Any]:
    valid = series.dropna()
    if valid.empty:
        return "", None
    idx = valid.idxmax()
    return finite_or_blank(valid.loc[idx]), idx


def best_min(series: pd.Series) -> Tuple[Any, Any]:
    valid = series.dropna()
    if valid.empty:
        return "", None
    idx = valid.idxmin()
    return finite_or_blank(valid.loc[idx]), idx


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def difference(value: Any, baseline: Any) -> Any:
    value_f = safe_float(value)
    base_f = safe_float(baseline)
    if value_f is None or base_f is None:
        return ""
    return value_f - base_f


def relative_difference_pct(value: Any, baseline: Any) -> Any:
    value_f = safe_float(value)
    base_f = safe_float(baseline)
    if value_f is None or base_f is None or abs(base_f) < 1e-15:
        return ""
    return 100.0 * (value_f - base_f) / abs(base_f)


def summary_stats_for_values(values: Sequence[Any]) -> Dict[str, Any]:
    numeric = [float(v) for v in values if safe_float(v) is not None]
    if not numeric:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "positive_count": 0}
    arr = np.asarray(numeric, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "positive_count": int(np.sum(arr > 0)),
        "negative_count": int(np.sum(arr < 0)),
        "zero_count": int(np.sum(np.isclose(arr, 0.0))),
        "positive_fraction": float(np.mean(arr > 0)),
    }


def find_run_dirs() -> Tuple[List[Path], List[str]]:
    warnings_out: List[str] = []
    scan_names = []
    for name in REQUESTED_DATASET_DIRS + EXTRA_DATASET_DIRS:
        if name not in scan_names:
            scan_names.append(name)

    roots: List[Path] = []
    for name in scan_names:
        path = RUNS_ROOT / name
        if path.exists() and path.is_dir():
            roots.append(path)
        elif name in REQUESTED_DATASET_DIRS:
            warnings_out.append(f"Requested run folder runs/{name}/ is missing.")

    run_dirs: List[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            if SUMMARY_DIR == path or SUMMARY_DIR in path.parents:
                continue
            artifact_names = ["config.json", "metrics.csv", "metrics_plot.png", "best_model.pt"]
            if any((path / artifact).exists() for artifact in artifact_names):
                run_dirs.append(path)
    return sorted(set(run_dirs)), warnings_out


def load_run(run_dir: Path) -> Dict[str, Any]:
    rel_parts = run_dir.relative_to(RUNS_ROOT).parts
    dataset_folder = rel_parts[0] if rel_parts else ""
    run_name = run_dir.name
    parsed = parse_run_name(run_name)

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    plot_path = run_dir / "metrics_plot.png"
    best_model_path = run_dir / "best_model.pt"
    log_files = sorted(
        [
            path
            for path in run_dir.glob("*")
            if path.is_file() and ("log" in path.name.lower() or path.suffix.lower() == ".log")
        ]
    )

    config: Dict[str, Any] = {}
    config_error = ""
    config_hash = ""
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_hash = sha256_file(config_path)
        except Exception as exc:  # noqa: BLE001 - report and continue.
            config_error = str(exc)

    metrics_df: Optional[pd.DataFrame] = None
    metrics_error = ""
    metrics_hash = ""
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            metrics_hash = sha256_file(metrics_path)
        except Exception as exc:  # noqa: BLE001 - report and continue.
            metrics_error = str(exc)

    cols = metric_columns(metrics_df)
    epoch_series = numeric_series(metrics_df, cols["epoch"]) if metrics_df is not None else pd.Series(dtype=float)
    observed_rows = int(len(metrics_df)) if metrics_df is not None else 0
    observed_metric_epochs = int(epoch_series.dropna().nunique()) if not epoch_series.empty else observed_rows
    first_epoch = finite_or_blank(epoch_series.dropna().iloc[0]) if not epoch_series.dropna().empty else ""
    last_epoch = finite_or_blank(epoch_series.dropna().iloc[-1]) if not epoch_series.dropna().empty else ""

    if len(rel_parts) >= 3:
        model_folder = rel_parts[1]
    else:
        model_folder = config.get("model") or parsed.get("model") or ""

    best_model_size = ""
    best_model_mtime = ""
    if best_model_path.exists():
        try:
            stat = best_model_path.stat()
            best_model_size = stat.st_size
            best_model_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except OSError:
            pass

    return {
        "run_dir": run_dir,
        "run_path": rel(run_dir),
        "dataset_folder": dataset_folder,
        "dataset_display": dataset_display_name(dataset_folder, str(config.get("dataset", ""))),
        "model_folder": model_folder,
        "run_name": run_name,
        "parsed": parsed,
        "config": config,
        "config_error": config_error,
        "metrics_error": metrics_error,
        "metrics_df": metrics_df,
        "metric_cols": cols,
        "config_exists": config_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "metrics_plot_exists": plot_path.exists(),
        "best_model_exists": best_model_path.exists(),
        "best_model_size_bytes": best_model_size,
        "best_model_mtime": best_model_mtime,
        "log_files_count": len(log_files),
        "log_files": ";".join(rel(path) for path in log_files),
        "config_sha256": config_hash,
        "metrics_sha256": metrics_hash,
        "observed_metric_rows": observed_rows,
        "observed_metric_epochs": observed_metric_epochs,
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "notes": [],
    }


def make_inventory(runs: List[Dict[str, Any]], run_notes: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        config = run["config"]
        notes = list(dict.fromkeys(run["notes"] + run_notes.get(run["run_path"], [])))
        has_blocking = any(note.startswith(("missing", "unreadable")) for note in notes)
        has_warning = bool(notes)
        status = "ok"
        if has_blocking:
            status = "incomplete"
        elif has_warning:
            status = "suspicious"
        rows.append(
            {
                "dataset_folder": run["dataset_folder"],
                "dataset_display": run["dataset_display"],
                "model_folder": run["model_folder"],
                "run_name": run["run_name"],
                "run_path": run["run_path"],
                "config_dataset": config.get("dataset", ""),
                "config_model": config.get("model", ""),
                "seed": config.get("seed", run["parsed"].get("seed", "")),
                "cutout_mode": config.get("cutout_mode", run["parsed"].get("cutout_mode", "")),
                "cutout_m": config.get("cutout_m", run["parsed"].get("cutout_m", "")),
                "cutout_area": config.get("cutout_area", run["parsed"].get("cutout_area", "")),
                "configured_epochs": config.get("epochs", ""),
                "observed_metric_rows": run["observed_metric_rows"],
                "observed_metric_epochs": run["observed_metric_epochs"],
                "first_epoch": run["first_epoch"],
                "last_epoch": run["last_epoch"],
                "grayscale": config.get("grayscale", ""),
                "include_regex": config.get("include_regex", ""),
                "teacher_checkpoint": config.get("teacher_checkpoint", ""),
                "teacher_checkpoint_sha256": config.get("teacher_checkpoint_sha256", ""),
                "config_exists": run["config_exists"],
                "metrics_exists": run["metrics_exists"],
                "metrics_plot_exists": run["metrics_plot_exists"],
                "best_model_exists": run["best_model_exists"],
                "best_model_size_bytes": run["best_model_size_bytes"],
                "best_model_mtime": run["best_model_mtime"],
                "log_files_count": run["log_files_count"],
                "log_files": run["log_files"],
                "config_sha256": run["config_sha256"],
                "metrics_sha256": run["metrics_sha256"],
                "status": status,
                "notes": "; ".join(notes),
            }
        )
    return rows


def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    config = run["config"]
    df = run["metrics_df"]
    cols = run["metric_cols"]
    mode = config.get("cutout_mode", run["parsed"].get("cutout_mode", ""))
    m_value = config.get("cutout_m", run["parsed"].get("cutout_m", ""))

    epoch = numeric_series(df, cols["epoch"]) if df is not None else pd.Series(dtype=float)
    train_acc = numeric_series(df, cols["train_acc"]) if df is not None else pd.Series(dtype=float)
    eval_acc = numeric_series(df, cols["eval_acc"]) if df is not None else pd.Series(dtype=float)
    train_loss = numeric_series(df, cols["train_loss"]) if df is not None else pd.Series(dtype=float)
    eval_loss = numeric_series(df, cols["eval_loss"]) if df is not None else pd.Series(dtype=float)

    best_eval_acc, best_eval_idx = best_max(eval_acc)
    best_eval_loss, best_eval_loss_idx = best_min(eval_loss)
    best_train_acc, best_train_acc_idx = best_max(train_acc)
    best_train_loss, best_train_loss_idx = best_min(train_loss)

    final_train_acc = last_valid_value(train_acc)
    final_eval_acc = last_valid_value(eval_acc)
    final_train_loss = last_valid_value(train_loss)
    final_eval_loss = last_valid_value(eval_loss)

    best_epoch = value_at_index(epoch, best_eval_idx)
    best_eval_loss_epoch = value_at_index(epoch, best_eval_loss_idx)
    best_train_acc_epoch = value_at_index(epoch, best_train_acc_idx)
    best_train_loss_epoch = value_at_index(epoch, best_train_loss_idx)

    train_acc_at_best_eval = value_at_index(train_acc, best_eval_idx)
    train_loss_at_best_eval = value_at_index(train_loss, best_eval_idx)
    eval_loss_at_best_eval = value_at_index(eval_loss, best_eval_idx)

    gap_best = difference(train_acc_at_best_eval, best_eval_acc)
    gap_final = difference(final_train_acc, final_eval_acc)

    eval_split = ""
    if df is not None:
        split_col = pick_column(df.columns, ["eval_split", "split", "val_split_name", "test_split_name"])
        if split_col and split_col in df.columns and not df[split_col].dropna().empty:
            eval_split = str(df[split_col].dropna().iloc[-1])

    return {
        "dataset_folder": run["dataset_folder"],
        "dataset_display": run["dataset_display"],
        "model_folder": run["model_folder"],
        "run_name": run["run_name"],
        "run_path": run["run_path"],
        "config_dataset": config.get("dataset", ""),
        "config_model": config.get("model", ""),
        "seed": config.get("seed", run["parsed"].get("seed", "")),
        "grayscale": config.get("grayscale", ""),
        "include_regex": config.get("include_regex", ""),
        "cutout_mode": mode,
        "cutout_m": m_value,
        "cutout_area": config.get("cutout_area", run["parsed"].get("cutout_area", "")),
        "condition_key": condition_key(mode, m_value),
        "condition_label": condition_label(mode, m_value),
        "eval_split": eval_split,
        "train_acc_col": cols["train_acc"],
        "eval_acc_col": cols["eval_acc"],
        "train_loss_col": cols["train_loss"],
        "eval_loss_col": cols["eval_loss"],
        "final_train_accuracy": final_train_acc,
        "final_eval_accuracy": final_eval_acc,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "best_train_accuracy": best_train_acc,
        "best_train_accuracy_epoch": best_train_acc_epoch,
        "best_eval_accuracy": best_eval_acc,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_train_loss_epoch": best_train_loss_epoch,
        "best_eval_loss": best_eval_loss,
        "best_eval_loss_epoch": best_eval_loss_epoch,
        "train_accuracy_at_best_epoch": train_acc_at_best_eval,
        "train_loss_at_best_epoch": train_loss_at_best_eval,
        "eval_loss_at_best_epoch": eval_loss_at_best_eval,
        "generalization_gap_best_epoch": gap_best,
        "generalization_gap_final": gap_final,
        "baseline_none_run": "",
        "best_acc_improvement_over_none": "",
        "relative_best_acc_improvement_over_none_pct": "",
        "baseline_random_run": "",
        "best_acc_improvement_over_random": "",
        "relative_best_acc_improvement_over_random_pct": "",
        "notes": "",
    }


def baseline_group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("dataset_folder", ""),
        row.get("config_dataset", ""),
        row.get("config_model", ""),
        row.get("seed", ""),
        row.get("grayscale", ""),
        row.get("include_regex", ""),
    )


def random_group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return baseline_group_key(row) + (comparable_number(row.get("cutout_m", "")), comparable_number(row.get("cutout_area", "")))


def attach_improvements(summary_rows: List[Dict[str, Any]]) -> None:
    none_baselines: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    random_baselines: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for row in summary_rows:
        if row.get("cutout_mode") == "none":
            key = baseline_group_key(row)
            if key not in none_baselines or safe_float(row.get("best_eval_accuracy")) is not None:
                none_baselines[key] = row
        if row.get("cutout_mode") == "random":
            key = random_group_key(row)
            if key not in random_baselines or safe_float(row.get("best_eval_accuracy")) is not None:
                random_baselines[key] = row

    for row in summary_rows:
        none = none_baselines.get(baseline_group_key(row))
        if none is not None:
            row["baseline_none_run"] = none.get("run_name", "")
            row["best_acc_improvement_over_none"] = difference(row.get("best_eval_accuracy"), none.get("best_eval_accuracy"))
            row["relative_best_acc_improvement_over_none_pct"] = relative_difference_pct(
                row.get("best_eval_accuracy"), none.get("best_eval_accuracy")
            )
        else:
            row["notes"] = append_note(row.get("notes", ""), "missing matching no-cutout baseline")

        if str(row.get("cutout_mode", "")).startswith("cam"):
            random = random_baselines.get(random_group_key(row))
            if random is not None:
                row["baseline_random_run"] = random.get("run_name", "")
                row["best_acc_improvement_over_random"] = difference(
                    row.get("best_eval_accuracy"), random.get("best_eval_accuracy")
                )
                row["relative_best_acc_improvement_over_random_pct"] = relative_difference_pct(
                    row.get("best_eval_accuracy"), random.get("best_eval_accuracy")
                )
            else:
                row["notes"] = append_note(row.get("notes", ""), "missing matching random baseline")


def append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing.split("; "):
        return existing
    return f"{existing}; {note}"


def add_check(
    checks: List[Dict[str, Any]],
    check_type: str,
    severity: str,
    status: str,
    *,
    run: Optional[Dict[str, Any]] = None,
    dataset_folder: str = "",
    model: str = "",
    seed: Any = "",
    run_name: str = "",
    condition: str = "",
    expected: Any = "",
    observed: Any = "",
    details: str = "",
    notes: str = "",
) -> None:
    if run is not None:
        config = run["config"]
        dataset_folder = dataset_folder or run["dataset_folder"]
        model = model or config.get("model", run["parsed"].get("model", ""))
        seed = seed if seed != "" else config.get("seed", run["parsed"].get("seed", ""))
        run_name = run_name or run["run_name"]
        condition = condition or condition_label(
            config.get("cutout_mode", run["parsed"].get("cutout_mode", "")),
            config.get("cutout_m", run["parsed"].get("cutout_m", "")),
        )
    checks.append(
        {
            "check_type": check_type,
            "severity": severity,
            "status": status,
            "dataset_folder": dataset_folder,
            "model": model,
            "seed": seed,
            "run_name": run_name,
            "condition": condition,
            "expected": expected,
            "observed": observed,
            "details": details,
            "notes": notes,
        }
    )


def numeric_array_equal(df_a: Optional[pd.DataFrame], df_b: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    if df_a is None or df_b is None:
        return False, "one or both metrics files could not be loaded"
    common_cols = [col for col in df_a.columns if col in df_b.columns]
    numeric_cols = []
    for col in common_cols:
        if normalize_name(col) in STRING_METRIC_COLUMNS:
            continue
        a = pd.to_numeric(df_a[col], errors="coerce")
        b = pd.to_numeric(df_b[col], errors="coerce")
        if a.notna().any() or b.notna().any():
            numeric_cols.append(col)
    if not numeric_cols:
        return False, "no shared numeric metric columns"
    a_num = df_a[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    b_num = df_b[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if a_num.shape != b_num.shape:
        return False, f"numeric shapes differ: {a_num.shape} vs {b_num.shape}"
    equal = bool(np.allclose(a_num, b_num, rtol=0.0, atol=0.0, equal_nan=True))
    return equal, f"columns={','.join(numeric_cols)} shape={a_num.shape}"


def run_identity_key(run: Dict[str, Any], include_m_area: bool = False) -> Tuple[Any, ...]:
    config = run["config"]
    base = (
        run["dataset_folder"],
        config.get("dataset", ""),
        config.get("model", run["parsed"].get("model", "")),
        config.get("seed", run["parsed"].get("seed", "")),
        config.get("grayscale", ""),
        config.get("include_regex", ""),
    )
    if include_m_area:
        return base + (
            comparable_number(config.get("cutout_m", run["parsed"].get("cutout_m", ""))),
            comparable_number(config.get("cutout_area", run["parsed"].get("cutout_area", ""))),
        )
    return base


def build_integrity_checks(
    runs: List[Dict[str, Any]],
    missing_folder_warnings: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    checks: List[Dict[str, Any]] = []
    run_notes: Dict[str, List[str]] = defaultdict(list)

    for warning_text in missing_folder_warnings:
        severity = "warning"
        notes = ""
        if "runs/rawmaltf/" in warning_text and (RUNS_ROOT / "drive_zip").exists():
            notes = "runs/drive_zip/ exists and is treated as RawMal-TF based on project context."
        add_check(
            checks,
            "missing_dataset_folder",
            severity,
            "warning",
            dataset_folder="rawmaltf",
            expected="folder exists",
            observed="missing",
            details=warning_text,
            notes=notes,
        )

    for run in runs:
        config = run["config"]
        parsed = run["parsed"]
        run_path = run["run_path"]
        if not run["config_exists"]:
            add_check(checks, "missing_config", "critical", "fail", run=run, expected="config.json", observed="missing")
            run_notes[run_path].append("missing config.json")
        if run["config_error"]:
            add_check(
                checks,
                "unreadable_config",
                "critical",
                "fail",
                run=run,
                expected="valid JSON",
                observed=run["config_error"],
            )
            run_notes[run_path].append("unreadable config.json")
        if not run["metrics_exists"]:
            add_check(checks, "missing_metrics", "critical", "fail", run=run, expected="metrics.csv", observed="missing")
            run_notes[run_path].append("missing metrics.csv")
        if run["metrics_error"]:
            add_check(
                checks,
                "unreadable_metrics",
                "critical",
                "fail",
                run=run,
                expected="readable CSV",
                observed=run["metrics_error"],
            )
            run_notes[run_path].append("unreadable metrics.csv")

        configured_epochs = safe_float(config.get("epochs", ""))
        observed_epochs = safe_float(run["observed_metric_epochs"])
        last_epoch = safe_float(run["last_epoch"])
        if configured_epochs is not None and run["metrics_exists"]:
            mismatch = False
            details = []
            if observed_epochs is not None and int(observed_epochs) != int(configured_epochs):
                mismatch = True
                details.append(f"observed_metric_epochs={run['observed_metric_epochs']}")
            if last_epoch is not None and int(last_epoch) != int(configured_epochs):
                mismatch = True
                details.append(f"last_epoch={run['last_epoch']}")
            if mismatch:
                add_check(
                    checks,
                    "configured_vs_observed_epoch_mismatch",
                    "warning",
                    "fail",
                    run=run,
                    expected=configured_epochs,
                    observed="; ".join(details),
                    details="Configured epochs do not match observed metrics.",
                )
                run_notes[run_path].append("configured epochs mismatch observed metrics")

        if run["dataset_folder"] in {"cifar100"} | RAWMALTF_FOLDERS and run["metrics_exists"]:
            appears_100 = int(run["observed_metric_epochs"] or 0) == 100 and int(safe_float(run["last_epoch"]) or 0) == 100
            if not appears_100:
                add_check(
                    checks,
                    "rawmaltf_cifar100_not_100_epochs",
                    "warning",
                    "fail",
                    run=run,
                    expected="100 epochs",
                    observed=f"observed_metric_epochs={run['observed_metric_epochs']}, last_epoch={run['last_epoch']}",
                )
                run_notes[run_path].append("RawMal-TF/CIFAR100 run does not appear to be 100 epochs")

        if run["dataset_folder"] == "malimg" and run["metrics_exists"]:
            short_run = int(run["observed_metric_epochs"] or 0) < 100 or int(safe_float(run["last_epoch"]) or 0) < 100
            if short_run:
                add_check(
                    checks,
                    "malimg_short_run",
                    "expected_possible",
                    "warning",
                    run=run,
                    expected="possibly shorter MalImg run",
                    observed=f"observed_metric_epochs={run['observed_metric_epochs']}, last_epoch={run['last_epoch']}",
                    notes="Marked expected_possible per task instructions.",
                )
                run_notes[run_path].append("MalImg short run marked expected_possible")

        df = run["metrics_df"]
        if df is not None:
            non_numeric_details = []
            nan_details = []
            for col in df.columns:
                norm_col = normalize_name(col)
                if norm_col in STRING_METRIC_COLUMNS:
                    continue
                converted = pd.to_numeric(df[col], errors="coerce")
                original_nonempty = df[col].notna() & (df[col].astype(str).str.strip() != "")
                bad_count = int((original_nonempty & converted.isna()).sum())
                if bad_count:
                    non_numeric_details.append(f"{col}:{bad_count}")
                if norm_col not in STRING_METRIC_COLUMNS and converted.isna().any() and converted.notna().any():
                    nan_details.append(f"{col}:{int(converted.isna().sum())}")
            if non_numeric_details or nan_details:
                add_check(
                    checks,
                    "nan_or_non_numeric_metrics",
                    "warning",
                    "fail",
                    run=run,
                    observed="; ".join(non_numeric_details + nan_details),
                    details="NaN or non-numeric values detected in metric-like columns.",
                )
                run_notes[run_path].append("NaN/non-numeric metric values")

            cols = run["metric_cols"]
            for label, col in [("train_accuracy", cols["train_acc"]), ("eval_accuracy", cols["eval_acc"])]:
                if col:
                    values = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(values) > 1 and values.nunique(dropna=True) <= 1:
                        add_check(
                            checks,
                            "suspiciously_constant_accuracy",
                            "warning",
                            "fail",
                            run=run,
                            observed=f"{label} column {col} has {values.nunique()} unique value",
                            details=f"value={values.iloc[0] if not values.empty else ''}",
                        )
                        run_notes[run_path].append(f"suspiciously constant {label}")

        folder_dataset = run["dataset_folder"]
        config_dataset = config.get("dataset", "")
        if config_dataset and folder_dataset != config_dataset:
            add_check(
                checks,
                "folder_config_dataset_mismatch",
                "warning",
                "fail",
                run=run,
                expected=folder_dataset,
                observed=config_dataset,
                details="Dataset folder differs from config dataset.",
            )
            run_notes[run_path].append("folder/config dataset mismatch")

        config_model = config.get("model", "")
        parsed_model = parsed.get("model", "")
        if config_model and parsed_model and config_model != parsed_model:
            add_check(
                checks,
                "folder_config_model_mismatch",
                "warning",
                "fail",
                run=run,
                expected=parsed_model,
                observed=config_model,
                details="Model inferred from run folder differs from config model.",
            )
            run_notes[run_path].append("folder/config model mismatch")

        config_mode = config.get("cutout_mode", "")
        parsed_mode = parsed.get("cutout_mode", "")
        if config_mode and parsed_mode and config_mode != parsed_mode:
            add_check(
                checks,
                "folder_config_cutout_mode_mismatch",
                "warning",
                "fail",
                run=run,
                expected=parsed_mode,
                observed=config_mode,
                details="Cutout mode inferred from run folder differs from config.",
            )
            run_notes[run_path].append("folder/config cutout mode mismatch")

        if str(config_mode).startswith("cam") and not str(config.get("teacher_checkpoint", "")).strip():
            add_check(
                checks,
                "cam_missing_teacher_checkpoint",
                "critical",
                "fail",
                run=run,
                expected="teacher_checkpoint populated",
                observed="missing",
            )
            run_notes[run_path].append("CAM run missing teacher checkpoint in config")

    by_metrics_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_config_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["metrics_sha256"]:
            by_metrics_hash[run["metrics_sha256"]].append(run)
        if run["config_sha256"]:
            by_config_hash[run["config_sha256"]].append(run)

    duplicate_metric_count = 0
    for digest, grouped in sorted(by_metrics_hash.items()):
        if len(grouped) > 1:
            duplicate_metric_count += 1
            names = "; ".join(item["run_path"] for item in grouped)
            add_check(
                checks,
                "duplicate_metrics_hash",
                "warning",
                "fail",
                expected="unique metrics.csv hash",
                observed=digest,
                details=names,
            )
            for item in grouped:
                run_notes[item["run_path"]].append("metrics.csv hash duplicated with another run")
    if duplicate_metric_count == 0:
        add_check(checks, "duplicate_metrics_hash", "info", "pass", details="No duplicate metrics.csv hashes detected.")

    duplicate_config_count = 0
    for digest, grouped in sorted(by_config_hash.items()):
        if len(grouped) > 1:
            duplicate_config_count += 1
            names = "; ".join(item["run_path"] for item in grouped)
            add_check(
                checks,
                "duplicate_config_hash",
                "warning",
                "fail",
                expected="unique config.json hash",
                observed=digest,
                details=names,
            )
            for item in grouped:
                run_notes[item["run_path"]].append("config.json hash duplicated with another run")
    if duplicate_config_count == 0:
        add_check(checks, "duplicate_config_hash", "info", "pass", details="No duplicate config.json hashes detected.")

    cam_pairs: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        mode = run["config"].get("cutout_mode", run["parsed"].get("cutout_mode", ""))
        if mode in {"cam_low", "cam_high"}:
            cam_pairs[run_identity_key(run, include_m_area=True)][mode] = run

    pair_checks = 0
    for key, pair in sorted(cam_pairs.items(), key=lambda item: str(item[0])):
        if "cam_low" not in pair or "cam_high" not in pair:
            continue
        pair_checks += 1
        low = pair["cam_low"]
        high = pair["cam_high"]
        raw_hash_equal = bool(low["metrics_sha256"] and low["metrics_sha256"] == high["metrics_sha256"])
        numeric_equal, numeric_details = numeric_array_equal(low["metrics_df"], high["metrics_df"])
        severity = "critical" if raw_hash_equal or numeric_equal else "info"
        status = "fail" if raw_hash_equal or numeric_equal else "pass"
        details = (
            f"low={low['run_path']}; high={high['run_path']}; "
            f"raw_csv_hash_equal={raw_hash_equal}; numeric_arrays_equal={numeric_equal}; {numeric_details}"
        )
        add_check(
            checks,
            "identical_cam_low_high_metrics",
            severity,
            status,
            dataset_folder=low["dataset_folder"],
            model=low["config"].get("model", low["parsed"].get("model", "")),
            seed=low["config"].get("seed", low["parsed"].get("seed", "")),
            condition=f"M{comparable_number(low['config'].get('cutout_m', ''))} area={low['config'].get('cutout_area', '')}",
            expected="cam_low and cam_high differ",
            observed=f"raw_csv_hash_equal={raw_hash_equal}; numeric_arrays_equal={numeric_equal}",
            details=details,
            notes="Publication-critical low/high identity check.",
        )
        if raw_hash_equal or numeric_equal:
            run_notes[low["run_path"]].append("cam_low metrics identical to cam_high counterpart")
            run_notes[high["run_path"]].append("cam_high metrics identical to cam_low counterpart")
    if pair_checks == 0:
        add_check(
            checks,
            "identical_cam_low_high_metrics",
            "warning",
            "skipped",
            details="No complete cam_low/cam_high pairs found.",
        )

    return checks, run_notes


def make_comparison_table(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[baseline_group_key(row)].append(row)

    rows: List[Dict[str, Any]] = []
    for key, grouped in sorted(groups.items(), key=lambda item: str(item[0])):
        first = grouped[0]
        output: Dict[str, Any] = {
            "dataset_folder": first["dataset_folder"],
            "dataset_display": first["dataset_display"],
            "config_dataset": first["config_dataset"],
            "config_model": first["config_model"],
            "seed": first["seed"],
            "grayscale": first["grayscale"],
            "include_regex": first["include_regex"],
            "notes": "",
        }
        by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in grouped:
            by_condition[row["condition_key"]].append(row)

        baseline_acc = ""
        none_rows = by_condition.get("none", [])
        if none_rows:
            baseline = max(none_rows, key=lambda row: safe_float(row.get("best_eval_accuracy")) or -np.inf)
            baseline_acc = baseline.get("best_eval_accuracy", "")

        for condition in CONDITION_ORDER:
            candidates = by_condition.get(condition, [])
            prefix = condition
            if candidates:
                selected = max(candidates, key=lambda row: safe_float(row.get("best_eval_accuracy")) or -np.inf)
                if len(candidates) > 1:
                    output["notes"] = append_note(output["notes"], f"multiple {condition} runs; selected best accuracy")
                output[f"{prefix}_run_name"] = selected["run_name"]
                output[f"{prefix}_best_accuracy"] = selected["best_eval_accuracy"]
                output[f"{prefix}_final_accuracy"] = selected["final_eval_accuracy"]
                output[f"{prefix}_best_epoch"] = selected["best_epoch"]
                output[f"{prefix}_improvement_over_none"] = difference(selected["best_eval_accuracy"], baseline_acc)
                output[f"{prefix}_relative_improvement_over_none_pct"] = relative_difference_pct(
                    selected["best_eval_accuracy"], baseline_acc
                )
            else:
                output[f"{prefix}_run_name"] = ""
                output[f"{prefix}_best_accuracy"] = ""
                output[f"{prefix}_final_accuracy"] = ""
                output[f"{prefix}_best_epoch"] = ""
                output[f"{prefix}_improvement_over_none"] = ""
                output[f"{prefix}_relative_improvement_over_none_pct"] = ""
        rows.append(output)
    return rows


def percent(value: Any, digits: int = 2) -> str:
    f = safe_float(value)
    if f is None:
        return ""
    return f"{100.0 * f:.{digits}f}%"


def signed_percent_point(value: Any, digits: int = 2) -> str:
    f = safe_float(value)
    if f is None:
        return ""
    return f"{100.0 * f:+.{digits}f} pp"


def plain_number(value: Any, digits: int = 2) -> str:
    f = safe_float(value)
    if f is None:
        return ""
    return f"{f:.{digits}f}"


def df_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(clean_records(records))


def save_plot(fig: plt.Figure, path: Path) -> None:
    global PLOTS_CREATED
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    CREATED_FILES.add(path)
    PLOTS_CREATED += 1


def safe_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "plot"


def ordered_conditions(df: pd.DataFrame) -> List[str]:
    keys = sorted(df["condition_key"].dropna().unique().tolist(), key=condition_sort_key)
    return keys


def create_plots(summary_rows: List[Dict[str, Any]], runs: List[Dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = df_from_records(summary_rows)
    if summary_df.empty:
        return

    colors = {
        "none": "#333333",
        "random_M4": "#4C78A8",
        "random_M8": "#72B7B2",
        "cam_low_M4": "#54A24B",
        "cam_high_M4": "#E45756",
        "cam_low_M8": "#B279A2",
        "cam_high_M8": "#F58518",
    }

    curve_records: List[Dict[str, Any]] = []
    for run in runs:
        df = run["metrics_df"]
        if df is None:
            continue
        cols = run["metric_cols"]
        config = run["config"]
        mode = config.get("cutout_mode", run["parsed"].get("cutout_mode", ""))
        m_value = config.get("cutout_m", run["parsed"].get("cutout_m", ""))
        key = condition_key(mode, m_value)
        label = condition_label(mode, m_value)
        epoch = numeric_series(df, cols["epoch"])
        train_acc = numeric_series(df, cols["train_acc"])
        eval_acc = numeric_series(df, cols["eval_acc"])
        train_loss = numeric_series(df, cols["train_loss"])
        eval_loss = numeric_series(df, cols["eval_loss"])
        for idx in range(len(df)):
            curve_records.append(
                {
                    "dataset_folder": run["dataset_folder"],
                    "dataset_display": run["dataset_display"],
                    "config_model": config.get("model", run["parsed"].get("model", "")),
                    "seed": config.get("seed", run["parsed"].get("seed", "")),
                    "run_name": run["run_name"],
                    "condition_key": key,
                    "condition_label": label,
                    "epoch": finite_or_blank(epoch.iloc[idx]) if idx < len(epoch) else "",
                    "train_accuracy": finite_or_blank(train_acc.iloc[idx]) if idx < len(train_acc) else "",
                    "eval_accuracy": finite_or_blank(eval_acc.iloc[idx]) if idx < len(eval_acc) else "",
                    "train_loss": finite_or_blank(train_loss.iloc[idx]) if idx < len(train_loss) else "",
                    "eval_loss": finite_or_blank(eval_loss.iloc[idx]) if idx < len(eval_loss) else "",
                }
            )

    curve_df = df_from_records(curve_records)
    if not curve_df.empty:
        group_cols = ["dataset_folder", "dataset_display", "config_model"]
        for (dataset_folder, dataset_display, model), group in curve_df.groupby(group_cols, dropna=False):
            base_name = safe_filename(f"{dataset_folder}_{model}")
            table_path = PLOTS_DIR / f"{base_name}_accuracy_curves.csv"
            group.to_csv(table_path, index=False)
            CREATED_FILES.add(table_path)

            fig, ax = plt.subplots(figsize=(9, 5.2))
            for condition in ordered_conditions(group):
                sub = group[group["condition_key"] == condition].sort_values("epoch")
                ax.plot(
                    sub["epoch"],
                    sub["eval_accuracy"].astype(float),
                    label=CONDITION_LABELS.get(condition, condition),
                    color=colors.get(condition),
                    linewidth=1.8,
                )
            ax.set_title(f"{dataset_display} / {model}: validation accuracy curves")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            save_plot(fig, PLOTS_DIR / f"{base_name}_accuracy_curves.png")

            loss_table_path = PLOTS_DIR / f"{base_name}_loss_curves.csv"
            group.to_csv(loss_table_path, index=False)
            CREATED_FILES.add(loss_table_path)

            fig, ax = plt.subplots(figsize=(9, 5.2))
            for condition in ordered_conditions(group):
                sub = group[group["condition_key"] == condition].sort_values("epoch")
                ax.plot(
                    sub["epoch"],
                    sub["eval_loss"].astype(float),
                    label=CONDITION_LABELS.get(condition, condition),
                    color=colors.get(condition),
                    linewidth=1.8,
                )
            ax.set_title(f"{dataset_display} / {model}: validation loss curves")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            save_plot(fig, PLOTS_DIR / f"{base_name}_loss_curves.png")

    for (dataset_folder, dataset_display, model), group in summary_df.groupby(
        ["dataset_folder", "dataset_display", "config_model"], dropna=False
    ):
        base_name = safe_filename(f"{dataset_folder}_{model}")
        group = group.copy()
        group["_sort"] = group["condition_key"].map(lambda key: condition_sort_key(str(key))[0])
        group = group.sort_values(["_sort", "condition_key"])
        bar_table = group[
            [
                "dataset_folder",
                "dataset_display",
                "config_model",
                "seed",
                "condition_key",
                "condition_label",
                "run_name",
                "best_eval_accuracy",
                "final_eval_accuracy",
                "best_epoch",
                "best_acc_improvement_over_none",
            ]
        ]
        table_path = PLOTS_DIR / f"{base_name}_best_accuracy_bars.csv"
        bar_table.to_csv(table_path, index=False)
        CREATED_FILES.add(table_path)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(group))
        y = pd.to_numeric(group["best_eval_accuracy"], errors="coerce").to_numpy(dtype=float)
        labels = group["condition_label"].tolist()
        ax.bar(x, y, color=[colors.get(key, "#888888") for key in group["condition_key"]])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, max(1.0, np.nanmax(y) * 1.08 if np.isfinite(y).any() else 1.0))
        ax.set_ylabel("Best accuracy")
        ax.set_title(f"{dataset_display} / {model}: best validation accuracy")
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, PLOTS_DIR / f"{base_name}_best_accuracy_bars.png")

        improvement_table = bar_table.copy()
        table_path = PLOTS_DIR / f"{base_name}_improvement_over_none_bars.csv"
        improvement_table.to_csv(table_path, index=False)
        CREATED_FILES.add(table_path)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        y = pd.to_numeric(group["best_acc_improvement_over_none"], errors="coerce").to_numpy(dtype=float)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.bar(x, y, color=[colors.get(key, "#888888") for key in group["condition_key"]])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Improvement over none")
        ax.set_title(f"{dataset_display} / {model}: best-accuracy improvement over baseline")
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, PLOTS_DIR / f"{base_name}_improvement_over_none_bars.png")

    heatmap = summary_df.pivot_table(
        index=["dataset_display", "config_model"],
        columns="condition_key",
        values="best_eval_accuracy",
        aggfunc="max",
    )
    heatmap = heatmap.reindex(columns=[col for col in CONDITION_ORDER if col in heatmap.columns])
    heatmap_path = PLOTS_DIR / "overall_best_accuracy_heatmap.csv"
    heatmap.reset_index().to_csv(heatmap_path, index=False)
    CREATED_FILES.add(heatmap_path)

    if not heatmap.empty:
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.65 * len(heatmap))))
        matrix = heatmap.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        ax.set_xticks(np.arange(len(heatmap.columns)))
        ax.set_xticklabels([CONDITION_LABELS.get(col, col) for col in heatmap.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(heatmap.index)))
        ax.set_yticklabels([f"{idx[0]} / {idx[1]}" for idx in heatmap.index])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{100 * matrix[i, j]:.1f}%", ha="center", va="center", color="white", fontsize=8)
        ax.set_title("Overall best validation accuracy")
        fig.colorbar(im, ax=ax, label="Accuracy")
        save_plot(fig, PLOTS_DIR / "overall_best_accuracy_heatmap.png")

    final_vs_best = summary_df[
        [
            "dataset_folder",
            "dataset_display",
            "config_model",
            "seed",
            "condition_key",
            "condition_label",
            "run_name",
            "final_eval_accuracy",
            "best_eval_accuracy",
        ]
    ].copy()
    table_path = PLOTS_DIR / "final_vs_best_accuracy.csv"
    final_vs_best.to_csv(table_path, index=False)
    CREATED_FILES.add(table_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    for condition in ordered_conditions(summary_df):
        sub = summary_df[summary_df["condition_key"] == condition]
        ax.scatter(
            pd.to_numeric(sub["final_eval_accuracy"], errors="coerce"),
            pd.to_numeric(sub["best_eval_accuracy"], errors="coerce"),
            label=CONDITION_LABELS.get(condition, condition),
            color=colors.get(condition),
            s=52,
            alpha=0.9,
        )
    all_vals = pd.to_numeric(summary_df[["final_eval_accuracy", "best_eval_accuracy"]].stack(), errors="coerce").dropna()
    if not all_vals.empty:
        lo = max(0.0, float(all_vals.min()) - 0.02)
        hi = min(1.0, float(all_vals.max()) + 0.02)
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Final accuracy")
    ax.set_ylabel("Best accuracy")
    ax.set_title("Final vs best validation accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    save_plot(fig, PLOTS_DIR / "final_vs_best_accuracy.png")

    best_epoch_table = summary_df[
        [
            "dataset_folder",
            "dataset_display",
            "config_model",
            "seed",
            "condition_key",
            "condition_label",
            "run_name",
            "best_epoch",
            "best_eval_accuracy",
        ]
    ].copy()
    table_path = PLOTS_DIR / "best_epoch_by_run.csv"
    best_epoch_table.to_csv(table_path, index=False)
    CREATED_FILES.add(table_path)

    best_epoch_plot = best_epoch_table.copy()
    best_epoch_plot["label"] = (
        best_epoch_plot["dataset_display"].astype(str)
        + " / "
        + best_epoch_plot["condition_label"].astype(str)
    )
    best_epoch_plot = best_epoch_plot.sort_values(["dataset_display", "condition_key"], key=lambda col: col.astype(str))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(best_epoch_plot))))
    y_pos = np.arange(len(best_epoch_plot))
    ax.barh(y_pos, pd.to_numeric(best_epoch_plot["best_epoch"], errors="coerce"), color="#4C78A8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(best_epoch_plot["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Best epoch")
    ax.set_title("Best validation epoch by run")
    ax.grid(True, axis="x", alpha=0.25)
    save_plot(fig, PLOTS_DIR / "best_epoch_by_run.png")

    raw_df = summary_df[summary_df["dataset_folder"].isin(RAWMALTF_FOLDERS)].copy()
    if not raw_df.empty:
        raw_df["_sort"] = raw_df["condition_key"].map(lambda key: condition_sort_key(str(key))[0])
        raw_df = raw_df.sort_values(["config_model", "_sort", "condition_key"])
        raw_table = raw_df[
            [
                "dataset_folder",
                "dataset_display",
                "config_model",
                "seed",
                "condition_key",
                "condition_label",
                "run_name",
                "best_eval_accuracy",
                "final_eval_accuracy",
                "best_epoch",
                "best_acc_improvement_over_none",
                "best_acc_improvement_over_random",
            ]
        ]
        table_path = PLOTS_DIR / "rawmaltf_condition_comparison.csv"
        raw_table.to_csv(table_path, index=False)
        CREATED_FILES.add(table_path)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(raw_df))
        ax.bar(
            x,
            pd.to_numeric(raw_df["best_eval_accuracy"], errors="coerce"),
            color=[colors.get(key, "#888888") for key in raw_df["condition_key"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(raw_df["condition_label"], rotation=35, ha="right")
        ax.set_ylabel("Best accuracy")
        ax.set_title("RawMal-TF: condition comparison")
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, PLOTS_DIR / "rawmaltf_condition_comparison.png")

        mode_table_records = []
        for row in raw_df.to_dict("records"):
            mode = row.get("cutout_mode", "")
            family = "CAM" if str(mode).startswith("cam") else str(mode)
            mode_table_records.append(
                {
                    "family": family,
                    "condition_key": row.get("condition_key", ""),
                    "condition_label": row.get("condition_label", ""),
                    "best_eval_accuracy": row.get("best_eval_accuracy", ""),
                    "improvement_over_none": row.get("best_acc_improvement_over_none", ""),
                }
            )
        mode_table = df_from_records(mode_table_records)
        table_path = PLOTS_DIR / "rawmaltf_random_vs_cam.csv"
        mode_table.to_csv(table_path, index=False)
        CREATED_FILES.add(table_path)

        family_summary = mode_table.groupby("family", dropna=False)["best_eval_accuracy"].max().reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.bar(family_summary["family"], pd.to_numeric(family_summary["best_eval_accuracy"], errors="coerce"), color="#4C78A8")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Best accuracy")
        ax.set_title("RawMal-TF: random vs CAM best accuracy")
        ax.grid(True, axis="y", alpha=0.25)
        save_plot(fig, PLOTS_DIR / "rawmaltf_random_vs_cam.png")

        m_table = raw_df[raw_df["cutout_mode"] != "none"][
            ["cutout_mode", "cutout_m", "condition_key", "condition_label", "best_eval_accuracy", "best_acc_improvement_over_none"]
        ].copy()
        table_path = PLOTS_DIR / "rawmaltf_m4_vs_m8.csv"
        m_table.to_csv(table_path, index=False)
        CREATED_FILES.add(table_path)
        if not m_table.empty:
            grouped_m = m_table.groupby("cutout_m", dropna=False)["best_eval_accuracy"].max().reset_index()
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            ax.bar(grouped_m["cutout_m"].astype(str), pd.to_numeric(grouped_m["best_eval_accuracy"], errors="coerce"), color="#72B7B2")
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("M")
            ax.set_ylabel("Best accuracy")
            ax.set_title("RawMal-TF: M4 vs M8 best accuracy")
            ax.grid(True, axis="y", alpha=0.25)
            save_plot(fig, PLOTS_DIR / "rawmaltf_m4_vs_m8.png")

        low_high_records = []
        for m_value, group in raw_df[raw_df["cutout_mode"].isin(["cam_low", "cam_high"])].groupby("cutout_m", dropna=False):
            low = group[group["cutout_mode"] == "cam_low"]
            high = group[group["cutout_mode"] == "cam_high"]
            if low.empty or high.empty:
                continue
            low_best = pd.to_numeric(low["best_eval_accuracy"], errors="coerce").max()
            high_best = pd.to_numeric(high["best_eval_accuracy"], errors="coerce").max()
            low_high_records.append(
                {
                    "cutout_m": m_value,
                    "cam_low_best_accuracy": low_best,
                    "cam_high_best_accuracy": high_best,
                    "cam_high_minus_low": high_best - low_best,
                }
            )
        low_high_table = df_from_records(low_high_records)
        table_path = PLOTS_DIR / "rawmaltf_cam_low_vs_high.csv"
        low_high_table.to_csv(table_path, index=False)
        CREATED_FILES.add(table_path)
        if not low_high_table.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            x = np.arange(len(low_high_table))
            width = 0.36
            ax.bar(x - width / 2, low_high_table["cam_low_best_accuracy"], width, label="cam_low", color=colors["cam_low_M4"])
            ax.bar(x + width / 2, low_high_table["cam_high_best_accuracy"], width, label="cam_high", color=colors["cam_high_M4"])
            ax.set_xticks(x)
            ax.set_xticklabels([f"M{comparable_number(v)}" for v in low_high_table["cutout_m"]])
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Best accuracy")
            ax.set_title("RawMal-TF: cam_low vs cam_high")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend()
            save_plot(fig, PLOTS_DIR / "rawmaltf_cam_low_vs_high.png")


def top_rows(summary_rows: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    rows = [row for row in summary_rows if safe_float(row.get("best_eval_accuracy")) is not None]
    return sorted(rows, key=lambda row: safe_float(row.get("best_eval_accuracy")) or -np.inf, reverse=True)[:n]


def best_by_dataset(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[row["dataset_folder"]].append(row)
    best_rows = []
    for dataset, rows in sorted(groups.items()):
        candidates = [row for row in rows if safe_float(row.get("best_eval_accuracy")) is not None]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: safe_float(row.get("best_eval_accuracy")) or -np.inf)
        best_rows.append(best)
    return best_rows


def make_summary_stats(
    summary_rows: List[Dict[str, Any]],
    inventory_rows: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary_df = df_from_records(summary_rows)
    inventory_df = df_from_records(inventory_rows)

    status_counts = inventory_df["status"].value_counts().to_dict() if not inventory_df.empty else {}
    major_checks = [
        check
        for check in checks
        if check["status"] in {"fail", "warning"} and check["severity"] in {"critical", "warning", "expected_possible"}
    ]
    major_warnings = []
    for check in major_checks:
        text = f"{check['check_type']}: {check.get('run_name') or check.get('dataset_folder')}: {check.get('observed') or check.get('details')}"
        if text not in major_warnings:
            major_warnings.append(text)

    per_dataset: Dict[str, Any] = {}
    if not summary_df.empty:
        for dataset, group in summary_df.groupby("dataset_folder", dropna=False):
            best = group.loc[pd.to_numeric(group["best_eval_accuracy"], errors="coerce").idxmax()]
            per_dataset[str(dataset)] = {
                "runs": int(len(group)),
                "dataset_display": best.get("dataset_display", ""),
                "models": sorted(str(v) for v in group["config_model"].dropna().unique().tolist()),
                "conditions": sorted(
                    (str(v) for v in group["condition_key"].dropna().unique().tolist()),
                    key=condition_sort_key,
                ),
                "best_run": best.get("run_name", ""),
                "best_condition": best.get("condition_label", ""),
                "best_accuracy": safe_float(best.get("best_eval_accuracy")),
                "best_epoch": safe_float(best.get("best_epoch")),
            }

    per_model: Dict[str, Any] = {}
    if not summary_df.empty:
        for model, group in summary_df.groupby("config_model", dropna=False):
            best = group.loc[pd.to_numeric(group["best_eval_accuracy"], errors="coerce").idxmax()]
            per_model[str(model)] = {
                "runs": int(len(group)),
                "datasets": sorted(str(v) for v in group["dataset_folder"].dropna().unique().tolist()),
                "best_run": best.get("run_name", ""),
                "best_dataset": best.get("dataset_display", ""),
                "best_condition": best.get("condition_label", ""),
                "best_accuracy": safe_float(best.get("best_eval_accuracy")),
            }

    raw_rows = [row for row in summary_rows if row.get("dataset_folder") in RAWMALTF_FOLDERS]
    best_raw = top_rows(raw_rows, n=1)

    improvement_values = [row.get("best_acc_improvement_over_none") for row in summary_rows if row.get("cutout_mode") != "none"]
    cam_vs_random_values = [
        row.get("best_acc_improvement_over_random")
        for row in summary_rows
        if str(row.get("cutout_mode", "")).startswith("cam")
    ]

    low_high_diffs: List[float] = []
    pair_groups: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        if row.get("cutout_mode") in {"cam_low", "cam_high"}:
            key = baseline_group_key(row) + (
                comparable_number(row.get("cutout_m", "")),
                comparable_number(row.get("cutout_area", "")),
            )
            pair_groups[key][row["cutout_mode"]] = row
    identical_accuracy_pairs = 0
    for pair in pair_groups.values():
        if "cam_low" in pair and "cam_high" in pair:
            low_acc = safe_float(pair["cam_low"].get("best_eval_accuracy"))
            high_acc = safe_float(pair["cam_high"].get("best_eval_accuracy"))
            if low_acc is not None and high_acc is not None:
                diff = high_acc - low_acc
                low_high_diffs.append(diff)
                if abs(diff) < 1e-15:
                    identical_accuracy_pairs += 1

    identical_metric_checks = [
        check
        for check in checks
        if check["check_type"] == "identical_cam_low_high_metrics" and check["status"] == "fail"
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_runs": len(summary_rows),
        "status_counts": status_counts,
        "successful_count": int(status_counts.get("ok", 0)),
        "suspicious_count": int(status_counts.get("suspicious", 0)),
        "incomplete_count": int(status_counts.get("incomplete", 0)),
        "per_dataset": per_dataset,
        "per_model": per_model,
        "best_runs": [
            {
                "dataset": row.get("dataset_display", ""),
                "model": row.get("config_model", ""),
                "seed": row.get("seed", ""),
                "condition": row.get("condition_label", ""),
                "run_name": row.get("run_name", ""),
                "best_accuracy": safe_float(row.get("best_eval_accuracy")),
                "best_epoch": safe_float(row.get("best_epoch")),
            }
            for row in top_rows(summary_rows, n=10)
        ],
        "best_rawmaltf_run": (
            {
                "dataset": best_raw[0].get("dataset_display", ""),
                "model": best_raw[0].get("config_model", ""),
                "seed": best_raw[0].get("seed", ""),
                "condition": best_raw[0].get("condition_label", ""),
                "run_name": best_raw[0].get("run_name", ""),
                "best_accuracy": safe_float(best_raw[0].get("best_eval_accuracy")),
                "best_epoch": safe_float(best_raw[0].get("best_epoch")),
            }
            if best_raw
            else {}
        ),
        "improvement_statistics_over_no_cutout": summary_stats_for_values(improvement_values),
        "random_vs_cam_statistics": summary_stats_for_values(cam_vs_random_values),
        "cam_low_vs_cam_high_statistics": {
            "cam_high_minus_cam_low_best_accuracy": summary_stats_for_values(low_high_diffs),
            "pairs_compared": len(low_high_diffs),
            "identical_best_accuracy_pairs": identical_accuracy_pairs,
            "identical_metric_pairs": len(identical_metric_checks),
        },
        "major_warnings": major_warnings,
    }


def markdown_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows available._"
    selected = rows[:max_rows]
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            cells.append(str(value).replace("|", "/"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + body)


def make_paper_summary(
    summary_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    checks: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> str:
    datasets = sorted({row["dataset_display"] for row in summary_rows})
    models = sorted({str(row["config_model"]) for row in summary_rows})
    conditions = sorted({row["condition_key"] for row in summary_rows}, key=condition_sort_key)

    best_dataset_rows = []
    for row in best_by_dataset(summary_rows):
        best_dataset_rows.append(
            {
                "Dataset": row["dataset_display"],
                "Model": row["config_model"],
                "Seed": row["seed"],
                "Condition": row["condition_label"],
                "Best acc": percent(row["best_eval_accuracy"]),
                "Final acc": percent(row["final_eval_accuracy"]),
                "Best epoch": plain_number(row["best_epoch"], 0),
                "vs none": signed_percent_point(row["best_acc_improvement_over_none"]),
            }
        )

    raw_rows = [row for row in summary_rows if row["dataset_folder"] in RAWMALTF_FOLDERS]
    raw_rows = sorted(raw_rows, key=lambda row: condition_sort_key(row["condition_key"]))
    raw_table_rows = [
        {
            "Condition": row["condition_label"],
            "Best acc": percent(row["best_eval_accuracy"]),
            "Final acc": percent(row["final_eval_accuracy"]),
            "Best epoch": plain_number(row["best_epoch"], 0),
            "vs none": signed_percent_point(row["best_acc_improvement_over_none"]),
            "vs random": signed_percent_point(row["best_acc_improvement_over_random"]),
        }
        for row in raw_rows
    ]

    sanity_rows = []
    for row in summary_rows:
        if row["dataset_folder"] in {"cifar100", "malimg"}:
            sanity_rows.append(
                {
                    "Dataset": row["dataset_display"],
                    "Condition": row["condition_label"],
                    "Best acc": percent(row["best_eval_accuracy"]),
                    "Final acc": percent(row["final_eval_accuracy"]),
                    "Best epoch": plain_number(row["best_epoch"], 0),
                    "vs none": signed_percent_point(row["best_acc_improvement_over_none"]),
                }
            )
    sanity_rows = sorted(sanity_rows, key=lambda row: (row["Dataset"], condition_sort_key(row["Condition"].replace(" ", "_"))))

    identical_failures = [
        check for check in checks if check["check_type"] == "identical_cam_low_high_metrics" and check["status"] == "fail"
    ]
    warning_checks = [
        check
        for check in checks
        if check["status"] in {"fail", "warning"}
        and check["severity"] in {"critical", "warning", "expected_possible"}
    ]

    best_raw = stats.get("best_rawmaltf_run", {})
    random_vs_cam = stats.get("random_vs_cam_statistics", {})
    over_none = stats.get("improvement_statistics_over_no_cutout", {})
    low_high = stats.get("cam_low_vs_cam_high_statistics", {})

    interpretation_lines = []
    if best_raw:
        interpretation_lines.append(
            f"- RawMal-TF best run in the available artifacts is `{best_raw.get('run_name')}` "
            f"({best_raw.get('condition')}) with best accuracy {percent(best_raw.get('best_accuracy'))} "
            f"at epoch {plain_number(best_raw.get('best_epoch'), 0)}."
        )
    if random_vs_cam.get("count", 0):
        mean_delta = random_vs_cam.get("mean")
        pos = random_vs_cam.get("positive_count", 0)
        count = random_vs_cam.get("count", 0)
        interpretation_lines.append(
            f"- Across CAM runs with matched random baselines, the mean CAM-minus-random best-accuracy delta is "
            f"{signed_percent_point(mean_delta)} ({pos}/{count} positive)."
        )
    if over_none.get("count", 0):
        interpretation_lines.append(
            f"- Across non-baseline runs with matched no-cutout baselines, the mean best-accuracy delta is "
            f"{signed_percent_point(over_none.get('mean'))}."
        )
    if low_high.get("pairs_compared", 0):
        lh_stats = low_high.get("cam_high_minus_cam_low_best_accuracy", {})
        interpretation_lines.append(
            f"- For matched cam_high minus cam_low best accuracy, the mean delta is "
            f"{signed_percent_point(lh_stats.get('mean'))} across {low_high.get('pairs_compared')} pairs."
        )
    if not interpretation_lines:
        interpretation_lines.append("- Not enough matched runs were available for comparative interpretation.")

    low_high_warning = ""
    if identical_failures:
        low_high_warning = (
            "\n**Publication-critical warning:** At least one cam_low/cam_high pair has identical raw CSV hashes "
            "or identical loaded numeric metric arrays. Do not claim low/high saliency behavior differs for those pairs.\n"
        )
    else:
        low_high_warning = (
            "\nNo cam_low/cam_high pair had identical raw CSV hashes or identical loaded numeric arrays in this artifact set.\n"
        )

    warning_rows = [
        {
            "Severity": check["severity"],
            "Check": check["check_type"],
            "Run/Dataset": check.get("run_name") or check.get("dataset_folder", ""),
            "Observed": check.get("observed") or check.get("details", ""),
        }
        for check in warning_checks[:30]
    ]

    comparison_preview = []
    for row in comparison_rows:
        comparison_preview.append(
            {
                "Dataset": row.get("dataset_display", ""),
                "Model": row.get("config_model", ""),
                "Seed": row.get("seed", ""),
                "none": percent(row.get("none_best_accuracy")),
                "random M4": percent(row.get("random_M4_best_accuracy")),
                "random M8": percent(row.get("random_M8_best_accuracy")),
                "cam_low M4": percent(row.get("cam_low_M4_best_accuracy")),
                "cam_high M4": percent(row.get("cam_high_M4_best_accuracy")),
                "cam_low M8": percent(row.get("cam_low_M8_best_accuracy")),
                "cam_high M8": percent(row.get("cam_high_M8_best_accuracy")),
            }
        )

    md = f"""# CAM-Guided Cutout Summary

Generated from existing artifacts under `runs/cifar100/`, `runs/malimg/`, and the available RawMal-TF folder `runs/drive_zip/`. No model retraining or run-artifact edits were performed.

## Research Context

This package summarizes CAM-guided cutout augmentation for image-based malware classification. The intended comparison is no cutout (`none`), standard random cutout (`random`), low-saliency CAM-guided cutout (`cam_low`), and high-saliency CAM-guided cutout (`cam_high`). RawMal-TF / `drive_zip`, especially grayscale-only runs, is treated as the main publication dataset; CIFAR100 is a sanity check; MalImg is secondary malware evidence.

## Artifact Coverage

- Runs processed: {len(summary_rows)}
- Datasets found: {", ".join(datasets)}
- Models found: {", ".join(models)}
- Conditions found: {", ".join(CONDITION_LABELS.get(condition, condition) for condition in conditions)}
- Inventory/status counts: {json.dumps(stats.get("status_counts", {}), sort_keys=True)}

{low_high_warning}

## Best Result by Dataset

{markdown_table(best_dataset_rows, [("Dataset", "Dataset"), ("Model", "Model"), ("Seed", "Seed"), ("Condition", "Condition"), ("Best acc", "Best acc"), ("Final acc", "Final acc"), ("Best epoch", "Best epoch"), ("vs none", "vs none")])}

## Paper-Friendly Comparison Preview

{markdown_table(comparison_preview, [("Dataset", "Dataset"), ("Model", "Model"), ("Seed", "Seed"), ("none", "none"), ("random M4", "random M4"), ("random M8", "random M8"), ("cam_low M4", "cam_low M4"), ("cam_high M4", "cam_high M4"), ("cam_low M8", "cam_low M8"), ("cam_high M8", "cam_high M8")])}

## RawMal-TF Focused Results

{markdown_table(raw_table_rows, [("Condition", "Condition"), ("Best acc", "Best acc"), ("Final acc", "Final acc"), ("Best epoch", "Best epoch"), ("vs none", "vs none"), ("vs random", "vs random")])}

## MalImg and CIFAR100 Summaries

{markdown_table(sanity_rows, [("Dataset", "Dataset"), ("Condition", "Condition"), ("Best acc", "Best acc"), ("Final acc", "Final acc"), ("Best epoch", "Best epoch"), ("vs none", "vs none")])}

## Interpretation

{chr(10).join(interpretation_lines)}

These statements are computed only from existing run artifacts. A positive CAM-minus-random statistic is evidence only for the matched runs present here; it should not be generalized beyond the current seed/model/dataset coverage.

## Warnings

{markdown_table(warning_rows, [("Severity", "Severity"), ("Check", "Check"), ("Run/Dataset", "Run/Dataset"), ("Observed", "Observed")])}

## Next-Step Recommendations

- Use `comparison_table.csv` and the RawMal-TF plots as the primary publication tables/figures.
- Treat single-seed comparisons as preliminary unless additional seeds are added later.
- Report missing literal `runs/rawmaltf/` path as a naming issue if the paper refers to RawMal-TF while artifacts use `drive_zip`.
- Before making saliency-specific claims, check `integrity_checks.csv` for cam_low/cam_high identity failures.
- Do not claim CAM is better than random unless the matched improvement columns and plots support that claim for the target dataset/model/seed.
"""
    return md


def make_readme() -> str:
    return """# Summary Package

This folder contains publication-oriented summaries generated from existing run artifacts only.

## Generated Files

- `run_inventory.csv`: one row per run with configuration fields, artifact flags, epoch coverage, hashes, status, and notes.
- `run_summary.csv`: final and best train/evaluation metrics, best epoch, generalization gaps, and matched improvements over no-cutout and random baselines.
- `comparison_table.csv`: wide paper-friendly comparison by dataset/model/seed for none, random M4/M8, cam_low M4/M8, and cam_high M4/M8.
- `integrity_checks.csv`: missing artifacts, epoch mismatches, metric issues, duplicate hashes, folder/config mismatches, CAM teacher checkpoint checks, and cam_low/cam_high identity checks.
- `summary_stats.json`: aggregate counts, best runs, improvement statistics, CAM-vs-random statistics, cam_low-vs-cam_high statistics, and major warnings.
- `paper_summary.md`: human-readable report for publication planning.
- `plots/`: PNG figures and the CSV tables used to create each plot.

## Rerun

From the repository root:

```bash
python runs/summary/generate_summary.py
```

The script reads `runs/cifar100/`, `runs/malimg/`, `runs/rawmaltf/` when present, and `runs/drive_zip/` as the available RawMal-TF artifact folder. It writes outputs only under `runs/summary/`.
"""


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs, missing_folder_warnings = find_run_dirs()
    runs = [load_run(path) for path in run_dirs]

    checks, run_notes = build_integrity_checks(runs, missing_folder_warnings)
    inventory_rows = make_inventory(runs, run_notes)
    summary_rows = [summarize_run(run) for run in runs]
    attach_improvements(summary_rows)
    comparison_rows = make_comparison_table(summary_rows)
    stats = make_summary_stats(summary_rows, inventory_rows, checks)

    write_csv(SUMMARY_DIR / "run_inventory.csv", inventory_rows)
    write_csv(SUMMARY_DIR / "run_summary.csv", summary_rows)
    write_csv(SUMMARY_DIR / "comparison_table.csv", comparison_rows)
    write_csv(SUMMARY_DIR / "integrity_checks.csv", checks)
    write_json(SUMMARY_DIR / "summary_stats.json", stats)
    write_text(SUMMARY_DIR / "paper_summary.md", make_paper_summary(summary_rows, comparison_rows, checks, stats))
    write_text(SUMMARY_DIR / "README.md", make_readme())

    create_plots(summary_rows, runs)

    major_warnings = stats.get("major_warnings", [])
    print(f"number of runs processed: {len(runs)}")
    print(f"number of files created: {len(CREATED_FILES)}")
    print(f"number of plots created: {PLOTS_CREATED}")
    print("major warnings:")
    if major_warnings:
        for warning in major_warnings[:20]:
            print(f"- {warning}")
        if len(major_warnings) > 20:
            print(f"- ... {len(major_warnings) - 20} more warnings in runs/summary/integrity_checks.csv")
    else:
        print("- none")
    print("path to README.md: runs/summary/README.md")


if __name__ == "__main__":
    main()
