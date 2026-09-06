"""Validation-free full-data DaT model training.

Cross-validation checkpoint selection lives in ``dat_tune.py`` and
``dat_masking_experiments.py``.  This module deliberately has a separate
fixed-budget path for competition models so the labeled population is never
used as its own validation set.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from cutout import CutoutAugmentedDataset
from dat_model import build_resnet18_3d
from dat_preprocessing import DatDataset, load_dat_records
from dat_provenance import REPO_ROOT, current_git_commit, fingerprint, portable_path, sha256_file
from dat_training import fit_dat_model_fixed_epochs


def _portable_or_key(path: str | Path) -> str:
    try:
        return portable_path(path)
    except ValueError:
        return f"external/{Path(path).name}"


def _load_stage1_teacher(stage1_model_dir: str | Path, config: dict[str, Any]):
    model_dir = Path(stage1_model_dir)
    checkpoint = model_dir / "final_model.pt"
    if not checkpoint.is_file():
        checkpoint = model_dir / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage 1 final checkpoint not found under {model_dir}.")
    model = build_resnet18_3d(
        num_classes=int(config.get("num_classes", 2)),
        n_input_channels=int(config.get("n_input_channels", 1)),
        dropout=float(config.get("dropout", 0.0)),
        base_channels=int(config.get("base_channels", 32)),
    )
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Stage 1 checkpoint does not contain a model state dictionary.")
    model.load_state_dict({str(k).removeprefix("module."): v for k, v in payload.items()}, strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), checkpoint, sha256_file(checkpoint)


def train_final_dat_model(
    data_dir: str | Path,
    best_config_path: str | Path,
    output_dir: str | Path,
    calibration_path: str | Path | None = None,
    *,
    selected: dict[str, Any] | None = None,
    stage1_model_dir: str | Path | None = None,
    calibration_payload: dict[str, Any] | None = None,
    seed: int = 42,
    num_workers: int = 0,
    max_train_batches: int = 0,
) -> dict[str, Any]:
    """Train exactly the frozen number of epochs and write an offline bundle."""
    best_config = json.loads(Path(best_config_path).read_text(encoding="utf-8"))
    records = load_dat_records(data_dir)
    preprocessing = best_config["preprocessing"]
    selected = dict(selected or {})
    condition = str(selected.get("condition", selected.get("cutout_mode", "none")))
    m_value = int(selected.get("M", selected.get("cutout_m", 0)) or 0)
    fraction = float(selected.get("fraction", selected.get("cutout_fraction", 0.0)) or 0.0)
    if condition not in {"none", "random", "cam_low", "cam_high"}:
        raise ValueError(f"Unsupported final DaT condition: {condition}")
    if condition != "none" and (m_value <= 0 or fraction <= 0):
        raise ValueError("Masked final models require positive M and fraction.")
    epoch_budget = int(best_config.get("final_training_epochs", best_config.get("epochs", 100)))
    if epoch_budget <= 0:
        raise ValueError("Stage 1 did not provide a positive final epoch budget.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = deepcopy(best_config)
    config.update({
        "stage": 1 if condition == "none" else 2,
        "condition": condition, "cutout_mode": condition,
        "cutout_m": m_value if condition != "none" else 0,
        "cutout_fraction": fraction, "epochs": epoch_budget,
        "final_training_epochs": epoch_budget, "num_workers": int(num_workers),
        "max_train_batches": int(max_train_batches or 0), "max_val_batches": 0,
        "debug": bool(max_train_batches),
    })
    cache_dir = REPO_ROOT / "artifacts" / "dat_parkinsons" / "cache" / "preprocessed"
    base_train_dataset = DatDataset(
        records, preprocessing, train=True,
        augment=bool(best_config.get("spatial_augmentation", False)),
        seed=seed, cache_dir=cache_dir,
    )
    teacher = None
    teacher_checkpoint = None
    teacher_hash = None
    if condition.startswith("cam_"):
        if stage1_model_dir is None:
            raise ValueError("Final CAM masking requires the exact Stage 1 final model directory.")
        teacher, teacher_checkpoint, teacher_hash = _load_stage1_teacher(stage1_model_dir, best_config)
        config.update({
            "teacher_checkpoint": _portable_or_key(teacher_checkpoint),
            "teacher_checkpoint_sha256": teacher_hash,
            "teacher_lineage": "stage1_final_unmasked_checkpoint",
        })
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train_dataset, cutout_mode=condition,
            cutout_m=m_value, cutout_size=None, cutout_area=fraction,
            mean=(0.0,), std=(1.0,), seed=seed,
            saliency_candidate_percent=float(config.get("saliency_candidate_percent", 10.0)),
            teacher_model=teacher, cam_layer=str(config.get("cam_layer", "auto")),
            cam_cache_dir=str(REPO_ROOT / "artifacts" / "dat_parkinsons" / "cam_cache" / "final_stage2"),
            cam_cache_settings={
                "dataset": "dat_parkinsons", "student_model": "resnet18_3d",
                "teacher_checkpoint_sha256": teacher_hash,
                "cam_layer": str(config.get("cam_layer", "auto")), "spatial_dims": 3,
                "preprocessing": preprocessing,
                "min_foreground_fraction": float(config.get("min_foreground_fraction", 0.75)),
            },
            min_foreground_fraction=float(config.get("min_foreground_fraction", 0.75)),
        )
    elif condition == "none":
        train_dataset = base_train_dataset
    else:
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train_dataset, cutout_mode=condition,
            cutout_m=m_value, cutout_size=None, cutout_area=fraction,
            mean=(0.0,), std=(1.0,), seed=seed,
            min_foreground_fraction=float(config.get("min_foreground_fraction", 0.75)),
        )
    result = fit_dat_model_fixed_epochs(
        train_dataset, config, seed=seed, run_dir=output,
        epochs=epoch_budget, max_train_batches=max_train_batches or None,
    )
    model_config = {
        "model": "resnet18_3d", "num_classes": int(config.get("num_classes", 2)),
        "n_input_channels": int(config.get("n_input_channels", 1)),
        "base_channels": int(config.get("base_channels", 32)),
        "dropout": float(config.get("dropout", 0.0)),
        "training_condition": condition, "training_cutout_m": m_value,
        "training_cutout_fraction": fraction,
    }
    (output / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2, sort_keys=True), encoding="utf-8")
    if calibration_payload is None:
        if calibration_path is None:
            raise ValueError("A calibration file or calibration payload is required.")
        calibration_payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    (output / "calibration.json").write_text(json.dumps(calibration_payload, indent=2, sort_keys=True), encoding="utf-8")
    provenance = {
        "pipeline": "stage1_unmasked" if condition == "none" else "stage2_masked",
        "stage": int(config["stage"]), "selected_condition": condition,
        "M": m_value, "fraction": fraction,
        "selected_stage1_config_fingerprint": best_config.get("config_fingerprint", fingerprint(best_config)),
        "preprocessing_fingerprint": best_config.get("preprocessing_fingerprint", fingerprint(preprocessing)),
        "final_epoch_budget": epoch_budget, "seed": int(seed),
        "checkpoint_selection": "final_scheduled_epoch",
        "research_valid": not bool(max_train_batches), "git_commit": current_git_commit(),
        "final_checkpoint_sha256": sha256_file(output / "final_model.pt"),
        "calibration_provenance": selected.get("calibration_provenance", "stage1_oof_logits_only" if condition == "none" else "selected_stage2_candidate_oof_logits_only"),
        "calibration_candidate": {"condition": condition, "M": m_value, "fraction": fraction},
    }
    if teacher_checkpoint is not None:
        provenance.update({
            "stage1_teacher_checkpoint": _portable_or_key(teacher_checkpoint),
            "stage1_teacher_checkpoint_sha256": teacher_hash,
            "teacher_checkpoint_path": _portable_or_key(teacher_checkpoint),
            "teacher_checkpoint_sha256": teacher_hash,
        })
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": output, "model_config": model_config, "provenance": provenance, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a final DaT model using a frozen CV epoch budget.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--best_config", required=True)
    parser.add_argument("--calibration", default="artifacts/dat_parkinsons/optimization/calibration.json")
    parser.add_argument("--output_dir", default="artifacts/dat_parkinsons/final_stage1_unmasked")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--selected_model", default="", help="Optional Stage 2 selection JSON.")
    parser.add_argument("--stage1_model_dir", default="", help="Exact Stage 1 model used for final CAM masking.")
    args = parser.parse_args()
    selected = None
    calibration_payload = None
    if args.selected_model:
        payload = json.loads(Path(args.selected_model).read_text(encoding="utf-8"))
        selected = payload.get("best_masked") or payload.get("selected") or {}
        calibration_payload = selected.get("calibration")
        if selected.get("condition") in {"random", "cam_low", "cam_high"} and calibration_payload is None:
            raise ValueError("A Stage 2 selected masked candidate must provide its own OOF-fitted calibration.")
    result = train_final_dat_model(
        args.data_dir, args.best_config, args.output_dir, args.calibration,
        selected=selected, stage1_model_dir=args.stage1_model_dir or None,
        calibration_payload=calibration_payload, seed=args.seed,
        num_workers=args.num_workers, max_train_batches=args.max_train_batches,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "model_config": result["model_config"]}, indent=2))


if __name__ == "__main__":
    main()
