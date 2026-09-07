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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint, sha256_file(checkpoint)


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
    selected_provided = selected is not None
    selected = dict(selected or {})
    if selected_provided and "condition" not in selected:
        raise ValueError("A Stage 2 selected candidate must explicitly contain condition.")
    condition = str(selected.get("condition", "none"))
    if selected_provided and condition == "none":
        raise ValueError("Submission #2 must use a masked Stage 2 candidate, not condition='none'.")
    m_value = int(selected["M"] if selected_provided else selected.get("M", 0))
    fraction = float(selected["fraction"] if selected_provided else selected.get("fraction", 0.0))
    if condition not in {"none", "random", "cam_low", "cam_high"}:
        raise ValueError(f"Unsupported final DaT condition: {condition}")
    if condition != "none" and (m_value <= 0 or fraction <= 0):
        raise ValueError("Masked final models require positive M and fraction.")
    stage1_final_epoch_budget = int(best_config.get("final_training_epochs", best_config.get("epochs", 100)))
    if calibration_payload is None and selected_provided:
        calibration_payload = selected.get("calibration")
    if selected_provided and calibration_payload is None:
        raise ValueError("Selected Stage 2 candidate must provide its candidate-specific calibration payload.")
    if selected_provided:
        required_recipe = (
            "cam_layer", "saliency_candidate_percent", "min_foreground_fraction",
            "final_stage2_training_epochs", "selected_candidate_fold_best_epochs",
            "calibration_provenance",
        )
        missing = [field for field in required_recipe if field not in selected]
        if missing:
            raise ValueError("Selected Stage 2 candidate is missing required recipe fields: " + ", ".join(missing))
        epoch_budget = int(selected["final_stage2_training_epochs"])
        cam_layer = str(selected["cam_layer"])
        saliency_candidate_percent = float(selected["saliency_candidate_percent"])
        min_foreground_fraction = float(selected["min_foreground_fraction"])
    else:
        epoch_budget = stage1_final_epoch_budget
        cam_layer = str(best_config.get("cam_layer", "auto"))
        saliency_candidate_percent = float(best_config.get("saliency_candidate_percent", 10.0))
        min_foreground_fraction = float(best_config.get("min_foreground_fraction", 0.75))
    if epoch_budget <= 0:
        raise ValueError("The final DaT model requires a positive epoch budget.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = deepcopy(best_config)
    config.update({
        "stage": 1 if condition == "none" else 2,
        "condition": condition, "cutout_mode": condition,
        "cutout_m": m_value if condition != "none" else 0,
        "cutout_fraction": fraction, "epochs": epoch_budget,
        "final_training_epochs": epoch_budget,
        "stage1_final_training_epochs": stage1_final_epoch_budget,
        "num_workers": 0 if condition.startswith("cam_") else int(num_workers),
        "max_train_batches": int(max_train_batches or 0), "max_val_batches": 0,
        "debug": bool(max_train_batches),
        "cam_layer": cam_layer,
        "saliency_candidate_percent": saliency_candidate_percent,
        "min_foreground_fraction": min_foreground_fraction,
    })
    if condition.startswith("cam_") and int(num_workers) > 0:
        print("[Stage 2] Final CAM training uses num_workers=0 because saliency and window caches are not proven complete.")
    if selected_provided:
        config.update({
            "student_max_cv_epochs": int(selected["student_max_cv_epochs"]),
            "student_seed_policy": selected["student_seed_policy"],
            "student_model": selected["student_model"],
            "selected_candidate_fold_best_epochs": [int(value) for value in selected["selected_candidate_fold_best_epochs"]],
            "final_stage2_training_epoch_rule": selected.get(
                "final_stage2_training_epoch_rule",
                "median_selected_stage2_fold_best_epoch_round_half_up",
            ),
            "selected_stage1_config_fingerprint": selected["selected_stage1_config_fingerprint"],
            "fold_assignment_fingerprint": selected["fold_assignment_fingerprint"],
            "selected_candidate_fold_metrics": selected.get("candidate_fold_metrics", []),
            "calibration_provenance": selected["calibration_provenance"],
            "selected_candidate_calibration": calibration_payload,
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
            saliency_candidate_percent=saliency_candidate_percent,
            teacher_model=teacher, cam_layer=cam_layer,
            cam_cache_dir=str(REPO_ROOT / "artifacts" / "dat_parkinsons" / "cam_cache" / "final_stage2"),
            cam_cache_settings={
                "dataset": "dat_parkinsons", "student_model": "resnet18_3d",
                "teacher_checkpoint_sha256": teacher_hash,
                "cam_layer": cam_layer, "spatial_dims": 3,
                "preprocessing": preprocessing,
                "min_foreground_fraction": min_foreground_fraction,
            },
            min_foreground_fraction=min_foreground_fraction,
        )
    elif condition == "none":
        train_dataset = base_train_dataset
    else:
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train_dataset, cutout_mode=condition,
            cutout_m=m_value, cutout_size=None, cutout_area=fraction,
            mean=(0.0,), std=(1.0,), seed=seed,
            min_foreground_fraction=min_foreground_fraction,
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
        "cam_layer": cam_layer,
        "saliency_candidate_percent": saliency_candidate_percent,
        "min_foreground_fraction": min_foreground_fraction,
        "final_training_epochs": epoch_budget,
    }
    if selected_provided:
        model_config.update({
            "student_max_cv_epochs": int(selected["student_max_cv_epochs"]),
            "final_stage2_training_epochs": epoch_budget,
            "selected_candidate_fold_best_epochs": [int(value) for value in selected["selected_candidate_fold_best_epochs"]],
            "calibration_provenance": selected["calibration_provenance"],
            "calibration_method": str((calibration_payload or {}).get("method", "raw")),
            "calibration_temperature": float((calibration_payload or {}).get("temperature", 1.0)),
            "selected_candidate_calibration": calibration_payload,
        })
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
        "condition": condition,
        "cam_layer": cam_layer,
        "saliency_candidate_percent": saliency_candidate_percent,
        "min_foreground_fraction": min_foreground_fraction,
        "selected_stage1_config_fingerprint": selected.get(
            "selected_stage1_config_fingerprint",
            best_config.get("config_fingerprint", fingerprint(best_config)),
        ),
        "stage1_selection_objective": best_config.get("selection_objective", "cross_fitted_calibrated_oof_log_loss"),
        "stage1_raw_oof_metrics": best_config.get("raw_oof_metrics"),
        "stage1_cross_fitted_calibrated_oof_metrics": best_config.get("cross_fitted_calibrated_oof_metrics"),
        "stage1_calibration_method": best_config.get("calibration_method"),
        "preprocessing_fingerprint": best_config.get("preprocessing_fingerprint", fingerprint(preprocessing)),
        "final_epoch_budget": epoch_budget,
        "final_stage2_training_epochs": epoch_budget if selected_provided else None,
        "stage1_final_training_epochs": stage1_final_epoch_budget,
        "selected_stage2_fold_best_epochs": selected.get("selected_candidate_fold_best_epochs") if selected_provided else None,
        "final_stage2_training_epoch_rule": selected.get("final_stage2_training_epoch_rule") if selected_provided else None,
        "seed": int(seed),
        "checkpoint_selection": "final_scheduled_epoch",
        "research_valid": not bool(max_train_batches), "git_commit": current_git_commit(),
        "final_checkpoint_sha256": sha256_file(output / "final_model.pt"),
        "calibration_provenance": selected.get("calibration_provenance", "stage1_oof_logits_only" if condition == "none" else "selected_stage2_candidate_oof_logits_only"),
        "calibration_candidate": {"condition": condition, "M": m_value, "fraction": fraction},
    }
    if selected_provided:
        provenance.update({
            "stage2_selection_objective": "cross_fitted_calibrated_oof_log_loss",
            "stage2_selection_score": selected.get("selection_score"),
            "stage2_raw_oof_metrics": {
                "log_loss": selected.get("raw_oof_log_loss"),
                "auroc": selected.get("raw_oof_auroc"),
                "brier_score": selected.get("raw_oof_brier_score"),
                "accuracy": selected.get("raw_oof_accuracy"),
            },
            "stage2_cross_fitted_calibrated_oof_metrics": {
                "log_loss": selected.get("cross_fitted_calibrated_oof_log_loss"),
                "auroc": selected.get("cross_fitted_calibrated_oof_auroc"),
                "brier_score": selected.get("cross_fitted_calibrated_oof_brier_score"),
                "accuracy": selected.get("cross_fitted_calibrated_oof_accuracy"),
                "ece": selected.get("cross_fitted_calibrated_oof_ece"),
            },
        })
    if selected_provided:
        provenance.update({
            "selected_stage2_recipe": {
                "condition": condition, "M": m_value, "fraction": fraction,
                "cam_layer": cam_layer,
                "saliency_candidate_percent": saliency_candidate_percent,
                "min_foreground_fraction": min_foreground_fraction,
            },
            "selected_candidate_fold_best_epochs": [int(value) for value in selected["selected_candidate_fold_best_epochs"]],
            "selected_candidate_fold_metrics": selected.get("candidate_fold_metrics", []),
            "student_max_cv_epochs": int(selected["student_max_cv_epochs"]),
            "student_seed_policy": selected["student_seed_policy"],
            "student_model": selected["student_model"],
            "fold_assignment_fingerprint": selected["fold_assignment_fingerprint"],
            "final_stage2_training_epoch_rule": selected.get(
                "final_stage2_training_epoch_rule",
                "median_selected_stage2_fold_best_epoch_round_half_up",
            ),
        })
    if teacher_checkpoint is not None:
        provenance.update({
            "teacher_lineage": "stage1_final_unmasked_checkpoint",
            "stage1_teacher_checkpoint": _portable_or_key(teacher_checkpoint),
            "stage1_teacher_checkpoint_sha256": teacher_hash,
            "teacher_checkpoint_path": _portable_or_key(teacher_checkpoint),
            "teacher_checkpoint_sha256": teacher_hash,
        })
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_dir": output, "model_config": model_config, "config": config, "provenance": provenance, "result": result}


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
