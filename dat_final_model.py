"""Train the final DaT model on all labeled training records using frozen settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

from cutout import CutoutAugmentedDataset
from dat_preprocessing import DatDataset, load_dat_records
from dat_training import fit_dat_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a final DaT model from a frozen Stage 1 configuration.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--best_config", required=True)
    parser.add_argument("--calibration", default="artifacts/dat_parkinsons/optimization/calibration.json")
    parser.add_argument("--output_dir", default="artifacts/dat_parkinsons/final_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--selected_model", default="", help="Optional dat_select_model.json; enables the selected Stage 2 condition.")
    args = parser.parse_args()

    best_config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    records = load_dat_records(args.data_dir)
    preprocessing = best_config["preprocessing"]
    selected = {"condition": "none", "M": 0, "fraction": 0.0}
    if args.selected_model:
        selection_payload = json.loads(Path(args.selected_model).read_text(encoding="utf-8"))
        selected.update(selection_payload.get("selected", {}))
    condition = str(selected["condition"])
    m_value = int(selected.get("M", 0))
    fraction = float(selected.get("fraction", 0.0) or 0.0)
    base_train_dataset = DatDataset(
        records, preprocessing, train=True,
        augment=bool(best_config.get("spatial_augmentation", False)),
        seed=args.seed, cache_dir="artifacts/dat_parkinsons/cache/preprocessed",
    )
    evaluation_dataset = DatDataset(
        records, preprocessing, train=False, augment=False,
        seed=args.seed, cache_dir="artifacts/dat_parkinsons/cache/preprocessed",
    )
    config = deepcopy(best_config)
    config.update({"stage": "final_model", "cutout_mode": condition, "cutout_m": m_value, "cutout_fraction": fraction, "num_workers": args.num_workers, "condition": condition})
    output = Path(args.output_dir)
    teacher = None
    teacher_hash = None
    if condition.startswith("cam_"):
        teacher_config = deepcopy(config)
        teacher_config.update({"cutout_mode": "none", "cutout_m": 0, "stage": "final_teacher"})
        teacher_dataset = DatDataset(records, preprocessing, train=True, augment=bool(best_config.get("spatial_augmentation", False)), seed=args.seed, cache_dir="artifacts/dat_parkinsons/cache/preprocessed")
        teacher_eval_dataset = DatDataset(records, preprocessing, train=False, augment=False, seed=args.seed, cache_dir="artifacts/dat_parkinsons/cache/preprocessed")
        teacher_result = fit_dat_model(teacher_dataset, teacher_eval_dataset, teacher_config, seed=args.seed + 10000, run_dir=output / "teacher", max_train_batches=args.max_train_batches or None)
        teacher = teacher_result["model"]
        digest = hashlib.sha256((output / "teacher" / "best_model.pt").read_bytes()).hexdigest()
        teacher_hash = digest
        config["teacher_checkpoint_sha256"] = teacher_hash
    if condition == "none":
        train_dataset = base_train_dataset
    else:
        train_dataset = CutoutAugmentedDataset(
            base_dataset=base_train_dataset,
            cutout_mode=condition,
            cutout_m=m_value,
            cutout_size=None,
            cutout_area=fraction,
            mean=(0.0,),
            std=(1.0,),
            seed=args.seed,
            saliency_candidate_percent=float(config.get("saliency_candidate_percent", 10.0)),
            teacher_model=teacher,
            cam_layer=str(config.get("cam_layer", "auto")),
            cam_cache_dir="artifacts/dat_parkinsons/cam_cache/final" if condition.startswith("cam_") else None,
            cam_cache_settings={
                "dataset": "dat_parkinsons", "student_model": "resnet18_3d", "teacher_checkpoint_sha256": teacher_hash,
                "cam_layer": str(config.get("cam_layer", "auto")), "spatial_dims": 3,
                "preprocessing": preprocessing, "target_spacing": preprocessing["target_spacing"],
                "target_shape": preprocessing["target_shape"],
                "min_foreground_fraction": float(config.get("min_foreground_fraction", 0.75)),
            } if condition.startswith("cam_") else None,
            min_foreground_fraction=float(config.get("min_foreground_fraction", 0.75)),
        )
    result = fit_dat_model(
        train_dataset, evaluation_dataset, config,
        seed=args.seed, run_dir=output,
        max_train_batches=args.max_train_batches or None,
    )
    model_config = {
        "model": "resnet18_3d",
        "num_classes": int(config.get("num_classes", 2)),
        "n_input_channels": int(config.get("n_input_channels", 1)),
        "base_channels": int(config.get("base_channels", 32)),
        "dropout": float(config.get("dropout", 0.0)),
        "training_condition": condition,
        "training_cutout_m": m_value,
        "training_cutout_fraction": fraction,
    }
    (output / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(args.calibration, output / "calibration.json")
    print(json.dumps({"output_dir": str(output), "model_config": model_config}, indent=2))


if __name__ == "__main__":
    main()
