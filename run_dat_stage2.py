"""The single normal command for the complete DaT Stage 1/Stage 2 workflow."""

from __future__ import annotations

import argparse
import json

from dat_preprocessing import DEFAULT_TARGET_SHAPE
from dat_pipeline import check_data, run_stage1, run_stage2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete DaT Stage 1/Stage 2 pipeline.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--best_config", default="artifacts/dat_parkinsons/optimization/best_config.json")
    parser.add_argument("--research_output_dir", default="runs/dat_parkinsons/optimization")
    parser.add_argument("--output_dir", default="runs/dat_parkinsons/resnet18_3d")
    parser.add_argument("--selected_model", default="artifacts/dat_parkinsons/selected_model.json")
    parser.add_argument("--summary_dir", default="runs/dat_parkinsons/summary")
    parser.add_argument("--stage1_model_dir", default="artifacts/dat_parkinsons/final_stage1_unmasked")
    parser.add_argument("--final_model_dir", default="artifacts/dat_parkinsons/final_stage2_masked")
    parser.add_argument("--stage1_submission_zip", default="submission/dat_stage1_unmasked.zip")
    parser.add_argument("--stage2_submission_zip", default="submission/dat_stage2_masked.zip")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--fold_scheme", choices=["stratified", "protocol_group"], default="stratified")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=None)
    parser.add_argument("--target_shape", nargs=3, type=int, default=list(DEFAULT_TARGET_SHAPE))
    parser.add_argument("--intensity_lower_percentile", type=float, default=1.0)
    parser.add_argument("--intensity_upper_percentile", type=float, default=99.0)
    parser.add_argument("--foreground_threshold", type=float, default=0.0)
    parser.add_argument("--crop_margin_mm", type=float, default=8.0)
    parser.add_argument("--calibration", choices=["raw", "temperature"], default="temperature")
    parser.add_argument("--calibration_method", choices=["raw", "temperature"], default="temperature")
    parser.add_argument("--search_space_json", default="")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augmentation", choices=["none", "mild"], default="none")
    parser.add_argument("--fold_assignments", default="")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--conditions", nargs="+", choices=["none", "random", "cam_low", "cam_high"], default=["none", "random", "cam_low", "cam_high"])
    parser.add_argument("--m_values", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--cam_layer", default="auto")
    parser.add_argument("--saliency_candidate_percent", type=float, default=10.0)
    parser.add_argument("--min_foreground_fraction", type=float, default=0.75)
    parser.add_argument("--cam_cache_dir", default="artifacts/dat_parkinsons/cam_cache")
    parser.add_argument("--preprocessed_cache_dir", default="artifacts/dat_parkinsons/cache/preprocessed")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-limit", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.check_data:
        print(json.dumps(check_data(args.data_dir, target_spacing=args.target_spacing, target_shape=args.target_shape, limit=args.check_limit), indent=2))
        return
    if args.stage1_only:
        print(json.dumps(run_stage1(args), indent=2, default=str))
        return
    print(json.dumps(run_stage2(args), indent=2, default=str))


if __name__ == "__main__":
    main()
