"""One-command DaT Stage 2: grid, integrity, calibration, final model, ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_dat_submission import build
from dat_masking_experiments import _parser as masking_parser, run as masking_run
from dat_select_model import select_candidates
from dat_stage2_summary import generate_summary
from dat_final_model import train_final_dat_model


def _parser() -> argparse.ArgumentParser:
    parser = masking_parser()
    parser.description = "Run the complete DaT Stage 2 masked pipeline."
    parser.add_argument("--selected_model", default="artifacts/dat_parkinsons/selected_model.json")
    parser.add_argument("--summary_dir", default="runs/dat_parkinsons/summary")
    parser.add_argument("--final_model_dir", default="artifacts/dat_parkinsons/final_stage2_masked")
    parser.add_argument("--stage1_model_dir", default="artifacts/dat_parkinsons/final_stage1_unmasked")
    parser.add_argument("--submission_zip", default="submission/dat_stage2_masked.zip")
    parser.add_argument("--calibration_method", choices=["raw", "temperature"], default="temperature")
    return parser


def run(args: argparse.Namespace) -> dict:
    grid_result = masking_run(args)
    best_config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    n_folds = int(best_config.get("cv_folds", grid_result["folds"]))
    selection = select_candidates(
        args.output_dir, expected_folds=n_folds, output_path=args.selected_model,
        conditions=args.conditions, m_values=args.m_values, fractions=args.fractions,
        calibration_method=args.calibration_method,
        frozen_config=best_config,
        summary_dir=args.summary_dir,
    )
    generate_summary(args.output_dir, args.summary_dir, expected_folds=n_folds, selection_path=args.selected_model,
                     conditions=args.conditions, m_values=args.m_values, fractions=args.fractions,
                     frozen_config=best_config)
    selected = selection["best_masked"]
    final = train_final_dat_model(
        args.data_dir, args.best_config, args.final_model_dir,
        selected=selected, stage1_model_dir=args.stage1_model_dir,
        calibration_payload=selected["calibration"], seed=args.seed,
        num_workers=args.num_workers, max_train_batches=0,
    )
    archive = build(args.final_model_dir, args.submission_zip)
    payload = {
        "stage2_research_output": str(args.output_dir), "summary_dir": str(args.summary_dir),
        "selection": str(args.selected_model), "selected_condition": selected["condition"],
        "selected_M": selected["M"], "selected_fraction": selected["fraction"],
        "final_model_dir": str(args.final_model_dir), "submission_zip": str(archive),
        "best_overall": selection["best_overall"], "best_masked": selection["best_masked"],
    }
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
