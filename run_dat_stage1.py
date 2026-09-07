"""One-command DaT Stage 1: tune, calibrate, train, and package Submission #1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_dat_submission import build
from dat_final_model import train_final_dat_model
from dat_tune import _parser as tune_parser, run as tune_run


def _parser() -> argparse.ArgumentParser:
    parser = tune_parser()
    parser.description = "Run the complete DaT Stage 1 unmasked pipeline."
    parser.add_argument("--research_output_dir", default="runs/dat_parkinsons/optimization")
    parser.add_argument("--final_model_dir", default="artifacts/dat_parkinsons/final_stage1_unmasked")
    parser.add_argument("--submission_zip", default="submission/dat_stage1_unmasked.zip")
    return parser


def run(args: argparse.Namespace) -> dict:
    result = tune_run(args)
    if not result["best_config"].get("research_valid", False):
        raise ValueError("Stage 1 smoke/debug or truncated runs cannot produce research or competition models. Remove batch limits and debug flags.")
    best_config_path = Path(args.output_dir) / "best_config.json"
    calibration_path = Path(args.output_dir) / "calibration.json"
    train_final_dat_model(
        args.data_dir, best_config_path, args.final_model_dir, calibration_path,
        seed=args.seed, num_workers=args.num_workers,
        max_train_batches=getattr(args, "max_train_batches", 0) or 0,
    )
    archive = build(args.final_model_dir, args.submission_zip)
    payload = {
        "stage1_research_output": str(Path(args.research_output_dir)),
        "best_config": str(best_config_path),
        "final_model_dir": str(args.final_model_dir),
        "submission_zip": str(archive),
        "final_training_epochs": result["final_training_epochs"],
        "research_valid": bool(result["best_config"].get("research_valid", False)),
    }
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
