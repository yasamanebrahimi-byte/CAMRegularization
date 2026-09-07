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
from dat_provenance import fingerprint
from run_dat_stage1 import _parser as stage1_parser, run as stage1_run


def _load_valid_stage1_config(path: Path) -> dict:
    """Load the persisted Stage 1 handoff and reject invalid research state."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Stage 1 configuration exists but is invalid:\n{path}\n{exc}"
        ) from exc

    problems = []
    if not isinstance(config, dict):
        problems.append("the JSON root must be an object")
    else:
        if config.get("dataset") != "dat_parkinsons":
            problems.append("dataset must be 'dat_parkinsons'")
        if config.get("model") != "resnet18_3d":
            problems.append("model must be 'resnet18_3d'")
        if type(config.get("stage")) is not int or config.get("stage") != 1:
            problems.append("stage must be 1")
        if config.get("research_valid") is not True:
            problems.append("research_valid must be true")
        if not isinstance(config.get("preprocessing"), dict):
            problems.append("preprocessing must be a JSON object")

        preprocessing_fingerprint = config.get("preprocessing_fingerprint")
        if preprocessing_fingerprint is not None and isinstance(config.get("preprocessing"), dict):
            if preprocessing_fingerprint != fingerprint(config["preprocessing"]):
                problems.append("preprocessing_fingerprint does not match preprocessing")

        config_fingerprint = config.get("config_fingerprint")
        if config_fingerprint is not None:
            fingerprinted_config = dict(config)
            fingerprinted_config.pop("config_fingerprint", None)
            # Stage 1 adds this artifact reference after calculating the config fingerprint.
            fingerprinted_config.pop("oof_artifact", None)
            if config_fingerprint != fingerprint(fingerprinted_config):
                problems.append("config_fingerprint does not match the selected configuration")

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Stage 1 configuration exists but is invalid:\n{path}\n{details}")
    return config


def _run_stage1_for_config(args: argparse.Namespace, best_config_path: Path) -> None:
    """Run the existing complete Stage 1 workflow with its own defaults."""
    stage1_args = stage1_parser().parse_args([
        "--data_dir", str(args.data_dir),
        "--output_dir", str(best_config_path.parent),
        "--final_model_dir", str(args.stage1_model_dir),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
    ])
    stage1_run(stage1_args)

    generated_path = best_config_path.parent / "best_config.json"
    if not best_config_path.is_file() and generated_path.is_file():
        # Stage 1's canonical filename is best_config.json. Preserve that
        # format while honoring an explicitly requested custom file path.
        generated_path.replace(best_config_path)


def _ensure_stage1(args: argparse.Namespace) -> Path:
    best_config_path = Path(args.best_config)

    if best_config_path.exists():
        if not best_config_path.is_file():
            raise ValueError(
                f"Stage 1 configuration exists but is invalid:\n{best_config_path}\n"
                "the path is not a file"
            )
        _load_valid_stage1_config(best_config_path)
        print(f"[Stage 2] Found Stage 1 configuration:\n{best_config_path}")
        print("[Stage 2] Skipping Stage 1.")
        return best_config_path

    print(f"[Stage 2] No Stage 1 configuration found at {best_config_path}.")
    print("[Stage 2] Running Stage 1 first...")
    _run_stage1_for_config(args, best_config_path)
    if not best_config_path.is_file():
        raise RuntimeError(
            "Stage 1 completed without creating the expected best_config.json "
            f"at {best_config_path}"
        )
    _load_valid_stage1_config(best_config_path)
    print("[Stage 2] Stage 1 complete.")
    print(f"[Stage 2] Using:\n{best_config_path}")
    return best_config_path


def _require_stage1_checkpoint(args) -> None:
    if not any(str(condition).startswith("cam_") for condition in args.conditions):
        return
    model_dir = Path(args.stage1_model_dir)
    if not any((model_dir / name).is_file() for name in ("final_model.pt", "best_model.pt")):
        raise FileNotFoundError(
            "[Stage 2] The valid Stage 1 configuration was found, but the final Stage 1 "
            f"checkpoint is missing under {model_dir}. Expected final_model.pt or best_model.pt. "
            "Run run_dat_stage1.py to create the final model before running CAM Stage 2 conditions."
        )


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
    _ensure_stage1(args)
    _require_stage1_checkpoint(args)
    print("[Stage 2] Starting masking experiments.")
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
    train_final_dat_model(
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
        "final_stage2_training_epochs": selected["final_stage2_training_epochs"],
        "final_model_dir": str(args.final_model_dir), "submission_zip": str(archive),
        "best_overall": selection["best_overall"], "best_masked": selection["best_masked"],
    }
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
