"""Fit the fixed DaT calibration transform from saved labeled OOF logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dat_calibration import calibrated_probabilities, cross_fitted_calibration, fit_calibration, save_calibration
from dat_metrics import compute_binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit raw or temperature calibration from training OOF predictions.")
    parser.add_argument("--oof_predictions", required=True)
    parser.add_argument("--method", choices=["raw", "temperature"], default="temperature")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = np.load(args.oof_predictions)
    logits = payload["logits"]
    targets = payload["targets"]
    calibration = fit_calibration(logits, targets, method=args.method)
    raw = compute_binary_metrics(targets, logits=logits)
    final_all_oof = compute_binary_metrics(targets, probabilities=calibrated_probabilities(logits, calibration))
    output = {
        "raw_oof_metrics": raw,
        "final_all_oof_calibrated_metrics": final_all_oof,
        "calibration_fit_data": "all_labeled_oof_predictions_for_deployment",
    }
    if "fold_ids" in payload:
        cross_fitted, fold_calibrations = cross_fitted_calibration(
            logits, targets,
            [np.flatnonzero(payload["fold_ids"] == fold).tolist() for fold in sorted(set(payload["fold_ids"].tolist()))],
            method=args.method,
        )
        cross_fitted_metrics = compute_binary_metrics(targets, probabilities=cross_fitted)
        output.update({
            "cross_fitted_calibrated_oof_metrics": cross_fitted_metrics,
            "cross_fitted_fold_calibrations": fold_calibrations,
            "selection_objective": "cross_fitted_calibrated_oof_log_loss",
            "calibration_selection_data": "other_folds_only_for_each_validation_fold",
        })
    calibration.update(output)
    save_calibration(args.output, calibration)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
