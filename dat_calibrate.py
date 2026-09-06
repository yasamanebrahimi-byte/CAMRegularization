"""Fit the fixed DaT calibration transform from saved labeled OOF logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dat_calibration import calibrated_probabilities, fit_calibration, save_calibration
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
    calibrated = compute_binary_metrics(targets, probabilities=calibrated_probabilities(logits, calibration))
    calibration.update({"raw_oof_log_loss": raw["log_loss"], "calibrated_oof_log_loss": calibrated["log_loss"], "calibration_fit_data": "labeled_training_oof_predictions_only"})
    save_calibration(args.output, calibration)
    print(json.dumps({"raw_oof_log_loss": raw["log_loss"], "calibrated_oof_log_loss": calibrated["log_loss"]}, indent=2))


if __name__ == "__main__":
    main()

