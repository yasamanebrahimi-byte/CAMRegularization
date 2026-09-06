"""Select a DaT Stage 2 candidate by cross-validated calibrated log loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dat_calibration import calibrated_probabilities, load_calibration
from dat_metrics import compute_binary_metrics


def _root_from_script() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose the best DaT masking candidate without using leaderboard data.")
    parser.add_argument("--summary_table", default="runs/dat_parkinsons/summary/tables/per_run_metrics.csv")
    parser.add_argument("--calibration", default="artifacts/dat_parkinsons/optimization/calibration.json")
    parser.add_argument("--output", default="artifacts/dat_parkinsons/selected_model.json")
    args = parser.parse_args()
    table = pd.read_csv(args.summary_table)
    calibration = load_calibration(args.calibration) if Path(args.calibration).is_file() else {"method": "raw", "temperature": 1.0}
    root = _root_from_script()
    candidates = []
    for keys, group in table.groupby(["condition", "M", "fraction"], dropna=False):
        all_logits = []
        all_targets = []
        for run_dir in group["run_dir"]:
            config_path = root / run_dir / "config.json"
            if not config_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            oof_path = config.get("oof_artifact")
            if not oof_path or not Path(oof_path).is_file():
                continue
            payload = np.load(oof_path)
            all_logits.append(payload["logits"])
            all_targets.append(payload["targets"])
        if all_logits:
            logits = np.concatenate(all_logits, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            raw_loss = compute_binary_metrics(targets, logits=logits)["log_loss"]
            calibrated_loss = compute_binary_metrics(targets, probabilities=calibrated_probabilities(logits, calibration))["log_loss"]
        else:
            fallback = group["minimum_validation_log_loss"].astype(float)
            if fallback.empty:
                continue
            raw_loss = float(fallback.mean())
            calibrated_loss = raw_loss
        if not np.isfinite(raw_loss):
            continue
        candidates.append({
            "condition": keys[0], "M": int(keys[1]), "fraction": float(keys[2]),
            "raw_cv_log_loss": float(raw_loss),
            "calibrated_cv_log_loss": float(calibrated_loss),
            "calibration_source": str(args.calibration),
        })
    if not candidates:
        raise ValueError("No selectable DaT candidates were found.")
    selected = min(candidates, key=lambda row: row["calibrated_cv_log_loss"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"selection_basis": "cross_validated_calibrated_log_loss", "selected": selected, "candidates": candidates}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
