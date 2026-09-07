"""DrivenData DaT inference entry point.

The packaged ZIP places this file at its root.  It performs inference only;
all model, preprocessing, and calibration parameters are loaded from the
bundle and no test-population statistics are estimated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dat_calibration_runtime import apply_fixed_calibration
from dat_model import load_model_from_bundle
from dat_preprocessing import preprocess_nifti


DATA_ROOT = Path("/code_execution/data")


def run_inference(
    data_root: Path = DATA_ROOT,
    bundle_root: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    bundle_root = Path(bundle_root or Path(__file__).resolve().parent)
    output_path = Path(output_path or bundle_root / "submission.csv")
    submission_format_path = Path(data_root) / "submission_format.csv"
    nifti_dir = Path(data_root) / "niftis"
    frame = pd.read_csv(submission_format_path)
    if list(frame.columns) != ["uid", "is_pathologic"]:
        raise ValueError("submission_format.csv must have exactly uid,is_pathologic columns.")
    if frame["uid"].isna().any() or frame["uid"].astype(str).duplicated().any():
        raise ValueError("submission_format.csv contains missing or duplicate UIDs.")
    template_uids = frame["uid"].astype(str).tolist()

    model_config = json.loads((bundle_root / "model_config.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((bundle_root / "preprocessing.json").read_text(encoding="utf-8"))
    calibration = json.loads((bundle_root / "calibration.json").read_text(encoding="utf-8"))
    weights_path = bundle_root / "model" / "weights.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = load_model_from_bundle(weights_path, model_config, device=device)
    print("Running inference")
    probabilities = []
    for uid in template_uids:
        nifti_path = nifti_dir / f"{uid}.nii.gz"
        if not nifti_path.is_file():
            raise FileNotFoundError("A required NIfTI scan is missing from the runtime data directory.")
        tensor = preprocess_nifti(nifti_path, preprocessing)
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(device)).detach().cpu().numpy()
        probability = float(apply_fixed_calibration(logits, calibration)[0])
        if not np.isfinite(probability):
            raise RuntimeError("Inference produced a non-finite probability.")
        probabilities.append(float(np.clip(probability, 1e-6, 1.0 - 1e-6)))
    output = frame[["uid", "is_pathologic"]].copy()
    output["is_pathologic"] = probabilities
    probability_array = pd.to_numeric(output["is_pathologic"], errors="coerce").to_numpy(dtype=float)
    if len(output) != len(frame) or output["uid"].astype(str).tolist() != template_uids:
        raise RuntimeError("Inference changed submission row count or UID order.")
    if not np.isfinite(probability_array).all() or np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise RuntimeError("Inference produced invalid submission probabilities.")
    if output.isna().any().any() or output["uid"].astype(str).duplicated().any():
        raise RuntimeError("Inference produced missing or duplicate submission values.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print("Writing submission.csv")
    print("Inference complete")
    return output_path


def main() -> None:
    run_inference()


if __name__ == "__main__":
    main()
