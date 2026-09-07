"""Build a root-level ``main.py`` DrivenData submission ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_FILES = ("main.py", "dat_model.py", "dat_preprocessing.py", "dat_calibration_runtime.py")
REQUIRED_BUNDLE_FILES = {
    "main.py", "dat_model.py", "dat_preprocessing.py", "dat_calibration_runtime.py",
    "model/weights.pt", "model_config.json", "preprocessing.json", "calibration.json",
}


def validate_submission_csv(path: str | Path, template_path: str | Path | None = None) -> dict[str, object]:
    """Validate the exact DrivenData two-column probability artifact."""
    path = Path(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["uid", "is_pathologic"]:
        raise ValueError("submission.csv must have exactly uid,is_pathologic columns in that order.")
    if frame["uid"].isna().any() or frame["uid"].astype(str).duplicated().any():
        raise ValueError("submission.csv contains missing or duplicate UIDs.")
    probabilities = pd.to_numeric(frame["is_pathologic"], errors="coerce").to_numpy(dtype=float)
    if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
        raise ValueError("submission.csv probabilities must be finite numeric values.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("submission.csv probabilities must lie in [0, 1].")
    if template_path is not None:
        template = pd.read_csv(template_path)
        if list(template.columns) != ["uid", "is_pathologic"]:
            raise ValueError("submission_format.csv must have exactly uid,is_pathologic columns.")
        template_uids = template["uid"].astype(str).tolist()
        output_uids = frame["uid"].astype(str).tolist()
        if output_uids != template_uids:
            raise ValueError("submission.csv UIDs do not exactly match submission_format.csv order.")
    return {"rows": int(len(frame)), "columns": list(frame.columns), "uids": frame["uid"].astype(str).tolist()}


def validate_submission_zip(path: str | Path) -> dict[str, object]:
    """Validate root layout and required offline assets in a submission ZIP."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Generated submission ZIP contains corrupt data.")
        names = set(archive.namelist())
    missing = REQUIRED_BUNDLE_FILES.difference(names)
    if missing:
        raise RuntimeError(f"Generated ZIP is missing required files: {sorted(missing)}")
    if "main.py" not in names:
        raise RuntimeError("Generated submission ZIP must place main.py at its root.")
    return {"file_count": int(len(names)), "required_files": sorted(REQUIRED_BUNDLE_FILES)}


def _read_json_bytes(source: Path, fallback: dict | None = None) -> bytes:
    if source.is_file():
        return source.read_bytes()
    if fallback is None:
        raise FileNotFoundError(str(source))
    return json.dumps(fallback, indent=2, sort_keys=True).encode("utf-8")


def build(model_dir: str | Path, output_zip: str | Path, source_root: str | Path | None = None) -> Path:
    model_dir = Path(model_dir)
    source_root = Path(source_root or Path(__file__).resolve().parent)
    output_zip = Path(output_zip)
    weights = model_dir / "weights.pt"
    if not weights.exists():
        # Full-data DaT bundles use the scheduled final epoch.  Prefer that
        # explicit name so Submission #1 and the final CAM teacher refer to
        # the same checkpoint bytes.
        weights = model_dir / "final_model.pt"
    if not weights.exists():
        weights = model_dir / "best_model.pt"
    required = [source_root / name for name in SOURCE_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    if not weights.is_file():
        missing.append(str(model_dir / "weights.pt or best_model.pt"))
    if missing:
        raise FileNotFoundError("Submission bundle inputs are missing: " + ", ".join(missing))

    model_config = _read_json_bytes(model_dir / "model_config.json", {
        "model": "resnet18_3d", "num_classes": 2, "n_input_channels": 1, "base_channels": 32, "dropout": 0.0,
    })
    preprocessing = _read_json_bytes(model_dir / "preprocessing.json")
    calibration = _read_json_bytes(model_dir / "calibration.json")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in required:
            archive.writestr(source.name, source.read_bytes())
        archive.writestr("model/weights.pt", weights.read_bytes())
        archive.writestr("model_config.json", model_config)
        archive.writestr("preprocessing.json", preprocessing)
        archive.writestr("calibration.json", calibration)
        provenance = model_dir / "provenance.json"
        if provenance.is_file():
            archive.writestr("provenance.json", provenance.read_bytes())
    validate_submission_zip(output_zip)
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a final DaT model for the official runtime.")
    parser.add_argument("--model_dir", default="artifacts/dat_parkinsons/final_stage1_unmasked")
    parser.add_argument("--output", default="submission/dat_stage1_unmasked.zip")
    args = parser.parse_args()
    print(json.dumps({"submission_zip": str(build(args.model_dir, args.output))}, indent=2))


if __name__ == "__main__":
    main()
