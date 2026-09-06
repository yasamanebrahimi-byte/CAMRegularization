"""Build a root-level ``main.py`` DrivenData submission ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


SOURCE_FILES = ("main.py", "dat_model.py", "dat_preprocessing.py", "dat_calibration_runtime.py")
REQUIRED_BUNDLE_FILES = {
    "main.py", "dat_model.py", "dat_preprocessing.py", "dat_calibration_runtime.py",
    "model/weights.pt", "model_config.json", "preprocessing.json", "calibration.json",
}


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
    with zipfile.ZipFile(output_zip) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Generated submission ZIP contains corrupt data.")
        names = set(archive.namelist())
    missing_bundle = REQUIRED_BUNDLE_FILES.difference(names)
    if missing_bundle:
        raise RuntimeError(f"Generated ZIP is missing required files: {sorted(missing_bundle)}")
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a final DaT model for the official runtime.")
    parser.add_argument("--model_dir", default="artifacts/dat_parkinsons/final_stage1_unmasked")
    parser.add_argument("--output", default="submission/dat_stage1_unmasked.zip")
    args = parser.parse_args()
    print(json.dumps({"submission_zip": str(build(args.model_dir, args.output))}, indent=2))


if __name__ == "__main__":
    main()
