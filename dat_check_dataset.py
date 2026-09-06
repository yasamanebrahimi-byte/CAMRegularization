"""Validate a local DaT training directory without exposing record-level data."""

from __future__ import annotations

import argparse
import json

from dat_preprocessing import default_preprocessing_config, estimate_target_spacing, load_dat_records, preprocess_nifti


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local DaT labels, NIfTI files, and preprocessing.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_spacing", nargs=3, type=float, default=None)
    parser.add_argument("--target_shape", nargs=3, type=int, default=[64, 96, 96])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    records = load_dat_records(args.data_dir)
    config = default_preprocessing_config(
        records,
        target_spacing=args.target_spacing or estimate_target_spacing(records),
        target_shape=args.target_shape,
    )
    for record in records[: args.limit or len(records)]:
        tensor = preprocess_nifti(record.path, config)
        if tuple(tensor.shape) != (1, *config["target_shape"]):
            raise RuntimeError("A preprocessed DaT tensor has an unexpected shape.")
    print(json.dumps({"status": "ok", "target_spacing": config["target_spacing"], "target_shape": config["target_shape"]}, indent=2))


if __name__ == "__main__":
    main()

