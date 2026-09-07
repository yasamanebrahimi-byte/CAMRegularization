"""Validate a local DaT submission CSV and/or offline submission ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dat_submission import validate_submission_csv, validate_submission_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DaT submission artifact locally.")
    parser.add_argument("--submission", default="", help="Path to submission.csv to validate.")
    parser.add_argument("--data_dir", default="", help="Runtime data directory containing submission_format.csv.")
    parser.add_argument("--zip", dest="zip_path", default="", help="Path to a submission ZIP to validate.")
    args = parser.parse_args()
    result = {}
    if args.submission:
        template = Path(args.data_dir) / "submission_format.csv" if args.data_dir else None
        result["submission"] = validate_submission_csv(args.submission, template if template and template.is_file() else None)
    if args.zip_path:
        result["zip"] = validate_submission_zip(args.zip_path)
    if not result:
        parser.error("Provide --submission and/or --zip.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
