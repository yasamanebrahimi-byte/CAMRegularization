"""Compatibility CLI for the authoritative DaT Stage 2 summary generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_ROOT = ROOT / "runs" / "dat_parkinsons" / "resnet18_3d"
SUMMARY_DIR = ROOT / "runs" / "dat_parkinsons" / "summary"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DaT Stage 2 runs.")
    parser.add_argument("--run_root", default=str(RUN_ROOT))
    parser.add_argument("--best_config", default="")
    parser.add_argument("--expected_folds", type=int, default=5)
    args = parser.parse_args()

    frozen = None
    if args.best_config and Path(args.best_config).is_file():
        frozen = json.loads(Path(args.best_config).read_text(encoding="utf-8"))

    from dat_stage2_summary import generate_summary

    generate_summary(
        args.run_root,
        SUMMARY_DIR,
        expected_folds=args.expected_folds,
        frozen_config=frozen,
    )


if __name__ == "__main__":
    main()
