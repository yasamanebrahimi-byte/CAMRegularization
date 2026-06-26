#!/usr/bin/env python3
"""Python entry point for regenerating the summary package."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).with_name("generate_summary.js")
    node = shutil.which("node")
    if node is None:
        print("Node.js is required to run the summary generator from this wrapper.", file=sys.stderr)
        print("You can still inspect the generated CSV, JSON, Markdown, and PNG files in this folder.", file=sys.stderr)
        return 1
    return subprocess.call([node, str(script)])


if __name__ == "__main__":
    raise SystemExit(main())
