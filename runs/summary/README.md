# Summary Package

This folder contains publication-oriented summaries generated from existing run artifacts under runs/cifar100/, runs/malimg/, and runs/rawmaltf/.

## Files

- run_inventory.csv: one row per run with config fields, artifact flags, checkpoint file metadata, status, and notes.
- run_summary.csv: final and best train/evaluation metrics, generalization gap, and improvements over matching no-cutout and random baselines.
- comparison_table.csv: paper-friendly wide table by dataset/model/seed for none, random M4/M8, cam_low M4/M8, and cam_high M4/M8.
- integrity_checks.csv: audit checks for missing files, epoch mismatches, NaNs, constant metrics, duplicate hashes, folder/config mismatches, missing CAM teachers, and identical CAM-low/CAM-high metrics.
- summary_stats.json: aggregate counts, best runs, improvement statistics, CAM comparisons, and major warnings.
- paper_summary.md: human-readable report for manuscript triage.
- plots/: PNG plots plus the CSV data used to build each plot.

## Rerun

From the repository root, run:

    python runs/summary/generate_summary.py

or:

    node runs/summary/generate_summary.js

The generator reads only the source run folders and writes only inside runs/summary/. Rerunning it refreshes the generated summary files in this folder.
