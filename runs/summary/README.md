# Summary Package

This folder contains publication-oriented summaries generated from existing run artifacts only.

## Generated Files

- `run_inventory.csv`: one row per run with configuration fields, artifact flags, epoch coverage, hashes, status, and notes.
- `run_summary.csv`: final and best train/evaluation metrics, best epoch, generalization gaps, and matched improvements over no-cutout and random baselines.
- `comparison_table.csv`: wide paper-friendly comparison by dataset/model/seed for none, random M4/M8, cam_low M4/M8, and cam_high M4/M8.
- `integrity_checks.csv`: missing artifacts, epoch mismatches, metric issues, duplicate hashes, folder/config mismatches, CAM teacher checkpoint checks, and cam_low/cam_high identity checks.
- `summary_stats.json`: aggregate counts, best runs, improvement statistics, CAM-vs-random statistics, cam_low-vs-cam_high statistics, and major warnings.
- `paper_summary.md`: human-readable report for publication planning.
- `plots/`: PNG figures and the CSV tables used to create each plot.

## Rerun

From the repository root:

```bash
python runs/summary/generate_summary.py
```

The script reads `runs/cifar100/`, `runs/malimg/`, `runs/rawmaltf/` when present, and `runs/drive_zip/` as the available RawMal-TF artifact folder. It writes outputs only under `runs/summary/`.
