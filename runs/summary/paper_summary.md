# CAM-Guided Cutout Summary

Generated from existing artifacts under `runs/cifar100/`, `runs/malimg/`, and the available RawMal-TF folder `runs/drive_zip/`. No model retraining or run-artifact edits were performed.

## Research Context

This package summarizes CAM-guided cutout augmentation for image-based malware classification. The intended comparison is no cutout (`none`), standard random cutout (`random`), low-saliency CAM-guided cutout (`cam_low`), and high-saliency CAM-guided cutout (`cam_high`). RawMal-TF / `drive_zip`, especially grayscale-only runs, is treated as the main publication dataset; CIFAR100 is a sanity check; MalImg is secondary malware evidence.

## Artifact Coverage

- Runs processed: 21
- Datasets found: CIFAR100, MalImg, RawMal-TF (drive_zip)
- Models found: resnet18
- Conditions found: none, random M4, random M8, cam_low M4, cam_high M4, cam_low M8, cam_high M8
- Inventory/status counts: {"ok": 21}


No cam_low/cam_high pair had identical raw CSV hashes or identical loaded numeric arrays in this artifact set.


## Best Result by Dataset

| Dataset | Model | Seed | Condition | Best acc | Final acc | Best epoch | vs none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR100 | resnet18 | 42 | cam_low M4 | 63.50% | 63.12% | 87 | +1.45 pp |
| RawMal-TF (drive_zip) | resnet18 | 42 | none | 72.64% | 72.47% | 59 | +0.00 pp |
| MalImg | resnet18 | 42 | cam_low M8 | 99.38% | 99.03% | 48 | +0.06 pp |

## Paper-Friendly Comparison Preview

| Dataset | Model | Seed | none | random M4 | random M8 | cam_low M4 | cam_high M4 | cam_low M8 | cam_high M8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR100 | resnet18 | 42 | 62.05% | 62.00% | 62.37% | 63.50% | 58.15% | 62.64% | 57.94% |
| RawMal-TF (drive_zip) | resnet18 | 42 | 72.64% | 70.30% | 70.50% | 70.09% | 69.66% | 70.27% | 69.46% |
| MalImg | resnet18 | 42 | 99.32% | 99.16% | 99.16% | 99.35% | 99.06% | 99.38% | 99.32% |

## RawMal-TF Focused Results

| Condition | Best acc | Final acc | Best epoch | vs none | vs random |
| --- | --- | --- | --- | --- | --- |
| none | 72.64% | 72.47% | 59 | +0.00 pp |  |
| random M4 | 70.30% | 70.08% | 90 | -2.34 pp |  |
| random M8 | 70.50% | 70.28% | 87 | -2.13 pp |  |
| cam_low M4 | 70.09% | 69.46% | 65 | -2.55 pp | -0.21 pp |
| cam_high M4 | 69.66% | 69.13% | 88 | -2.98 pp | -0.64 pp |
| cam_low M8 | 70.27% | 69.58% | 72 | -2.36 pp | -0.23 pp |
| cam_high M8 | 69.46% | 68.91% | 62 | -3.18 pp | -1.04 pp |

## MalImg and CIFAR100 Summaries

| Dataset | Condition | Best acc | Final acc | Best epoch | vs none |
| --- | --- | --- | --- | --- | --- |
| CIFAR100 | none | 62.05% | 61.95% | 99 | +0.00 pp |
| CIFAR100 | random M4 | 62.00% | 61.76% | 82 | -0.06 pp |
| CIFAR100 | random M8 | 62.37% | 61.33% | 83 | +0.31 pp |
| CIFAR100 | cam_low M4 | 63.50% | 63.12% | 87 | +1.45 pp |
| CIFAR100 | cam_high M4 | 58.15% | 57.51% | 85 | -3.91 pp |
| CIFAR100 | cam_low M8 | 62.64% | 61.48% | 81 | +0.59 pp |
| CIFAR100 | cam_high M8 | 57.94% | 56.46% | 88 | -4.11 pp |
| MalImg | none | 99.32% | 99.06% | 97 | +0.00 pp |
| MalImg | random M4 | 99.16% | 98.45% | 43 | -0.16 pp |
| MalImg | random M8 | 99.16% | 98.45% | 36 | -0.16 pp |
| MalImg | cam_low M4 | 99.35% | 98.71% | 21 | +0.03 pp |
| MalImg | cam_high M4 | 99.06% | 98.71% | 34 | -0.26 pp |
| MalImg | cam_low M8 | 99.38% | 99.03% | 48 | +0.06 pp |
| MalImg | cam_high M8 | 99.32% | 99.06% | 64 | +0.00 pp |

## Interpretation

- RawMal-TF best run in the available artifacts is `resnet18_seed42_none` (none) with best accuracy 72.64% at epoch 59.
- Across CAM runs with matched random baselines, the mean CAM-minus-random best-accuracy delta is -0.68 pp (5/12 positive).
- Across non-baseline runs with matched no-cutout baselines, the mean best-accuracy delta is -1.21 pp.
- For matched cam_high minus cam_low best accuracy, the mean delta is -1.94 pp across 6 pairs.

These statements are computed only from existing run artifacts. A positive CAM-minus-random statistic is evidence only for the matched runs present here; it should not be generalized beyond the current seed/model/dataset coverage.

## Warnings

| Severity | Check | Run/Dataset | Observed |
| --- | --- | --- | --- |
| warning | missing_dataset_folder | rawmaltf | missing |

## Next-Step Recommendations

- Use `comparison_table.csv` and the RawMal-TF plots as the primary publication tables/figures.
- Treat single-seed comparisons as preliminary unless additional seeds are added later.
- Report missing literal `runs/rawmaltf/` path as a naming issue if the paper refers to RawMal-TF while artifacts use `drive_zip`.
- Before making saliency-specific claims, check `integrity_checks.csv` for cam_low/cam_high identity failures.
- Do not claim CAM is better than random unless the matched improvement columns and plots support that claim for the target dataset/model/seed.
