# CAM Cutout Validation Summary

Generated: 2026-08-04 16:30:18 UTC

## Research Question

This project asks whether saliency-guided cutout improves validation performance or stability relative to standard random cutout. The four conditions are no cutout, random cutout, low-saliency cutout, and high-saliency cutout. The primary estimates are means across seeds 42, 43, and 44 for each matched dataset, M, cutout-area, and condition combination.

The CSV `eval_*` metrics are validation metrics, not held-out test results.

## Valid Run Inventory

Discovered 168 run folders with configs and metrics. After deduplicating no-cutout baselines, 150 runs are analysis-valid: 144 augmented runs and 6 selected no-cutout baselines. The missing-or-invalid table contains 0 missing expected combinations and 1 invalid or excluded run records.

The aggregate table has 50 dataset/condition/M/area rows: one no-cutout row per dataset plus separate rows for every M and area for random, low-saliency, and high-saliency cutout. No aggregate row averages across different areas, M values, or conditions.

## No-Cutout Duplicate Handling

No-cutout runs appear under multiple area directories even though area does not apply. The generator compares metric hashes and normalized config hashes, then selects exactly one no-cutout observation per dataset and seed. Repeated baseline copies are excluded from seed counts and paired comparisons.

Baseline duplicate findings: 6 dataset/seed baseline groups, 17 exact duplicate copies excluded, 1 differing copy excluded, and 1 group with any differing baseline content. Details are in `tables/duplicate_baselines.csv`.

## Highest Mean Performance

- CIFAR-100: Low-saliency cutout, M4, area 0.10: 63.51%; Low-saliency cutout, M8, area 0.10: 63.31%; Low-saliency cutout, M8, area 0.05: 63.22%.
- RawMal-TF: No cutout: 72.83%; Random cutout, M4, area 0.30: 71.55%; Low-saliency cutout, M4, area 0.30: 71.43%.

## Lowest Across-Seed Variance

- CIFAR-100: Low-saliency cutout, M8, area 0.05: variance 0.00000100, SD 0.10 pp; High-saliency cutout, M8, area 0.10: variance 0.00000183, SD 0.14 pp; Random cutout, M8, area 0.10: variance 0.00000245, SD 0.16 pp.
- RawMal-TF: No cutout: variance 0.00000265, SD 0.16 pp; High-saliency cutout, M8, area 0.05: variance 0.00000556, SD 0.24 pp; High-saliency cutout, M8, area 0.30: variance 0.00000930, SD 0.30 pp.

## CAM Versus Random Cutout

- CIFAR-100: Low-saliency cutout versus random has 7 area/M cells positive for all available seeds, 0 negative for all available seeds, and 1 mixed. Cell mean paired effects range from +0.12 pp to +7.91 pp.
- CIFAR-100: High-saliency cutout versus random has 0 area/M cells positive for all available seeds, 8 negative for all available seeds, and 0 mixed. Cell mean paired effects range from -7.08 pp to -1.52 pp.
- RawMal-TF: Low-saliency cutout versus random has 1 area/M cells positive for all available seeds, 1 negative for all available seeds, and 6 mixed. Cell mean paired effects range from -0.56 pp to +0.46 pp.
- RawMal-TF: High-saliency cutout versus random has 0 area/M cells positive for all available seeds, 6 negative for all available seeds, and 2 mixed. Cell mean paired effects range from -2.00 pp to -0.21 pp.

## Effects by Area and M

- CIFAR-100: largest mean CAM advantage is Low-saliency cutout minus random cutout at M8, area 0.30 (+7.91 pp); largest mean CAM deficit is High-saliency cutout minus random cutout at M8, area 0.20 (-7.08 pp).
- RawMal-TF: largest mean CAM advantage is Low-saliency cutout minus random cutout at M4, area 0.20 (+0.46 pp); largest mean CAM deficit is High-saliency cutout minus random cutout at M4, area 0.30 (-2.00 pp).

## CIFAR-100 Versus RawMal-TF

- CIFAR-100: the highest aggregate mean best validation accuracy is 63.51%; the no-cutout mean is 62.65%.
- RawMal-TF: the highest aggregate mean best validation accuracy is 72.83%; the no-cutout mean is 72.83%.
- For low-saliency minus random, CIFAR-100 cell mean paired effects range from +0.12 pp to +7.91 pp; RawMal-TF ranges from -0.56 pp to +0.46 pp.
- For high-saliency minus random, CIFAR-100 cell mean paired effects range from -7.08 pp to -1.52 pp; RawMal-TF ranges from -2.00 pp to -0.21 pp.

## Seed Consistency

The paired-effect table computes each comparison within seed before aggregating. Apparent improvements are strongest when all three seed-level differences in the matched area/M cell have the same sign; mixed-sign cells should be read as seed-sensitive rather than reliable treatment wins.

## M8 Minus M4

The M8-minus-M4 paired effects are descriptive because M changes the number of augmented samples and optimizer updates. The generator computes M8 minus M4 within the same dataset, seed, area, and cutout condition before reporting means and variability.

## Plot Notes

The plot directory contains 80 figures, each saved as high-resolution PNG and PDF. Plotted accuracies are percentages, plotted differences are percentage points, and every error bar or shaded band represents sample standard deviation across seeds.

## Limitations

Only three seeds are available, so variance and t-based 95% confidence intervals are exploratory. The files support validation accuracy, validation loss, and stability summaries, but they do not support:

- held-out test accuracy;
- macro-F1;
- per-family metrics;
- confusion matrices;
- calibration;
- sample-level predictions;
- saliency-faithfulness measurements;
- zero-padding overlap;
- wall-clock or GPU-efficiency analysis.

No unavailable result is fabricated here.
