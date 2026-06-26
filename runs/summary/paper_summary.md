# CAM-Regularization Run Summary

## Research Context

This package summarizes existing artifacts for CAM-guided cutout augmentation in image-based malware classification. It compares no cutout (none), standard random cutout (random), low-saliency CAM-guided cutout (cam_low), and high-saliency CAM-guided cutout (cam_high). RawMal-TF appears in configs as drive_zip and is treated as the primary publication dataset; CIFAR100 is a sanity check, and MalImg is secondary malware evidence.

All conclusions below are computed only from files already present under runs/cifar100/, runs/malimg/, and runs/rawmaltf/.

## Inventory

- Total runs processed: 42
- Status counts: {"suspicious":22,"ok":16,"expected_possible":4}
- Integrity checks emitted: 43

Datasets/models found:
- cifar100: densenet121 (7 runs), resnet18 (7 runs)
- malimg: densenet121 (7 runs), resnet18 (7 runs)
- rawmaltf: densenet121 (7 runs), resnet18 (7 runs)

## Publication-Critical Warning

**At least one cam_low/cam_high pair has identical metric arrays or identical raw metric CSV hashes. Do not claim low- and high-saliency CAM behavior differs for those pairs.**

| Dataset | Model | Seed | Run | Related run | Details |
| --- | --- | --- | --- | --- | --- |
| cifar100 | densenet121 | 42 | densenet121_seed42_cam_low_M4_area0.1 | densenet121_seed42_cam_high_M4_area0.1 | cam_low=densenet121_seed42_cam_low_M4_area0.1 and cam_high=densenet121_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| cifar100 | densenet121 | 42 | densenet121_seed42_cam_low_M8_area0.1 | densenet121_seed42_cam_high_M8_area0.1 | cam_low=densenet121_seed42_cam_low_M8_area0.1 and cam_high=densenet121_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| cifar100 | resnet18 | 42 | resnet18_seed42_cam_low_M4_area0.1 | resnet18_seed42_cam_high_M4_area0.1 | cam_low=resnet18_seed42_cam_low_M4_area0.1 and cam_high=resnet18_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| cifar100 | resnet18 | 42 | resnet18_seed42_cam_low_M8_area0.1 | resnet18_seed42_cam_high_M8_area0.1 | cam_low=resnet18_seed42_cam_low_M8_area0.1 and cam_high=resnet18_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| malimg | densenet121 | 42 | densenet121_seed42_cam_low_M8_area0.1 | densenet121_seed42_cam_high_M8_area0.1 | cam_low=densenet121_seed42_cam_low_M8_area0.1 and cam_high=densenet121_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| malimg | resnet18 | 42 | resnet18_seed42_cam_low_M4_area0.1 | resnet18_seed42_cam_high_M4_area0.1 | cam_low=resnet18_seed42_cam_low_M4_area0.1 and cam_high=resnet18_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| malimg | resnet18 | 42 | resnet18_seed42_cam_low_M8_area0.1 | resnet18_seed42_cam_high_M8_area0.1 | cam_low=resnet18_seed42_cam_low_M8_area0.1 and cam_high=resnet18_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| rawmaltf | densenet121 | 42 | densenet121_seed42_cam_low_M4_area0.1 | densenet121_seed42_cam_high_M4_area0.1 | cam_low=densenet121_seed42_cam_low_M4_area0.1 and cam_high=densenet121_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| rawmaltf | densenet121 | 42 | densenet121_seed42_cam_low_M8_area0.1 | densenet121_seed42_cam_high_M8_area0.1 | cam_low=densenet121_seed42_cam_low_M8_area0.1 and cam_high=densenet121_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| rawmaltf | resnet18 | 42 | resnet18_seed42_cam_low_M4_area0.1 | resnet18_seed42_cam_high_M4_area0.1 | cam_low=resnet18_seed42_cam_low_M4_area0.1 and cam_high=resnet18_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| rawmaltf | resnet18 | 42 | resnet18_seed42_cam_low_M8_area0.1 | resnet18_seed42_cam_high_M8_area0.1 | cam_low=resnet18_seed42_cam_low_M8_area0.1 and cam_high=resnet18_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |

## Best Runs Overall

| Dataset | Model | Run | Condition | Best acc | Final acc | Best epoch | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| malimg | densenet121 | densenet121_seed42_cam_high_M4_area0.1 | cam_high M4 | 99.55% | 99.45% | 66 | ok |
| malimg | densenet121 | densenet121_seed42_cam_high_M8_area0.1 | cam_high M8 | 99.29% | 99.13% | 13 | suspicious |
| malimg | densenet121 | densenet121_seed42_cam_low_M8_area0.1 | cam_low M8 | 99.29% | 99.13% | 13 | suspicious |
| malimg | densenet121 | densenet121_seed42_random_M4_area0.1 | random M4 | 99.29% | 99.03% | 23 | ok |
| malimg | densenet121 | densenet121_seed42_random_M8_area0.1 | random M8 | 99.29% | 99.13% | 14 | expected_possible |
| malimg | densenet121 | densenet121_seed42_cam_low_M4_area0.1 | cam_low M4 | 99.22% | 99.22% | 17 | expected_possible |
| malimg | resnet18 | resnet18_seed42_cam_high_M4_area0.1 | cam_high M4 | 99.06% | 99.06% | 20 | suspicious |
| malimg | resnet18 | resnet18_seed42_cam_low_M4_area0.1 | cam_low M4 | 99.06% | 99.06% | 20 | suspicious |
| malimg | resnet18 | resnet18_seed42_cam_high_M8_area0.1 | cam_high M8 | 98.80% | 98.80% | 18 | suspicious |
| malimg | resnet18 | resnet18_seed42_cam_low_M8_area0.1 | cam_low M8 | 98.80% | 98.80% | 18 | suspicious |
| malimg | resnet18 | resnet18_seed42_random_M8_area0.1 | random M8 | 98.71% | 98.71% | 20 | expected_possible |
| malimg | densenet121 | densenet121_seed42_none | none | 98.44% | 98.21% | 23 | ok |
| malimg | resnet18 | resnet18_seed42_none | none | 98.41% | 98.15% | 54 | ok |
| malimg | resnet18 | resnet18_seed42_random_M4_area0.1 | random M4 | 98.34% | 97.92% | 12 | expected_possible |
| cifar100 | resnet18 | resnet18_seed42_cam_high_M4_area0.1 | cam_high M4 | 78.68% | 78.56% | 99 | suspicious |

_Showing 15 of 42 rows._

## RawMal-TF Focus

| Model | Run | Condition | Best acc | Final acc | Best epoch | Vs none | Vs random | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| densenet121 | densenet121_seed42_cam_high_M8_area0.1 | cam_high M8 | 75.02% | 74.32% | 98 | 6.15% | 1.10% | suspicious |
| densenet121 | densenet121_seed42_cam_low_M8_area0.1 | cam_low M8 | 75.02% | 74.32% | 98 | 6.15% | 1.10% | suspicious |
| resnet18 | resnet18_seed42_cam_high_M8_area0.1 | cam_high M8 | 74.99% | 74.99% | 100 | 7.33% | 0.66% | suspicious |
| resnet18 | resnet18_seed42_cam_low_M8_area0.1 | cam_low M8 | 74.99% | 74.99% | 100 | 7.33% | 0.66% | suspicious |
| resnet18 | resnet18_seed42_random_M8_area0.1 | random M8 | 74.33% | 73.70% | 95 | 6.67% |  | ok |
| densenet121 | densenet121_seed42_cam_high_M4_area0.1 | cam_high M4 | 74.20% | 73.70% | 93 | 5.34% | 0.17% | suspicious |
| densenet121 | densenet121_seed42_cam_low_M4_area0.1 | cam_low M4 | 74.20% | 73.70% | 93 | 5.34% | 0.17% | suspicious |
| densenet121 | densenet121_seed42_random_M4_area0.1 | random M4 | 74.03% | 73.57% | 88 | 5.16% |  | ok |
| densenet121 | densenet121_seed42_random_M8_area0.1 | random M8 | 73.92% | 73.75% | 90 | 5.06% |  | ok |
| resnet18 | resnet18_seed42_random_M4_area0.1 | random M4 | 73.46% | 73.18% | 98 | 5.80% |  | ok |
| resnet18 | resnet18_seed42_cam_high_M4_area0.1 | cam_high M4 | 73.13% | 72.54% | 95 | 5.47% | -0.33% | suspicious |
| resnet18 | resnet18_seed42_cam_low_M4_area0.1 | cam_low M4 | 73.13% | 72.54% | 95 | 5.47% | -0.33% | suspicious |
| densenet121 | densenet121_seed42_none | none | 68.87% | 67.60% | 98 | 0.00% |  | ok |
| resnet18 | resnet18_seed42_none | none | 67.66% | 65.70% | 91 | 0.00% |  | ok |

Interpretation for RawMal-TF should prioritize the computed improvement columns. A CAM method should be described as better than random only where Vs random is positive for the matching model, seed, M, and area.

## MalImg Summary

| Model | Run | Condition | Best acc | Best epoch | Vs none | Status |
| --- | --- | --- | --- | --- | --- | --- |
| densenet121 | densenet121_seed42_cam_high_M4_area0.1 | cam_high M4 | 99.55% | 66 | 1.10% | ok |
| densenet121 | densenet121_seed42_cam_high_M8_area0.1 | cam_high M8 | 99.29% | 13 | 0.84% | suspicious |
| densenet121 | densenet121_seed42_cam_low_M8_area0.1 | cam_low M8 | 99.29% | 13 | 0.84% | suspicious |
| densenet121 | densenet121_seed42_random_M4_area0.1 | random M4 | 99.29% | 23 | 0.84% | ok |
| densenet121 | densenet121_seed42_random_M8_area0.1 | random M8 | 99.29% | 14 | 0.84% | expected_possible |
| densenet121 | densenet121_seed42_cam_low_M4_area0.1 | cam_low M4 | 99.22% | 17 | 0.78% | expected_possible |
| resnet18 | resnet18_seed42_cam_high_M4_area0.1 | cam_high M4 | 99.06% | 20 | 0.65% | suspicious |
| resnet18 | resnet18_seed42_cam_low_M4_area0.1 | cam_low M4 | 99.06% | 20 | 0.65% | suspicious |
| resnet18 | resnet18_seed42_cam_high_M8_area0.1 | cam_high M8 | 98.80% | 18 | 0.40% | suspicious |
| resnet18 | resnet18_seed42_cam_low_M8_area0.1 | cam_low M8 | 98.80% | 18 | 0.40% | suspicious |
| resnet18 | resnet18_seed42_random_M8_area0.1 | random M8 | 98.71% | 20 | 0.30% | expected_possible |
| densenet121 | densenet121_seed42_none | none | 98.44% | 23 | 0.00% | ok |
| resnet18 | resnet18_seed42_none | none | 98.41% | 54 | 0.00% | ok |
| resnet18 | resnet18_seed42_random_M4_area0.1 | random M4 | 98.34% | 12 | -0.06% | expected_possible |

MalImg short runs are marked expected_possible in integrity checks because some MalImg runs may have 20 observed epochs.

## CIFAR100 Sanity Check

| Model | Run | Condition | Best acc | Best epoch | Vs none | Status |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18 | resnet18_seed42_cam_high_M4_area0.1 | cam_high M4 | 78.68% | 99 | 1.48% | suspicious |
| resnet18 | resnet18_seed42_cam_low_M4_area0.1 | cam_low M4 | 78.68% | 99 | 1.48% | suspicious |
| resnet18 | resnet18_seed42_cam_high_M8_area0.1 | cam_high M8 | 78.49% | 100 | 1.29% | suspicious |
| resnet18 | resnet18_seed42_cam_low_M8_area0.1 | cam_low M8 | 78.49% | 100 | 1.29% | suspicious |
| resnet18 | resnet18_seed42_random_M4_area0.1 | random M4 | 78.29% | 100 | 1.10% | ok |
| resnet18 | resnet18_seed42_random_M8_area0.1 | random M8 | 78.26% | 97 | 1.07% | ok |
| resnet18 | resnet18_seed42_none | none | 77.20% | 90 | 0.00% | ok |
| densenet121 | densenet121_seed42_random_M4_area0.1 | random M4 | 62.34% | 99 | 0.64% | ok |
| densenet121 | densenet121_seed42_cam_high_M4_area0.1 | cam_high M4 | 62.18% | 95 | 0.48% | suspicious |
| densenet121 | densenet121_seed42_cam_low_M4_area0.1 | cam_low M4 | 62.18% | 95 | 0.48% | suspicious |
| densenet121 | densenet121_seed42_cam_high_M8_area0.1 | cam_high M8 | 61.83% | 96 | 0.12% | suspicious |
| densenet121 | densenet121_seed42_cam_low_M8_area0.1 | cam_low M8 | 61.83% | 96 | 0.12% | suspicious |
| densenet121 | densenet121_seed42_none | none | 61.70% | 100 | 0.00% | ok |
| densenet121 | densenet121_seed42_random_M8_area0.1 | random M8 | 61.39% | 99 | -0.31% | ok |

## Aggregate Statistics

- Improvement over no-cutout: {"count":36,"mean":0.02447819444444445,"median":0.010829000000000089,"min":-0.00311399999999995,"max":0.07333400000000001,"positive_count":34,"zero_count":0,"negative_count":2}
- CAM vs matching random: {"count":24,"mean":0.0028270416666666687,"median":0.0022289999999999255,"min":-0.003306999999999949,"max":0.010983999999999994,"positive_count":17,"zero_count":2,"negative_count":5}
- CAM low vs high: {"count":12,"mean":-0.00026875000000000276,"median":0,"min":-0.0032250000000000334,"max":0,"positive_count":0,"zero_count":11,"negative_count":1,"low_better_count":0,"high_better_count":1,"tie_count":11,"identical_metric_pair_count":11}

## Warnings

| Check | Severity | Dataset | Model | Run | Details |
| --- | --- | --- | --- | --- | --- |
| duplicate_metrics_hash | critical | cifar100 | densenet121 | densenet121_seed42_cam_high_M4_area0.1 | metrics_hash=4ba861f4e72b3f1a80e3bb4f91d7df5c667df4cc6d6d0cd698f552cd037a7a19; duplicate among: runs/cifar100/densenet121/densenet121_seed42_cam_high_M4_area0.1, runs/cifar100/densenet121/densenet121_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | densenet121 | densenet121_seed42_cam_low_M4_area0.1 | metrics_hash=4ba861f4e72b3f1a80e3bb4f91d7df5c667df4cc6d6d0cd698f552cd037a7a19; duplicate among: runs/cifar100/densenet121/densenet121_seed42_cam_high_M4_area0.1, runs/cifar100/densenet121/densenet121_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | densenet121 | densenet121_seed42_cam_high_M8_area0.1 | metrics_hash=211610abf3d4782d697232cbaccbaf6abc4f9b99d2505f7e5f194e4248901125; duplicate among: runs/cifar100/densenet121/densenet121_seed42_cam_high_M8_area0.1, runs/cifar100/densenet121/densenet121_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | densenet121 | densenet121_seed42_cam_low_M8_area0.1 | metrics_hash=211610abf3d4782d697232cbaccbaf6abc4f9b99d2505f7e5f194e4248901125; duplicate among: runs/cifar100/densenet121/densenet121_seed42_cam_high_M8_area0.1, runs/cifar100/densenet121/densenet121_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | resnet18 | resnet18_seed42_cam_high_M4_area0.1 | metrics_hash=f81cd2ce7a317a2cb0a5828760abdb9f91e12dc8aa312e905b27f0023dd90dd7; duplicate among: runs/cifar100/resnet18/resnet18_seed42_cam_high_M4_area0.1, runs/cifar100/resnet18/resnet18_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | resnet18 | resnet18_seed42_cam_low_M4_area0.1 | metrics_hash=f81cd2ce7a317a2cb0a5828760abdb9f91e12dc8aa312e905b27f0023dd90dd7; duplicate among: runs/cifar100/resnet18/resnet18_seed42_cam_high_M4_area0.1, runs/cifar100/resnet18/resnet18_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | resnet18 | resnet18_seed42_cam_high_M8_area0.1 | metrics_hash=5918cdc6a99badc470c445416a2ad1dd90d946c211c1ddac0bc6e49077585c59; duplicate among: runs/cifar100/resnet18/resnet18_seed42_cam_high_M8_area0.1, runs/cifar100/resnet18/resnet18_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | cifar100 | resnet18 | resnet18_seed42_cam_low_M8_area0.1 | metrics_hash=5918cdc6a99badc470c445416a2ad1dd90d946c211c1ddac0bc6e49077585c59; duplicate among: runs/cifar100/resnet18/resnet18_seed42_cam_high_M8_area0.1, runs/cifar100/resnet18/resnet18_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | malimg | densenet121 | densenet121_seed42_cam_high_M8_area0.1 | metrics_hash=d17bc6de8152f73520bf4d3b74513e890a1f7745cb39b30a57c4c3be79fe1e6b; duplicate among: runs/malimg/densenet121/densenet121_seed42_cam_high_M8_area0.1, runs/malimg/densenet121/densenet121_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | malimg | densenet121 | densenet121_seed42_cam_low_M8_area0.1 | metrics_hash=d17bc6de8152f73520bf4d3b74513e890a1f7745cb39b30a57c4c3be79fe1e6b; duplicate among: runs/malimg/densenet121/densenet121_seed42_cam_high_M8_area0.1, runs/malimg/densenet121/densenet121_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | malimg | resnet18 | resnet18_seed42_cam_high_M4_area0.1 | metrics_hash=a829b9517d1e6f99bc41808096f200e75e615c52350d61c910611e29b3c79dd8; duplicate among: runs/malimg/resnet18/resnet18_seed42_cam_high_M4_area0.1, runs/malimg/resnet18/resnet18_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | malimg | resnet18 | resnet18_seed42_cam_low_M4_area0.1 | metrics_hash=a829b9517d1e6f99bc41808096f200e75e615c52350d61c910611e29b3c79dd8; duplicate among: runs/malimg/resnet18/resnet18_seed42_cam_high_M4_area0.1, runs/malimg/resnet18/resnet18_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | malimg | resnet18 | resnet18_seed42_cam_high_M8_area0.1 | metrics_hash=1962be9ad44f82233771737e6c9fe4128aa97fd215edd3a255a51b18bb3c03f3; duplicate among: runs/malimg/resnet18/resnet18_seed42_cam_high_M8_area0.1, runs/malimg/resnet18/resnet18_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | malimg | resnet18 | resnet18_seed42_cam_low_M8_area0.1 | metrics_hash=1962be9ad44f82233771737e6c9fe4128aa97fd215edd3a255a51b18bb3c03f3; duplicate among: runs/malimg/resnet18/resnet18_seed42_cam_high_M8_area0.1, runs/malimg/resnet18/resnet18_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | densenet121 | densenet121_seed42_cam_high_M4_area0.1 | metrics_hash=1d0a402e9ee443c5610c58792ab4c6e44ccd502b845e2d4f78169a995de7150f; duplicate among: runs/rawmaltf/densenet121/densenet121_seed42_cam_high_M4_area0.1, runs/rawmaltf/densenet121/densenet121_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | densenet121 | densenet121_seed42_cam_low_M4_area0.1 | metrics_hash=1d0a402e9ee443c5610c58792ab4c6e44ccd502b845e2d4f78169a995de7150f; duplicate among: runs/rawmaltf/densenet121/densenet121_seed42_cam_high_M4_area0.1, runs/rawmaltf/densenet121/densenet121_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | densenet121 | densenet121_seed42_cam_high_M8_area0.1 | metrics_hash=4c384cf098b9395b43ef77019137bc219f565068aed1083090503bff566ce524; duplicate among: runs/rawmaltf/densenet121/densenet121_seed42_cam_high_M8_area0.1, runs/rawmaltf/densenet121/densenet121_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | densenet121 | densenet121_seed42_cam_low_M8_area0.1 | metrics_hash=4c384cf098b9395b43ef77019137bc219f565068aed1083090503bff566ce524; duplicate among: runs/rawmaltf/densenet121/densenet121_seed42_cam_high_M8_area0.1, runs/rawmaltf/densenet121/densenet121_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | resnet18 | resnet18_seed42_cam_high_M4_area0.1 | metrics_hash=17eb71cae0a43cc39eb5d2f24f37c8cebc004f1347f55b150571b66d99c754e8; duplicate among: runs/rawmaltf/resnet18/resnet18_seed42_cam_high_M4_area0.1, runs/rawmaltf/resnet18/resnet18_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | resnet18 | resnet18_seed42_cam_low_M4_area0.1 | metrics_hash=17eb71cae0a43cc39eb5d2f24f37c8cebc004f1347f55b150571b66d99c754e8; duplicate among: runs/rawmaltf/resnet18/resnet18_seed42_cam_high_M4_area0.1, runs/rawmaltf/resnet18/resnet18_seed42_cam_low_M4_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | resnet18 | resnet18_seed42_cam_high_M8_area0.1 | metrics_hash=1609c7925643d65d722114a74fd33ec7bbd25cc864029031a2fb706eaa704bd5; duplicate among: runs/rawmaltf/resnet18/resnet18_seed42_cam_high_M8_area0.1, runs/rawmaltf/resnet18/resnet18_seed42_cam_low_M8_area0.1 |
| duplicate_metrics_hash | critical | rawmaltf | resnet18 | resnet18_seed42_cam_low_M8_area0.1 | metrics_hash=1609c7925643d65d722114a74fd33ec7bbd25cc864029031a2fb706eaa704bd5; duplicate among: runs/rawmaltf/resnet18/resnet18_seed42_cam_high_M8_area0.1, runs/rawmaltf/resnet18/resnet18_seed42_cam_low_M8_area0.1 |
| identical_cam_low_high_metrics | critical | cifar100 | densenet121 | densenet121_seed42_cam_low_M4_area0.1 | cam_low=densenet121_seed42_cam_low_M4_area0.1 and cam_high=densenet121_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | cifar100 | densenet121 | densenet121_seed42_cam_low_M8_area0.1 | cam_low=densenet121_seed42_cam_low_M8_area0.1 and cam_high=densenet121_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | cifar100 | resnet18 | resnet18_seed42_cam_low_M4_area0.1 | cam_low=resnet18_seed42_cam_low_M4_area0.1 and cam_high=resnet18_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | cifar100 | resnet18 | resnet18_seed42_cam_low_M8_area0.1 | cam_low=resnet18_seed42_cam_low_M8_area0.1 and cam_high=resnet18_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | malimg | densenet121 | densenet121_seed42_cam_low_M8_area0.1 | cam_low=densenet121_seed42_cam_low_M8_area0.1 and cam_high=densenet121_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | malimg | resnet18 | resnet18_seed42_cam_low_M4_area0.1 | cam_low=resnet18_seed42_cam_low_M4_area0.1 and cam_high=resnet18_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | malimg | resnet18 | resnet18_seed42_cam_low_M8_area0.1 | cam_low=resnet18_seed42_cam_low_M8_area0.1 and cam_high=resnet18_seed42_cam_high_M8_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |
| identical_cam_low_high_metrics | critical | rawmaltf | densenet121 | densenet121_seed42_cam_low_M4_area0.1 | cam_low=densenet121_seed42_cam_low_M4_area0.1 and cam_high=densenet121_seed42_cam_high_M4_area0.1 same_raw_csv_hash=true, same_numeric_arrays=true |

_Showing 30 of 33 rows._

## Next-Step Recommendations

- Treat RawMal-TF / drive_zip grayscale-only results as the main publication evidence.
- Before making a low-vs-high saliency claim, resolve any identical cam_low and cam_high metric warnings.
- For any run with fewer than the expected 100 epochs outside MalImg, rerun or exclude it from headline comparisons.
- Use comparison_table.csv for paper tables and integrity_checks.csv for audit notes.
