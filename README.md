# CAMRegularization

CAM-guided masking experiments for image classification. This repo trains CNNs with one of four masking strategies and compares generalization:

- `none`: no masking
- `random`: random cutout
- `cam_high`: mask high-saliency regions
- `cam_low`: mask low-saliency regions

The project supports baseline training, grid-based hyperparameter tuning, greedy mask tuning, and Optuna mask tuning.

## Available Models

Registered models (from `model_registry.py`):

- `resnet18`
- `resnet34`
- `resnet50`
- `vgg16_bn`
- `densenet121`
- `mobilenet_v3_small`
- `mobilenet_v3_large`
- `efficientnet_b0`
- `vit_b_16`
- `convnext_tiny`
- `swin_t`

### Recommended model starters

Use these as practical defaults before tuning:

- `resnet18` (best default for quick iteration)
  - `--lr 0.1 --weight_decay 5e-4 --epochs 100 --batch_size 128 --scheduler cosine --warmup_epochs 5`
- `resnet34`
  - `--lr 0.1 --weight_decay 5e-4 --epochs 120 --batch_size 96 --scheduler cosine --warmup_epochs 5`
- `resnet50`
  - `--lr 0.05 --weight_decay 1e-4 --epochs 120 --batch_size 64 --scheduler cosine --warmup_epochs 5`
- `vgg16_bn`
  - `--lr 0.01 --weight_decay 5e-4 --epochs 100 --batch_size 64 --scheduler multistep --milestones 60,80`
- `densenet121`
  - `--lr 0.05 --weight_decay 1e-4 --epochs 120 --batch_size 64 --scheduler cosine --warmup_epochs 5`
- `mobilenet_v3_small` / `mobilenet_v3_large`
  - `--lr 0.05 --weight_decay 1e-4 --epochs 120 --batch_size 128 --scheduler cosine --warmup_epochs 5`
- `efficientnet_b0`
  - `--lr 0.03 --weight_decay 1e-4 --epochs 120 --batch_size 64 --scheduler cosine --warmup_epochs 5`
- `vit_b_16`
  - `--lr 0.01 --weight_decay 1e-4 --epochs 120 --batch_size 64 --scheduler cosine --warmup_epochs 5`
- `convnext_tiny`
  - `--lr 0.005 --weight_decay 1e-4 --epochs 120 --batch_size 64 --scheduler cosine --warmup_epochs 5`
- `swin_t`
  - `--lr 0.001 --weight_decay 5e-2 --epochs 120 --batch_size 32 --scheduler cosine --warmup_epochs 5`

For CAM masking (`cam_high`, `cam_low`), start with:

- `--mask_warmup_epochs 10 --mask_prob 0.75 --mask_area 0.2 --mask_block 8 --cam_layer auto`
- and reduce batch size on 224px datasets if needed (for example `--batch_size 16` or `32`).

## Available Datasets

Registered datasets (from `dataset_registry.py`):

- `cifar100` (100 classes, default size 32)
- `tiny_imagenet` (200 classes, default size 64)
- `cub200` (200 classes, default size 224)
- `imagenette` (10 classes, default size 224)
- `cifar100_c` (100 classes, default size 32; clean CIFAR-100 train, corrupted test)
- `malimg` (25 classes, default size 224)
- `malware_classification` (9 classes, default size 224)
- `big2015` (alias of `malware_classification`, 9 classes, default size 224)
- `drive_zip` (generic image dataset loaded from a ZIP file, default size 224)

### Recommended dataset starters

- `cifar100`
  - Great first sanity-check dataset.
  - Start with: `--dataset cifar100 --model resnet18 --batch_size 128 --epochs 100 --val_split 0.1`
- `tiny_imagenet`
  - Medium-size benchmark with 64px images.
  - Start with: `--dataset tiny_imagenet --model resnet18 --batch_size 128 --epochs 120 --val_split 0.1`
- `cub200`
  - Fine-grained classification; usually benefits from longer training.
  - Start with: `--dataset cub200 --model resnet50 --batch_size 32 --epochs 120 --val_split 0.1`
- `imagenette`
  - ImageNet-style subset benchmark with automatic download through torchvision.
  - Start with: `--dataset imagenette --model resnet50 --batch_size 64 --epochs 100 --val_split 0.1`
- `cifar100_c`
  - Corruption robustness benchmark setup (default: gaussian_noise severity 5 at test time).
  - Start with: `--dataset cifar100_c --model resnet18 --batch_size 128 --epochs 100 --val_split 0.1`
- `malimg`
  - Requires explicit `train/val/test` folder split.
  - Start with: `--dataset malimg --model resnet18 --batch_size 32 --epochs 15 --masking none`
  - CAM start: `--masking cam_high --mask_warmup_epochs 7 --mask_area 0.2 --batch_size 16 --amp`
- `malware_classification` / `big2015`
  - Requires Microsoft Malware Classification structure (`trainLabels.csv` and `train/` with `.bytes` files).
  - Start with: `--dataset big2015 --model resnet18 --batch_size 32 --epochs 30 --val_split 0.1`

### Recommended mask thresholds by dataset

For `comparison.py`, `--threshold` controls which pixels are treated as low-saliency (`cam <= threshold`).

Use these as starting points, then sweep around them.

| Dataset                              | Start `--threshold` | Suggested sweep    | Notes                                                                             |
| ------------------------------------ | ------------------: | ------------------ | --------------------------------------------------------------------------------- |
| `cifar100`                           |              `0.20` | `0.15, 0.20, 0.30` | Existing runs used `0.15-0.30`; `0.20` is a good middle ground.                   |
| `cifar100_c`                         |              `0.15` | `0.10, 0.15, 0.20` | Start slightly lower than CIFAR-100 to avoid over-masking under corruption shift. |
| `tiny_imagenet`                      |              `0.05` | `0.04, 0.05, 0.08` | Recent runs clustered around `0.04-0.08`.                                         |
| `cub200`                             |              `0.04` | `0.03, 0.04, 0.06` | Fine-grained dataset; conservative masking usually works better first.            |
| `imagenette`                         |              `0.06` | `0.04, 0.06, 0.08` | Practical ImageNet-style starter range.                                           |
| `malimg`                             |              `0.08` | `0.05, 0.08, 0.15` | Existing runs used `0.05` and `0.15`; midpoint is a stable default.               |
| `malware_classification` / `big2015` |              `0.08` | `0.05, 0.08, 0.12` | Similar malware-domain behavior to `malimg`; start moderate.                      |
| `drive_zip`                          |              `0.05` | `0.04, 0.05, 0.10` | Existing runs used `0.04` and `0.10`; start near the lower end first.             |

Tip: if `low_saliency` looks too destructive in previews, lower `--threshold`; if masks are barely visible, raise it.

Quick baseline template:

```bash
python train.py --dataset <dataset> --model <model> --data_dir ./data --masking none --run_name baseline
```

### Copy/Paste Starter Commands

| Dataset         | Goal                       | Command                                                                                                                                                                                      |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cifar100`      | Fast baseline sanity check | `python train.py --dataset cifar100 --model resnet18 --data_dir ./data --epochs 100 --batch_size 128 --val_split 0.1 --masking none --run_name baseline_cifar100`                            |
| `tiny_imagenet` | Medium-scale baseline      | `python train.py --dataset tiny_imagenet --model resnet18 --data_dir ./data --epochs 120 --batch_size 128 --val_split 0.1 --masking none --run_name baseline_tiny_imagenet`                  |
| `cub200`        | Fine-grained baseline      | `python train.py --dataset cub200 --model resnet50 --data_dir ./data --epochs 120 --batch_size 32 --val_split 0.1 --masking none --run_name baseline_cub200`                                 |
| `imagenette`    | ImageNet-style subset      | `python train.py --dataset imagenette --model resnet50 --data_dir ./data --epochs 100 --batch_size 64 --val_split 0.1 --masking none --run_name baseline_imagenette`                         |
| `cifar100_c`    | Corruption robustness      | `python train.py --dataset cifar100_c --model resnet18 --data_dir ./data --epochs 100 --batch_size 128 --val_split 0.1 --masking none --run_name baseline_cifar100_c`                        |
| `malimg`        | Malware image baseline     | `python train.py --dataset malimg --model resnet18 --data_dir ./data --epochs 15 --batch_size 32 --masking none --run_name baseline_malimg`                                                  |
| `malimg`        | CAM-high starter           | `python train.py --dataset malimg --model resnet18 --data_dir ./data --epochs 15 --batch_size 16 --masking cam_high --mask_warmup_epochs 7 --mask_area 0.2 --amp --run_name cam_high_malimg` |
| `big2015`       | Bytes-to-image starter     | `python train.py --dataset big2015 --model resnet18 --data_dir ./data --epochs 30 --batch_size 32 --val_split 0.1 --masking none --run_name baseline_big2015`                                |

Notes:

- `tiny_imagenet`, `cub200`, `imagenette`, and `cifar100_c` now support cloud-style automatic download/extraction in `--data_dir`.
- `big2015` / `malware_classification` expects `trainLabels.csv` and `train/` with `.bytes` files under `--data_dir`.
- `malimg` expects an explicit `train/val/test` split directory.
- `drive_zip` expects a ZIP archive either under `--data_dir` or at `DRIVE_DATASET_ZIP` (mounted local path in Colab).
  Supported ZIP layouts:
  `train/val/test` with class folders, `train/test` with class folders, or a single class-folder root (auto split into train/val/test).
- If CAM runs hit CUDA memory limits, reduce `--batch_size` first.

Colab + Google Drive example:

```bash
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export DRIVE_DATASET_ZIP=/content/drive/MyDrive/path/to/your_dataset.zip
python train.py --dataset drive_zip --data_dir ./data --model resnet18 --run_name drive_zip_baseline
```

## Module Responsibilities

This project is intentionally organized so each file has a narrow role.

- `train.py`: end-to-end training entry point and training loop orchestration.
- `engine.py`: core epoch-level routines (`train_one_epoch`, `evaluate`, `warmup_model`).
- `cam_masking.py`: HiResCAM generation and CAM target layer resolution.
- `dataset_registry.py`: dataset loading and dataset-specific metadata.
- `model_registry.py`: model construction and model-specific defaults.
- `utils.py`: pure utility helpers (seed, metrics, shared parameter/context transforms).
- `IOutils.py`: argument parsing, run directory setup, and JSON/CSV persistence helpers.
- `graphics.py`: plotting and visualization helpers (metrics, tuning plots, CAM preview panels).
- `tune.py`: base hyperparameter grid search.

## Typical Use Cases

### 1) Train a single baseline model

Use when you want one run with fixed hyperparameters.

```bash
python train.py --dataset cifar100 --model resnet18 --masking none --run_name baseline
```

Artifacts:

- `runs/<model>/<dataset>/<run_name>/config.json`
- `runs/<model>/<dataset>/<run_name>/metrics.csv`
- `runs/<model>/<dataset>/<run_name>/metrics_plot.png`

### 2) Compare masking strategies directly

Use when you want an apples-to-apples comparison between `none`, `random`, `cam_high`, and `cam_low`.

Example runs:

```bash
python train.py --masking none
python train.py --masking random --mask_prob 0.75 --mask_area 0.2 --mask_block 8
python train.py --masking cam_high --mask_warmup_epochs 15 --cam_layer auto
python train.py --masking cam_low  --mask_warmup_epochs 15 --cam_layer auto
```

### 3) Tune base (non-mask) hyperparameters

Use when you want strong baseline training settings before mask tuning.

```bash
python tune.py --dataset cifar100 --model resnet18 --runs_root ./runs
```

Output:

- `runs/<model>_<dataset>/tuning_results/tuning_results.json`
- `runs/<model>_<dataset>/tuning_results/ranked_by_val.csv`
- tuning summary plots

### 4) Compare CAM masking variants

Use when you want to train teacher models, generate HiResCAM-masked dataset variants, and train downstream models under a controlled evaluation protocol.

```bash
python comparison.py --dataset cifar100 --input_models resnet18 resnet34 --output_models densenet121 resnet50 --enable_original --enable_low_saliency
```

This runs:

- teacher/input training on the original dataset (`--input_models`)
- variant dataset generation from merged HiResCAM heatmaps
- downstream/output training (`--output_models`) on each enabled variant
- cross-distribution evaluation and result export

Variants are opt-in (no variants generated unless enabled):

- `--enable_original`: `original`
- `--enable_low_saliency`: `low_saliency` (mask where merged CAM `<= --threshold`; use `--mask_top_of_threshold` for high-saliency masking)

Additional controls:

- `--enable_random_control`: adds `random_sparsity` (same per-image mask ratio as `low_saliency`, random pixel locations)
- `--enable_shuffled_cam_control`: adds `shuffled_cam_low_saliency` (low-saliency masks from CAMs of other images)

Key output files:

- `runs/.../comparison_config.json`
- `runs/.../comparison_results.json`
- `runs/.../<output_model>/<variant>/metrics.csv`
- `runs/.../<output_model>/validation_comparison_plot.png`

## Comparison Protocol (Publishable-Grade)

The comparison pipeline now reports a 2x2 core matrix for each downstream model using `original` and `low_saliency` test sets:

- train(original) -> test(original)
- train(original) -> test(low_saliency)
- train(low_saliency) -> test(original)
- train(low_saliency) -> test(low_saliency)

How to read it:

- `train(low_saliency) -> test(original)` estimates augmentation transfer/generalization.
- `train(original) -> test(low_saliency)` estimates preprocessing-only effect.
- If gains appear only when testing on masked images, the effect is likely distribution/preprocessing-driven.
- If gains persist on `test(original)`, this supports true downstream learning benefit.

The matrix is printed to console/log and saved under each variant's `eval_metrics` in `comparison_results.json`.

## Suggested Research Reporting Checklist

For stronger paper evidence, report at least:

- Multiple datasets (for example `cifar100`, `tiny_imagenet`, one malware dataset)
- Multiple teacher sets (`1` teacher and aggregated teachers)
- Multiple downstream architectures
- Cross-eval matrix above (not just train/test on matched distributions)
- Faithfulness controls (`random_sparsity`, `shuffled_cam_low_saliency`)
- Seed sweeps (`>=3` seeds) with mean and standard deviation

## Configuration Notes

- CAM masking (`cam_high` and `cam_low`) is delayed until `mask_warmup_epochs`.
- `val_split > 0` enables validation tracking and best-val model selection logic.
- Reported training metrics include top-1 accuracy and weighted F1.

## Visualization

To generate mask preview panels:

```bash
python graphics.py --dataset cifar100 --model resnet18 --preview_split test --out_dir ./mask_images
```

This writes panel images for `none`, `random`, `cam_high`, and `cam_low` to the chosen output directory.

## Environment

- Python 3.10+
- PyTorch
- pandas
- matplotlib
- Optuna (required only for `tune_optuna.py`)

If Optuna is not installed, base training and non-Optuna scripts still work.
