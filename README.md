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

For CAM masking (`cam_high`, `cam_low`), start with:

- `--mask_warmup_epochs 10 --mask_prob 0.75 --mask_area 0.2 --mask_block 8 --cam_layer auto`
- and reduce batch size on 224px datasets if needed (for example `--batch_size 16` or `32`).

## Available Datasets

Registered datasets (from `dataset_registry.py`):

- `cifar100` (100 classes, default size 32)
- `tiny_imagenet` (200 classes, default size 64)
- `cub200` (200 classes, default size 224)
- `malimg` (25 classes, default size 224)
- `malware_classification` (9 classes, default size 224)
- `big2015` (alias of `malware_classification`, 9 classes, default size 224)

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
- `malimg`
  - Requires explicit `train/val/test` folder split.
  - Start with: `--dataset malimg --model resnet18 --batch_size 32 --epochs 15 --masking none`
  - CAM start: `--masking cam_high --mask_warmup_epochs 7 --mask_area 0.2 --batch_size 16 --amp`
- `malware_classification` / `big2015`
  - Requires Microsoft Malware Classification structure (`trainLabels.csv` and `train/` with `.bytes` files).
  - Start with: `--dataset big2015 --model resnet18 --batch_size 32 --epochs 30 --val_split 0.1`

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
| `malimg`        | Malware image baseline     | `python train.py --dataset malimg --model resnet18 --data_dir ./data --epochs 15 --batch_size 32 --masking none --run_name baseline_malimg`                                                  |
| `malimg`        | CAM-high starter           | `python train.py --dataset malimg --model resnet18 --data_dir ./data --epochs 15 --batch_size 16 --masking cam_high --mask_warmup_epochs 7 --mask_area 0.2 --amp --run_name cam_high_malimg` |
| `big2015`       | Bytes-to-image starter     | `python train.py --dataset big2015 --model resnet18 --data_dir ./data --epochs 30 --batch_size 32 --val_split 0.1 --masking none --run_name baseline_big2015`                                |

Notes:

- `big2015` / `malware_classification` expects `trainLabels.csv` and `train/` with `.bytes` files under `--data_dir`.
- `malimg` expects an explicit `train/val/test` split directory.
- If CAM runs hit CUDA memory limits, reduce `--batch_size` first.

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

Use when you want to generate and evaluate external HiResCAM-masked dataset variants.

```bash
python comparison.py --dataset cifar100 --model resnet18 --input_models resnet18 resnet34
```
- `--reduction_factor`

## Configuration Notes

- CAM masking (`cam_high` and `cam_low`) is delayed until `mask_warmup_epochs`.
- `val_split > 0` enables validation tracking and best-val model selection logic.
- Reported training metrics include top-1 accuracy and macro F1.

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
