# CAMRegularization

CAM-guided masking experiments for image classification. This repo trains CNNs with one of four masking strategies and compares generalization:

- `none`: no masking
- `random`: random cutout
- `cam_high`: mask high-saliency regions
- `cam_low`: mask low-saliency regions

The project supports baseline training, grid-based hyperparameter tuning, greedy mask tuning, and Optuna mask tuning.

## Module Responsibilities

This project is intentionally organized so each file has a narrow role.

- `train.py`: end-to-end training entry point and training loop orchestration.
- `engine.py`: core epoch-level routines (`train_one_epoch`, `evaluate`, `warmup_model`).
- `cam_masking.py`: GradCAM generation and masking operations.
- `dataset_registry.py`: dataset loading and dataset-specific metadata.
- `model_registry.py`: model construction and model-specific defaults.
- `utils.py`: pure utility helpers (seed, metrics, shared parameter/context transforms).
- `IOutils.py`: argument parsing, run directory setup, and JSON/CSV persistence helpers.
- `graphics.py`: plotting and visualization helpers (metrics, tuning plots, CAM preview panels).
- `tune.py`: base hyperparameter grid search.
- `mask_tune.py`: full mask-parameter grid search (optionally per masking mode).
- `greedy_tune.py`: greedy stage-wise mask-parameter search.
- `tune_optuna.py`: Optuna multi-fidelity mask tuning.

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

### 4) Tune mask hyperparameters with full grid search

Use when exhaustive mask-search coverage is preferred.

```bash
python mask_tune.py --dataset cifar100 --model resnet18 --masking_type all
```

To tune only one masking mode:

```bash
python mask_tune.py --masking_type cam_high
```

### 5) Tune mask hyperparameters with greedy search

Use when you need lower compute than full grid search.

```bash
python greedy_tune.py --dataset cifar100 --model resnet18 --masking_type all
```

### 6) Tune mask hyperparameters with Optuna

Use when you want adaptive search + pruning.

```bash
python tune_optuna.py --dataset cifar100 --model resnet18 --runs_root ./runs --n_jobs 1
```

Optional controls:

- `--masking_type {all,random,cam_high,cam_low}`
- `--min_resource_epochs`
- `--max_resource_epochs`
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
