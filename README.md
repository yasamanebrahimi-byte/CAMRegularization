# CAMRegularization

Image classification training experiments with optional cutout augmentation. The active workflow is standard training through `train.py`, with dataset and model selection handled by the registries.

Supported cutout modes:

- `none`: train on the original training set.
- `random`: add random square cutout copies during training.
- `cam_low`: add cutout copies from low-saliency regions selected with a teacher model.
- `cam_high`: add cutout copies from high-saliency regions selected with a teacher model.

Validation and test data are not cut out. When cutout is enabled, each training sample yields the original plus `--cutout_m` augmented copies.

## Quick Start

Baseline training:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --cutout_mode none --run_name baseline_cifar100
```

Random cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --cutout_mode random --cutout_m 4 --cutout_area 0.10 --run_name random_cutout_cifar100
```

Train a teacher checkpoint for CAM cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --cutout_mode none --run_name teacher_cifar100
```

Use the teacher checkpoint for low-saliency cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --cutout_mode cam_low --cutout_m 4 --cutout_area 0.10 --teacher_model resnet18 --teacher_checkpoint ./runs/teacher_cifar100/best_model.pt --cam_layer auto --run_name cam_low_cifar100
```

Use the teacher checkpoint for high-saliency cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --cutout_mode cam_high --cutout_m 4 --cutout_area 0.10 --teacher_model resnet18 --teacher_checkpoint ./runs/teacher_cifar100/best_model.pt --cam_layer auto --run_name cam_high_cifar100
```

CAM cutout modes require both `--teacher_model` and `--teacher_checkpoint` when `--cutout_m > 0`. CAM saliency maps are cached as CPU `.pt` tensors under `--cam_cache_dir`; when omitted, the default is `data/cam_cache/<dataset>/<teacher_model>/<teacher_checkpoint_hash>/`. CAM window coordinates are cached under `--cam_cache_dir/windows` so later epochs can reuse the selected `top/left/size` without repeating CAM pooling and top-k selection. For fast CAM training with workers, precompute the CAM cache first, then rerun training in cache-only worker mode.

## Fast CAM Cache Workflow

For CAM-low or CAM-high runs with `--num_workers > 0`, first populate the saliency cache with workers disabled and deterministic train transforms. Add `--cam_precompute_windows` when you also want to warm the window cache for `aug_index=1..cutout_m` before training, which is especially useful on 224x224 datasets such as MalImg and RawMal-TF:

```bash
python train.py \
  --dataset cifar100 \
  --data_dir ./data \
  --model resnet18 \
  --out_dir runs/cifar100/resnet18/42 \
  --run_name resnet18_seed42_cam_cache_precompute \
  --seed 42 \
  --epochs 100 \
  --batch_size 128 \
  --num_workers 0 \
  --cutout_mode cam_low \
  --cutout_m 1 \
  --cutout_area 0.10 \
  --saliency_candidate_percent 10.0 \
  --teacher_model resnet18 \
  --teacher_checkpoint runs/cifar100/resnet18/42/resnet18_seed42_none/best_model.pt \
  --cam_layer auto \
  --cam_cache_dir /content/cam_cache/cifar100/resnet18/seed42 \
  --deterministic_train_transforms \
  --cam_precompute_only \
  --cam_precompute_windows
```

Then run CAM training with workers against the populated cache:

```bash
python train.py \
  --dataset cifar100 \
  --data_dir ./data \
  --model resnet18 \
  --out_dir runs/cifar100/resnet18/42 \
  --run_name resnet18_seed42_cam_low_M4_area0.1 \
  --seed 42 \
  --epochs 100 \
  --batch_size 128 \
  --num_workers 4 \
  --cutout_mode cam_low \
  --cutout_m 4 \
  --cutout_area 0.10 \
  --saliency_candidate_percent 10.0 \
  --teacher_model resnet18 \
  --teacher_checkpoint runs/cifar100/resnet18/42/resnet18_seed42_none/best_model.pt \
  --cam_layer auto \
  --cam_cache_dir /content/cam_cache/cifar100/resnet18/seed42 \
  --deterministic_train_transforms
```

The same cache can be reused for `cam_high` when the dataset, teacher checkpoint, image transforms, input size, CAM layer, and cache settings are the same. For fair comparisons, if CAM runs use `--deterministic_train_transforms`, rerun the `none` and `random` baselines with `--deterministic_train_transforms` too.

Quick local CAM validation without dataset downloads:

```bash
python validate_cam_cutout.py
```

## Current Cutout Flags

- `--cutout_mode {none,random,cam_low,cam_high}`: choose the training augmentation mode.
- `--cutout_m`: number of cutout copies to add per training sample.
- `--cutout_size`: square cutout side length in pixels.
- `--cutout_area`: square cutout area as a fraction of image area. Use this or `--cutout_size` when `--cutout_m > 0`.
- `--teacher_model`: model architecture used by the teacher checkpoint for CAM cutout.
- `--teacher_checkpoint`: path to a saved teacher checkpoint, usually a `best_model.pt` from a prior `train.py` run.
- `--cam_layer`: teacher layer used for CAM generation. `auto` selects a suitable layer from the model.
- `--cam_cache_dir`: directory for cached CAM saliency maps. Defaults to `data/cam_cache/<dataset>/<teacher_model>/<teacher_checkpoint_hash>/`.
- `--saliency_candidate_percent`: percent of candidate windows considered for CAM-based placement.
- `--cam_precompute_only`: populate the CAM saliency cache and exit without training. Requires CAM cutout, a teacher checkpoint, and `--deterministic_train_transforms`.
- `--cam_precompute_windows`: populate CAM cutout window coordinates for `aug_index=1..cutout_m`; use with `--cam_precompute_only` to exit after precomputing, or without it to warm windows before training.
- `--debug_cam_timing`: log lightweight timing diagnostics for CAM cache path creation, saliency loading, window selection, and masking.
- `--grayscale`: load supported image datasets as grayscale.
- `--include_regex`: include only matching input paths for supported file-based datasets.

## Training Flags

Common options:

- `--dataset`: registered dataset name.
- `--model`: registered model name.
- `--data_dir`: dataset root or cache directory.
- `--out_dir`: output directory for run artifacts. Defaults to `./runs`.
- `--run_name`: run directory name under `--out_dir`. If omitted, `train.py` creates a timestamped name.
- `--epochs`, `--batch_size`, `--num_workers`, `--val_split`, `--seed`.
- `--deterministic_train_transforms`: use deterministic training transforms instead of stochastic train-time augmentation.
- `--optimizer {sgd,adamw}`, `--lr`, `--momentum`, `--weight_decay`, `--scheduler {cosine,multistep}`.
- `--warmup_epochs`, `--min_lr`, `--gamma`, `--milestones`, `--label_smoothing`, `--nesterov`, `--amp`.

Run `python train.py --help` for the full parser output.

## Artifacts

Each run writes artifacts under `--out_dir/<run_name>/`:

- `config.json`: resolved run arguments.
- `metrics.csv`: per-epoch train and eval metrics.
- `metrics_plot.png`: loss and top-1 accuracy curves.
- `best_model.pt`: best tracked checkpoint and final test metrics.
- `<run_name>_<timestamp>.log`: run log.

## Available Datasets

Registered datasets from `dataset_registry.py`:

- `cifar100` (100 classes, default size 32)
- `tiny_imagenet` (200 classes, default size 64)
- `cub200` (200 classes, default size 224)
- `malimg` (25 classes, default size 224)
- `malware_classification` (9 classes, default size 224)
- `big2015` (alias of `malware_classification`, 9 classes, default size 224)
- `imagenette` (10 classes, default size 224)
- `cifar100_c` (100 classes, default size 32; clean CIFAR-100 train, corrupted test)
- `drive_zip` (generic image dataset loaded from a ZIP file, default size 224)

Dataset notes:

- `tiny_imagenet`, `cub200`, `imagenette`, and `cifar100_c` support local cache/download behavior through `--data_dir`.
- `malimg` expects an explicit `train/val/test` split directory.
- `malware_classification` and `big2015` expect Microsoft Malware Classification data with `trainLabels.csv` and a `train/` folder containing `.bytes` files.
- `drive_zip` expects a ZIP archive under `--data_dir` or at `DRIVE_DATASET_ZIP`.
- `drive_zip` supports `train/val/test`, `train/test`, or one class-folder root that can be split automatically.

## Available Models

Registered models from `model_registry.py`:

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

## Example Dataset Starters

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --epochs 100 --batch_size 128 --val_split 0.1 --cutout_mode none --run_name baseline_cifar100
```

```bash
python train.py --dataset tiny_imagenet --model resnet18 --data_dir ./data --epochs 120 --batch_size 128 --val_split 0.1 --cutout_mode none --run_name baseline_tiny_imagenet
```

```bash
python train.py --dataset cub200 --model resnet50 --data_dir ./data --epochs 120 --batch_size 32 --val_split 0.1 --cutout_mode none --run_name baseline_cub200
```

```bash
python train.py --dataset malimg --model resnet18 --data_dir ./data --epochs 15 --batch_size 32 --cutout_mode none --run_name baseline_malimg
```

```bash
python train.py --dataset drive_zip --model resnet18 --data_dir ./data --epochs 30 --batch_size 32 --cutout_mode random --cutout_m 4 --cutout_area 0.10 --grayscale --include_regex "train|val|test" --run_name drive_zip_random
```

## Module Responsibilities

- `train.py`: command-line entry point and end-to-end training orchestration.
- `engine.py`: epoch-level training and evaluation routines.
- `cutout.py`: training dataset wrapper for random and CAM-guided cutout.
- `cam_masking.py`: HiResCAM generation and CAM target layer resolution.
- `dataset_registry.py`: dataset loading and dataset-specific metadata.
- `model_registry.py`: model construction and model-specific defaults.
- `logger.py`: run logging setup.
- `utils.py`: seed, tensor, metric, and input-size helpers.
- `IOutils.py`: argument parsing, run directory setup, and JSON/CSV persistence helpers.
- `graphics.py`: metrics plotting.

## Environment

- Python 3.10+
- PyTorch and torchvision
- pandas
- matplotlib
