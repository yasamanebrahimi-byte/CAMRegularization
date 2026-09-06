# CAMRegularization

Image classification training experiments with optional cutout augmentation. The active workflow is standard training through `train.py`, with dataset and model selection handled by the registries.

This repository separates general code capabilities from completed experiment results. The current committed experiment set covers CIFAR-100 and RawMal-TF only. RawMal-TF is loaded through the internal `drive_zip` dataset identifier because the code reads the dataset from a ZIP archive, but research-facing text, tables, plots, and reports should refer to it as RawMal-TF.

Supported cutout modes:

- `none`: train on the original training set.
- `random`: add random square cutout copies during training.
- `cam_low`: add cutout copies from low-saliency regions selected with a teacher model.
- `cam_high`: add cutout copies from high-saliency regions selected with a teacher model.

Validation and test data are not cut out. When cutout is enabled, each training sample yields the original plus `--cutout_m` augmented copies.

## Quick Start

The current run folders use this area-based layout:

```text
runs/<dataset_id>/<model>/<seed>/<area>/<run_name>/
```

For RawMal-TF runs, `<dataset_id>` is `drive_zip`. No-cutout runs are stored under area directories for matched comparisons even though the area value does not affect the no-cutout condition.

Baseline training:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --out_dir runs/cifar100/resnet18/42/0.1 --seed 42 --epochs 100 --cutout_mode none --run_name resnet18_seed42_none
```

Random cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --out_dir runs/cifar100/resnet18/42/0.1 --seed 42 --epochs 100 --cutout_mode random --cutout_m 4 --cutout_area 0.10 --run_name resnet18_seed42_random_M4_area0.1
```

Train a teacher checkpoint for CAM cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --out_dir runs/cifar100/resnet18/42/0.1 --seed 42 --epochs 100 --cutout_mode none --run_name resnet18_seed42_none
```

Use the teacher checkpoint for low-saliency cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --out_dir runs/cifar100/resnet18/42/0.1 --seed 42 --epochs 100 --cutout_mode cam_low --cutout_m 4 --cutout_area 0.10 --teacher_model resnet18 --teacher_checkpoint runs/cifar100/resnet18/42/0.1/resnet18_seed42_none/best_model.pt --cam_layer auto --run_name resnet18_seed42_cam_low_M4_area0.1
```

Use the teacher checkpoint for high-saliency cutout:

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --out_dir runs/cifar100/resnet18/42/0.1 --seed 42 --epochs 100 --cutout_mode cam_high --cutout_m 4 --cutout_area 0.10 --teacher_model resnet18 --teacher_checkpoint runs/cifar100/resnet18/42/0.1/resnet18_seed42_none/best_model.pt --cam_layer auto --run_name resnet18_seed42_cam_high_M4_area0.1
```

CAM cutout modes require both `--teacher_model` and `--teacher_checkpoint` when `--cutout_m > 0`. CAM saliency maps are cached as CPU `.pt` tensors under `--cam_cache_dir`; when omitted, the default is `data/cam_cache/<dataset>/<teacher_model>/<teacher_checkpoint_hash>/`. CAM window coordinates are cached under `--cam_cache_dir/windows` so later epochs can reuse the selected `top/left/size` without repeating CAM pooling and top-k selection. For fast CAM training with workers, precompute the CAM cache first, then rerun training in cache-only worker mode.

## Fast CAM Cache Workflow

For CAM-low or CAM-high runs with `--num_workers > 0`, first populate the saliency cache with workers disabled and deterministic train transforms. Add `--cam_precompute_windows` when you also want to warm the window cache for `aug_index=1..cutout_m` before training, which is especially useful for 224x224 RawMal-TF runs:

```bash
python train.py \
  --dataset cifar100 \
  --data_dir ./data \
  --model resnet18 \
  --out_dir runs/cifar100/resnet18/42/0.1 \
  --run_name resnet18_seed42_cam_low_M4_area0.1_cache_precompute \
  --seed 42 \
  --epochs 100 \
  --batch_size 128 \
  --num_workers 0 \
  --cutout_mode cam_low \
  --cutout_m 1 \
  --cutout_area 0.10 \
  --saliency_candidate_percent 10.0 \
  --teacher_model resnet18 \
  --teacher_checkpoint runs/cifar100/resnet18/42/0.1/resnet18_seed42_none/best_model.pt \
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
  --out_dir runs/cifar100/resnet18/42/0.1 \
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
  --teacher_checkpoint runs/cifar100/resnet18/42/0.1/resnet18_seed42_none/best_model.pt \
  --cam_layer auto \
  --cam_cache_dir /content/cam_cache/cifar100/resnet18/seed42 \
  --deterministic_train_transforms
```

The same cache can be reused for `cam_high` when the dataset, teacher checkpoint, image transforms, input size, CAM layer, and cache settings are the same. For fair comparisons, if CAM runs use `--deterministic_train_transforms`, rerun the `none` and `random` baselines with `--deterministic_train_transforms` too.

Quick local CAM validation without dataset downloads:

```bash
python validate_cam_cutout.py
```

## DaT Parkinson's Challenge (3D addition)

The repository also contains a separate research workflow for the DrivenData [DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/). It does not include competition data. Each labeled examination is a compressed three-dimensional NIfTI scan (`<uid>.nii.gz`) and `train_labels.csv` contains `uid,is_pathologic`, with 0.0 for normal and 1.0 for abnormal.

`dat_preprocessing.py` discovers an extracted dataset or an outer `.zip`/`.tar.gz`, converts scans to a canonical orientation, resamples using spacing estimated from labeled training scans only, performs a foreground-aware crop and fixed crop/pad, then applies per-scan positive-voxel percentile scaling. The model input is always `[1,D,H,W]`. The registered identifiers are `dat_parkinsons` and `resnet18_3d`; the existing 2D identifiers and completed run outputs are unchanged.

Stage 1 (`run_dat_stage1.py`) uses no cutout. It builds fixed stratified folds before any augmentation, searches a small configurable set of training parameters, records the winning fold epochs, derives a frozen median epoch budget, calibrates from labeled OOF predictions only, trains the final unmasked model without validation-based checkpoint selection, and packages Submission #1. `dat_tune.py` remains available as the reusable optimization/calibration layer. For a robustness diagnostic, `--fold_scheme protocol_group` uses rounded original scan shape/spacing signatures; these groups are not hospital-center labels.

Stage 2 (`run_dat_stage2.py`) reads the frozen Stage 1 configuration and varies only `none`, `random`, `cam_low`, or `cam_high`, M4/M8, and fractions 0.05/0.10/0.20/0.30. There is exactly one no-cutout baseline per fold and 24 masked cells per fold. The grid is resumable and integrity-checked. 3D fractions describe a cube whose volume is approximately the requested fraction. Random, low-saliency, and high-saliency use the same foreground-valid candidate cubes; CAM teachers are trained separately within each fold using only that fold's training partition and the frozen Stage 1 epoch budget. Validation scans are never masked. Every candidate receives its own cross-fitted OOF temperature calibration. Stage 2 selection reports `best_overall` for research and uses `best_masked` for Submission #2. For a CAM final condition, saliency comes from the exact full-data Stage 1 final checkpoint; no new final teacher is trained. 3D HiResCAM and CPU saliency/window caches are implemented in the existing dimension-aware `cam_masking.py` and `cutout.py` modules.

The epoch budgets have distinct meanings. `best_config["epochs"]` is the maximum research/CV budget and is retained for every Stage 2 student, because masking can change convergence speed and students are not capped at the Stage 1 median. `best_config["final_training_epochs"]` is the Stage 1-derived fixed duration used by the full-data unmasked Stage 1 model and by each fold-specific Stage 2 CAM teacher. After `best_masked` is selected, `best_masked["final_stage2_training_epochs"]` is derived by applying the same round-half-up median rule to that candidate's own fold-best student epochs; Submission #2 trains on all labeled data for exactly that new fixed duration, without validation checkpoint selection.

Research outputs and competition assets are separate:

- Lightweight Stage 2 run metrics/configurations may live under `runs/dat_parkinsons/`; `runs/dat_parkinsons/summary/generate_summary.py` is independent from the existing CIFAR-100/RawMal-TF summary generator.
- Checkpoints, fold assignments containing UIDs, OOF predictions, preprocessed volumes, CAM caches, and local competition data belong under ignored `artifacts/`, `cache/`, or another local-only directory. Never commit them.
- `dat_final_model.py` trains the selected model using labeled training data only. `build_dat_submission.py` creates a ZIP with `main.py` at its root, fixed preprocessing, model weights, and fixed calibration. The packaged `main.py` reads `/code_execution/data/submission_format.csv` and `/code_execution/data/niftis/<uid>.nii.gz`, processes cases independently, and writes exactly `submission.csv` with `uid,is_pathologic` probabilities in the input row order. It does not train or fit anything at inference time.

Recommended complete local workflow:

```bash
python run_dat_stage1.py \
  --data_dir /path/to/dat_training \
  --cv_folds 5 \
  --trials 4 \
  --epochs 100
```

This leaves `artifacts/dat_parkinsons/final_stage1_unmasked/`, `runs/dat_parkinsons/optimization/`, and `submission/dat_stage1_unmasked.zip`.

```bash
python run_dat_stage2.py \
  --data_dir /path/to/dat_training \
  --best_config artifacts/dat_parkinsons/optimization/best_config.json
```

This leaves `runs/dat_parkinsons/resnet18_3d/`, `runs/dat_parkinsons/summary/`, `artifacts/dat_parkinsons/selected_model.json`, `artifacts/dat_parkinsons/final_stage2_masked/`, and `submission/dat_stage2_masked.zip`. The two submission ZIPs and final model directories are independent and remain on disk together.

Lower-level commands remain available for debugging and research development:

```bash
# Check local labels, NIfTI discovery, spacing, and fixed output tensors.
python dat_check_dataset.py --data_dir /path/to/dat_training

# Stage 1; use --target_shape and --trials overrides for a short smoke run.
python dat_tune.py --data_dir /path/to/dat_training --output_dir artifacts/dat_parkinsons/optimization --cv_folds 5 --trials 4 --epochs 100

# Optional standalone calibration from the Stage 1 OOF logits.
python dat_calibrate.py --oof_predictions artifacts/dat_parkinsons/optimization/oof_predictions.npz --method temperature --output artifacts/dat_parkinsons/optimization/calibration.json

# Stage 2 frozen masking grid.
python dat_masking_experiments.py --data_dir /path/to/dat_training --best_config artifacts/dat_parkinsons/optimization/best_config.json

# DaT-specific summary and cross-validated candidate selection.
python runs/dat_parkinsons/summary/generate_summary.py --best_config artifacts/dat_parkinsons/optimization/best_config.json
python dat_select_model.py

# Train/package lower-level assets only when developing a custom workflow.
python dat_final_model.py --data_dir /path/to/dat_training --best_config artifacts/dat_parkinsons/optimization/best_config.json --output_dir artifacts/dat_parkinsons/final_stage1_unmasked
```

For the current official runtime, clone [competition-sfmn-parkinsons-runtime](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime), put smoke-test data in `data-demo/submission_format.csv` and `data-demo/niftis/`, set `SUBMISSION_IMAGE` as described by that repository, then run `just pull`, `just pack-submission`, `just check-submission`, and `DATA_DIR=/path/to/data-demo just test-submission`. The runtime unpacks the ZIP into `/code_execution/`, runs `python main.py`, and copies the root `submission.csv` back to the local submission directory. No network access or test-set fitting is needed by the submission.

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

Committed run folders are intentionally lightweight and live under `runs/<dataset_id>/<model>/<seed>/<area>/<run_name>/`. They contain only:

- `config.json`: resolved run arguments.
- `metrics.csv`: per-epoch training and validation trajectories. The `eval_*` columns are validation metrics, not held-out test results.
- `metrics_plot.png`: loss and top-1 accuracy curves.

Checkpoint and log files may be produced during local training, but they are not part of the committed run archive.

## Completed Experiment Scope

The completed experiments are separate from the broader dataset and model support in the code. The committed analysis scope is:

- Datasets: CIFAR-100 and RawMal-TF.
- Internal dataset identifiers: `cifar100` and `drive_zip`; write RawMal-TF in research-facing text.
- Model: ResNet18.
- Seeds: 42, 43, and 44.
- Epochs: 100.
- Cutout areas: 0.05, 0.10, 0.20, and 0.30.
- Cutout multiplicities: M4 and M8.
- Conditions: no cutout, random cutout, low-saliency cutout, and high-saliency cutout.

## Summary Analysis

`runs/summary/generate_summary.py` reads the committed run folders and writes summary tables, plots, `summary_report.md`, and `integrity_report.json` under `runs/summary/`. It reports per-condition means across seeds, sample variance, sample standard deviation, and paired effects computed within seed before aggregation. The summary is validation-based: it uses training and validation trajectories from `metrics.csv`, not held-out test results.

## Available Datasets

These are code capabilities registered in `dataset_registry.py`; they are not all completed experiment datasets.

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
- `dat_parkinsons` (labeled three-dimensional NIfTI scans, two classes, target shape configured in `dat_preprocessing.py`)

Dataset notes:

- `tiny_imagenet`, `cub200`, `imagenette`, and `cifar100_c` support local cache/download behavior through `--data_dir`.
- `malimg` expects an explicit `train/val/test` split directory.
- `malware_classification` and `big2015` expect Microsoft Malware Classification data with `trainLabels.csv` and a `train/` folder containing `.bytes` files.
- `drive_zip` expects a ZIP archive under `--data_dir` or at `DRIVE_DATASET_ZIP`.
- `drive_zip` supports `train/val/test`, `train/test`, or one class-folder root that can be split automatically.

## Available Models

These are code capabilities registered in `model_registry.py`; the completed experiments above use ResNet18.

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
- `resnet18_3d` (one-channel, two-logit 3D ResNet18-style model for DaT)

## Current Experiment Starters

```bash
python train.py --dataset cifar100 --model resnet18 --data_dir ./data --out_dir runs/cifar100/resnet18/42/0.05 --seed 42 --epochs 100 --batch_size 128 --val_split 0.1 --cutout_mode none --run_name resnet18_seed42_none
```

```bash
python train.py --dataset drive_zip --model resnet18 --data_dir ./data --out_dir runs/drive_zip/resnet18/42/0.05 --seed 42 --epochs 100 --batch_size 32 --val_split 0.1 --cutout_mode random --cutout_m 4 --cutout_area 0.05 --grayscale --run_name resnet18_seed42_random_M4_area0.05
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
- nibabel, scipy, numpy, pandas, matplotlib, scikit-learn

Install the reproducible research/test dependencies with `pip install -r requirements.txt`. The competition runtime remains inference-only: hidden competition scans are never used for training, calibration, preprocessing fitting, or model selection.
- matplotlib
