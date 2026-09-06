from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cam_masking import compute_saliency_map
from cutout import CutoutAugmentedDataset
from dat_calibration import apply_temperature, fit_temperature
from dat_metrics import compute_binary_metrics
from dat_model import build_resnet18_3d, load_model_from_bundle
from dat_preprocessing import DatDataset, default_preprocessing_config, load_dat_records, preprocess_nifti
from main import run_inference


class Toy3DTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 1, 1, 1] = 1.0
            self.conv.weight[0, 0, 0, 1, 1] = 0.3
            self.conv.weight[1, 0, 1, 1, 0] = 0.7
            self.fc.weight.copy_(torch.tensor([[1.0, 0.2], [0.2, 1.0]]))

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))


def _make_data(tmp_path: Path):
    niftis = tmp_path / "niftis"
    niftis.mkdir(parents=True)
    rows = []
    for index, shape in enumerate(((10, 12, 8), (12, 10, 9), (9, 11, 10), (11, 9, 8), (10, 10, 9), (9, 12, 8))):
        array = np.zeros(shape, dtype=np.uint16)
        array[2:-2, 3:-3, 2:-2] = (index + 1) * 10
        array[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1000
        uid = f"synthetic_{index}"
        affine = np.diag([1.5 + 0.1 * index, 2.0, 2.5, 1.0])
        nib.save(nib.Nifti1Image(array, affine), niftis / f"{uid}.nii.gz")
        rows.append({"uid": uid, "is_pathologic": float(index % 2)})
    pd.DataFrame(rows).to_csv(tmp_path / "train_labels.csv", index=False)
    return tmp_path


def test_nifti_discovery_resampling_and_determinism(tmp_path):
    root = _make_data(tmp_path)
    records = load_dat_records(root)
    config = default_preprocessing_config(records, target_spacing=(2.0, 2.0, 2.0), target_shape=(8, 12, 12))
    first = preprocess_nifti(records[0].path, config)
    second = preprocess_nifti(records[0].path, config)
    assert first.shape == (1, 8, 12, 12)
    assert torch.equal(first, second)
    dataset = DatDataset(records, config, train=False)
    image, target = dataset[1]
    assert image.shape == (1, 8, 12, 12)
    assert target.item() == 1
    assert dataset.get_foreground_mask(1).shape == (8, 12, 12)


def test_archive_dataset_discovery(tmp_path):
    root = _make_data(tmp_path / "source")
    archive_path = tmp_path / "dat_outer.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in root.rglob("*"):
            if source.is_file():
                archive.write(source, Path("outer") / source.relative_to(root))
    records = load_dat_records(archive_path)
    assert len(records) == 6
    assert {record.uid for record in records} == {f"synthetic_{index}" for index in range(6)}


def test_resnet3d_forward_and_probability_shape():
    model = build_resnet18_3d(num_classes=2, n_input_channels=1, base_channels=2)
    logits = model(torch.rand(2, 1, 8, 12, 12))
    assert logits.shape == (2, 2)
    assert torch.softmax(logits, dim=1)[:, 1].shape == (2,)
    saliency = compute_saliency_map(model, torch.rand(1, 8, 12, 12))
    assert saliency.shape == (8, 12, 12)


def test_3d_hirescam_and_cutout_modes(tmp_path):
    teacher = Toy3DTeacher().eval()
    image = torch.rand(1, 8, 12, 12)
    saliency = compute_saliency_map(teacher, image, cam_layer="conv")
    assert saliency.shape == (8, 12, 12)
    assert torch.isfinite(saliency).all()
    assert 0.0 <= float(saliency.min()) <= float(saliency.max()) <= 1.0

    foreground = torch.zeros(8, 12, 12, dtype=torch.bool)
    foreground[2:6, 4:8, 4:8] = True
    base_image = foreground.float().unsqueeze(0)

    class Base(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return base_image.clone(), 1

        def get_foreground_mask(self, index):
            return foreground.clone()

    for mode in ("random", "cam_low", "cam_high"):
        kwargs = {
            "base_dataset": Base(), "cutout_mode": mode, "cutout_m": 1,
            "cutout_size": 3, "cutout_area": None, "mean": (0.0,), "std": (1.0,),
            "seed": 7, "min_foreground_fraction": 0.75,
        }
        if mode.startswith("cam_"):
            kwargs.update({"teacher_model": teacher, "cam_layer": "conv", "cam_cache_dir": str(tmp_path / mode)})
        dataset = CutoutAugmentedDataset(**kwargs)
        masked, target = dataset[1]
        assert masked.shape == (1, 8, 12, 12)
        assert target == 1
        changed = (base_image - masked).abs() > 1e-6
        coords = torch.nonzero(changed[0])
        assert coords.numel() > 0
        assert int(coords[:, 0].min()) >= 2 and int(coords[:, 0].max()) < 6
        assert int(coords[:, 1].min()) >= 4 and int(coords[:, 1].max()) < 8
        assert int(coords[:, 2].min()) >= 4 and int(coords[:, 2].max()) < 8

    random_dataset = CutoutAugmentedDataset(
        base_dataset=Base(), cutout_mode="random", cutout_m=2, cutout_size=3,
        cutout_area=None, mean=(0.0,), std=(1.0,), seed=7,
        min_foreground_fraction=0.75,
    )
    assert len(random_dataset) == 3
    _ = random_dataset[2]

    invalid_cam_dataset = CutoutAugmentedDataset(
        base_dataset=Base(), cutout_mode="cam_low", cutout_m=1, cutout_size=3,
        cutout_area=None, mean=(0.0,), std=(1.0,), seed=7,
        teacher_model=teacher, cam_layer="missing.layer", cam_cache_dir=str(tmp_path / "bad_cam"),
    )
    with pytest.raises(RuntimeError, match="CAM cutout failed.*Could not find CAM layer"):
        invalid_cam_dataset[1]

    cam_cache = list((tmp_path / "cam_low").rglob("*.pt"))
    window_cache = list((tmp_path / "cam_low" / "windows").rglob("*.json"))
    assert cam_cache and window_cache

    # Reuse both CPU caches without making a teacher available.  The M=2
    # request forces a new window entry while reusing the existing saliency.
    cached_only = CutoutAugmentedDataset(
        base_dataset=Base(), cutout_mode="cam_low", cutout_m=2,
        cutout_size=3, cutout_area=None, mean=(0.0,), std=(1.0,), seed=7,
        teacher_model=None, cam_layer="conv", cam_cache_dir=str(tmp_path / "cam_low"),
        min_foreground_fraction=0.75,
    )
    cached_masked, _ = cached_only[2]
    assert cached_masked.shape == base_image.shape


def test_probability_metrics_and_temperature_scaling():
    logits = np.asarray([[-1.0, 1.0], [1.0, -1.0], [-0.3, 0.3], [0.3, -0.3]])
    targets = np.asarray([1, 0, 1, 0])
    metrics = compute_binary_metrics(targets, logits=logits)
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["brier_score"])
    temperature = fit_temperature(logits, targets)
    assert temperature > 0
    assert np.all(np.isfinite(apply_temperature(logits, temperature)))


def test_stage1_smoke_driver(tmp_path):
    from argparse import Namespace
    from dat_tune import run

    root = _make_data(tmp_path / "dataset")
    args = Namespace(
        data_dir=str(root), output_dir=str(tmp_path / "optimization"), cv_folds=2,
        trials=1, seed=3, epochs=1, patience=0, batch_size=2, num_workers=0,
        target_spacing=[2.0, 2.0, 2.0], target_shape=[8, 12, 12],
        intensity_lower_percentile=1.0, intensity_upper_percentile=99.0,
        foreground_threshold=0.0, crop_margin_mm=2.0, calibration="temperature",
        search_space_json="", max_train_batches=1, max_val_batches=1, base_channels=2, amp=False, augmentation="none",
    )
    result = run(args)
    output = Path(args.output_dir)
    assert (output / "best_config.json").is_file()
    assert (output / "cv_trials.csv").is_file()
    assert (output / "calibration.json").is_file()
    assert np.isfinite(result["raw_metrics"]["log_loss"])

    from dat_final_model import main as final_main
    previous_argv = sys.argv
    sys.argv = [
        "dat_final_model.py", "--data_dir", str(root),
        "--best_config", str(output / "best_config.json"),
        "--calibration", str(output / "calibration.json"),
        "--output_dir", str(tmp_path / "final_model"), "--max_train_batches", "1",
    ]
    try:
        final_main()
    finally:
        sys.argv = previous_argv
    assert (tmp_path / "final_model" / "model_config.json").is_file()
    assert (tmp_path / "final_model" / "best_model.pt").is_file()

    from dat_masking_experiments import run as run_masking
    masking_args = Namespace(
        data_dir=str(root), best_config=str(output / "best_config.json"),
        output_dir=str(tmp_path / "runs" / "resnet18_3d"), fold_assignments=str(output / "fold_assignments.json"),
        fold=0, conditions=["none", "random", "cam_low", "cam_high"], m_values=[1], fractions=[0.10],
        seed=3, num_workers=0, max_train_batches=1, max_val_batches=0, cam_layer="conv1",
        saliency_candidate_percent=10.0, min_foreground_fraction=0.25,
        cam_cache_dir=str(tmp_path / "cam_cache"), preprocessed_cache_dir=str(tmp_path / "preprocessed"),
    )
    run_masking(masking_args)
    assert list((tmp_path / "runs").rglob("metrics.csv"))


def test_submission_main_schema_and_fixed_bundle(tmp_path):
    data_root = _make_data(tmp_path / "data")
    runtime_data = tmp_path / "runtime_data"
    (runtime_data / "niftis").mkdir(parents=True)
    submission_frame = pd.DataFrame({"uid": ["synthetic_4", "synthetic_0"], "is_pathologic": [np.nan, np.nan]})
    submission_frame.to_csv(runtime_data / "submission_format.csv", index=False)
    for uid in submission_frame["uid"]:
        source = data_root / "niftis" / f"{uid}.nii.gz"
        (runtime_data / "niftis" / source.name).write_bytes(source.read_bytes())
    bundle = tmp_path / "bundle"
    (bundle / "model").mkdir(parents=True)
    model = build_resnet18_3d(num_classes=2, n_input_channels=1, base_channels=2)
    torch.save(model.state_dict(), bundle / "model" / "weights.pt")
    config = default_preprocessing_config(load_dat_records(data_root), target_spacing=(2, 2, 2), target_shape=(8, 12, 12))
    (bundle / "model_config.json").write_text(json.dumps({"num_classes": 2, "n_input_channels": 1, "base_channels": 2, "dropout": 0.0}))
    (bundle / "preprocessing.json").write_text(json.dumps(config))
    (bundle / "calibration.json").write_text(json.dumps({"method": "raw", "temperature": 1.0}))
    output_path = run_inference(runtime_data, bundle, bundle / "submission.csv")
    result = pd.read_csv(output_path)
    assert list(result.columns) == ["uid", "is_pathologic"]
    assert result["uid"].tolist() == submission_frame["uid"].tolist()
    assert result["is_pathologic"].between(0, 1).all()


def test_submission_builder_places_main_at_zip_root(tmp_path):
    from build_dat_submission import build

    model_dir = tmp_path / "model_bundle"
    (model_dir / "model").mkdir(parents=True)
    model = build_resnet18_3d(num_classes=2, n_input_channels=1, base_channels=2)
    torch.save(model.state_dict(), model_dir / "weights.pt")
    (model_dir / "model_config.json").write_text(json.dumps({"num_classes": 2, "n_input_channels": 1, "base_channels": 2, "dropout": 0.0}))
    (model_dir / "preprocessing.json").write_text(json.dumps({"target_spacing": [2, 2, 2], "target_shape": [8, 12, 12]}))
    (model_dir / "calibration.json").write_text(json.dumps({"method": "raw", "temperature": 1.0}))
    archive = build(model_dir, tmp_path / "submission.zip")
    import zipfile
    with zipfile.ZipFile(archive) as handle:
        assert "main.py" in handle.namelist()
        assert "model/weights.pt" in handle.namelist()
