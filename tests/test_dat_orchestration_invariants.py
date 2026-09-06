from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch import nn

from dat_calibration import fit_candidate_calibration
from dat_masking_experiments import _run_is_valid, expected_grid
from dat_select_model import integrity_check


def test_default_stage2_grid_has_one_baseline_and_24_masked_cells_per_fold():
    cells = expected_grid(5)
    assert len(cells) == 125
    for fold in range(5):
        fold_cells = [cell for cell in cells if cell["fold"] == fold]
        assert sum(cell["condition"] == "none" for cell in fold_cells) == 1
        assert sum(cell["condition"] != "none" for cell in fold_cells) == 24


def test_candidate_calibration_is_independent_and_cross_fitted():
    targets = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    confident = np.asarray([[-6, 6] if target else [6, -6] for target in targets], dtype=float)
    uncertain = confident / 8.0
    folds = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    first = fit_candidate_calibration(confident, targets, folds)
    second = fit_candidate_calibration(uncertain, targets, folds)
    assert first["final_calibration"]["method"] == "temperature"
    assert second["final_calibration"]["method"] == "temperature"
    assert first["final_calibration"]["temperature"] != second["final_calibration"]["temperature"]
    assert first["final_calibration"]["provenance"] == "candidate_oof_logits_only"
    assert first["final_calibration"]["n_cv_folds"] == 4


def test_fixed_final_fit_has_no_validation_and_uses_every_scheduled_epoch(monkeypatch, tmp_path: Path):
    import dat_training

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            return self.fc(x.flatten(1))

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.full((1, 2, 2, 2), float(index)), torch.tensor(index % 2)

    monkeypatch.setattr(dat_training, "build_dat_model", lambda config: TinyModel())
    result = dat_training.fit_dat_model_fixed_epochs(
        TinyDataset(), {"epochs": 3, "batch_size": 2, "optimizer": "adamw", "learning_rate": 1e-3, "scheduler": "cosine"},
        seed=4, run_dir=tmp_path / "final",
    )
    assert result["final_epoch"] == 3
    assert result["epochs_completed"] == 3
    assert all(np.isnan(row["val_log_loss"]) for row in result["metrics_rows"])
    assert (tmp_path / "final" / "final_model.pt").is_file()


def test_integrity_checker_rejects_missing_exact_grid_cell(tmp_path: Path):
    report = integrity_check(tmp_path, expected_folds=2)
    assert report["passed"] is False
    assert report["expected_cell_count"] == 50
    assert report["missing_cells"]


def test_resume_validation_accepts_only_complete_finite_cells(tmp_path: Path):
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    artifact = tmp_path / "oof.npz"
    np.savez_compressed(artifact, logits=np.zeros((2, 2)), targets=np.asarray([0, 1]))
    config = {
        "stage": 2, "fold": 0, "condition": "random", "cutout_m": 4, "cutout_fraction": 0.1,
        "oof_artifact": str(artifact), "completed": True, "checkpoint_selection": "minimum_validation_log_loss",
        "research_valid": True, "best_epoch": 1, "epochs_completed": 1,
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    pd.DataFrame({"epoch": [1], "val_log_loss": [0.5]}).to_csv(run_dir / "metrics.csv", index=False)
    assert _run_is_valid(run_dir)[0]
    pd.DataFrame({"epoch": [1], "val_log_loss": [np.nan]}).to_csv(run_dir / "metrics.csv", index=False)
    assert not _run_is_valid(run_dir)[0]


def test_integrity_checker_does_not_count_teacher_directories(tmp_path: Path):
    teacher = tmp_path / "teachers" / "fold_0"
    teacher.mkdir(parents=True)
    (teacher / "config.json").write_text(json.dumps({"stage": "stage2_teacher", "condition": "none"}))
    report = integrity_check(tmp_path, expected_folds=1)
    assert report["discovered_cell_count"] == 0


def _tiny_dataset(root: Path) -> Path:
    niftis = root / "niftis"
    niftis.mkdir(parents=True)
    rows = []
    for index in range(4):
        array = np.zeros((8, 8, 8), dtype=np.float32)
        array[2:6, 2:6, 2:6] = index + 1
        uid = f"tiny_{index}"
        nib.save(nib.Nifti1Image(array, np.eye(4)), niftis / f"{uid}.nii.gz")
        rows.append({"uid": uid, "is_pathologic": index % 2})
    pd.DataFrame(rows).to_csv(root / "train_labels.csv", index=False)
    return root


def test_two_command_orchestration_keeps_independent_stage_assets(tmp_path: Path):
    from argparse import Namespace
    from run_dat_stage1 import run as run_stage1
    from run_dat_stage2 import run as run_stage2

    data = _tiny_dataset(tmp_path / "data")
    optimization = tmp_path / "optimization"
    stage1 = Namespace(
        data_dir=str(data), output_dir=str(optimization), research_output_dir=str(tmp_path / "stage1_research"),
        final_model_dir=str(tmp_path / "final_stage1_unmasked"), submission_zip=str(tmp_path / "dat_stage1_unmasked.zip"),
        cv_folds=2, fold_scheme="stratified", trials=1, seed=11, epochs=1, patience=0, batch_size=2,
        num_workers=0, target_spacing=[1.0, 1.0, 1.0], target_shape=[8, 8, 8],
        intensity_lower_percentile=1.0, intensity_upper_percentile=99.0, foreground_threshold=0.0,
        crop_margin_mm=1.0, calibration="temperature", search_space_json="", max_train_batches=0,
        max_val_batches=0, base_channels=1, amp=False, augmentation="none",
    )
    first = run_stage1(stage1)
    assert Path(first["final_model_dir"], "provenance.json").is_file()
    assert Path(first["submission_zip"]).is_file()
    stage2 = Namespace(
        data_dir=str(data), best_config=str(optimization / "best_config.json"),
        output_dir=str(tmp_path / "stage2_runs"), fold_assignments=str(optimization / "fold_assignments.json"), fold=-1,
        conditions=["none", "random"], m_values=[1], fractions=[0.10], seed=11, num_workers=0,
        max_train_batches=0, max_val_batches=0, debug=False, cam_layer="auto", saliency_candidate_percent=10.0,
        min_foreground_fraction=0.25, cam_cache_dir=str(tmp_path / "cam_cache"),
        preprocessed_cache_dir=str(tmp_path / "preprocessed"), selected_model=str(tmp_path / "selected.json"),
        summary_dir=str(tmp_path / "summary"), final_model_dir=str(tmp_path / "final_stage2_masked"),
        stage1_model_dir=str(tmp_path / "final_stage1_unmasked"), submission_zip=str(tmp_path / "dat_stage2_masked.zip"),
        calibration_method="temperature",
    )
    second = run_stage2(stage2)
    assert Path(second["final_model_dir"], "provenance.json").is_file()
    assert Path(second["submission_zip"]).is_file()
    assert Path(first["submission_zip"]).read_bytes()
    assert second["best_masked"]["condition"] == "random"
    runtime = tmp_path / "runtime"
    (runtime / "niftis").mkdir(parents=True)
    frame = pd.DataFrame({"uid": ["tiny_3", "tiny_0"], "is_pathologic": [np.nan, np.nan]})
    frame.to_csv(runtime / "submission_format.csv", index=False)
    for uid in frame["uid"]:
        shutil.copy2(data / "niftis" / f"{uid}.nii.gz", runtime / "niftis" / f"{uid}.nii.gz")
    from main import run_inference
    for name, archive in (("stage1", first["submission_zip"]), ("stage2", second["submission_zip"])):
        unpacked = tmp_path / f"unpacked_{name}"
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(unpacked)
        result = pd.read_csv(run_inference(runtime, unpacked, unpacked / "submission.csv"))
        assert list(result.columns) == ["uid", "is_pathologic"]
        assert result["uid"].tolist() == frame["uid"].tolist()
        assert result["is_pathologic"].between(0, 1).all()
