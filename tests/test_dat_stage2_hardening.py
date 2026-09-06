from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import dat_final_model
import dat_masking_experiments
from dat_provenance import REPO_ROOT, sha256_file
from dat_select_model import integrity_check, select_candidates


def _masking_args(tmp_path: Path, **overrides):
    values = {
        "cam_cache_dir": str(tmp_path / "cam_cache"),
        "preprocessed_cache_dir": str(tmp_path / "preprocessed"),
        "num_workers": 3,
        "seed": 11,
        "max_train_batches": 0,
        "max_val_batches": 0,
        "debug": False,
        "cam_layer": "layer3.1.conv2",
        "saliency_candidate_percent": 20.0,
        "min_foreground_fraction": 0.60,
    }
    values.update(overrides)
    return Namespace(**values)


def _stage1_config():
    preprocessing = {"target_shape": [4, 4, 4], "target_spacing": [1.0, 1.0, 1.0]}
    return {
        "model": "resnet18_3d", "epochs": 10, "final_training_epochs": 3,
        "patience": 4, "batch_size": 2, "optimizer": "adamw",
        "learning_rate": 1e-3, "weight_decay": 1e-4, "scheduler": "cosine",
        "preprocessing": preprocessing, "preprocessing_fingerprint": "prep-fp",
        "config_fingerprint": "stage1-fp", "fold_assignment_fingerprint": "fold-fp",
        "base_channels": 1, "num_classes": 2, "n_input_channels": 1,
        "dropout": 0.0, "spatial_augmentation": False, "amp": False,
    }


def test_stage2_student_and_teacher_budgets_are_distinct(tmp_path):
    config = _stage1_config()
    args = _masking_args(tmp_path)
    student = dat_masking_experiments._student_expected_config(
        config, 0, "cam_low", 8, 0.10, args, "teacher-sha", "fold-fp"
    )
    assert student["epochs"] == 10
    assert student["student_max_cv_epochs"] == 10
    assert student["final_training_epochs"] == 3
    assert dat_masking_experiments._effective_num_workers("random", args) == 3
    assert dat_masking_experiments._effective_num_workers("cam_low", args) == 0


def test_candidate_selection_persists_recipe_fold_epochs_and_stage2_budget(tmp_path):
    runs = tmp_path / "runs"
    oof_root = tmp_path / "oof"
    for fold, best_epoch in enumerate((5, 7)):
        run_dir = runs / f"fold_{fold}" / "fraction_0.10" / f"resnet18_3d_fold{fold}_random_M8_fraction0.10"
        run_dir.mkdir(parents=True)
        artifact = oof_root / f"fold{fold}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, logits=np.asarray([[0.0, 0.5], [0.5, 0.0]]), targets=np.asarray([0, 1]), fold=np.asarray([fold, fold]))
        frame = pd.DataFrame({
            "epoch": [1, best_epoch], "val_log_loss": [0.9, 0.4 + fold * 0.01],
            "val_accuracy": [0.5, 0.75], "val_auroc": [0.5, 0.8],
            "val_brier_score": [0.3, 0.2], "val_ece": [0.2, 0.1],
        })
        frame.to_csv(run_dir / "metrics.csv", index=False)
        try:
            artifact_value = artifact.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            artifact_value = str(artifact)
        config = {
            **_stage1_config(), "stage": 2, "fold": fold, "condition": "random",
            "cutout_mode": "random", "cutout_m": 8, "cutout_fraction": 0.10,
            "teacher_checkpoint_sha256": None, "selected_stage1_config_fingerprint": "stage1-fp",
            "fold_assignment_fingerprint": "fold-fp", "cam_layer": "layer3.1.conv2",
            "saliency_candidate_percent": 20.0, "min_foreground_fraction": 0.60,
            "student_seed_policy": "base_seed_plus_fold", "student_model": "resnet18_3d",
            "student_max_cv_epochs": 10, "early_stopping_patience": 4,
            "seed": 11 + fold, "num_workers": 3, "requested_num_workers": 3,
            "oof_artifact": artifact_value, "completed": True,
            "checkpoint_selection": "minimum_validation_log_loss", "research_valid": True,
            "best_epoch": best_epoch, "epochs_completed": best_epoch,
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    selected = select_candidates(
        runs, expected_folds=2, output_path=tmp_path / "selected.json",
        conditions=("random",), m_values=(8,), fractions=(0.10,),
        frozen_config=_stage1_config(), summary_dir=tmp_path / "summary",
    )
    best = selected["best_masked"]
    assert best["cam_layer"] == "layer3.1.conv2"
    assert best["saliency_candidate_percent"] == 20.0
    assert best["min_foreground_fraction"] == 0.60
    assert best["selected_candidate_fold_best_epochs"] == [5, 7]
    assert best["final_stage2_training_epochs"] == 6
    assert (tmp_path / "summary" / "candidate_fold_metrics.csv").is_file()


def test_integrity_rejects_candidate_recipe_drift(tmp_path):
    # A complete grid is unnecessary here: the consistency issue itself must
    # be surfaced clearly by the integrity report.
    for fold, layer in enumerate(("layer3.1.conv2", "layer4.1.conv2")):
        run_dir = tmp_path / f"fold_{fold}" / "fraction_0.10" / f"random_{fold}"
        run_dir.mkdir(parents=True)
        artifact = tmp_path / f"oof{fold}.npz"
        np.savez_compressed(artifact, logits=np.zeros((2, 2)), targets=np.asarray([0, 1]), fold=np.asarray([fold, fold]))
        pd.DataFrame({"epoch": [1], "val_log_loss": [0.5], "val_accuracy": [0.5], "val_auroc": [0.5], "val_brier_score": [0.25], "val_ece": [0.1]}).to_csv(run_dir / "metrics.csv", index=False)
        config = {
            "stage": 2, "fold": fold, "condition": "random", "cutout_m": 8, "cutout_fraction": 0.10,
            "cam_layer": layer, "saliency_candidate_percent": 20.0, "min_foreground_fraction": 0.60,
            "student_seed_policy": "base_seed_plus_fold", "student_model": "resnet18_3d",
            "selected_stage1_config_fingerprint": "stage1-fp", "preprocessing_fingerprint": "prep-fp",
            "fold_assignment_fingerprint": "fold-fp", "student_max_cv_epochs": 10, "epochs": 10,
            "early_stopping_patience": 4, "patience": 4, "oof_artifact": str(artifact),
            "completed": True, "checkpoint_selection": "minimum_validation_log_loss",
            "research_valid": True, "best_epoch": 1, "epochs_completed": 1,
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    report = integrity_check(tmp_path, expected_folds=2, conditions=("random",), m_values=(8,), fractions=(0.10,))
    assert not report["passed"]
    assert any(issue["field"] == "cam_layer" for issue in report["candidate_recipe_issues"])


def test_teacher_resume_validates_lineage_and_freezes_device(monkeypatch, tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 2)

        def forward(self, x):
            return self.fc(x)

    calls = {"fit": 0}

    def fake_fit(dataset, config, *, seed, run_dir, epochs, max_train_batches):
        calls["fit"] += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        model = TinyModel()
        torch.save({"model_state_dict": model.state_dict()}, run_dir / "final_model.pt")
        persisted = {**config, "completed": True, "checkpoint_selection": "final_scheduled_epoch", "final_epoch": epochs}
        (run_dir / "config.json").write_text(json.dumps(persisted), encoding="utf-8")
        return {"model": model}

    monkeypatch.setattr(dat_masking_experiments, "build_dat_model", lambda config: TinyModel())
    monkeypatch.setattr(dat_masking_experiments, "fit_dat_model_fixed_epochs", fake_fit)
    monkeypatch.setattr(dat_masking_experiments, "DatDataset", lambda *args, **kwargs: object())
    args = _masking_args(tmp_path, num_workers=0)
    config = _stage1_config()
    first, checkpoint, first_sha = dat_masking_experiments._train_teacher(
        ["record"], [0], config["preprocessing"], config, 0, args, fold_hash="fold-fp"
    )
    assert calls["fit"] == 1
    assert first.training is False
    assert all(not parameter.requires_grad for parameter in first.parameters())
    second, _, second_sha = dat_masking_experiments._train_teacher(
        ["record"], [0], config["preprocessing"], config, 0, args, fold_hash="fold-fp"
    )
    assert calls["fit"] == 1
    assert first_sha == second_sha == sha256_file(checkpoint)
    changed_fingerprint = {**config, "config_fingerprint": "different"}
    dat_masking_experiments._train_teacher(["record"], [0], changed_fingerprint["preprocessing"], changed_fingerprint, 0, args, fold_hash="fold-fp")
    assert calls["fit"] == 2
    changed_epochs = {**config, "final_training_epochs": 4}
    dat_masking_experiments._train_teacher(["record"], [0], changed_epochs["preprocessing"], changed_epochs, 0, args, fold_hash="fold-fp")
    assert calls["fit"] == 3


def test_final_cam_uses_exact_stage1_checkpoint_and_selected_recipe(monkeypatch, tmp_path):
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.zeros(1), torch.tensor(0)

    captured = {}
    fit_epochs = []
    stage1_checkpoint = tmp_path / "stage1" / "final_model.pt"
    stage1_checkpoint.parent.mkdir()
    stage1_checkpoint.write_bytes(b"exact-stage1-final-checkpoint")
    stage1_sha = sha256_file(stage1_checkpoint)

    def fake_cutout(**kwargs):
        captured.update(kwargs)
        return TinyDataset()

    def fake_fit(dataset, config, *, seed, run_dir, epochs, max_train_batches):
        fit_epochs.append(epochs)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final_model.pt").write_bytes(b"final-student")
        (run_dir / "config.json").write_text(json.dumps({**config, "completed": True}), encoding="utf-8")
        return {"model": None, "epochs_completed": epochs}

    monkeypatch.setattr(dat_final_model, "load_dat_records", lambda data_dir: ["record"])
    monkeypatch.setattr(dat_final_model, "DatDataset", lambda *args, **kwargs: TinyDataset())
    monkeypatch.setattr(dat_final_model, "CutoutAugmentedDataset", fake_cutout)
    monkeypatch.setattr(dat_final_model, "fit_dat_model_fixed_epochs", fake_fit)
    monkeypatch.setattr(dat_final_model, "_load_stage1_teacher", lambda directory, config: (nn.Identity(), stage1_checkpoint, stage1_sha))
    config_path = tmp_path / "best_config.json"
    config_path.write_text(json.dumps(_stage1_config()), encoding="utf-8")
    selected = {
        "condition": "cam_low", "M": 8, "fraction": 0.10,
        "cam_layer": "layer3.1.conv2", "saliency_candidate_percent": 20.0,
        "min_foreground_fraction": 0.60, "student_max_cv_epochs": 10,
        "student_seed_policy": "base_seed_plus_fold", "student_model": "resnet18_3d",
        "selected_stage1_config_fingerprint": "stage1-fp", "fold_assignment_fingerprint": "fold-fp",
        "selected_candidate_fold_best_epochs": [5, 6, 7],
        "final_stage2_training_epoch_rule": "median_selected_stage2_fold_best_epoch_round_half_up",
        "final_stage2_training_epochs": 6,
        "calibration_provenance": "candidate_own_fold_OOF_logits_only",
        "calibration": {"method": "temperature", "temperature": 1.2},
    }
    result = dat_final_model.train_final_dat_model(
        tmp_path, config_path, tmp_path / "final_stage2", selected=selected,
        stage1_model_dir=tmp_path / "stage1", calibration_payload=selected["calibration"],
    )
    assert fit_epochs == [6]
    assert captured["cam_layer"] == "layer3.1.conv2"
    assert captured["saliency_candidate_percent"] == 20.0
    assert captured["min_foreground_fraction"] == 0.60
    assert result["provenance"]["stage1_teacher_checkpoint_sha256"] == stage1_sha
    assert result["provenance"]["final_stage2_training_epochs"] == 6
    assert result["model_config"]["final_stage2_training_epochs"] == 6
    assert not (tmp_path / "final_stage2" / "final_teacher").exists()
