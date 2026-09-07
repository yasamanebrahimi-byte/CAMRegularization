from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import dat_calibration
from build_dat_submission import validate_submission_csv
from dat_calibration import cross_fitted_calibration
from dat_select_model import select_best_overall_and_masked
from dat_training import evaluate_dat_model
from dat_tune import (
    _load_resumable_stage1_trial,
    evaluate_stage1_oof,
    generate_unique_trial_values,
    select_best_stage1_trial,
)


def test_stage1_selection_uses_calibrated_oof_score_over_raw_score():
    targets = np.asarray([0, 1] * 6, dtype=np.int64)
    fold_ids = np.repeat(np.arange(3), 4)
    logits_a = np.asarray([[4.081838242770365, -5.1113300626283635], [0.8361976934515577, -1.1355392122558596], [-0.9052985842208917, -0.4311943261795318], [-4.039972258294502, -0.46386475528837895], [-1.7304261525498834, 6.645999033289765], [0.4515732264558435, -0.7052615886831908], [-0.5625748363027008, -1.3360926922179002], [-2.1103011024102427, -0.7816019544693095], [0.9638907770135717, -0.4771072131467334], [1.9155174059195281, -0.39960425813316], [0.048519130153329246, 3.091641702425624], [1.0902110453752891, -1.010457471228036]])
    logits_b = np.asarray([[-0.3656779491954698, 1.0810502635096042], [3.8701760681977055, -0.539240654683827], [-0.4871173581582091, 2.0046272025513825], [-1.7729198863211741, -0.583440464879728], [1.7650779349129677, 1.1607000323817982], [0.18303340656470438, 1.3402087096569588], [-5.656324613687525, 2.04261363500016], [-1.9192895196162834, -3.337239685311939], [0.5528915190419993, 1.4010897706987802], [-0.8895349113655682, -2.1528116802016153], [0.052249667068067246, -0.10549461648575854], [2.811196332036185, 1.4948159749587009]])
    evaluated_a = evaluate_stage1_oof(logits_a, targets, fold_ids)
    evaluated_b = evaluate_stage1_oof(logits_b, targets, fold_ids)
    # This synthetic pair intentionally has the required ranking conflict:
    # raw loss favors A, while the leakage-safe calibrated OOF estimate favors B.
    assert evaluated_a["raw_metrics"]["log_loss"] < evaluated_b["raw_metrics"]["log_loss"]
    assert evaluated_a["cross_fitted_metrics"]["log_loss"] > evaluated_b["cross_fitted_metrics"]["log_loss"]
    rows = [
        {"trial_id": 0, "raw_oof_log_loss": evaluated_a["raw_metrics"]["log_loss"], "cross_fitted_calibrated_oof_log_loss": evaluated_a["cross_fitted_metrics"]["log_loss"]},
        {"trial_id": 1, "raw_oof_log_loss": evaluated_b["raw_metrics"]["log_loss"], "cross_fitted_calibrated_oof_log_loss": evaluated_b["cross_fitted_metrics"]["log_loss"]},
    ]
    assert select_best_stage1_trial(rows)["trial_id"] == 1
    assert "final_all_oof_metrics" in evaluated_b


def test_cross_fitted_calibration_excludes_validation_observations(monkeypatch):
    logits = np.asarray([[-3.0, 3.0], [-2.0, 2.0], [2.0, -2.0], [3.0, -3.0]])
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    folds = [[0, 1], [2, 3]]
    original = dat_calibration.fit_calibration
    calls = []

    def spy(train_logits, train_targets, method="temperature"):
        calls.append((np.asarray(train_logits).copy(), np.asarray(train_targets).copy()))
        return original(train_logits, train_targets, method=method)

    monkeypatch.setattr(dat_calibration, "fit_calibration", spy)
    cross_fitted_calibration(logits, targets, folds, method="temperature")
    assert len(calls) == 2
    assert np.array_equal(calls[0][0], logits[[2, 3]])
    assert np.array_equal(calls[1][0], logits[[0, 1]])


def test_unique_trial_generation_is_deterministic_and_excludes_reserved():
    space = {"learning_rate": [1e-3, 1e-3, 1e-4], "optimizer": ["adamw", "sgd"]}
    first, count = generate_unique_trial_values(space, 4, 7)
    second, second_count = generate_unique_trial_values(space, 4, 7)
    assert count == second_count == 4
    assert first == second
    assert len({json.dumps(value, sort_keys=True) for value in first.values()}) == 4
    remaining, remaining_count = generate_unique_trial_values(
        space, 3, 7, reserved_values=[first[0], first[1]]
    )
    assert remaining_count == 2
    assert not ({json.dumps(value, sort_keys=True) for value in remaining.values()} & {
        json.dumps(first[0], sort_keys=True), json.dumps(first[1], sort_keys=True)
    })


def test_resume_reconstructs_historical_fold_membership_and_scores_calibrated_metric(tmp_path):
    trial_dir = tmp_path / "trial_000_old"
    trial_dir.mkdir()
    logits = np.asarray([[-1.0, 1.0], [1.0, -1.0], [-0.5, 0.5], [0.5, -0.5]])
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    current_fold_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    # Historical artifacts omitted fold_ids; reconstruction is allowed because
    # the current target order and fold partition are explicitly supplied.
    np.savez_compressed(trial_dir / "oof_logits.npz", logits=logits, targets=targets)
    loaded = _load_resumable_stage1_trial(
        trial_dir, {"calibration": "raw"}, targets, current_fold_ids, calibration_method="raw"
    )
    assert np.array_equal(loaded[2], current_fold_ids)
    assert loaded[3]["selection_score"] == loaded[3]["cross_fitted_metrics"]["log_loss"]


def test_evaluation_loss_is_sample_weighted_on_uneven_batches():
    class FixedModel(nn.Module):
        def forward(self, images):
            return images[:, :2]

    logits = torch.tensor([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    metrics, _, _ = evaluate_dat_model(FixedModel(), loader, criterion, torch.device("cpu"))
    expected = float(nn.CrossEntropyLoss()(logits, targets).item())
    assert metrics["loss"] == pytest.approx(expected)


def test_submission_validation_preserves_nonalphabetical_template_order(tmp_path):
    template = tmp_path / "submission_format.csv"
    output = tmp_path / "submission.csv"
    pd.DataFrame({"uid": ["uid_b", "uid_a", "uid_c"], "is_pathologic": ["", "", ""]}).to_csv(template, index=False)
    pd.DataFrame({"uid": ["uid_b", "uid_a", "uid_c"], "is_pathologic": [0.2, 0.8, 0.5]}).to_csv(output, index=False)
    result = validate_submission_csv(output, template)
    assert result["uids"] == ["uid_b", "uid_a", "uid_c"]
    pd.DataFrame({"uid": ["uid_a", "uid_b", "uid_c"], "is_pathologic": [0.2, 0.8, 0.5]}).to_csv(output, index=False)
    with pytest.raises(ValueError, match="order"):
        validate_submission_csv(output, template)


def test_stage2_keeps_best_overall_and_best_masked_distinct():
    candidates = [
        {"condition": "none", "M": 0, "fraction": 0.0, "selection_score": 0.20},
        {"condition": "random", "M": 4, "fraction": 0.10, "selection_score": 0.30},
        {"condition": "cam_low", "M": 8, "fraction": 0.20, "selection_score": 0.25},
    ]
    overall, masked = select_best_overall_and_masked(candidates)
    assert overall["condition"] == "none"
    assert masked["condition"] == "cam_low"
