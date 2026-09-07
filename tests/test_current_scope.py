from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import dat_stage2_summary
from IOutils import build_parser
from dataset_registry import DATASET_REGISTRY, get_available_datasets, get_dataset_metadata
from model_registry import get_available_models, get_model


def test_dataset_registry_contains_only_current_scope():
    assert get_available_datasets() == ["cifar100", "drive_zip", "dat_parkinsons"]
    for name in get_available_datasets():
        assert callable(DATASET_REGISTRY[name]["loader"])
        assert get_dataset_metadata(name)["num_classes"] > 0


@pytest.mark.parametrize(
    "name",
    [
        "malimg",
        "big2015",
        "malware_classification",
        "tiny_imagenet",
        "cub200",
        "imagenette",
        "cifar100_c",
    ],
)
def test_removed_dataset_names_are_rejected(name):
    with pytest.raises(ValueError, match="not found"):
        get_dataset_metadata(name)


def test_current_model_registry_builds_only_current_models():
    assert get_available_models() == ["resnet18", "resnet18_3d"]
    model_2d = get_model("resnet18", num_classes=3, input_size=32)
    model_3d = get_model("resnet18_3d", num_classes=2, base_channels=2)
    model_2d.eval()
    model_3d.eval()
    with torch.no_grad():
        assert model_2d(torch.rand(1, 3, 32, 32)).shape == (1, 3)
        assert model_3d(torch.rand(1, 1, 8, 16, 16)).shape == (1, 2)


@pytest.mark.parametrize(
    "name",
    [
        "resnet34",
        "resnet50",
        "vgg16_bn",
        "densenet121",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "vit_b_16",
        "convnext_tiny",
        "swin_t",
    ],
)
def test_removed_model_names_are_rejected(name):
    with pytest.raises(ValueError, match="not found"):
        get_model(name, num_classes=2)


def test_training_cli_uses_reduced_registry_scope():
    parser = build_parser()
    assert parser.parse_args(["--dataset", "drive_zip", "--model", "resnet18"]).dataset == "drive_zip"
    assert parser.parse_args(["--dataset", "dat_parkinsons", "--model", "resnet18_3d"]).model == "resnet18_3d"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "malimg"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "densenet121"])


def test_dat_summary_cli_delegates_to_authoritative_generator(monkeypatch):
    wrapper_path = Path(__file__).parents[1] / "runs" / "dat_parkinsons" / "summary" / "generate_summary.py"
    spec = importlib.util.spec_from_file_location("dat_summary_wrapper", wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wrapper)

    calls = {}

    def fake_generate_summary(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(dat_stage2_summary, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_summary.py", "--run_root", "runs/dat_parkinsons/resnet18_3d", "--expected_folds", "3"],
    )

    wrapper.main()

    assert calls["args"] == ("runs/dat_parkinsons/resnet18_3d", wrapper.SUMMARY_DIR)
    assert calls["kwargs"] == {"expected_folds": 3, "frozen_config": None}
