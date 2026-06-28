import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import cutout
from cam_masking import compute_saliency_map
from cutout import CutoutAugmentedDataset


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 2, bias=False)

        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 1, 1] = 1.0
            self.conv.weight[0, 1, 0, 1] = 0.5
            self.conv.weight[1, 2, 1, 1] = 1.0
            self.conv.weight[1, 0, 1, 0] = 0.25
            self.fc.weight.copy_(torch.tensor([[1.0, 0.2], [0.2, 1.0]]))

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TinyImageDataset(Dataset):
    def __init__(self):
        base = torch.linspace(0.05, 1.0, steps=16 * 16, dtype=torch.float32).reshape(1, 16, 16)
        self.images = [base.repeat(3, 1, 1), torch.flip(base, dims=[1]).repeat(3, 1, 1)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].clone(), index % 2


def _build_teacher():
    teacher = ToyTeacher().eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _assert_valid_saliency():
    teacher = _build_teacher()
    image, _ = TinyImageDataset()[0]
    saliency = compute_saliency_map(teacher, image, cam_layer="conv")
    assert saliency.ndim == 2, saliency.shape
    assert saliency.shape == image.shape[-2:], saliency.shape
    assert torch.isfinite(saliency).all()
    assert float(saliency.min()) >= -1e-6
    assert float(saliency.max()) <= 1.0 + 1e-6
    assert float(saliency.max() - saliency.min()) > 1e-6


def _dataset(mode, teacher, cam_layer="conv", cutout_m=1, cam_cache_dir=None, cam_cache_settings=None):
    return CutoutAugmentedDataset(
        base_dataset=TinyImageDataset(),
        cutout_mode=mode,
        cutout_m=cutout_m,
        cutout_size=4,
        cutout_area=None,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        seed=123,
        teacher_model=teacher,
        cam_layer=cam_layer,
        cam_cache_dir=cam_cache_dir,
        cam_cache_settings=cam_cache_settings,
        debug_log_limit=0,
    )


def _assert_cam_mode_raises_without_random_fallback(mode):
    ds = _dataset(mode, _build_teacher(), cam_layer="missing_layer")
    try:
        ds[1]
    except RuntimeError as exc:
        message = str(exc)
        assert "CAM cutout failed" in message
        assert "dataset index 0" in message
        assert f"cutout_mode={mode}" in message
        assert "missing_layer" in message
    else:
        raise AssertionError(f"{mode} did not raise when CAM layer was broken")


def _assert_random_still_works():
    ds = _dataset("random", teacher=None, cam_layer="missing_layer")
    image, target = ds[1]
    assert image.shape == (3, 16, 16)
    assert target == 0



def _load_cpu_tensor(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _assert_saliency_cache_reuses_cpu_pt():
    settings = {
        "dataset": "tiny",
        "grayscale": False,
        "student_model": "toy",
        "teacher_model": "toy",
        "teacher_checkpoint": {"path": "toy.pt", "sha256": "toy", "mtime_ns": 0},
        "cam_layer": "conv",
        "input_size": 16,
    }
    original_compute = cutout.compute_saliency_map
    calls = {"count": 0}

    def counted_compute(*args, **kwargs):
        calls["count"] += 1
        return original_compute(*args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            cutout.compute_saliency_map = counted_compute
            ds = _dataset(
                "cam_high",
                _build_teacher(),
                cutout_m=2,
                cam_cache_dir=tmpdir,
                cam_cache_settings=settings,
            )
            image, target = ds[1]
            image2, target2 = ds[2]
            assert image.shape == (3, 16, 16)
            assert image2.shape == (3, 16, 16)
            assert target == 0
            assert target2 == 0
            assert calls["count"] == 1, calls

            cache_files = sorted(Path(tmpdir).rglob("*.pt"))
            assert len(cache_files) == 1, cache_files
            saliency = _load_cpu_tensor(cache_files[0])
            assert torch.is_tensor(saliency)
            assert saliency.device.type == "cpu"
            assert saliency.shape == (16, 16)

            def fail_compute(*_args, **_kwargs):
                raise AssertionError("cached CAM path should not recompute saliency")

            cutout.compute_saliency_map = fail_compute
            cached_ds = _dataset(
                "cam_high",
                teacher=None,
                cutout_m=2,
                cam_cache_dir=tmpdir,
                cam_cache_settings=settings,
            )
            cached_image, cached_target = cached_ds[1]
            assert cached_image.shape == (3, 16, 16)
            assert cached_target == 0
        finally:
            cutout.compute_saliency_map = original_compute


def _assert_window_cache_reuses_coordinates():
    settings = {
        "dataset": "tiny",
        "grayscale": False,
        "student_model": "toy",
        "teacher_model": "toy",
        "teacher_checkpoint": {"path": "toy.pt", "sha256": "toy", "mtime_ns": 0},
        "cam_layer": "conv",
        "input_size": 16,
    }
    original_compute = cutout.compute_saliency_map
    original_select = cutout._select_cam_window

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            ds = _dataset(
                "cam_low",
                _build_teacher(),
                cutout_m=2,
                cam_cache_dir=tmpdir,
                cam_cache_settings=settings,
            )
            first_image, _ = ds[1]
            second_image, _ = ds[2]
            assert first_image.shape == (3, 16, 16)
            assert second_image.shape == (3, 16, 16)
            assert sorted((Path(tmpdir) / "windows").rglob("*.json"))

            def fail_compute(*_args, **_kwargs):
                raise AssertionError("window cache should avoid saliency loading")

            def fail_select(*_args, **_kwargs):
                raise AssertionError("window cache should avoid CAM window selection")

            cutout.compute_saliency_map = fail_compute
            cutout._select_cam_window = fail_select
            cached_ds = _dataset(
                "cam_low",
                teacher=None,
                cutout_m=2,
                cam_cache_dir=tmpdir,
                cam_cache_settings=settings,
            )
            cached_image, cached_target = cached_ds[1]
            cached_image2, cached_target2 = cached_ds[2]
            assert cached_image.shape == (3, 16, 16)
            assert cached_image2.shape == (3, 16, 16)
            assert cached_target == 0
            assert cached_target2 == 0
        finally:
            cutout.compute_saliency_map = original_compute
            cutout._select_cam_window = original_select


def main():
    _assert_valid_saliency()
    _assert_saliency_cache_reuses_cpu_pt()
    _assert_window_cache_reuses_coordinates()
    for mode in ("cam_low", "cam_high"):
        valid_ds = _dataset(mode, _build_teacher())
        image, target = valid_ds[1]
        assert image.shape == (3, 16, 16)
        assert target == 0
        _assert_cam_mode_raises_without_random_fallback(mode)
    _assert_random_still_works()
    print("CAM cutout validation passed.")


if __name__ == "__main__":
    main()
