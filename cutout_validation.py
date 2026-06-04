import sys

import torch

from IOutils import build_parser
from dataset_registry import (
    get_dataset_loaders,
    get_normalization_params,
    get_default_input_size,
    get_num_classes,
    infer_num_classes_from_loader,
)
from cutout import CutoutAugmentedDataset
from train import _load_teacher_model
from utils import infer_input_size_from_loader


def _is_square_cutout(original: torch.Tensor, augmented: torch.Tensor, mean, std) -> bool:
    if original.shape != augmented.shape:
        return False

    channels = int(original.shape[0])
    black = torch.tensor(mean, dtype=original.dtype).clone()
    if black.numel() == 1:
        black = black.repeat(channels)
    if black.numel() != channels:
        black = black[:channels]
    std_v = torch.tensor(std, dtype=original.dtype)
    if std_v.numel() == 1:
        std_v = std_v.repeat(channels)
    if std_v.numel() != channels:
        std_v = std_v[:channels]
    black = (0.0 - black) / std_v

    black = black[:, None, None]
    black_mask = (augmented == black).all(dim=0)
    diff_mask = (augmented != original).any(dim=0)
    mask = black_mask & diff_mask
    if mask.sum() == 0:
        return False

    rows = torch.nonzero(mask.any(dim=1)).flatten()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return False

    top = int(rows.min().item())
    bottom = int(rows.max().item())
    left = int(cols.min().item())
    right = int(cols.max().item())
    height = bottom - top + 1
    width = right - left + 1
    if height != width:
        return False

    box = mask[top:bottom + 1, left:right + 1]
    fill_ratio = float(box.float().mean().item())
    return fill_ratio >= 0.8


def _build_wrapper(base_dataset, args, mean, std, cutout_m, cutout_mode, teacher_model):
    return CutoutAugmentedDataset(
        base_dataset=base_dataset,
        cutout_mode=cutout_mode,
        cutout_m=cutout_m,
        cutout_size=args.cutout_size if args.cutout_size > 0 else None,
        cutout_area=args.cutout_area,
        mean=mean,
        std=std,
        seed=args.seed,
        saliency_candidate_percent=args.saliency_candidate_percent,
        teacher_model=teacher_model,
        cam_layer=args.cam_layer,
    )


def main():
    args = build_parser().parse_args()
    train_dl, val_dl, test_dl = get_dataset_loaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        grayscale=args.grayscale,
        include_regex=args.include_regex,
    )

    base_len = len(train_dl.dataset)
    mean, std = get_normalization_params(args.dataset)
    num_classes = infer_num_classes_from_loader(train_dl)
    if num_classes is None:
        num_classes = get_num_classes(args.dataset)
    input_size = infer_input_size_from_loader(train_dl, get_default_input_size(args.dataset))

    wrapper_none = _build_wrapper(train_dl.dataset, args, mean, std, 0, "none", None)
    if len(wrapper_none) != base_len:
        raise AssertionError(f"Expected no-cutout length {base_len}, got {len(wrapper_none)}")

    wrapper_4 = _build_wrapper(train_dl.dataset, args, mean, std, 4, "random", None)
    if len(wrapper_4) != base_len * 5:
        raise AssertionError("cutout_m=4 length check failed")

    wrapper_8 = _build_wrapper(train_dl.dataset, args, mean, std, 8, "random", None)
    if len(wrapper_8) != base_len * 9:
        raise AssertionError("cutout_m=8 length check failed")

    if val_dl is not None and isinstance(val_dl.dataset, CutoutAugmentedDataset):
        raise AssertionError("Validation dataset should not be augmented")
    if isinstance(test_dl.dataset, CutoutAugmentedDataset):
        raise AssertionError("Test dataset should not be augmented")

    teacher_model = None
    cutout_mode = str(args.cutout_mode or "none").lower()
    if cutout_mode in {"cam_low", "cam_high"}:
        teacher_model = _load_teacher_model(args, num_classes=num_classes, input_size=input_size, logger=None)

    shape_wrapper = _build_wrapper(
        train_dl.dataset,
        args,
        mean,
        std,
        cutout_m=1,
        cutout_mode=cutout_mode if cutout_mode != "none" else "random",
        teacher_model=teacher_model,
    )
    original, _ = shape_wrapper[0]
    augmented, _ = shape_wrapper[1]
    if not _is_square_cutout(original, augmented, mean, std):
        raise AssertionError(f"Square cutout check failed for mode '{cutout_mode}'.")

    print("Cutout validation checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Cutout validation failed: {exc}")
        sys.exit(1)
