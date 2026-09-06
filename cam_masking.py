"""Dimension-aware HiResCAM for the existing 2D and new 3D workflows."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


def _get_submodule(model: nn.Module, module_name: str) -> nn.Module:
    if hasattr(model, "get_submodule"):
        return model.get_submodule(module_name)
    current: nn.Module = model
    for token in module_name.split("."):
        current = current[int(token)] if token.isdigit() else getattr(current, token)
    return current


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_cam_target_module(model: nn.Module, cam_layer: str = "auto"):
    requested = (cam_layer or "auto").strip()
    if requested.lower() != "auto":
        try:
            return requested, _get_submodule(model, requested)
        except Exception as exc:
            raise ValueError(f"Could not find CAM layer '{requested}' in model '{type(model).__name__}'.") from exc

    conv_candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, (nn.Conv2d, nn.Conv3d))
    ]
    if conv_candidates:
        conv3d_candidates = [module for _name, module in conv_candidates if isinstance(module, nn.Conv3d)]
        if conv3d_candidates:
            # Keep a useful spatial grid for 3D ResNet-style models: the last
            # block can be nearly 1x1x1 after downsampling on small volumes.
            # Prefer the final convolution in the penultimate residual stage.
            preferred_names = (
                "layer3.1.conv2", "layer3.0.conv2", "layer2.1.conv2",
                "layer2.0.conv2", "conv1",
            )
            by_name = dict(conv_candidates)
            for name in preferred_names:
                module = by_name.get(name)
                if isinstance(module, nn.Conv3d):
                    return name, module
            return conv_candidates[-1]
        return conv_candidates[-1]

    feature_candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and not isinstance(module, nn.Sequential)
    ]
    if feature_candidates:
        return feature_candidates[-1]
    raise ValueError(f"Unable to automatically select CAM layer for model '{type(model).__name__}'.")


class HiResCAM:
    """Element-wise gradient * activation, summed over channels."""

    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._hook = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inp, out):
        if not torch.is_tensor(out):
            self.activations = None
            self.gradients = None
            return
        self.activations = out
        self.gradients = None
        if out.requires_grad:
            out.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        self.gradients = grad

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        if cam.ndim not in (4, 5) or cam.size(1) != 1:
            raise RuntimeError(f"HiResCAM produced invalid shape {tuple(cam.shape)}; expected [B,1,*spatial].")
        if cam.numel() == 0:
            raise RuntimeError("HiResCAM produced an empty saliency map.")
        if not torch.isfinite(cam).all():
            raise RuntimeError("HiResCAM produced NaN or Inf values.")
        spatial_dims = tuple(range(2, cam.ndim))
        cam_min = cam.amin(dim=spatial_dims, keepdim=True)
        cam_max = cam.amax(dim=spatial_dims, keepdim=True)
        denom = cam_max - cam_min
        if torch.any(denom <= 1e-12):
            raise RuntimeError("HiResCAM saliency map has no dynamic range.")
        return (cam - cam_min) / denom

    def cam(self, x: torch.Tensor, target_class=None) -> torch.Tensor:
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            acts = self.activations
            if acts is None:
                raise RuntimeError("HiResCAM activations are None. Hook not firing?")
            if acts.ndim not in (4, 5):
                raise RuntimeError(f"HiResCAM activations have invalid shape {tuple(acts.shape)}; expected 2D or 3D feature maps.")
            if not acts.requires_grad:
                raise RuntimeError("HiResCAM activations do not require gradients. Check grad/no_grad context.")
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise RuntimeError(f"HiResCAM expected model logits with shape [B, classes], got {type(logits).__name__}.")

            indices = torch.arange(x.size(0), device=x.device)
            if target_class is None:
                predictions = logits.argmax(dim=1)
            elif isinstance(target_class, int):
                predictions = torch.full((x.size(0),), int(target_class), device=x.device, dtype=torch.long)
            else:
                predictions = torch.as_tensor(target_class, device=x.device, dtype=torch.long)
                if predictions.ndim == 0:
                    predictions = predictions.repeat(x.size(0))
                predictions = predictions.reshape(-1)
            if predictions.numel() != x.size(0):
                raise RuntimeError("HiResCAM target_class must provide one class per input image.")
            if torch.any(predictions < 0) or torch.any(predictions >= logits.size(1)):
                raise RuntimeError("HiResCAM target_class contains an out-of-range class index.")

            score = logits[indices, predictions]
            gradients = torch.autograd.grad(score.sum(), acts, retain_graph=False, create_graph=False)[0]
            if gradients.shape != acts.shape:
                raise RuntimeError(f"HiResCAM gradients have invalid shape {tuple(gradients.shape)}; expected {tuple(acts.shape)}.")
            if not torch.isfinite(gradients).all():
                raise RuntimeError("HiResCAM gradients contain NaN or Inf values.")

        cam = (gradients.detach() * acts.detach()).sum(dim=1, keepdim=True)
        cam = self._normalize_cam(F.relu(cam))
        mode = "bilinear" if cam.ndim == 4 else "trilinear"
        return F.interpolate(cam, size=x.shape[2:], mode=mode, align_corners=False)


def _is_3d_model(model: nn.Module) -> bool:
    return any(isinstance(module, nn.Conv3d) for module in model.modules())


def compute_saliency_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class=None,
    cam_layer: str = "auto",
) -> torch.Tensor:
    """Return a normalized CPU map: [H,W] for 2D or [D,H,W] for 3D."""
    if not torch.is_tensor(image_tensor):
        raise RuntimeError("compute_saliency_map expected image_tensor to be a torch.Tensor.")
    is_3d = _is_3d_model(model)
    if is_3d:
        if image_tensor.ndim == 4:
            x = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 5:
            x = image_tensor
        else:
            raise RuntimeError(f"3D CAM expected [C,D,H,W] or [B,C,D,H,W], got {tuple(image_tensor.shape)}.")
    else:
        if image_tensor.ndim == 3:
            x = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4:
            x = image_tensor
        else:
            raise RuntimeError(f"2D CAM expected [C,H,W] or [B,C,H,W], got {tuple(image_tensor.shape)}.")
    if x.numel() == 0:
        raise RuntimeError("compute_saliency_map received an empty image tensor.")
    if not torch.is_floating_point(x):
        raise RuntimeError("compute_saliency_map requires a floating point image tensor.")

    model.eval()
    device = _model_device(model)
    x = x.detach().clone().to(device).requires_grad_(True)
    _, target_module = resolve_cam_target_module(model, cam_layer)
    cam_runner = HiResCAM(model, target_module)
    try:
        cam = cam_runner.cam(x, target_class=target_class)
    finally:
        cam_runner.close()

    cam = cam[:, 0]
    expected_ndim = 4 if is_3d else 3
    if cam.ndim != expected_ndim or cam.size(0) < 1:
        raise RuntimeError(f"compute_saliency_map produced invalid CAM shape {tuple(cam.shape)}.")
    if tuple(cam.shape[1:]) != tuple(x.shape[2:]):
        raise RuntimeError(f"compute_saliency_map produced CAM shape {tuple(cam.shape[1:])}, expected input shape {tuple(x.shape[2:])}.")
    if cam.numel() == 0 or not torch.isfinite(cam).all():
        raise RuntimeError("compute_saliency_map produced an empty or non-finite saliency map.")

    spatial_dims = tuple(range(1, cam.ndim))
    cam_min = cam.amin(dim=spatial_dims, keepdim=True)
    cam_max = cam.amax(dim=spatial_dims, keepdim=True)
    denom = cam_max - cam_min
    if torch.any(denom <= 1e-12):
        raise RuntimeError("compute_saliency_map produced a saliency map with no dynamic range.")
    cam = (cam - cam_min) / denom
    result = cam[0].detach().cpu()
    if result.ndim != (3 if is_3d else 2):
        raise RuntimeError(f"compute_saliency_map produced invalid saliency shape {tuple(result.shape)}.")
    if not torch.isfinite(result).all() or float(result.min()) < -1e-6 or float(result.max()) > 1.0 + 1e-6:
        raise RuntimeError("compute_saliency_map produced invalid normalized values.")
    if float(result.max() - result.min()) <= 1e-6:
        raise RuntimeError("compute_saliency_map produced a saliency map with no dynamic range.")
    return result
