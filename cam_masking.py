import torch
import torch.nn.functional as F
import torch.nn as nn


def _get_submodule(model: nn.Module, module_name: str) -> nn.Module:
    if hasattr(model, "get_submodule"):
        return model.get_submodule(module_name)

    cur: nn.Module = model
    for token in module_name.split("."):
        if token.isdigit():
            cur = cur[int(token)]
        else:
            cur = getattr(cur, token)
    return cur


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
        if name and isinstance(module, nn.Conv2d)
    ]
    if conv_candidates:
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
    """
    HiResCAM: element-wise gradient * activation, summed over channels.
    Unlike GradCAM which uses globally-averaged gradient weights,
    HiResCAM preserves spatial gradient information for higher-resolution maps.

    Reference: https://arxiv.org/abs/2011.08891
    """

    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._hook = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, _inp, out):
        if not torch.is_tensor(out):
            self.activations = None
            self.gradients = None
            return

        self.activations = out  # [B, C, H, W]
        self.gradients = None
        if out.requires_grad:
            out.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        self.gradients = grad

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _normalize_cam(self, cam):
        # cam: [B, 1, H, W]
        if cam.ndim != 4 or cam.size(1) != 1:
            raise RuntimeError(f"HiResCAM produced invalid shape {tuple(cam.shape)}; expected [B, 1, H, W].")
        if cam.numel() == 0:
            raise RuntimeError("HiResCAM produced an empty saliency map.")
        if not torch.isfinite(cam).all():
            raise RuntimeError("HiResCAM produced NaN or Inf values.")

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        denom = cam_max - cam_min
        if torch.any(denom <= 1e-12):
            raise RuntimeError("HiResCAM saliency map has no dynamic range.")
        return (cam - cam_min) / denom

    def cam(self, x, target_class=None):
        """
        Returns HiResCAM in input resolution: [B, 1, H_in, W_in].
        Element-wise gradient * activation product (no global average pooling).
        Computes CAM for the predicted class unless target_class is provided.
        """
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)  # forward hook saves activations
            acts = self.activations  # [B,C,H,W]
            if acts is None:
                raise RuntimeError("HiResCAM activations are None. Hook not firing?")
            if acts.ndim != 4:
                raise RuntimeError(f"HiResCAM activations have invalid shape {tuple(acts.shape)}; expected [B, C, H, W].")
            if not acts.requires_grad:
                raise RuntimeError("HiResCAM activations do not require gradients. Check grad/no_grad context.")
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise RuntimeError(f"HiResCAM expected model logits with shape [B, classes], got {type(logits).__name__}.")

            idx = torch.arange(x.size(0), device=x.device)
            if target_class is None:
                pred = logits.argmax(dim=1)
            else:
                if isinstance(target_class, int):
                    pred = torch.full((x.size(0),), int(target_class), device=x.device, dtype=torch.long)
                else:
                    pred = torch.as_tensor(target_class, device=x.device, dtype=torch.long)
                    if pred.ndim == 0:
                        pred = pred.repeat(x.size(0))
                    pred = pred.reshape(-1)
            if pred.numel() != x.size(0):
                raise RuntimeError("HiResCAM target_class must provide one class per input image.")
            if torch.any(pred < 0) or torch.any(pred >= logits.size(1)):
                raise RuntimeError("HiResCAM target_class contains an out-of-range class index.")

            score = logits[idx, pred]
            grads = torch.autograd.grad(score.sum(), acts, retain_graph=False, create_graph=False)[0]
            if self.gradients is None:
                self.gradients = grads
            if self.gradients.shape != acts.shape:
                raise RuntimeError(
                    f"HiResCAM gradients have invalid shape {tuple(self.gradients.shape)}; "
                    f"expected {tuple(acts.shape)}."
                )
            if not torch.isfinite(self.gradients).all():
                raise RuntimeError("HiResCAM gradients contain NaN or Inf values.")

        acts_detached = acts.detach()
        grads_detached = self.gradients.detach()
        self.activations = None
        self.gradients = None

        cam = (grads_detached * acts_detached).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = self._normalize_cam(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam


def compute_saliency_map(model: nn.Module, image_tensor: torch.Tensor, target_class=None, cam_layer: str = "auto") -> torch.Tensor:
    """Return a normalized HiResCAM saliency map for a single image tensor.

    Returns a [H, W] tensor in [0, 1] on CPU.
    """
    if not torch.is_tensor(image_tensor):
        raise RuntimeError("compute_saliency_map expected image_tensor to be a torch.Tensor.")
    if image_tensor.ndim == 3:
        x = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4:
        x = image_tensor
    else:
        raise RuntimeError(f"compute_saliency_map expected a [C, H, W] or [B, C, H, W] tensor, got {tuple(image_tensor.shape)}.")
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
        cam = cam_runner.cam(x, target_class=target_class)  # [B, 1, H, W]
    finally:
        cam_runner.close()

    cam = cam[:, 0]
    if cam.ndim != 3 or cam.size(0) < 1:
        raise RuntimeError(f"compute_saliency_map produced invalid CAM shape {tuple(cam.shape)}; expected [B, H, W].")
    if cam.shape[-2:] != x.shape[-2:]:
        raise RuntimeError(
            f"compute_saliency_map produced CAM shape {tuple(cam.shape[-2:])}, "
            f"expected input shape {tuple(x.shape[-2:])}."
        )
    if cam.numel() == 0:
        raise RuntimeError("compute_saliency_map produced an empty saliency map.")
    if not torch.isfinite(cam).all():
        raise RuntimeError("compute_saliency_map produced NaN or Inf values.")

    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True)
    denom = cam_max - cam_min
    if torch.any(denom <= 1e-12):
        raise RuntimeError("compute_saliency_map produced a saliency map with no dynamic range.")
    cam = (cam - cam_min) / denom
    cam = cam[0].detach().cpu()

    if cam.ndim != 2:
        raise RuntimeError(f"compute_saliency_map produced invalid saliency shape {tuple(cam.shape)}; expected [H, W].")
    if not torch.isfinite(cam).all():
        raise RuntimeError("compute_saliency_map produced NaN or Inf values after normalization.")
    if float(cam.min()) < -1e-6 or float(cam.max()) > 1.0 + 1e-6:
        raise RuntimeError("compute_saliency_map produced values outside [0, 1].")
    return cam
