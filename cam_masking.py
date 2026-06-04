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
        self._hook = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, _inp, out):
        self.activations = out  # [B, C, H, W]

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _normalize_cam(self, cam):
        # cam: [B, 1, H, W]
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return cam

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
            if not acts.requires_grad:
                raise RuntimeError("HiResCAM activations do not require gradients. Check grad/no_grad context.")

            # class score for predicted labels
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
            score = logits[idx, pred]
            grads = torch.autograd.grad(score.sum(), acts, retain_graph=False, create_graph=False)[0]  # [B,C,H,W]

        acts_detached = acts.detach()
        grads_detached = grads.detach()
        self.activations = None

        # HiResCAM: element-wise product (no global average pooling of gradients)
        cam = (grads_detached * acts_detached).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        cam = self._normalize_cam(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # to input size
        return cam


def compute_saliency_map(model: nn.Module, image_tensor: torch.Tensor, target_class=None, cam_layer: str = "auto") -> torch.Tensor:
    """Return a normalized HiResCAM saliency map for a single image tensor.

    Returns a [H, W] tensor in [0, 1] on CPU.
    """
    if image_tensor.ndim == 3:
        x = image_tensor.unsqueeze(0)
    else:
        x = image_tensor

    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)

    _, target_module = resolve_cam_target_module(model, cam_layer)
    cam_runner = HiResCAM(model, target_module)
    try:
        cam = cam_runner.cam(x, target_class=target_class)  # [B, 1, H, W]
    finally:
        cam_runner.close()

    cam = cam[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-6)
    cam = cam[0].detach().cpu()
    return cam

