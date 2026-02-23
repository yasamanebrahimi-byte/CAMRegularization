import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self._hook = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B, C, H, W]

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @torch.no_grad()
    def _normalize_cam(self, cam):
        # cam: [B, 1, H, W]
        cam = cam - cam.amin(dim=(2,3), keepdim=True)
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-6)
        return cam

    def cam(self, x, y):
        """
        Returns CAM in input resolution: [B, 1, H_in, W_in]
        Uses grad w.r.t activations via torch.autograd.grad (no param .grad accumulation).
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # forward hook saves activations
        acts = self.activations  # [B,C,H,W]
        if acts is None:
            raise RuntimeError("GradCAM activations are None. Hook not firing?")

        # class score for ground-truth labels
        idx = torch.arange(x.size(0), device=x.device)
        score = logits[idx, y]  # [B]
        grads = torch.autograd.grad(score.sum(), acts, retain_graph=False, create_graph=False)[0]  # [B,C,H,W]

        weights = grads.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        cam = self._normalize_cam(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # to input size
        return cam

def _rand_int(low, high, device):
    return torch.randint(low, high, (1,), device=device).item()

def apply_random_cutout(x, area_frac=0.2, block=8, fill=0.0):
    """
    x: [B,3,H,W] (normalized ok). Masks same number of blocks per sample to match area_frac approximately.
    """
    B, C, H, W = x.shape
    out = x.clone()
    mask_area = int(area_frac * H * W)
    block_area = block * block
    n_blocks = max(1, mask_area // max(1, block_area))

    for b in range(B):
        for _ in range(n_blocks):
            top = _rand_int(0, max(1, H - block + 1), x.device)
            left = _rand_int(0, max(1, W - block + 1), x.device)
            out[b, :, top:top+block, left:left+block] = fill
    return out

def apply_cam_cutout(x, cam, area_frac=0.2, block=8, fill=1.0, mode="high",
                     thr=0.7):
    """
    thr: for mode="high", candidates are cam >= thr
         for mode="low",  candidates are cam <= thr  (use e.g. thr=0.3)
    """
    B, C, H, W = x.shape
    out = x.clone()
    mask_area = int(area_frac * H * W)
    block_area = block * block
    n_blocks = max(1, mask_area // max(1, block_area))

    cam2d = cam[:, 0]  # [B,H,W] in [0,1]

    for b in range(B):
        for _ in range(n_blocks):
            if mode == "high":
                coords = (cam2d[b] >= thr).nonzero(as_tuple=False)
            else:
                coords = (cam2d[b] <= thr).nonzero(as_tuple=False)

            # fallback if threshold too strict
            if coords.numel() == 0:
                cy = torch.randint(0, H, (1,), device=x.device).item()
                cx = torch.randint(0, W, (1,), device=x.device).item()
            else:
                j = torch.randint(0, coords.size(0), (1,), device=x.device).item()
                cy, cx = coords[j].tolist()

            top = max(0, min(H - block, cy - block // 2))
            left = max(0, min(W - block, cx - block // 2))
            out[b, :, top:top+block, left:left+block] = fill

            # optionally "invalidate" region so you don't re-mask same area
            if mode == "high":
                cam2d[b, top:top+block, left:left+block] = 0.0
            else:
                cam2d[b, top:top+block, left:left+block] = 1.0

    return out