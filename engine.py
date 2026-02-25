import torch
from utils import accuracy_top1, accuracy_top5
from cam_masking import apply_random_cutout, apply_cam_cutout

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, log_every,
                    epoch=0, masking_cfg=None, cam_runner=None):
    model.train()
    running_loss, running_acc1, running_acc5 = 0.0, 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Apply masking BEFORE random crop
        if masking_cfg is not None:
            ms = masking_cfg
            do_mask = (epoch >= ms["warmup_epochs"]) and (ms["strategy"] != "none")
            if do_mask and (torch.rand(1, device=device).item() < ms["prob"]):
                if ms["strategy"] == "random":
                    x = apply_random_cutout(x, area_frac=ms["area"], block=ms["block"], fill=0.0)
                else:
                    if cam_runner is None:
                        raise RuntimeError("CAM masking requested but cam_runner is None.")
                    was_training = model.training
                    model.eval()
                    x_cam = x.detach().requires_grad_(True)
                    cam = cam_runner.cam(x_cam)
                    model.train(was_training)
                    mode = "high" if ms["strategy"] == "cam_high" else "low"
                    x = apply_cam_cutout(x, cam.detach(), area_frac=ms["area"], block=ms["block"], fill=0.0, mode=mode)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        acc1 = accuracy_top1(logits.detach(), y)
        acc5 = accuracy_top5(logits.detach(), y)
        running_loss += loss.item()
        running_acc1 += acc1
        running_acc5 += acc5
        # Per-batch progress logging removed to keep console/log concise
    return running_loss / len(loader), running_acc1 / len(loader), running_acc5 / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, acc1_sum, acc5_sum = 0.0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()
        acc1_sum += accuracy_top1(logits, y)
        acc5_sum += accuracy_top5(logits, y)
    return loss_sum / len(loader), acc1_sum / len(loader), acc5_sum / len(loader)