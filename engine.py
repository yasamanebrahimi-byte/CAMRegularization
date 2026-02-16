import time,torch
import torch.nn.functional as F
from utils import accuracy_topk

def _mixup(x,y,alpha):
    if alpha<=0: return x,y,None
    lam=torch.distributions.Beta(alpha,alpha).sample().item()
    idx=torch.randperm(x.size(0),device=x.device)
    x=lam*x+(1.0-lam)*x[idx]
    return x,y,("mixup",lam,idx)

def _rand_bbox(W,H,lam):
    cut_rat=(1.0-lam)**0.5
    cut_w=int(W*cut_rat); cut_h=int(H*cut_rat)
    cx=torch.randint(0,W,(1,)).item(); cy=torch.randint(0,H,(1,)).item()
    x1=max(cx-cut_w//2,0); x2=min(cx+cut_w//2,W)
    y1=max(cy-cut_h//2,0); y2=min(cy+cut_h//2,H)
    return x1,y1,x2,y2

def _cutmix(x,y,alpha):
    if alpha<=0: return x,y,None
    lam=torch.distributions.Beta(alpha,alpha).sample().item()
    idx=torch.randperm(x.size(0),device=x.device)
    W=x.size(3); H=x.size(2)
    x1,y1,x2,y2=_rand_bbox(W,H,lam)
    x2b=x.clone()
    x2b[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    lam=1.0-((x2-x1)*(y2-y1)/(W*H))
    return x2b,y,("cutmix",lam,idx)

def _loss_with_mixing(logits,y,mixinfo,criterion,ls=0.0):
    if mixinfo is None: return criterion(logits,y)
    kind,lam,idx=mixinfo
    y2=y[idx]
    if ls and ls>0:
        # label smoothing with mixed targets: use KL on smoothed one-hot
        n=logits.size(1)
        def smooth_targets(t):
            oh=F.one_hot(t,num_classes=n).float()
            return oh*(1.0-ls)+ls/float(n)
        p=F.log_softmax(logits,dim=1)
        t1=smooth_targets(y); t2=smooth_targets(y2)
        return -(lam*(t1*p).sum(dim=1)+(1.0-lam)*(t2*p).sum(dim=1)).mean()
    # standard CE mixing
    return lam*F.cross_entropy(logits,y)+(1.0-lam)*F.cross_entropy(logits,y2)

def train_one_epoch(model,loader,optimizer,criterion,scaler,device,log_every=100,mixup_alpha=0.0,cutmix_alpha=0.0,grad_clip=0.0,label_smoothing=0.0):
    model.train()
    loss_sum,acc1_sum,acc5_sum=0.0,0.0,0.0
    t0=time.time()
    for i,(x,y) in enumerate(loader, start=1):
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        mixinfo=None
        if cutmix_alpha and cutmix_alpha>0 and torch.rand(1).item()<0.5:
            x,y,mixinfo=_cutmix(x,y,cutmix_alpha)
        elif mixup_alpha and mixup_alpha>0:
            x,y,mixinfo=_mixup(x,y,mixup_alpha)

        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(scaler is not None)):
            logits=model(x)
            if mixinfo is None:
                if label_smoothing and label_smoothing>0 and hasattr(criterion,"label_smoothing"):
                    loss=criterion(logits,y)
                else:
                    loss=F.cross_entropy(logits,y,label_smoothing=float(label_smoothing))
            else:
                loss=_loss_with_mixing(logits,y,mixinfo,criterion,ls=float(label_smoothing))

        if scaler is None:
            loss.backward()
            if grad_clip and grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),float(grad_clip))
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip>0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(grad_clip))
            scaler.step(optimizer); scaler.update()

        acc1,acc5=accuracy_topk(logits.detach(),y,ks=(1,5))
        loss_sum+=loss.item(); acc1_sum+=acc1; acc5_sum+=acc5

        if i%log_every==0 or i==len(loader):
            dt=time.time()-t0
            print(f"[train] {i}/{len(loader)} loss {loss_sum/i:.4f} acc1 {acc1_sum/i*100:.2f}% acc5 {acc5_sum/i*100:.2f}% {i/dt:.2f} it/s")

    n=len(loader)
    return loss_sum/n, acc1_sum/n, acc5_sum/n

@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval()
    loss_sum,acc1_sum,acc5_sum=0.0,0.0,0.0
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        logits=model(x)
        loss=criterion(logits,y)
        acc1,acc5=accuracy_topk(logits,y,ks=(1,5))
        loss_sum+=loss.item(); acc1_sum+=acc1; acc5_sum+=acc5
    n=len(loader)
    return loss_sum/n, acc1_sum/n, acc5_sum/n
