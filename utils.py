import os,random,math
import torch

def set_seed(seed,deterministic=False):
    seed=int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    else:
        torch.backends.cudnn.benchmark=True

@torch.no_grad()
def accuracy_topk(logits,targets,ks=(1,)):
    maxk=max(ks)
    _,pred=logits.topk(maxk,dim=1,largest=True,sorted=True)
    pred=pred.t()
    correct=pred.eq(targets.view(1,-1).expand_as(pred))
    out=[]
    for k in ks:
        out.append(correct[:k].reshape(-1).float().mean().item())
    return out

def save_ckpt(path,model,optimizer,epoch,best_metric,extra=None):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":int(epoch),"best_metric":float(best_metric)}
    if extra: payload.update(extra)
    torch.save(payload,path)

class EMA:
    def __init__(self,model,decay=0.999):
        self.decay=float(decay)
        self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.backup=None
    @torch.no_grad()
    def update(self,model):
        d=self.decay
        msd=model.state_dict()
        for k,v in msd.items():
            self.shadow[k].mul_(d).add_(v.detach(),alpha=1.0-d)
    def apply_to(self,model):
        self.backup={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow,strict=True)
    def restore(self,model):
        if self.backup is not None:
            model.load_state_dict(self.backup,strict=True)
            self.backup=None

def cosine_warmup_lr(optimizer,epochs,warmup_epochs,min_lr=0.0):
    warmup_epochs=max(0,int(warmup_epochs))
    epochs=int(epochs)
    min_lr=float(min_lr)
    base_lrs=[g["lr"] for g in optimizer.param_groups]
    def lr_lambda(e):
        e=int(e)
        if warmup_epochs>0 and e<warmup_epochs:
            return float(e+1)/float(warmup_epochs)
        t=(e-warmup_epochs)/max(1,epochs-warmup_epochs)
        return (min_lr/max(base_lrs)+0.5*(1.0-min_lr/max(base_lrs))*(1.0+math.cos(math.pi*t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda)
