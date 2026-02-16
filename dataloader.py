import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

CIFAR100_MEAN=(0.5071,0.4867,0.4408)
CIFAR100_STD=(0.2675,0.2565,0.2761)

def build_transforms(aug="basic", ra_n=2, ra_m=9, re_prob=0.0):
    base=[T.ToTensor(),T.Normalize(CIFAR100_MEAN,CIFAR100_STD)]
    if aug=="none":
        train=T.Compose(base); test=T.Compose(base); return train,test
    aug_list=[T.RandomCrop(32,padding=4),T.RandomHorizontalFlip()]
    if aug=="randaug":
        aug_list.append(T.RandAugment(num_ops=int(ra_n),magnitude=int(ra_m)))
    if re_prob and re_prob>0:
        aug_list.append(T.RandomErasing(p=float(re_prob)))
    train=T.Compose(aug_list+base)
    test=T.Compose(base)
    return train,test

def cifar100_loaders(data_dir,batch_size,num_workers,val_split=0.0,seed=42,aug="basic",ra_n=2,ra_m=9,re_prob=0.0,pin_memory=True):
    train_tfms,test_tfms=build_transforms(aug=aug,ra_n=ra_n,ra_m=ra_m,re_prob=re_prob)
    full_train=torchvision.datasets.CIFAR100(root=data_dir,train=True,download=True,transform=train_tfms)
    test_ds=torchvision.datasets.CIFAR100(root=data_dir,train=False,download=True,transform=test_tfms)
    if val_split and val_split>0:
        n=len(full_train); n_val=int(n*val_split); n_train=n-n_val
        g=torch.Generator().manual_seed(int(seed))
        train_ds,val_ds=random_split(full_train,[n_train,n_val],generator=g)
        val_ds.dataset.transform=test_tfms  # eval transforms
    else:
        train_ds, val_ds = full_train, None
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory,persistent_workers=(num_workers>0))
    val_dl=None if val_ds is None else DataLoader(val_ds,batch_size=256,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,persistent_workers=(num_workers>0))
    test_dl=DataLoader(test_ds,batch_size=256,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,persistent_workers=(num_workers>0))
    return train_dl,val_dl,test_dl
