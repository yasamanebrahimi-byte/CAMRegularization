import itertools,subprocess,sys,os

def run(cmd):
    print("\n"+" ".join(cmd))
    r=subprocess.run(cmd)
    if r.returncode!=0: raise SystemExit(r.returncode)

def main():
    base=[sys.executable,"train.py","--epochs","30","--val_split","0.1","--amp","--aug","randaug","--random_erasing","0.25","--label_smoothing","0.1","--mixup","0.2","--ema","0.999","--nesterov"]
    grid={
        "lr":["0.05","0.1","0.2"],
        "weight_decay":["1e-4","5e-4","1e-3"],
        "ra_m":["7","9","11"],
    }
    keys=list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        name="_".join([f"{k}{v}" for k,v in zip(keys,vals)])
        cmd=base+["--run_name",name]
        for k,v in zip(keys,vals): cmd+=["--"+k,str(v)]
        run(cmd)

if __name__=="__main__":
    main()
