import itertools
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils import best_val_from_metrics, plot_tuning_results


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class TuningConfig:
    runs_root: Path = Path("./runs_cifar100_resnet18")
    tuning_dirname: str = "tuning_results"
    train_entry: str = "train.py"
    timeout_sec: int = 3600  # 1 hour


PARAM_GRID: Dict[str, List[Any]] = {
    "epochs": [100],
    "lr": [0.05, 0.1, 0.2],
    "weight_decay": [5e-4, 2e-3],
    "momentum": [0.95],
    "nesterov": [False, True],
    "label_smoothing": [0.0, 0.1],
    "scheduler": ["cosine", "multistep"],
    "warmup_epochs": [0, 5],
    "min_lr": [0.0],
    "gamma": [0.1],
    "milestones": ["100,150"],
    "dropout": [0.0],
    "val_split": [0.1],
}

FIXED_PARAMS: Dict[str, Any] = {
    "batch_size": 128,
    "num_workers": 2,
    "seed": 42,
    "log_every": 100,
    "amp": True,
}


# -----------------------------
# Helpers
# -----------------------------

FINAL_TEST_RE = re.compile(r"Final test:.*?acc1\s+([\d.]+)%", re.IGNORECASE)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cartesian_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def prune_combinations(combos: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply simple pruning rules to cut unhelpful configs."""
    kept = []
    for c in combos:
        # Only test warmup with cosine (common choice)
        if c["scheduler"] == "multistep" and c["warmup_epochs"] > 0:
            continue
        kept.append(c)
    return kept


def compute_multistep_milestones(epochs: int) -> str:
    if epochs == 150:
        return "90,120"
    if epochs == 200:
        return "100,150"
    return f"{int(0.5 * epochs)},{int(0.75 * epochs)}"


def with_scheduler_dependent_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy where multistep milestones are consistent with the epoch budget."""
    out = dict(params)
    if out.get("scheduler") == "multistep":
        out["milestones"] = compute_multistep_milestones(int(out["epochs"]))
    return out


def format_run_name(all_params: Dict[str, Any], fixed_params: Dict[str, Any]) -> str:
    wd = float(all_params["weight_decay"])
    ms = all_params["milestones"] if all_params["scheduler"] == "multistep" else "na"
    return (
        f"tune_ep{all_params['epochs']}_bs{fixed_params['batch_size']}_lr{all_params['lr']}"
        f"_wd{wd:.0e}_m{all_params['momentum']}_nest{int(bool(all_params['nesterov']))}"
        f"_ls{all_params['label_smoothing']}_sch{all_params['scheduler']}_wu{all_params['warmup_epochs']}"
        f"_ms{ms}"
    )


def build_train_cmd(train_entry: str, run_name: str, params: Dict[str, Any]) -> List[str]:
    """
    Build CLI args in a consistent way:
    - boolean flags (amp/nesterov) become --flag when True
    - everything else becomes --key value
    """
    cmd = ["python", train_entry, "--run_name", run_name]

    bool_flags = {"amp", "nesterov"}
    for key, value in params.items():
        if key in bool_flags:
            if bool(value):
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])

    return cmd


def parse_final_test_acc1(stdout: str) -> Optional[float]:
    """
    Return acc1 as fraction (0.0-1.0) if present.
    Looks for: 'Final test: ... acc1 XX.XX%'
    """
    m = FINAL_TEST_RE.search(stdout or "")
    if not m:
        return None
    return float(m.group(1)) / 100.0


def tail(text: Optional[str], n: int = 500) -> str:
    if not text:
        return ""
    return text[-n:]


def metrics_csv_path(cfg: TuningConfig, run_name: str) -> Path:
    return cfg.runs_root / run_name / "metrics.csv"


# -----------------------------
# Main
# -----------------------------

def tune_hyperparameters(cfg: TuningConfig = TuningConfig()) -> None:
    tuning_dir = ensure_dir(cfg.runs_root / cfg.tuning_dirname)

    combos = prune_combinations(cartesian_product(PARAM_GRID))
    print(f"Running {len(combos)} training configurations...")
    print(f"Results will be saved to {tuning_dir}\n")

    results: List[Dict[str, Any]] = []

    for idx, grid_params in enumerate(combos, 1):
        all_params = {**FIXED_PARAMS, **grid_params}
        all_params = with_scheduler_dependent_params(all_params)

        run_name = format_run_name(all_params, FIXED_PARAMS)
        cmd = build_train_cmd(cfg.train_entry, run_name, all_params)

        print(f"[{idx}/{len(combos)}] Running: {run_name}")
        print(f"  Command: {' '.join(cmd)}")

        result_info = run_single_training_run(cfg, run_name, all_params, cmd)
        results.append(result_info)
        print()

    results_file = tuning_dir / "tuning_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    print(f"\nTuning complete! Results saved to {results_file}")
    print_summary(cfg, results)
    plot_tuning_results(results, tuning_dir)


def run_single_training_run(
    cfg: TuningConfig, run_name: str, params: Dict[str, Any], cmd: List[str]
) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_sec)
        final_test_acc1 = parse_final_test_acc1(proc.stdout)

        if proc.returncode == 0:
            status = "success"
            if final_test_acc1 is not None:
                print(f"Completed successfully - Test Acc: {final_test_acc1 * 100:.2f}%")
            else:
                print("Completed successfully")
        else:
            status = "failed"
            print(f"Failed with exit code {proc.returncode}")
            if proc.stderr:
                for line in [l for l in proc.stderr.splitlines() if l.strip()][-10:]:
                    print(f"    {line}")

        info: Dict[str, Any] = {
            "run_name": run_name,
            "params": params,
            "status": status,
            "exit_code": proc.returncode,
            "stderr": tail(proc.stderr, 500),
        }
        if final_test_acc1 is not None:
            info["final_test_acc1"] = final_test_acc1
        return info

    except subprocess.TimeoutExpired:
        print("Timeout (exceeded 1 hour)")
        return {"run_name": run_name, "params": params, "status": "timeout"}

    except Exception as e:
        print(f"Exception: {e}")
        return {"run_name": run_name, "params": params, "status": "error", "error": str(e)}


def print_summary(cfg: TuningConfig, results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    other = [r for r in results if r.get("status") not in {"success", "failed"}]

    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Other: {len(other)}")

    best_test = max((r for r in successful if "final_test_acc1" in r), default=None, key=lambda r: r["final_test_acc1"])
    if best_test:
        acc = best_test["final_test_acc1"]
        p = best_test["params"]
        print(f"\nBest test accuracy (for reference): {acc * 100:.2f}% ({best_test['run_name']})")
        print(f"   lr={p['lr']}, epochs={p['epochs']}, wd={p['weight_decay']}, val_split={p['val_split']}")

    ranked: List[Tuple[Dict[str, Any], float]] = []
    for r in successful:
        mpath = metrics_csv_path(cfg, r["run_name"])
        best_val = best_val_from_metrics(mpath)
        if best_val is not None:
            ranked.append((r, best_val))

    ranked.sort(key=lambda x: x[1], reverse=True)

    if ranked:
        print("\nTop 10 runs by BEST val_acc1 (max over epochs):")
        for i, (r, best_val) in enumerate(ranked[:10], 1):
            p = r["params"]
            print(f"  {i}. {r['run_name']}: best_val_acc1={best_val:.6f}")
            print(
                "     "
                f"lr={p['lr']}, ep={p['epochs']}, wd={p['weight_decay']}, mom={p['momentum']}, "
                f"nest={p['nesterov']}, ls={p['label_smoothing']}, sch={p['scheduler']}, "
                f"wu={p['warmup_epochs']}, ms={p.get('milestones','')}"
            )

        rows = []
        for r, best_val in ranked:
            p = r["params"]
            rows.append(
                {
                    "run_name": r["run_name"],
                    "best_val_acc1": best_val,
                    "lr": p["lr"],
                    "epochs": p["epochs"],
                    "weight_decay": p["weight_decay"],
                    "momentum": p["momentum"],
                    "nesterov": p["nesterov"],
                    "label_smoothing": p["label_smoothing"],
                    "scheduler": p["scheduler"],
                    "warmup_epochs": p["warmup_epochs"],
                    "milestones": p.get("milestones", ""),
                }
            )

        out_csv = cfg.runs_root / cfg.tuning_dirname / "ranked_by_val.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved ranking to {out_csv}")


if __name__ == "__main__":
    tune_hyperparameters()
