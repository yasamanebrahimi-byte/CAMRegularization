"""Leakage-safe fold construction for labeled DaT examinations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from dat_preprocessing import DatRecord, _canonical_array_and_spacing


def make_stratified_folds(records: Sequence[DatRecord], n_splits: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(records) < n_splits:
        raise ValueError("There must be at least one labeled record per fold.")
    rng = np.random.default_rng(int(seed))
    buckets = [[] for _ in range(int(n_splits))]
    for label in sorted(set(record.label for record in records)):
        indices = np.asarray([i for i, record in enumerate(records) if record.label == label], dtype=np.int64)
        rng.shuffle(indices)
        for position, index in enumerate(indices.tolist()):
            buckets[position % n_splits].append(int(index))
    folds = []
    all_indices = set(range(len(records)))
    for validation in buckets:
        validation = sorted(validation)
        training = sorted(all_indices.difference(validation))
        if not validation or not training:
            raise ValueError("A stratified fold is empty; reduce n_splits or provide more labeled records.")
        folds.append((training, validation))
    return folds


def protocol_signature(record: DatRecord, decimals: int = 1) -> tuple:
    volume, spacing = _canonical_array_and_spacing(record.path)
    return tuple(int(v) for v in volume.shape), tuple(round(float(v), decimals) for v in spacing)


def make_protocol_group_folds(records: Sequence[DatRecord], n_splits: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Optional robustness diagnostic; protocol groups are not hospital-center labels."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    groups: dict[tuple, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(protocol_signature(record), []).append(index)
    ordered_groups = list(groups.items())
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ordered_groups)
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for signature, indices in sorted(ordered_groups, key=lambda item: -len(item[1])):
        target = min(range(n_splits), key=lambda fold: fold_sizes[fold])
        fold_groups[target].extend(indices)
        fold_sizes[target] += len(indices)
    result = []
    all_indices = set(range(len(records)))
    for validation in fold_groups:
        validation = sorted(validation)
        training = sorted(all_indices.difference(validation))
        if not validation or not training:
            raise ValueError("A protocol-group fold is empty; reduce n_splits or provide more protocol groups.")
        result.append((training, validation))
    return result


def save_fold_assignments(
    path: str | Path,
    records: Sequence[DatRecord],
    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    seed: int,
    grouped: bool = False,
) -> None:
    payload = {
        "version": 1,
        "seed": int(seed),
        "n_splits": len(folds),
        "grouped_protocol_diagnostic": bool(grouped),
        "folds": [
            {
                "train_uids": [records[int(index)].uid for index in train],
                "validation_uids": [records[int(index)].uid for index in validation],
            }
            for train, validation in folds
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_fold_assignments(path: str | Path, records: Sequence[DatRecord]) -> list[tuple[list[int], list[int]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_uid = {record.uid: index for index, record in enumerate(records)}
    folds = []
    for fold in payload.get("folds", []):
        try:
            train = [by_uid[uid] for uid in fold["train_uids"]]
            validation = [by_uid[uid] for uid in fold["validation_uids"]]
        except KeyError as exc:
            raise ValueError("Fold assignment file does not match the current labeled dataset.") from exc
        folds.append((train, validation))
    if not folds:
        raise ValueError("Fold assignment file contains no folds.")
    return folds
