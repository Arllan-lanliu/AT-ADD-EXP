#!/usr/bin/env python3
"""Create reproducible random train/dev splits for AT-ADD Track 1.

The split is sampled from the union of the original train and dev protocols.
For each (type, label) stratum, the new dev set keeps exactly the same row
count as the original dev set. This preserves the train/dev ratio and the
real/fake balance for speech, sound, singing, and music.

Example:
python make_random_split.py \
  --train-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/train.csv \
  --dev-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv \
  --out-dir /home/new_disk/liulan/workspace/released_model/ADD-TRACK1/protocol_random_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.0"

DEFAULT_TRAIN_CSV = Path("/home/new_disk/liulan/workspace/dataset/at_add_track2/labels/train.csv")
DEFAULT_DEV_CSV = Path("/home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv")
DEFAULT_OUT_DIR = Path("/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/protocol_random_seed42")
EXPECTED_TYPES = ("music", "singing", "sound", "speech")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly resplit original train+dev while preserving per-type "
            "real/fake counts and original train/dev counts."
        )
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--dev-csv", type=Path, default=DEFAULT_DEV_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--type-col", default="type")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--train-out-name", default="train_random.csv")
    parser.add_argument("--dev-out-name", default="val_random.csv")
    return parser.parse_args()


def read_protocol(path: Path, split_name: str) -> tuple[list[dict[str, str]], list[str]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{split_name} CSV has no header: {path}")
        rows = [dict(row, _orig_split=split_name) for row in reader]
        return rows, list(reader.fieldnames)


def read_protocols(train_csv: Path, dev_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    train_rows, train_columns = read_protocol(train_csv, "train")
    dev_rows, dev_columns = read_protocol(dev_csv, "dev")
    if train_columns != dev_columns:
        raise ValueError(
            "train/dev CSV columns differ.\n"
            f"train: {train_columns}\n"
            f"dev  : {dev_columns}"
        )
    return train_rows + dev_rows, train_columns


def _norm(value: str) -> str:
    return str(value).strip().lower()


def validate_columns(rows: list[dict[str, str]], type_col: str, label_col: str) -> None:
    if not rows:
        raise ValueError("No rows found in train/dev protocols.")
    columns = set(rows[0].keys())
    for col in (type_col, label_col):
        if col not in columns:
            raise ValueError(f"Required column {col!r} not found. Columns: {sorted(columns)}")


def group_rows(
    rows: list[dict[str, str]],
    type_col: str,
    label_col: str,
) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(_norm(row[type_col]), _norm(row[label_col]))].append(idx)
    return dict(groups)


def count_by_split_and_stratum(
    rows: list[dict[str, str]],
    type_col: str,
    label_col: str,
) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        counts[(row["_orig_split"], _norm(row[type_col]), _norm(row[label_col]))] += 1
    return counts


def make_random_split(
    rows: list[dict[str, str]],
    type_col: str,
    label_col: str,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    groups = group_rows(rows, type_col, label_col)
    original_counts = count_by_split_and_stratum(rows, type_col, label_col)

    rng = random.Random(seed)
    dev_indices: set[int] = set()
    allocation: dict[str, dict[str, dict[str, int]]] = {}

    for type_name, label_name in sorted(groups):
        indices = groups[(type_name, label_name)]
        target_dev = original_counts[("dev", type_name, label_name)]
        target_train = original_counts[("train", type_name, label_name)]
        if target_train + target_dev != len(indices):
            raise RuntimeError(
                f"Internal count mismatch for {(type_name, label_name)!r}: "
                f"train={target_train}, dev={target_dev}, total={len(indices)}"
            )
        if target_dev > len(indices):
            raise RuntimeError(
                f"Cannot sample {target_dev} dev rows from {len(indices)} rows "
                f"for {(type_name, label_name)!r}."
            )

        sampled = rng.sample(indices, target_dev)
        dev_indices.update(sampled)
        allocation.setdefault(type_name, {})[label_name] = {
            "train": int(target_train),
            "dev": int(target_dev),
            "total": int(len(indices)),
        }

    train_rows: list[dict[str, str]] = []
    dev_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        output_row = {k: v for k, v in row.items() if not k.startswith("_")}
        if idx in dev_indices:
            dev_rows.append(output_row)
        else:
            train_rows.append(output_row)

    report = build_report(train_rows, dev_rows, type_col, label_col, allocation)
    meta = {
        "script_version": SCRIPT_VERSION,
        "seed": seed,
        "mode": "random_stratified_by_type_label",
        "type_col": type_col,
        "label_col": label_col,
        "expected_types": list(EXPECTED_TYPES),
        "allocation": allocation,
        "report": report,
    }
    return train_rows, dev_rows, meta


def nested_counts(
    rows: list[dict[str, str]],
    type_col: str,
    label_col: str,
) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counts[(_norm(row[type_col]), _norm(row[label_col]))] += 1

    result: dict[str, dict[str, int]] = {}
    for (type_name, label_name), count in sorted(counts.items()):
        result.setdefault(type_name, {})[label_name] = int(count)
    return result


def label_counts(rows: list[dict[str, str]], label_col: str) -> dict[str, int]:
    counts = Counter(_norm(row[label_col]) for row in rows)
    return {str(k): int(v) for k, v in sorted(counts.items())}


def type_counts(rows: list[dict[str, str]], type_col: str) -> dict[str, int]:
    counts = Counter(_norm(row[type_col]) for row in rows)
    return {str(k): int(v) for k, v in sorted(counts.items())}


def build_report(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    type_col: str,
    label_col: str,
    allocation: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    return {
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(dev_rows)),
        "train_label_counts": label_counts(train_rows, label_col),
        "val_label_counts": label_counts(dev_rows, label_col),
        "train_type_counts": type_counts(train_rows, type_col),
        "val_type_counts": type_counts(dev_rows, type_col),
        "train_type_label_counts": nested_counts(train_rows, type_col, label_col),
        "val_type_label_counts": nested_counts(dev_rows, type_col, label_col),
        "target_allocation": allocation,
    }


def print_report(meta: dict[str, Any]) -> None:
    report = meta["report"]
    print("===== RANDOM TYPE/LABEL STRATIFIED SPLIT =====")
    print(f"seed: {meta['seed']}")
    print(f"train samples: {report['train_rows']}")
    print(f"val samples  : {report['val_rows']}")
    print(f"train real/fake: {report['train_label_counts']}")
    print(f"val real/fake  : {report['val_label_counts']}")
    print("per-type label counts:")
    for type_name in sorted(report["target_allocation"]):
        train_counts = report["train_type_label_counts"].get(type_name, {})
        val_counts = report["val_type_label_counts"].get(type_name, {})
        print(f"  {type_name}: train={train_counts}, val={val_counts}")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    original_columns: list[str],
    out_dir: Path,
    train_out_name: str,
    dev_out_name: str,
    meta: dict[str, Any],
) -> None:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / train_out_name
    dev_path = out_dir / dev_out_name
    meta_path = out_dir / "split_meta.json"

    meta["output_files"] = {
        "train": train_path.name,
        "val": dev_path.name,
        "meta": meta_path.name,
    }

    write_csv(train_path, train_rows, original_columns)
    write_csv(dev_path, dev_rows, original_columns)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"saved: {train_path}")
    print(f"saved: {dev_path}")
    print(f"saved: {meta_path}")


def main() -> None:
    args = parse_args()
    rows, original_columns = read_protocols(args.train_csv, args.dev_csv)
    validate_columns(rows, args.type_col, args.label_col)

    observed_types = sorted({_norm(row[args.type_col]) for row in rows})
    missing_types = sorted(set(EXPECTED_TYPES) - set(observed_types))
    if missing_types:
        print(f"WARNING: expected type(s) not found: {missing_types}")

    train_rows, dev_rows, meta = make_random_split(
        rows,
        args.type_col,
        args.label_col,
        args.seed,
    )
    print_report(meta)
    save_outputs(
        train_rows,
        dev_rows,
        original_columns,
        args.out_dir,
        args.train_out_name,
        args.dev_out_name,
        meta,
    )


if __name__ == "__main__":
    main()
