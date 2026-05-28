#!/usr/bin/env python3
"""Build 5-fold AT-ADD Track-2 protocols from train + dev labels.

The split is stratified by (generator, label): rows in every generator/label
bucket are shuffled with a fixed seed and distributed round-robin into 5 folds.
For fold N, val/dev is fold N and train is the union of the remaining folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - this repo normally depends on PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_LABEL = Path("/data/liulan/workspace/dataset/at_add_track2/labels/train.csv")
DEFAULT_DEV_LABEL = Path("/data/liulan/workspace/dataset/at_add_track2/labels/dev.csv")
DEFAULT_OUT_DIR = ROOT / "protocol_k_fold"
DEFAULT_TEMPLATE = ROOT / "multi_head/conf_multi_head/xlsrbeats_random_seed2026.yaml"
DEFAULT_TRAIN_DEV_AUDIO = Path("/data/liulan/workspace/dataset/at_add_track2/train_dev")
DEFAULT_EVAL_AUDIO = Path("/data/liulan/workspace/dataset/at_add_track2/eval")
CSV_FIELDS = ["name", "type", "label", "generator"]


def _read_protocol(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = [c for c in CSV_FIELDS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; got {reader.fieldnames}")
        rows = []
        for row in reader:
            clean = {c: (row.get(c) or "").strip() for c in CSV_FIELDS}
            if not clean["name"]:
                continue
            rows.append(clean)
    return rows


def _write_protocol(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _nested_counts(rows: list[dict[str, str]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row[outer]][row[inner]] += 1
    for key in sorted(grouped):
        out[key] = dict(sorted(grouped[key].items()))
    return out


def _counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))


def _report(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "label_counts": _counts(rows, "label"),
        "type_counts": _counts(rows, "type"),
        "generator_counts": _counts(rows, "generator"),
        "generator_label_counts": _nested_counts(rows, "generator", "label"),
        "type_label_counts": _nested_counts(rows, "type", "label"),
    }


def _make_folds(rows: list[dict[str, str]], n_splits: int, seed: int) -> list[list[dict[str, str]]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["generator"], row["label"])].append(row)

    rng = random.Random(seed)
    folds: list[list[dict[str, str]]] = [[] for _ in range(n_splits)]
    for key in sorted(buckets):
        group = list(buckets[key])
        rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % n_splits].append(row)

    for fold in folds:
        fold.sort(key=lambda r: r["name"])
    return folds


def _load_template(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to generate experiment config YAML files")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return raw


def _write_config(
    template: dict[str, Any],
    path: Path,
    fold_dir: Path,
    fold_idx: int,
    train_dev_audio: Path,
    eval_audio: Path,
    ckpt_root: str,
) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to generate experiment config YAML files")

    cfg = json.loads(json.dumps(template))
    data = cfg.setdefault("data", {})
    data["atadd_t2_train_audio"] = str(train_dev_audio)
    data["atadd_t2_train_label"] = str(fold_dir / "train_random.csv")
    data["atadd_t2_dev_audio"] = str(train_dev_audio)
    data["atadd_t2_dev_label"] = str(fold_dir / "val_random.csv")
    data["atadd_t2_eval_audio"] = str(eval_audio)
    cfg["out_fold"] = f"{ckpt_root}/fold_{fold_idx}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def build_k_fold(args: argparse.Namespace) -> None:
    train_rows = _read_protocol(args.train_label)
    dev_rows = _read_protocol(args.dev_label)
    all_rows = train_rows + dev_rows

    names = [row["name"] for row in all_rows]
    dup_names = [name for name, count in Counter(names).items() if count > 1]
    if dup_names:
        raise ValueError(f"Duplicate filenames found, examples: {dup_names[:10]}")

    folds = _make_folds(all_rows, args.n_splits, args.seed)
    template = _load_template(args.config_template)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = args.out_dir / "configs"

    fold_reports = {}
    for i, val_rows in enumerate(folds):
        val_names = {row["name"] for row in val_rows}
        fold_train = [row for row in all_rows if row["name"] not in val_names]
        fold_dir = args.out_dir / f"fold_{i}"

        _write_protocol(fold_dir / "train_random.csv", fold_train)
        _write_protocol(fold_dir / "val_random.csv", val_rows)
        _write_config(
            template,
            configs_dir / f"fold_{i}.yaml",
            fold_dir,
            i,
            args.train_dev_audio,
            args.eval_audio,
            args.ckpt_root,
        )

        fold_reports[f"fold_{i}"] = {
            "train": _report(fold_train),
            "val": _report(val_rows),
            "files": {
                "train": str(fold_dir / "train_random.csv"),
                "val": str(fold_dir / "val_random.csv"),
                "config": str(configs_dir / f"fold_{i}.yaml"),
            },
        }

    meta = {
        "script_version": "1.0",
        "seed": args.seed,
        "mode": "k_fold_stratified_by_generator_label",
        "n_splits": args.n_splits,
        "stratify_cols": ["generator", "label"],
        "source_files": {
            "train_label": str(args.train_label),
            "dev_label": str(args.dev_label),
        },
        "source_report": {
            "train": _report(train_rows),
            "dev": _report(dev_rows),
            "merged": _report(all_rows),
        },
        "output_files": {
            "fold_dir_pattern": "fold_{i}/",
            "train": "train_random.csv",
            "val": "val_random.csv",
            "configs_dir": "configs/",
            "meta": "split_meta.json",
        },
        "folds": fold_reports,
    }
    with (args.out_dir / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"source train rows: {len(train_rows)}")
    print(f"source dev rows  : {len(dev_rows)}")
    print(f"merged rows      : {len(all_rows)}")
    print(f"output dir       : {args.out_dir}")
    for i, fold in enumerate(folds):
        print(f"fold_{i}: val={len(fold)} train={len(all_rows) - len(fold)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 5-fold protocols for AT-ADD Track-2")
    parser.add_argument("--train_label", type=Path, default=DEFAULT_TRAIN_LABEL)
    parser.add_argument("--dev_label", type=Path, default=DEFAULT_DEV_LABEL)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config_template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--train_dev_audio", type=Path, default=DEFAULT_TRAIN_DEV_AUDIO)
    parser.add_argument("--eval_audio", type=Path, default=DEFAULT_EVAL_AUDIO)
    parser.add_argument("--ckpt_root", type=str, default="./ckpt_t2_multi_head_k_fold")
    args = parser.parse_args()
    if args.n_splits < 2:
        parser.error("--n_splits must be >= 2")
    return args


if __name__ == "__main__":
    build_k_fold(parse_args())
