#!/usr/bin/env python3
"""Create generator-disjoint protocol splits for AT-ADD Track 1.

This script only writes new protocol CSV files. It does not modify dataset or
training code.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd

"""

stage 1:
python make_gendisjoint_split.py \
  --train-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/train.csv \
  --dev-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv \
  --out-dir /home/new_disk/liulan/workspace/released_model/ADD-TRACK1/protocol_gendisjoint_v1_stage1 \
  --heldout-generators WaveGlow Tacotron2 GradTTS Llasa1B StarGANv2VC

stage 2:
python make_gendisjoint_split.py \
  --train-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/train.csv \
  --dev-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv \
  --out-dir /home/new_disk/liulan/workspace/released_model/ADD-TRACK1/protocol_gendisjoint_v1_stage2 \
  --heldout-generators GradTTS Llasa1B 

"""
SCRIPT_VERSION = "1.0"

LABEL_CANDIDATES = ("label", "target", "class", "bonafide", "is_fake")
GENERATOR_CANDIDATES = (
    "generator",
    "gen",
    "attack_id",
    "method",
    "spoof_type",
    "system",
    "model",
    "vocoder",
    "tts_model",
)
SOURCE_CANDIDATES = (
    "source",
    "corpus",
    "dataset",
    "speaker_source",
    "source_dataset",
    "origin",
    "database",
    "db",
    "real_source",
)
DEFAULT_HELDOUT_GENERATORS = [
    "WaveGlow",
    "Tacotron2",
    "GradTTS",
    "Llasa",
    "StarGANv2-VC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build generator-disjoint train/validation protocol CSVs."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--heldout-generators",
        nargs="*",
        default=DEFAULT_HELDOUT_GENERATORS,
        help="Fake generator names to hold out for validation.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Merge train+dev into train_full.csv without holdout.",
    )
    parser.add_argument(
        "--real-strategy",
        choices=("random_stratified", "official_dev_real", "source_holdout", "none"),
        default="random_stratified",
    )
    parser.add_argument(
        "--heldout-real-sources",
        nargs="*",
        default=None,
        help="Real source names to hold out when --real-strategy=source_holdout.",
    )
    parser.add_argument("--real-val-ratio", type=float, default=0.25)
    parser.add_argument("--min-real-val", type=int, default=2000)
    parser.add_argument("--max-real-val", type=int, default=6000)
    parser.add_argument("--max-real-val-frac", type=float, default=0.30)
    parser.add_argument("--label-col", default="auto")
    parser.add_argument("--generator-col", default="auto")
    return parser.parse_args()


def read_protocols(train_csv: Path, dev_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    train_csv = train_csv.expanduser().resolve()
    dev_csv = dev_csv.expanduser().resolve()
    if not train_csv.is_file():
        raise FileNotFoundError(f"train CSV not found: {train_csv}")
    if not dev_csv.is_file():
        raise FileNotFoundError(f"dev CSV not found: {dev_csv}")

    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    if list(train_df.columns) != list(dev_df.columns):
        raise ValueError(
            "train/dev CSV columns differ.\n"
            f"train: {list(train_df.columns)}\n"
            f"dev  : {list(dev_df.columns)}"
        )

    original_columns = list(train_df.columns)
    train_df = train_df.copy()
    dev_df = dev_df.copy()
    train_df["_orig_split"] = "train"
    dev_df["_orig_split"] = "dev"
    return pd.concat([train_df, dev_df], ignore_index=True), original_columns


def _infer_one_column(
    columns: list[str],
    requested: str,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    if requested != "auto":
        if requested not in columns:
            raise ValueError(f"{role} column {requested!r} not found. Columns: {columns}")
        return requested

    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    raise ValueError(
        f"Could not auto-detect {role} column. Columns: {columns}. "
        f"Pass --{role}-col explicitly."
    )


def infer_columns(
    df: pd.DataFrame,
    label_col_arg: str,
    generator_col_arg: str,
) -> dict[str, str | None]:
    columns = [c for c in df.columns if not c.startswith("_")]
    label_col = _infer_one_column(columns, label_col_arg, LABEL_CANDIDATES, "label")
    generator_col = _infer_one_column(
        columns, generator_col_arg, GENERATOR_CANDIDATES, "generator"
    )

    source_col = None
    lower_to_original = {c.lower(): c for c in columns}
    for cand in SOURCE_CANDIDATES:
        if cand.lower() in lower_to_original:
            source_col = lower_to_original[cand.lower()]
            break

    duration_col = None
    for col in columns:
        low = col.lower()
        if low in ("duration", "dur", "duration_sec", "seconds", "length_sec"):
            duration_col = col
            break

    type_col = None
    for col in columns:
        if col.lower() in ("type", "audio_type", "class_type", "category"):
            type_col = col
            break

    language_col = None
    for col in columns:
        if col.lower() in ("language", "lang", "locale"):
            language_col = col
            break

    return {
        "label": label_col,
        "generator": generator_col,
        "source": source_col,
        "duration": duration_col,
        "type": type_col,
        "language": language_col,
    }


def normalize_label_mask(df: pd.DataFrame, label_col: str) -> tuple[pd.Series, pd.Series]:
    values = df[label_col]
    norm = values.astype(str).str.strip().str.lower()

    real_tokens = {"real", "bonafide", "bona-fide", "genuine", "human", "0", "false"}
    fake_tokens = {"fake", "spoof", "generated", "synthetic", "1", "true"}
    known = real_tokens | fake_tokens
    unknown = sorted(set(norm.dropna().unique()) - known)
    if unknown:
        raise ValueError(
            f"Unrecognized label values in {label_col!r}: {unknown}. "
            "Expected real/fake style labels or binary 0/1."
        )

    real_mask = norm.isin(real_tokens)
    fake_mask = norm.isin(fake_tokens)
    if not real_mask.any():
        raise ValueError("No real samples found after label normalization.")
    if not fake_mask.any():
        raise ValueError("No fake samples found after label normalization.")
    return real_mask, fake_mask


def validate_heldout_generators(
    df: pd.DataFrame,
    fake_mask: pd.Series,
    generator_col: str,
    heldout_generators: list[str],
) -> list[str]:
    if not heldout_generators:
        raise ValueError("--heldout-generators cannot be empty unless --final is used.")

    available = sorted(
        str(g).strip()
        for g in df.loc[fake_mask, generator_col].dropna().unique().tolist()
    )
    available_set = set(available)
    missing = [g for g in heldout_generators if g not in available_set]
    if missing:
        lines = ["Held-out generator(s) not found:"]
        for name in missing:
            matches = get_close_matches(name, available, n=5, cutoff=0.0)
            lines.append(f"  {name!r}; closest candidates: {matches}")
        lines.append("Available fake generators:")
        lines.extend(f"  {g}" for g in available)
        raise ValueError("\n".join(lines))
    return heldout_generators


def _duration_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(["missing"] * len(series), index=series.index)
    try:
        return pd.qcut(numeric, q=5, duplicates="drop").astype(str).fillna("missing")
    except ValueError:
        return numeric.fillna(-1).astype(str)


def _stratified_sample_indices(
    df: pd.DataFrame,
    n: int,
    strata_cols: list[str],
    seed: int,
) -> list[int]:
    if n <= 0:
        return []
    if n >= len(df):
        return list(df.index)

    rng = random.Random(seed)
    if not strata_cols:
        return rng.sample(list(df.index), n)

    grouped = list(df.groupby(strata_cols, dropna=False, sort=True))
    total = len(df)
    allocations: list[tuple[Any, pd.DataFrame, int, float]] = []
    allocated = 0
    for key, group in grouped:
        raw = n * len(group) / total
        count = min(len(group), int(math.floor(raw)))
        allocations.append((key, group, count, raw - math.floor(raw)))
        allocated += count

    remaining = n - allocated
    allocations.sort(key=lambda item: item[3], reverse=True)
    for i, (key, group, count, frac) in enumerate(allocations):
        if remaining <= 0:
            break
        room = len(group) - count
        if room <= 0:
            continue
        add = min(room, remaining)
        allocations[i] = (key, group, count + add, frac)
        remaining -= add

    selected: list[int] = []
    for _key, group, count, _frac in allocations:
        if count <= 0:
            continue
        selected.extend(rng.sample(list(group.index), count))

    if len(selected) < n:
        remaining_pool = sorted(set(df.index) - set(selected))
        selected.extend(rng.sample(remaining_pool, min(n - len(selected), len(remaining_pool))))
    return selected


def sample_real_validation(
    df: pd.DataFrame,
    real_mask: pd.Series,
    fake_val_count: int,
    columns: dict[str, str | None],
    args: argparse.Namespace,
) -> list[int]:
    real_df = df.loc[real_mask].copy()
    if args.real_strategy == "none":
        return []

    if args.real_strategy == "official_dev_real":
        return list(real_df.index[real_df["_orig_split"] == "dev"])

    if args.real_strategy == "source_holdout":
        source_col = columns["source"]
        if source_col is None:
            raise ValueError("--real-strategy=source_holdout requires a source column.")
        if not args.heldout_real_sources:
            raise ValueError(
                "--real-strategy=source_holdout requires --heldout-real-sources."
            )
        source_values = real_df[source_col].astype(str).str.strip()
        return list(real_df.index[source_values.isin(args.heldout_real_sources)])

    if args.real_strategy != "random_stratified":
        raise ValueError(f"Unsupported real strategy: {args.real_strategy}")

    total_real = len(real_df)
    requested = min(
        max(args.min_real_val, round(args.real_val_ratio * fake_val_count)),
        args.max_real_val,
        math.floor(args.max_real_val_frac * total_real),
    )
    if requested <= 0:
        print("WARNING: requested real validation count is 0; no real samples selected.")
        return []
    if requested > total_real:
        print(
            f"WARNING: requested {requested} real validation samples but only "
            f"{total_real} real samples are available; using all real samples."
        )
        requested = total_real

    strata_cols = ["_orig_split"]
    temp_cols: list[str] = []
    for key in ("duration", "language", "type"):
        col = columns.get(key)
        if not col:
            continue
        if key == "duration":
            tmp = "_duration_bucket"
            real_df[tmp] = _duration_bucket(real_df[col])
            temp_cols.append(tmp)
            strata_cols.append(tmp)
        else:
            strata_cols.append(col)

    if strata_cols == ["_orig_split"]:
        # _orig_split always exists. This keeps official train/dev proportion stable.
        pass
    return _stratified_sample_indices(real_df, int(requested), strata_cols, args.seed)


def make_split(
    df: pd.DataFrame,
    real_mask: pd.Series,
    fake_mask: pd.Series,
    columns: dict[str, str | None],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    generator_col = str(columns["generator"])
    heldout = validate_heldout_generators(
        df, fake_mask, generator_col, args.heldout_generators
    )
    generators = df[generator_col].astype(str).str.strip()
    val_fake_mask = fake_mask & generators.isin(heldout)
    if not val_fake_mask.any():
        raise ValueError("Held-out fake sample count is 0.")

    val_real_indices = sample_real_validation(
        df, real_mask, int(val_fake_mask.sum()), columns, args
    )
    val_mask = val_fake_mask.copy()
    if val_real_indices:
        val_mask.loc[val_real_indices] = True

    val_df = df.loc[val_mask].copy()
    train_df = df.loc[~val_mask].copy()

    meta = {
        "script_version": SCRIPT_VERSION,
        "seed": args.seed,
        "heldout_generators": heldout,
        "real_strategy": args.real_strategy,
        "real_val_ratio": args.real_val_ratio,
        "min_real_val": args.min_real_val,
        "max_real_val": args.max_real_val,
        "max_real_val_frac": args.max_real_val_frac,
        "columns": columns,
    }
    return train_df, val_df, meta


def _label_counts(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    return {
        str(k): int(v)
        for k, v in df[label_col].astype(str).str.strip().str.lower().value_counts().items()
    }


def _fake_generators(df: pd.DataFrame, label_col: str, generator_col: str) -> list[str]:
    norm = df[label_col].astype(str).str.strip().str.lower()
    fake = norm.isin({"fake", "spoof", "generated", "synthetic", "1", "true"})
    return sorted(df.loc[fake, generator_col].astype(str).str.strip().unique().tolist())


def _print_counter(title: str, values: pd.Series) -> None:
    print(title)
    for key, value in values.astype(str).str.strip().value_counts().sort_index().items():
        print(f"  {key!r}: {int(value)}")


def _duration_summary(df: pd.DataFrame, duration_col: str) -> dict[str, float]:
    s = pd.to_numeric(df[duration_col], errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
    }


def print_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    columns: dict[str, str | None],
    meta: dict[str, Any],
    *,
    final: bool,
) -> dict[str, Any]:
    label_col = str(columns["label"])
    generator_col = str(columns["generator"])
    source_col = columns["source"]
    duration_col = columns["duration"]

    report: dict[str, Any] = {"final": final}

    if final:
        print("===== FINAL TRAIN FULL =====")
        print(f"total samples: {len(train_df)}")
        print(f"real/fake counts: {_label_counts(train_df, label_col)}")
        _print_counter("generator distribution:", train_df[generator_col])
        report["train_full_rows"] = int(len(train_df))
        report["train_full_label_counts"] = _label_counts(train_df, label_col)
        return report

    assert val_df is not None
    print("===== GENERATOR-DISJOINT SPLIT =====")
    print(f"train samples: {len(train_df)}")
    print(f"val samples  : {len(val_df)}")
    print(f"train real/fake: {_label_counts(train_df, label_col)}")
    print(f"val real/fake  : {_label_counts(val_df, label_col)}")

    train_fake_gens = _fake_generators(train_df, label_col, generator_col)
    val_fake_gens = _fake_generators(val_df, label_col, generator_col)
    print("train fake generators:")
    for g in train_fake_gens:
        print(f"  {g}")
    print("val fake generators:")
    for g in val_fake_gens:
        print(f"  {g}")

    overlap = sorted(set(train_fake_gens) & set(val_fake_gens))
    print(f"fake generator overlap: {overlap}")
    if overlap:
        raise RuntimeError(f"train/val fake generators are not disjoint: {overlap}")

    print("held-out generator counts in val:")
    val_fake_norm = val_df[label_col].astype(str).str.strip().str.lower().isin(
        {"fake", "spoof", "generated", "synthetic", "1", "true"}
    )
    held_counts = (
        val_df.loc[val_fake_norm, generator_col]
        .astype(str)
        .str.strip()
        .value_counts()
        .sort_index()
    )
    for gen in meta["heldout_generators"]:
        print(f"  {gen}: {int(held_counts.get(gen, 0))}")

    train_real = train_df[label_col].astype(str).str.strip().str.lower().isin(
        {"real", "bonafide", "bona-fide", "genuine", "human", "0", "false"}
    )
    val_real = val_df[label_col].astype(str).str.strip().str.lower().isin(
        {"real", "bonafide", "bona-fide", "genuine", "human", "0", "false"}
    )
    print("real statistics:")
    print(f"  strategy: {meta['real_strategy']}")
    print(f"  train real: {int(train_real.sum())}")
    print(f"  val real  : {int(val_real.sum())}")
    if source_col:
        _print_counter("train real source distribution:", train_df.loc[train_real, source_col])
        _print_counter("val real source distribution:", val_df.loc[val_real, source_col])
    else:
        print(
            "WARNING: No source column found. Real samples are selected by "
            "random_stratified strategy; this split is generator-disjoint, "
            "not source-disjoint."
        )

    if duration_col:
        print("duration summary:")
        print(f"  train: {_duration_summary(train_df, duration_col)}")
        print(f"  val  : {_duration_summary(val_df, duration_col)}")

    if "_orig_split" in train_df.columns:
        print("original split counts:")
        print(f"  train output: {dict(train_df['_orig_split'].value_counts().sort_index())}")
        print(f"  val output  : {dict(val_df['_orig_split'].value_counts().sort_index())}")

    report.update(
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_label_counts": _label_counts(train_df, label_col),
            "val_label_counts": _label_counts(val_df, label_col),
            "train_fake_generators": train_fake_gens,
            "val_fake_generators": val_fake_gens,
            "fake_generator_overlap": overlap,
            "heldout_generator_val_counts": {
                gen: int(held_counts.get(gen, 0)) for gen in meta["heldout_generators"]
            },
            "train_orig_split_counts": {
                str(k): int(v) for k, v in train_df["_orig_split"].value_counts().items()
            },
            "val_orig_split_counts": {
                str(k): int(v) for k, v in val_df["_orig_split"].value_counts().items()
            },
        }
    )
    return report


def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    original_columns: list[str],
    out_dir: Path,
    meta: dict[str, Any],
    *,
    final: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "split_meta.json"

    if final:
        train_full_path = out_dir / "train_full.csv"
        train_df.loc[:, original_columns].to_csv(train_full_path, index=False)
        print(f"saved: {train_full_path}")
    else:
        assert val_df is not None
        train_path = out_dir / "train_gendisjoint.csv"
        val_path = out_dir / "val_gendisjoint.csv"
        train_df.loc[:, original_columns].to_csv(train_path, index=False)
        val_df.loc[:, original_columns].to_csv(val_path, index=False)
        print(f"saved: {train_path}")
        print(f"saved: {val_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved: {meta_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    df, original_columns = read_protocols(args.train_csv, args.dev_csv)
    columns = infer_columns(df, args.label_col, args.generator_col)
    real_mask, fake_mask = normalize_label_mask(df, str(columns["label"]))

    if args.final:
        meta = {
            "script_version": SCRIPT_VERSION,
            "seed": args.seed,
            "mode": "final",
            "columns": columns,
        }
        report = print_report(df, None, columns, meta, final=True)
        meta["report"] = report
        save_outputs(df, None, original_columns, args.out_dir, meta, final=True)
        return

    train_df, val_df, meta = make_split(df, real_mask, fake_mask, columns, args)
    report = print_report(train_df, val_df, columns, meta, final=False)
    meta["report"] = report
    save_outputs(train_df, val_df, original_columns, args.out_dir, meta, final=False)


if __name__ == "__main__":
    main()
