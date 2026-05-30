#!/usr/bin/env python3
"""Create type-balanced generator-disjoint protocol splits for AT-ADD Track 1.

Compared with make_gendisjoint_split.py, this v2 script does not require a
hand-picked held-out generator list. It automatically selects held-out fake
generators per audio type so that validation covers speech/sound/singing/music
more evenly while keeping fake generators disjoint between train and validation.

Example:

python make_gendisjoint_split_v2.py \
  --train-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/train.csv \
  --dev-csv /home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv \
  --out-dir /home/new_disk/liulan/workspace/released_model/ADD-TRACK1/protocol_gendisjoint_v2
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "2.0"

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
TYPE_CANDIDATES = ("type", "audio_type", "class_type", "category")
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

REAL_TOKENS = {"real", "bonafide", "bona-fide", "genuine", "human", "0", "false"}
FAKE_TOKENS = {"fake", "spoof", "generated", "synthetic", "1", "true"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build type-balanced generator-disjoint train/dev protocol CSVs."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fake-val-ratio",
        type=float,
        default=0.15,
        help=(
            "Target fake validation ratio inside each audio type. Since whole "
            "generators are held out, the actual ratio is approximate."
        ),
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Audio types to balance. Default: all types found in fake samples.",
    )
    parser.add_argument(
        "--min-generators-per-type",
        type=int,
        default=1,
        help="Minimum held-out generators for each selected audio type.",
    )
    parser.add_argument(
        "--max-generators-per-type",
        type=int,
        default=None,
        help="Optional cap on held-out generators for each audio type.",
    )
    parser.add_argument(
        "--force-heldout-generators",
        nargs="*",
        default=None,
        help="Always hold out these fake generators in addition to automatic picks.",
    )
    parser.add_argument(
        "--exclude-heldout-generators",
        nargs="*",
        default=None,
        help="Never choose these generators automatically.",
    )
    parser.add_argument(
        "--real-ratio-to-fake",
        type=float,
        default=0.25,
        help="Real validation samples per type = fake validation count * this ratio.",
    )
    parser.add_argument("--min-real-val-per-type", type=int, default=300)
    parser.add_argument("--max-real-val-per-type", type=int, default=2000)
    parser.add_argument("--max-real-val-frac-per-type", type=float, default=0.30)
    parser.add_argument(
        "--real-strategy",
        choices=("type_stratified", "official_dev_real", "none"),
        default="type_stratified",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Merge train+dev into train_full.csv without holdout.",
    )
    parser.add_argument("--label-col", default="auto")
    parser.add_argument("--generator-col", default="auto")
    parser.add_argument("--type-col", default="auto")
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
    type_col_arg: str,
) -> dict[str, str | None]:
    columns = [c for c in df.columns if not c.startswith("_")]
    label_col = _infer_one_column(columns, label_col_arg, LABEL_CANDIDATES, "label")
    generator_col = _infer_one_column(
        columns, generator_col_arg, GENERATOR_CANDIDATES, "generator"
    )
    type_col = _infer_one_column(columns, type_col_arg, TYPE_CANDIDATES, "type")

    source_col = None
    lower_to_original = {c.lower(): c for c in columns}
    for cand in SOURCE_CANDIDATES:
        if cand.lower() in lower_to_original:
            source_col = lower_to_original[cand.lower()]
            break

    return {
        "label": label_col,
        "generator": generator_col,
        "type": type_col,
        "source": source_col,
    }


def normalize_label_mask(df: pd.DataFrame, label_col: str) -> tuple[pd.Series, pd.Series]:
    norm = df[label_col].astype(str).str.strip().str.lower()
    unknown = sorted(set(norm.dropna().unique()) - (REAL_TOKENS | FAKE_TOKENS))
    if unknown:
        raise ValueError(
            f"Unrecognized label values in {label_col!r}: {unknown}. "
            "Expected real/fake style labels or binary 0/1."
        )

    real_mask = norm.isin(REAL_TOKENS)
    fake_mask = norm.isin(FAKE_TOKENS)
    if not real_mask.any():
        raise ValueError("No real samples found after label normalization.")
    if not fake_mask.any():
        raise ValueError("No fake samples found after label normalization.")
    return real_mask, fake_mask


def _norm_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _validate_generator_names(
    available: set[str],
    names: list[str] | None,
    option_name: str,
) -> set[str]:
    if not names:
        return set()
    out = {str(name).strip() for name in names if str(name).strip()}
    missing = sorted(out - available)
    if missing:
        raise ValueError(
            f"{option_name} contains unknown generator(s): {missing}. "
            f"Available: {sorted(available)}"
        )
    return out


def _generator_stats(
    fake_df: pd.DataFrame,
    generator_col: str,
    type_col: str,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for gen, gen_df in fake_df.groupby(generator_col, dropna=False, sort=True):
        gen_name = str(gen).strip()
        type_counts = Counter(_norm_values(gen_df[type_col]))
        primary_type = sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        stats[gen_name] = {
            "total": int(len(gen_df)),
            "type_counts": {str(k): int(v) for k, v in sorted(type_counts.items())},
            "primary_type": str(primary_type),
        }
    return stats


def _choose_generators_for_type(
    candidates: list[str],
    stats: dict[str, dict[str, Any]],
    type_name: str,
    target_count: int,
    min_count: int,
    max_count: int | None,
    seed: int,
) -> list[str]:
    if not candidates:
        return []

    upper = len(candidates) if max_count is None else min(max_count, len(candidates))
    lower = min(min_count, upper)
    best: tuple[float, int, int, list[str]] | None = None

    # Candidate groups are small in AT-ADD. Enumerating combinations gives a
    # stable split whose count is genuinely closest to the per-type target.
    from itertools import combinations

    for size in range(lower, upper + 1):
        for combo in combinations(sorted(candidates), size):
            type_total = sum(
                int(stats[g]["type_counts"].get(type_name, 0)) for g in combo
            )
            all_total = sum(int(stats[g]["total"]) for g in combo)
            distance = abs(type_total - target_count)
            over_penalty = 0 if type_total <= target_count else 0.1
            score = (distance + over_penalty, size, all_total, list(combo))
            if best is None or score < best:
                best = score

    if best is None:
        return []

    selected = list(best[3])
    random.Random(seed + sum(ord(ch) for ch in type_name)).shuffle(selected)
    return sorted(selected)


def select_heldout_generators(
    df: pd.DataFrame,
    fake_mask: pd.Series,
    columns: dict[str, str | None],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    generator_col = str(columns["generator"])
    type_col = str(columns["type"])
    fake_df = df.loc[fake_mask].copy()
    fake_df[generator_col] = _norm_values(fake_df[generator_col])
    fake_df[type_col] = _norm_values(fake_df[type_col])

    stats = _generator_stats(fake_df, generator_col, type_col)
    available = set(stats)
    forced = _validate_generator_names(
        available, args.force_heldout_generators, "--force-heldout-generators"
    )
    excluded = _validate_generator_names(
        available, args.exclude_heldout_generators, "--exclude-heldout-generators"
    )
    blocked = excluded - forced

    fake_type_counts = {
        str(k): int(v) for k, v in fake_df[type_col].value_counts().sort_index().items()
    }
    selected_types = args.types or sorted(fake_type_counts)
    missing_types = sorted(set(selected_types) - set(fake_type_counts))
    if missing_types:
        raise ValueError(
            f"--types contains type(s) without fake samples: {missing_types}. "
            f"Available: {sorted(fake_type_counts)}"
        )

    primary_groups: dict[str, list[str]] = {t: [] for t in selected_types}
    for gen, gen_stats in stats.items():
        if gen in blocked:
            continue
        primary_type = str(gen_stats["primary_type"])
        if primary_type in primary_groups:
            primary_groups[primary_type].append(gen)

    selected_by_type: dict[str, list[str]] = {}
    selected: set[str] = set(forced)
    for type_name in selected_types:
        candidates = [g for g in primary_groups.get(type_name, []) if g not in selected]
        target = max(1, round(fake_type_counts[type_name] * args.fake_val_ratio))
        picked = _choose_generators_for_type(
            candidates,
            stats,
            type_name,
            target,
            max(0, args.min_generators_per_type),
            args.max_generators_per_type,
            args.seed,
        )
        selected.update(picked)
        selected_by_type[type_name] = picked

    if not selected:
        raise ValueError("No held-out generators selected. Check ratios and filters.")

    selected_list = sorted(selected)
    selection_report = {
        "strategy": "type_balanced_auto_generator_holdout",
        "fake_val_ratio": args.fake_val_ratio,
        "selected_types": selected_types,
        "fake_type_counts": fake_type_counts,
        "selected_by_type": selected_by_type,
        "forced_heldout_generators": sorted(forced),
        "excluded_heldout_generators": sorted(excluded),
        "generator_stats": stats,
    }
    return selected_list, selection_report


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
    val_fake_df: pd.DataFrame,
    columns: dict[str, str | None],
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    if args.real_strategy == "none":
        return [], {"strategy": "none", "selected_by_type": {}}

    type_col = str(columns["type"])
    real_df = df.loc[real_mask].copy()
    real_df[type_col] = _norm_values(real_df[type_col])
    val_fake_types = _norm_values(val_fake_df[type_col])
    fake_val_type_counts = Counter(val_fake_types)

    if args.real_strategy == "official_dev_real":
        indices = list(real_df.index[real_df["_orig_split"] == "dev"])
        selected_counts = Counter(_norm_values(real_df.loc[indices, type_col]))
        return indices, {
            "strategy": "official_dev_real",
            "selected_by_type": {str(k): int(v) for k, v in sorted(selected_counts.items())},
        }

    selected_indices: list[int] = []
    selected_by_type: dict[str, int] = {}
    for type_name, fake_count in sorted(fake_val_type_counts.items()):
        type_real_df = real_df[real_df[type_col] == type_name]
        if type_real_df.empty:
            selected_by_type[str(type_name)] = 0
            continue

        requested = min(
            max(args.min_real_val_per_type, round(fake_count * args.real_ratio_to_fake)),
            args.max_real_val_per_type,
            math.floor(args.max_real_val_frac_per_type * len(type_real_df)),
        )
        requested = min(int(requested), len(type_real_df))
        picked = _stratified_sample_indices(
            type_real_df,
            requested,
            ["_orig_split"],
            args.seed + sum(ord(ch) for ch in str(type_name)),
        )
        selected_indices.extend(picked)
        selected_by_type[str(type_name)] = len(picked)

    return selected_indices, {
        "strategy": "type_stratified",
        "real_ratio_to_fake": args.real_ratio_to_fake,
        "min_real_val_per_type": args.min_real_val_per_type,
        "max_real_val_per_type": args.max_real_val_per_type,
        "max_real_val_frac_per_type": args.max_real_val_frac_per_type,
        "selected_by_type": selected_by_type,
    }


def make_split(
    df: pd.DataFrame,
    real_mask: pd.Series,
    fake_mask: pd.Series,
    columns: dict[str, str | None],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    generator_col = str(columns["generator"])
    heldout_generators, selection_report = select_heldout_generators(
        df, fake_mask, columns, args
    )

    generators = _norm_values(df[generator_col])
    val_fake_mask = fake_mask & generators.isin(heldout_generators)
    if not val_fake_mask.any():
        raise ValueError("Held-out fake sample count is 0.")

    val_fake_df = df.loc[val_fake_mask].copy()
    val_real_indices, real_report = sample_real_validation(
        df, real_mask, val_fake_df, columns, args
    )

    val_mask = val_fake_mask.copy()
    if val_real_indices:
        val_mask.loc[val_real_indices] = True

    val_df = df.loc[val_mask].copy()
    train_df = df.loc[~val_mask].copy()

    meta = {
        "script_version": SCRIPT_VERSION,
        "seed": args.seed,
        "heldout_generators": heldout_generators,
        "columns": columns,
        "selection": selection_report,
        "real_selection": real_report,
    }
    return train_df, val_df, meta


def _label_counts(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    return {
        str(k): int(v)
        for k, v in df[label_col].astype(str).str.strip().str.lower().value_counts().items()
    }


def _fake_generators(df: pd.DataFrame, label_col: str, generator_col: str) -> list[str]:
    norm = df[label_col].astype(str).str.strip().str.lower()
    fake = norm.isin(FAKE_TOKENS)
    return sorted(df.loc[fake, generator_col].astype(str).str.strip().unique().tolist())


def _type_counts(df: pd.DataFrame, type_col: str) -> dict[str, int]:
    return {
        str(k): int(v)
        for k, v in df[type_col].astype(str).str.strip().value_counts().sort_index().items()
    }


def _fake_type_counts(df: pd.DataFrame, label_col: str, type_col: str) -> dict[str, int]:
    norm = df[label_col].astype(str).str.strip().str.lower()
    return _type_counts(df.loc[norm.isin(FAKE_TOKENS)], type_col)


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
    type_col = str(columns["type"])

    report: dict[str, Any] = {"final": final}
    if final:
        print("===== FINAL TRAIN FULL =====")
        print(f"total samples: {len(train_df)}")
        print(f"real/fake counts: {_label_counts(train_df, label_col)}")
        print(f"type counts: {_type_counts(train_df, type_col)}")
        report["train_full_rows"] = int(len(train_df))
        report["train_full_label_counts"] = _label_counts(train_df, label_col)
        report["train_full_type_counts"] = _type_counts(train_df, type_col)
        return report

    assert val_df is not None
    print("===== TYPE-BALANCED GENERATOR-DISJOINT SPLIT V2 =====")
    print(f"train samples: {len(train_df)}")
    print(f"val samples  : {len(val_df)}")
    print(f"train real/fake: {_label_counts(train_df, label_col)}")
    print(f"val real/fake  : {_label_counts(val_df, label_col)}")
    print(f"train type counts: {_type_counts(train_df, type_col)}")
    print(f"val type counts  : {_type_counts(val_df, type_col)}")
    print(f"val fake type counts: {_fake_type_counts(val_df, label_col, type_col)}")

    train_fake_gens = _fake_generators(train_df, label_col, generator_col)
    val_fake_gens = _fake_generators(val_df, label_col, generator_col)
    overlap = sorted(set(train_fake_gens) & set(val_fake_gens))
    print("held-out fake generators:")
    for gen in val_fake_gens:
        print(f"  {gen}")
    print(f"fake generator overlap: {overlap}")
    if overlap:
        raise RuntimeError(f"train/val fake generators are not disjoint: {overlap}")

    held_counts = (
        val_df.loc[
            val_df[label_col].astype(str).str.strip().str.lower().isin(FAKE_TOKENS),
            generator_col,
        ]
        .astype(str)
        .str.strip()
        .value_counts()
        .sort_index()
    )
    print("held-out generator counts in val:")
    for gen in meta["heldout_generators"]:
        print(f"  {gen}: {int(held_counts.get(gen, 0))}")

    print("original split counts:")
    print(f"  train output: {dict(train_df['_orig_split'].value_counts().sort_index())}")
    print(f"  val output  : {dict(val_df['_orig_split'].value_counts().sort_index())}")

    report.update(
        {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_label_counts": _label_counts(train_df, label_col),
            "val_label_counts": _label_counts(val_df, label_col),
            "train_type_counts": _type_counts(train_df, type_col),
            "val_type_counts": _type_counts(val_df, type_col),
            "val_fake_type_counts": _fake_type_counts(val_df, label_col, type_col),
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
        train_path = out_dir / "train_gendisjoint_v2.csv"
        val_path = out_dir / "val_gendisjoint_v2.csv"
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

    if not 0.0 < args.fake_val_ratio < 1.0:
        raise ValueError("--fake-val-ratio must be in (0, 1).")
    if args.real_ratio_to_fake < 0.0:
        raise ValueError("--real-ratio-to-fake must be non-negative.")

    df, original_columns = read_protocols(args.train_csv, args.dev_csv)
    columns = infer_columns(df, args.label_col, args.generator_col, args.type_col)
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
