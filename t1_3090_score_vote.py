#!/usr/bin/env python3
"""Vote/ensemble score files for AT-ADD submissions.

The script accepts multiple logit files and/or multiple prediction-label files.

For logit files, rows with the same utterance id are aggregated across files by
mean and max, then thresholded into prediction CSVs.

For prediction-label files, rows with the same utterance id are aggregated by
majority vote into a prediction CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable


A1 = "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_layer/xlsr_3_11_24"
A2 = "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_layer/xlsr_11"

B2 = "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_layer_train_seed/xlsr_3_11_24_seed666"

C1 = "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_train_dev_split/xlsr_3_11_24_cat_proj_v1_random_seed2026"
C2 = "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_train_dev_split/xlsr_3_11_24_cat_proj_v1_random_seed42"

A1_binary = A1+"/result/atadd-track1_binary_eval.csv"
A1_logits = A1+"/result/atadd-track1_logits_eval.csv"
A2_binary = A2+"/result/atadd-track1_binary_eval.csv"
A2_logits = A2+"/result/atadd-track1_logits_eval.csv"

B2_binary = B2+"/result/atadd-track1_binary_eval.csv"
B2_logits = B2+"/result/atadd-track1_logits_eval.csv"

C1_binary = C1+"/result/atadd-track1_binary_eval.csv"
C1_logits = C1+"/result/atadd-track1_logits_eval.csv"
C2_binary = C2+"/result/atadd-track1_binary_eval.csv"
C2_logits = C2+"/result/atadd-track1_logits_eval.csv"

pred_labels = [
    A1_binary,
    A2_binary,
    B2_binary,
    C1_binary,
    C2_binary,
]
logits_files = [
    A1_logits,
    A2_logits,
    B2_logits,
    C1_logits,
    C2_logits,
]
DEFAULT_OUTPUT_DIR = Path(
    "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_vote/A1_A2_B2_C1_C2/"
)



# 输出格式参考
DEFAULT_TEMPLATE = ( "/home/new_disk/liulan/workspace/released_model/ADD-TRACK1/ckpt_t1_vote/train_seed_vote/vote_binary.csv"
)

UTT_CANDIDATES = ("name")
LOGIT_CANDIDATES = ("predict")
LABEL_CANDIDATES = ("predict")  



def parse_path_list(values: list[str] | None) -> list[Path]:
    """Parse repeated CLI values, JSON lists, and comma-separated path lists."""
    if not values:
        return []

    paths: list[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value.startswith("["):
            loaded = json.loads(value)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON list of paths, got: {value}")
            paths.extend(str(item) for item in loaded)
        elif "," in value:
            paths.extend(item.strip() for item in value.split(",") if item.strip())
        else:
            paths.append(value)
    return [Path(path).expanduser() for path in paths]


def resolve_existing_file(path: Path) -> Path | None:
    """Resolve common output-name variants used by different inference scripts."""
    if path.is_file():
        return path

    candidates: list[Path] = []
    name = path.name
    if name.startswith("atadd-"):
        candidates.append(path.with_name(name.removeprefix("atadd-")))
    if name.endswith(".csv") and not name.endswith("_binary.csv"):
        candidates.append(path.with_name(name[:-4] + "_binary.csv"))
    if name.startswith("atadd-") and name.endswith(".csv"):
        candidates.append(path.with_name(name.removeprefix("atadd-")[:-4] + "_binary.csv"))

    for candidate in candidates:
        if candidate.is_file():
            print(f"INFO: use existing file instead: {candidate}")
            return candidate
    return None


def keep_existing_files(paths: list[Path], kind: str) -> list[Path]:
    """Return existing files and warn about bad paths before voting starts."""
    existing: list[Path] = []
    missing: list[Path] = []

    for path in paths:
        resolved = resolve_existing_file(path)
        if resolved is not None:
            existing.append(resolved)
        else:
            missing.append(path)

    for path in missing:
        print(f"WARNING: skip missing {kind} file: {path}")

    if paths and not existing:
        raise SystemExit(
            f"No valid {kind} files found. Please check the paths configured "
            f"at the top of score_vote.py or passed by command line."
        )
    return existing


def sniff_dialect(path: Path) -> csv.Dialect:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel_tab if path.suffix.lower() in {".tsv", ".tab"} else csv.excel


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def choose_column(
    fieldnames: Iterable[str],
    explicit: str | None,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    fields = list(fieldnames)
    if explicit:
        if explicit in fields:
            return explicit
        normalized_explicit = normalize_name(explicit)
        for field in fields:
            if normalize_name(field) == normalized_explicit:
                return field
        raise ValueError(f"Cannot find {role} column {explicit!r}; available columns: {fields}")

    normalized = {normalize_name(field): field for field in fields}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    if role == "utt" and fields:
        return fields[0]
    if role in {"logit", "label"} and len(fields) >= 2:
        return fields[1]
    raise ValueError(f"Cannot infer {role} column from columns: {fields}")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    dialect = sniff_dialect(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a header row")
        return reader.fieldnames, list(reader)


def read_template_header(template: Path) -> list[str] | None:
    if not template.exists():
        return None
    fields, _ = read_rows(template)
    return fields


def write_logits(path: Path, utt_order: list[str], values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "score"])
        writer.writeheader()
        for utt in utt_order:
            writer.writerow({"name": utt, "score": f"{values[utt]:.10g}"})


def write_pred(
    path: Path,
    utt_order: list[str],
    preds: dict[str, str | int],
    template_header: list[str] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = template_header or ["name", "predict"]
    if len(fieldnames) < 2:
        raise ValueError(f"Prediction output needs at least two columns, got: {fieldnames}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for utt in utt_order:
            row = {field: "" for field in fieldnames}
            row[fieldnames[0]] = utt
            row[fieldnames[1]] = str(preds[utt])
            writer.writerow(row)


def vote_logits(
    logit_files: list[Path],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    threshold: float = 0.5,
    prefix: str = "logits_vote",
    utt_col: str | None = None,
    logit_col: str | None = None,
    template: Path = DEFAULT_TEMPLATE,
) -> dict[str, Path]:
    scores: dict[str, list[float]] = defaultdict(list)
    utt_order_map: OrderedDict[str, None] = OrderedDict()

    for path in logit_files:
        fields, rows = read_rows(path)
        utt_field = choose_column(fields, utt_col, UTT_CANDIDATES, "utt")
        logit_field = choose_column(fields, logit_col, LOGIT_CANDIDATES, "logit")
        for row in rows:
            utt = row[utt_field].strip()
            if not utt:
                continue
            scores[utt].append(float(row[logit_field]))
            utt_order_map.setdefault(utt, None)

    if not scores:
        raise ValueError("No logits were loaded.")

    utt_order = list(utt_order_map)
    mean_logits = {utt: sum(values) / len(values) for utt, values in scores.items()}
    mean_preds = {utt: "real" if mean_logits[utt] >= threshold else "fake" for utt in utt_order}
    max_logits = {utt: max(values) for utt, values in scores.items()}
    max_preds = {utt: "real" if max_logits[utt] >= threshold else "fake" for utt in utt_order}
    template_header = read_template_header(template)

    paths = {
        "mean_logits": output_dir / f"{prefix}_logits_mean.csv",
        "mean_pred": output_dir / f"atadd-track2_pred_{prefix}_mean.csv",
        "max_logits": output_dir / f"{prefix}_logits_max.csv",
        "max_pred": output_dir / f"atadd-track2_pred_{prefix}_max.csv",
    }
    write_logits(paths["mean_logits"], utt_order, mean_logits)
    write_pred(paths["mean_pred"], utt_order, mean_preds, template_header)
    write_logits(paths["max_logits"], utt_order, max_logits)
    write_pred(paths["max_pred"], utt_order, max_preds, template_header)
    return paths


def majority_label(labels: list[str]) -> str:
    counts = Counter(labels)
    best_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return tied[0]

    binary_rank = {"fake": 0, "real": 1}
    normalized_tied = [label.lower() for label in tied]
    if all(label in binary_rank for label in normalized_tied):
        return max(tied, key=lambda label: binary_rank[label.lower()])

    def tie_key(label: str) -> tuple[int, float | str]:
        try:
            return (1, float(label))
        except ValueError:
            return (0, label)

    return max(tied, key=tie_key)


def vote_pred_labels(
    pred_label_files: list[Path],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prefix: str = "binary_vote",
    utt_col: str | None = None,
    label_col: str | None = None,
    template: Path = DEFAULT_TEMPLATE,
) -> dict[str, Path]:
    labels: dict[str, list[str]] = defaultdict(list)
    utt_order_map: OrderedDict[str, None] = OrderedDict()

    for path in pred_label_files:
        fields, rows = read_rows(path)
        utt_field = choose_column(fields, utt_col, UTT_CANDIDATES, "utt")
        label_field = choose_column(fields, label_col, LABEL_CANDIDATES, "label")
        for row in rows:
            utt = row[utt_field].strip()
            if not utt:
                continue
            labels[utt].append(row[label_field].strip())
            utt_order_map.setdefault(utt, None)

    if not labels:
        raise ValueError("No prediction labels were loaded.")

    utt_order = list(utt_order_map)
    preds = {utt: majority_label(values) for utt, values in labels.items()}
    template_header = read_template_header(template)
    paths = {
        "majority_pred": output_dir / f"atadd-track2_pred_{prefix}_majority.csv",
    }
    write_pred(paths["majority_pred"], utt_order, preds, template_header)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate AT-ADD logits by mean/max and pred labels by majority vote."
    )
    parser.add_argument("--logits", nargs="*", default=[], help="Logit CSV/TSV files.")
    parser.add_argument("--pred_labels", nargs="*", default=[], help="Pred-label CSV/TSV files.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for logit preds.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="score_vote", help="Prefix for output filenames.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="CSV whose header is reused for pred outputs.")
    parser.add_argument("--utt_col", default=None, help="Utterance-id column name.")
    parser.add_argument("--logit_col", default=None, help="Logit/score column name.")
    parser.add_argument("--label_col", default=None, help="Prediction-label column name.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logit_files = parse_path_list(args.logits) or parse_path_list(logits_files)
    pred_label_files = parse_path_list(args.pred_labels) or parse_path_list(pred_labels)
    logit_files = keep_existing_files(logit_files, "logits")
    pred_label_files = keep_existing_files(pred_label_files, "pred_labels")

    if not logit_files and not pred_label_files:
        raise SystemExit(
            "Please provide --logits/--pred_labels, or edit logits_files/pred_labels "
            "at the top of score_vote.py."
        )

    written: dict[str, Path] = {}
    if logit_files:
        written.update(
            vote_logits(
                logit_files=logit_files,
                output_dir=args.output_dir,
                threshold=args.threshold,
                utt_col=args.utt_col,
                logit_col=args.logit_col,
                template=args.template,
            )
        )
    if pred_label_files:
        written.update(
            vote_pred_labels(
                pred_label_files=pred_label_files,
                output_dir=args.output_dir,
                utt_col=args.utt_col,
                label_col=args.label_col,
                template=args.template,
            )
        )

    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
