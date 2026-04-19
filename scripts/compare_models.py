"""
Compare multiple models on the AT-ADD Track-2 dev set.
Computes accuracy and macro-F1 for the overall set and each audio type
(speech, sound, singing, music), then prints a formatted comparison table.

Supported result file formats (tried in priority order per model):
  1. analysis_dev_attention/result/atadd-track2_logits_dev.csv
        columns: name, predict, type, label  (fully self-contained)
  2. result/atadd-track2_binary_dev.csv
        columns: name, predict               (needs external reference)
  3. result/atadd-track2_logits_dev.csv or result/atadd-track2_logits.csv
        columns: name, score                 (threshold 0.5 > real; needs ref)
"""

import csv
import os
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────

CKPT_ROOT = os.path.join(os.path.dirname(__file__), "..", "ckpt_t2")

# Priority list of relative paths to try inside each model directory.
# Tuples of (relative_path, file_format)
# file_format:
#   "full"   – name, predict, type, label
#   "binary" – name, predict
#   "score"  – name, score  (predict = "real" if score >= SCORE_THRESHOLD)
SCORE_THRESHOLD = 0.5

RESULT_CANDIDATES = [
    ("analysis_dev_attention/result/atadd-track2_logits_dev.csv", "full"),
    ("result/atadd-track2_binary_dev.csv", "binary"),
    ("result/atadd-track2_logits_dev.csv", "score"),
    ("result/atadd-track2_logits.csv", "score"),
]

# Audio types to report individually (in display order)
AUDIO_TYPES = ["speech", "sound", "singing", "music"]


# ── helper functions ──────────────────────────────────────────────────────────

def f1_binary(y_true, y_pred, pos_label):
    """Compute F1 for a single class (pos_label)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(y_true, y_pred):
    """Return (accuracy, macro_f1) for binary real/fake predictions."""
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    acc    = sum(t == p for t, p in zip(y_true, y_pred)) / n
    f1_r   = f1_binary(y_true, y_pred, "real")
    f1_f   = f1_binary(y_true, y_pred, "fake")
    macro_f1 = (f1_r + f1_f) / 2
    return acc, macro_f1


def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_reference(rows):
    """Build {name -> {"type": ..., "label": ...}} from full-format rows."""
    return {r["name"]: {"type": r["type"], "label": r["label"]} for r in rows}


def load_model_predictions(model_dir, reference):
    """
    Try each candidate file in order; return list of dicts with keys
    name, predict, type, label.  Returns None if no usable file found.
    """
    for rel_path, fmt in RESULT_CANDIDATES:
        full_path = os.path.join(model_dir, rel_path)
        if not os.path.isfile(full_path):
            continue
        rows = read_csv_dicts(full_path)
        if not rows:          # empty file
            continue

        if fmt == "full":
            # Validate expected columns
            if "predict" in rows[0] and "type" in rows[0] and "label" in rows[0]:
                return rows, rel_path

        elif fmt == "binary":
            if "predict" not in rows[0]:
                continue
            if reference is None:
                continue
            enriched = []
            for r in rows:
                ref = reference.get(r["name"])
                if ref is None:
                    continue
                enriched.append({
                    "name":    r["name"],
                    "predict": r["predict"],
                    "type":    ref["type"],
                    "label":   ref["label"],
                })
            if enriched:
                return enriched, rel_path

        elif fmt == "score":
            if "score" not in rows[0]:
                continue
            if reference is None:
                continue
            enriched = []
            for r in rows:
                ref = reference.get(r["name"])
                if ref is None:
                    continue
                score   = float(r["score"])
                predict = "real" if score >= SCORE_THRESHOLD else "fake"
                enriched.append({
                    "name":    r["name"],
                    "predict": predict,
                    "type":    ref["type"],
                    "label":   ref["label"],
                })
            if enriched:
                return enriched, rel_path

    return None, None


# ── discover models and build a shared reference ──────────────────────────────

def discover_models(ckpt_root):
    """Return sorted list of model directory names under ckpt_root."""
    return sorted(
        d for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d))
    )


def build_reference(ckpt_root, model_names):
    """
    Build a {name -> {type, label}} reference from the first full-format
    file we can find across all models.
    """
    for model in model_names:
        model_dir = os.path.join(ckpt_root, model)
        for rel_path, fmt in RESULT_CANDIDATES:
            if fmt != "full":
                continue
            full_path = os.path.join(model_dir, rel_path)
            if not os.path.isfile(full_path):
                continue
            rows = read_csv_dicts(full_path)
            if rows and "type" in rows[0] and "label" in rows[0]:
                ref = load_reference(rows)
                print(f"[info] Reference built from: {model}/{rel_path}  ({len(ref)} samples)")
                return ref
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model_names = discover_models(CKPT_ROOT)
    reference   = build_reference(CKPT_ROOT, model_names)

    # Collect per-model results
    results = []

    for model in model_names:
        model_dir = os.path.join(CKPT_ROOT, model)
        rows, src = load_model_predictions(model_dir, reference)

        if rows is None:
            print(f"[skip] {model}  – no usable result file found")
            continue

        # Overall
        y_true = [r["label"]   for r in rows]
        y_pred = [r["predict"] for r in rows]
        oa, of1 = compute_metrics(y_true, y_pred)

        # Per audio type
        type_metrics = {}
        type_data = defaultdict(lambda: ([], []))
        for r in rows:
            t = r["type"]
            type_data[t][0].append(r["label"])
            type_data[t][1].append(r["predict"])
        for t in AUDIO_TYPES:
            if t in type_data:
                ta, tf1 = compute_metrics(*type_data[t])
            else:
                ta, tf1 = float("nan"), float("nan")
            type_metrics[t] = (ta, tf1)

        results.append({
            "model":       model,
            "src":         src,
            "n":           len(rows),
            "overall":     (oa, of1),
            "per_type":    type_metrics,
        })

    if not results:
        print("No results found.")
        return

    # ── print table ───────────────────────────────────────────────────────────

    col_width  = max(len(r["model"]) for r in results) + 2
    type_cols  = AUDIO_TYPES
    metric_hdr = ["Acc", "F1"]   # two sub-columns per type

    # Header line 1: Model | Overall || speech || sound || singing || music
    header_types = ["Overall"] + [t.capitalize() for t in type_cols]
    # Each type block occupies two sub-columns (Acc, F1)

    def fmt_pct(v):
        return f"{v*100:6.2f}" if v == v else "   N/A"

    # Build column separator string
    sep_unit = "-" * 15  # Acc + F1 per type

    print("\n" + "=" * (col_width + 1 + len(header_types) * 16))
    print(f"{'Model':<{col_width}} | " +
          " | ".join(f"{'  '+h+'  ':^13}" for h in header_types))
    print(f"{'':<{col_width}} | " +
          " | ".join(f"{'Acc':>6}  {'F1':>5}" for _ in header_types))
    print("-" * (col_width + 1 + len(header_types) * 16))

    for r in results:
        oa, of1 = r["overall"]
        row_str = f"{r['model']:<{col_width}} | {fmt_pct(oa)} {fmt_pct(of1)}"
        for t in type_cols:
            ta, tf1 = r["per_type"][t]
            row_str += f" | {fmt_pct(ta)} {fmt_pct(tf1)}"
        print(row_str)

    print("=" * (col_width + 1 + len(header_types) * 16))
    print("\nMetrics: Acc = Accuracy (%), F1 = Macro-F1 (%)")
    print(f"Score threshold for logit files: {SCORE_THRESHOLD}")

    # ── also save as CSV ──────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "..", "model_comparison_dev.csv")
    fieldnames = ["model", "source_file", "n_samples",
                  "overall_acc", "overall_f1"] + \
                 [f"{t}_acc" for t in AUDIO_TYPES] + \
                 [f"{t}_f1"  for t in AUDIO_TYPES]

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            oa, of1 = r["overall"]
            row = {
                "model":       r["model"],
                "source_file": r["src"],
                "n_samples":   r["n"],
                "overall_acc": f"{oa*100:.4f}" if oa == oa else "",
                "overall_f1":  f"{of1*100:.4f}" if of1 == of1 else "",
            }
            for t in AUDIO_TYPES:
                ta, tf1 = r["per_type"][t]
                row[f"{t}_acc"] = f"{ta*100:.4f}" if ta == ta else ""
                row[f"{t}_f1"]  = f"{tf1*100:.4f}" if tf1 == tf1 else ""
            writer.writerow(row)

    print(f"\nResults also saved to: {out_path}")


if __name__ == "__main__":
    main()
