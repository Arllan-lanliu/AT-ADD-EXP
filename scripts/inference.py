"""
Inference script: load a trained model and write prediction scores to CSV.

Usage
-----
    python scripts/inference.py \
        --model_path  ./ckpt_t2/my_experiment \
        --gpu         0 \
        --batch_size  160 \
        --eval_task   atadd-track2 \
        --threshold   0.5

Fixed vs EER threshold for binary ``real``/``fake`` export
------------------------------------------------------------
* ``--eval_threshold_mode fixed`` (default): use ``--threshold`` (default 0.5;
  or ``score_threshold`` from saved training config when injected).
* ``--eval_threshold_mode eer``: set EER operating threshold from a **dev** score
  CSV (same format as ``score_file``: ``name,score``), aligned with labels in
  ``--val_label_csv`` (default: dev label path from config). Example::

    # 1) Score dev set (writes e.g. result/atadd-track2_logits_dev.csv)
    python scripts/inference.py --model_path ... --eval_audio <dev_wav_dir> \\
        --score_file result/atadd-track2_logits_dev.csv

    # 2) Score eval and binarize with dev-derived EER threshold
    python scripts/inference.py --model_path ... --eval_threshold_mode eer \\
        --val_score_file result/atadd-track2_logits_dev.csv
"""
import os
import sys
import json
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as a top-level script from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model
from data.dataset import atadd_eval_dataset
from utils import metrics as em

torch.multiprocessing.set_start_method('spawn', force=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _init_args(argv=None):
    parser = argparse.ArgumentParser(description="AT-ADD score generator")

    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved model directory (must contain args.json)")
    parser.add_argument('--gpu', type=str, default="0",
                        help="CUDA device index")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Inference batch size (default: from args.json)")
    parser.add_argument('--eval_audio', type=str, default=None,
                        help="Directory of audio to score (default: eval path from config)")
    parser.add_argument('--score_file', type=str, default=None,
                        help="Override path for the output logits CSV")
    parser.add_argument('--eval_task', type=str, default=None,
                        choices=["atadd-track1", "atadd-track2"],
                        help="Evaluation task (default: from args.json)")
    parser.add_argument('--threshold', type=float, default=None,
                        help="Fixed binary threshold: real if score >= threshold (default: score_threshold from config, else 0.5)")
    parser.add_argument(
        '--eval_threshold_mode', type=str, default=None,
        choices=['fixed', 'eer'],
        help="fixed | eer (default: from saved training config, else fixed)",
    )
    parser.add_argument(
        '--val_score_file', type=str, default=None,
        help="Dev logits CSV (name,score) for EER threshold; required when eval_threshold_mode=eer",
    )
    parser.add_argument(
        '--val_label_csv', type=str, default=None,
        help="Dev label CSV (name,type,label,...); default from config atadd_t*_dev_label",
    )

    # First pass to get --model_path so we can load the training config.
    temp_args, _ = parser.parse_known_args(argv)

    config_yaml = os.path.join(temp_args.model_path, "config.yaml")
    args_json   = os.path.join(temp_args.model_path, "args.json")

    if os.path.exists(config_yaml):
        # New format: load ATADDConfig from saved YAML
        from utils.config import ATADDConfig
        train_cfg = ATADDConfig.from_yaml(config_yaml)
        train_ns  = train_cfg.to_namespace()
        train_dict = vars(train_ns)
    elif os.path.exists(args_json):
        # Legacy format: flat args.json
        with open(args_json, "r") as f:
            train_dict = json.load(f)
    else:
        raise FileNotFoundError(
            f"Neither config.yaml nor args.json found in {temp_args.model_path}"
        )

    # Inject training args as extra argparse arguments with their saved defaults.
    _inference_reserved = frozenset({
        "eval_threshold_mode", "val_score_file", "val_label_csv", "threshold",
    })
    for key, value in train_dict.items():
        if key in _inference_reserved:
            continue
        if key not in vars(temp_args) and not key.startswith("_"):
            if isinstance(value, bool):
                parser.add_argument(
                    f'--{key}',
                    action='store_true' if value else 'store_false',
                    default=value,
                )
            else:
                parser.add_argument(
                    f'--{key}',
                    type=type(value) if value is not None else str,
                    default=value,
                )

    args = parser.parse_args(argv)

    # Fill in defaults that depend on other fields.
    if args.batch_size is None:
        args.batch_size = train_dict.get('batch_size', 24)
    if args.eval_task is None:
        args.eval_task = train_dict.get("train_task", "atadd-track2")
    if args.eval_audio is None:
        key = ("atadd_t1_eval_audio" if args.eval_task == "atadd-track1"
               else "atadd_t2_eval_audio")
        args.eval_audio = train_dict.get(key)
    if args.score_file is None:
        result_dir = os.path.join(args.model_path, 'result')
        os.makedirs(result_dir, exist_ok=True)
        args.score_file = os.path.join(result_dir, f'{args.eval_task}_logits_eval.csv')
        args.binary_score_file = os.path.join(result_dir, f'{args.eval_task}_binary_eval.csv')

    if args.eval_threshold_mode is None:
        args.eval_threshold_mode = train_dict.get("eval_threshold_mode", "fixed")
    if args.threshold is None:
        args.threshold = float(
            getattr(args, "score_threshold", train_dict.get("score_threshold", 0.5))
        )
    else:
        args.threshold = float(args.threshold)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda   = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if args.eval_threshold_mode == "eer":
        if not args.val_score_file:
            raise ValueError(
                "eval_threshold_mode=eer requires --val_score_file pointing to a dev logits CSV "
                "(name,score), e.g. from a prior inference run on the dev audio directory."
            )
        label_csv = args.val_label_csv
        if not label_csv:
            key = (
                "atadd_t1_dev_label"
                if args.eval_task == "atadd-track1"
                else "atadd_t2_dev_label"
            )
            label_csv = train_dict.get(key)
        if not label_csv or not os.path.isfile(label_csv):
            raise ValueError(
                "eval_threshold_mode=eer requires a dev label CSV "
                "(set --val_label_csv or atadd_t*_dev_label in training config)."
            )
        if not os.path.isfile(args.val_score_file):
            raise FileNotFoundError(f"val_score_file not found: {args.val_score_file}")
        val_eer, eer_thr = _eer_threshold_from_dev_csv(args.val_score_file, label_csv)
        args._decision_threshold = float(eer_thr)
        args._dev_eer_for_threshold = float(val_eer)
        args._val_label_csv_used = label_csv
    else:
        args._decision_threshold = args.threshold
        args._dev_eer_for_threshold = None
        args._val_label_csv_used = None

    print("GPU                 :", args.gpu)
    print("Eval task           :", args.eval_task)
    print("Eval audio          :", args.eval_audio)
    print("Eval threshold mode :", args.eval_threshold_mode)
    if args.eval_threshold_mode == "eer":
        print("Val score file      :", args.val_score_file)
        print("Val label CSV       :", label_csv)
        print("Dev EER (calib)     :", args._dev_eer_for_threshold)
        print("Decision threshold  :", args._decision_threshold, "(EER)")
    else:
        print("Decision threshold  :", args._decision_threshold, "(fixed)")
    print("Score file          :", args.score_file)
    print("Binary file         :", args.binary_score_file)

    return args


def _dev_label_name_to_binary(path: str) -> dict:
    """Map basename -> 0 (real) / 1 (fake) from AT-ADD dev/train CSV."""
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            lab = row["label"].strip().lower()
            out[name] = 0 if lab == "real" else 1
    return out


def _eer_threshold_from_dev_csv(score_csv: str, label_csv: str):
    """Return (eer, threshold) using real=0 / fake=1; fake if score < threshold."""
    label_map = _dev_label_name_to_binary(label_csv)
    scores_list, labels_list = [], []
    n_missing = 0
    with open(score_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "name" not in reader.fieldnames:
            raise ValueError(f"{score_csv} must have a 'name' column")
        for row in reader:
            name = row["name"].strip()
            if name not in label_map:
                n_missing += 1
                continue
            scores_list.append(float(row["score"]))
            labels_list.append(label_map[name])
    if not scores_list:
        raise ValueError(
            f"No overlapping names between {score_csv} and {label_csv} "
            f"(missing_label_rows={n_missing})."
        )
    if n_missing:
        print(f"[warn] {n_missing} score rows had no label match (skipped).")
    scores_np = np.asarray(scores_list, dtype=np.float64)
    labels_np = np.asarray(labels_list, dtype=np.int64)
    return em.compute_eer(scores_np[labels_np == 0], scores_np[labels_np == 1])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model_for_inference(args) -> torch.nn.Module:
    """Build model from registry and move to args.device."""
    return build_model(args).to(args.device)


# ---------------------------------------------------------------------------
# Score generation
# ---------------------------------------------------------------------------

def gen_score(model: torch.nn.Module, args) -> None:
    """Run inference and write logit scores to ``args.score_file``."""
    dataset = atadd_eval_dataset(
        path_to_audio=args.eval_audio,
        audio_length=args.audio_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=args.cuda,
    )

    with torch.no_grad(), open(args.score_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score"])

        for waveform, filenames in tqdm(loader, desc="scoring"):
            waveform = waveform.to(args.device, non_blocking=True)
            _, outputs = model(waveform)
            if getattr(args, "base_loss", "ce") == "bce":
                scores = torch.sigmoid(outputs[:, 0]).detach().cpu().numpy()
            else:
                scores = F.softmax(outputs, dim=1)[:, 0].detach().cpu().numpy()

            for fn, score in zip(filenames, scores):
                writer.writerow([fn.strip(), float(score)])


def gen_binary_score(score_file: str, binary_file: str, threshold: float = 0.5) -> None:
    """Convert continuous logit scores to binary real/fake predictions."""
    with open(score_file, "r", encoding="utf-8-sig", newline="") as fin, \
         open(binary_file, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["name", "predict"])

        for row in reader:
            score   = float(row["score"])
            predict = "real" if score >= threshold else "fake"
            writer.writerow([row["name"].strip(), predict])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_model_checkpoint(model_path: str) -> str:
    """Locate the best model checkpoint in *model_path*.

    Priority:
    1. ``checkpoint_all_dev/best.pt``  — best F1 on the full dev set
    2. ``checkpoint_sample_dev/top3.json`` → highest-F1 entry
    3. Legacy ``atadd_model.pt`` at the root (backward compatibility)
    """
    # 1. Full-dev best
    all_dev_best = os.path.join(model_path, "checkpoint_all_dev", "best.pt")
    if os.path.exists(all_dev_best):
        print(f"Using checkpoint_all_dev/best.pt  (best F1 on full dev set)")
        return all_dev_best

    # 2. Sample-dev top-3 — pick the best entry (first entry is best after sorting)
    top3_json = os.path.join(model_path, "checkpoint_sample_dev", "top3.json")
    if os.path.exists(top3_json):
        with open(top3_json) as f:
            entries = json.load(f)
        if entries:
            best_entry = entries[0]   # top3.json is stored best-first
            ckpt = best_entry["path"]
            if os.path.exists(ckpt):
                metric = best_entry.get("metric", "metric")
                val    = best_entry.get("metric_val", "?")
                print(f"Using checkpoint_sample_dev/{os.path.basename(ckpt)}"
                      f"  (sample-dev best {metric}={val:.4f},"
                      f" step={best_entry['step']})")
                return ckpt

    # 3. Legacy
    legacy = os.path.join(model_path, "atadd_model.pt")
    if os.path.exists(legacy):
        print(f"Using legacy atadd_model.pt")
        return legacy

    raise FileNotFoundError(
        f"No model checkpoint found in {model_path!r}.\n"
        f"Expected one of:\n"
        f"  {os.path.join(model_path, 'checkpoint_all_dev', 'best.pt')}\n"
        f"  {top3_json} (with valid paths)\n"
        f"  {legacy}"
    )


if __name__ == "__main__":
    args = _init_args()

    ckpt_path  = _find_model_checkpoint(args.model_path)
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    print("Model:", args.model)
    model = build_model_for_inference(args)
    model.load_state_dict(checkpoint)
    model.eval()

    gen_score(model, args)
    print(f"Logit scores saved to: {args.score_file}")

    gen_binary_score(args.score_file, args.binary_score_file, args._decision_threshold)
    print(f"Binary predictions saved to: {args.binary_score_file}")

    meta_path = os.path.join(
        os.path.dirname(args.binary_score_file),
        f"{args.eval_task}_binary_threshold_meta.json",
    )
    meta = {
        "eval_threshold_mode": args.eval_threshold_mode,
        "decision_threshold": args._decision_threshold,
        "score_file": args.score_file,
        "binary_score_file": args.binary_score_file,
    }
    if args.eval_threshold_mode == "eer":
        meta["val_score_file"] = args.val_score_file
        meta["val_label_csv"] = args._val_label_csv_used
        meta["dev_eer"] = args._dev_eer_for_threshold
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"Threshold metadata: {meta_path}")
