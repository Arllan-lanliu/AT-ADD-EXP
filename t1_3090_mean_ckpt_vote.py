#!/usr/bin/env python3
"""Checkpoint-mean inference for Track1 speech models.

Pass multiple checkpoints from the same model architecture with ``--ckpts``.
The script averages their weights, scores dev/eval, and reports dev F1.
When ``--ckpts`` is omitted, it keeps the old splitv1_stage2 preset behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import atadd_eval_dataset, pad_dataset, torchaudio_load
from model.model import build_model
from utils.config import ATADDConfig


"""
同一个model的不同ckpt平均后推理dev和eval
python t1_3090_mean_ckpt_vote.py \
  --model-dir ckpt_t1_ssl_layer/xlsr_24_last_splitv1_stage1 \
  --out-root ckpt_t1_vote/xlsr_24_last_splitv1_stage1 \
  --ensemble-name mean_ckpt10000_20000 \
  --ckpts \
    ckpt_t1_ssl_layer/xlsr_24_last_splitv1_stage1/checkpoint_steps/step_10000.pt \
    ckpt_t1_ssl_layer/xlsr_24_last_splitv1_stage1/checkpoint_steps/step_20000.pt 


"""

MODEL_DIR = ROOT / "ckpt_t1_train_dev_split/xlsr_3_11_24_cat_proj_v1_splitv1_stage2"
OUT_ROOT = ROOT / "ckpt_t1_vote/splitv1_stage2"
DEV_AUDIO = Path("/home/new_disk/liulan/workspace/dataset/at_add_track2/dev")
DEV_LABEL = Path("/home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv")
EVAL_AUDIO = Path("/home/new_disk/liulan/workspace/dataset/at_add_track1/eval_t1")


class ProtocolEvalDataset(Dataset):
    """Eval dataset from a protocol CSV, optionally filtered by audio type."""

    def __init__(
        self,
        audio_dir: Path,
        label_csv: Path,
        audio_length: int,
        filter_type: str | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.audio_length = audio_length
        self.items: list[tuple[str, str, str]] = []
        wanted_type = filter_type.strip().lower() if filter_type else None

        with Path(label_csv).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                audio_type = row["type"].strip().lower()
                if wanted_type and audio_type != wanted_type:
                    continue
                self.items.append((row["name"].strip(), audio_type, row["label"].strip().lower()))

        if not self.items:
            raise ValueError(f"No rows found in {label_csv} with filter_type={filter_type!r}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        filename, audio_type, label = self.items[idx]
        filepath = self.audio_dir / filename
        waveform, _sr = torchaudio_load(str(filepath))
        waveform = pad_dataset(waveform, self.audio_length)
        return waveform, filename, audio_type, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean Track1 checkpoints and score dev/eval.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--dev-audio", type=Path, default=DEV_AUDIO)
    parser.add_argument("--dev-label", type=Path, default=DEV_LABEL)
    parser.add_argument("--eval-audio", type=Path, default=EVAL_AUDIO)
    parser.add_argument("--dev-filter-type", default="speech")
    parser.add_argument(
        "--ckpts",
        type=Path,
        nargs="+",
        default=None,
        help="Checkpoints from the same model to average. Relative paths are resolved from cwd, repo root, then model-dir.",
    )
    parser.add_argument(
        "--ensemble-name",
        default="mean_ckpt",
        help="Output subdirectory name when --ckpts is provided.",
    )
    parser.add_argument(
        "--skip-dev",
        action="store_true",
        help="Only score eval; useful when dev labels/audio are unavailable.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only score dev and report dev F1.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def load_train_args(model_dir: Path, gpu: str, batch_size: int) -> SimpleNamespace:
    config_yaml = model_dir / "config.yaml"
    if not config_yaml.is_file():
        raise FileNotFoundError(f"config.yaml not found: {config_yaml}")

    cfg = ATADDConfig.from_yaml(config_yaml)
    args = cfg.to_namespace()
    args.gpu = gpu
    args.batch_size = batch_size
    args.eval_task = "atadd-track1"
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def resolve_checkpoint_path(model_dir: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        ROOT / path,
        model_dir / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def sample_dev_checkpoints(model_dir: Path) -> list[Path]:
    top3_json = model_dir / "checkpoint_sample_dev/top3.json"
    if not top3_json.is_file():
        raise FileNotFoundError(f"top3.json not found: {top3_json}")

    with top3_json.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if len(entries) != 3:
        raise ValueError(f"Expected exactly 3 sample-dev checkpoints, got {len(entries)}")

    ckpts = [resolve_checkpoint_path(model_dir, entry["path"]) for entry in entries]
    for ckpt in ckpts:
        if not ckpt.is_file():
            raise FileNotFoundError(f"sample-dev checkpoint not found: {ckpt}")
    return ckpts


def all_dev_checkpoint(model_dir: Path) -> Path:
    ckpt = model_dir / "checkpoint_all_dev/best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"all-dev checkpoint not found: {ckpt}")
    return ckpt


def unwrap_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like object: {type(obj)!r}")
    return obj


def remap_legacy_layer_fusion_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map checkpoints saved before SSL layer fusion was wrapped in a submodule."""
    legacy_prefixes = (
        "layer_proj",
        "per_layer_ln",
        "cat_linear_head",
        "layer_weights",
        "mhfa_w_k",
        "mhfa_w_v",
        "mhfa_proj_k",
        "mhfa_proj_v",
        "mhfa_fuse_out",
        "mhfa_query",
        "mhfa_head_proj",
        "mhfa_dropout_layer",
    )
    frontend_prefixes = ("frontend", "xlsr", "wavlm", "beats")
    remapped: dict[str, torch.Tensor] = {}
    changed = 0

    for key, value in state_dict.items():
        new_key = key
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] in frontend_prefixes and parts[1] in legacy_prefixes:
            new_key = ".".join([parts[0], "layer_fusion_mod", *parts[1:]])
            changed += 1
        remapped[new_key] = value

    if changed:
        print(f"[compat] remapped {changed} legacy layer-fusion checkpoint keys")
    return remapped


def mean_checkpoints(ckpt_paths: Iterable[Path]) -> dict[str, torch.Tensor]:
    paths = [Path(p) for p in ckpt_paths]
    if not paths:
        raise ValueError("No checkpoints provided for averaging.")

    avg_state: dict[str, torch.Tensor] | None = None
    float_keys: set[str] = set()
    n = len(paths)

    for i, ckpt_path in enumerate(paths, start=1):
        print(f"[ckpt {i}/{n}] loading: {ckpt_path}")
        state = remap_legacy_layer_fusion_keys(
            unwrap_state_dict(torch.load(ckpt_path, map_location="cpu"))
        )

        if avg_state is None:
            avg_state = {}
            for key, value in state.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    avg_state[key] = value.detach().float().clone().div_(n)
                    float_keys.add(key)
                elif torch.is_tensor(value):
                    avg_state[key] = value.detach().clone()
                else:
                    avg_state[key] = value
            continue

        missing = set(avg_state) ^ set(state)
        if missing:
            raise ValueError(f"Checkpoint keys differ for {ckpt_path}: {sorted(missing)[:20]}")

        for key, avg_value in avg_state.items():
            value = state[key]
            if torch.is_tensor(avg_value) and torch.is_tensor(value):
                if avg_value.shape != value.shape:
                    raise ValueError(
                        f"Checkpoint tensor shape differs for {key}: "
                        f"{tuple(avg_value.shape)} vs {tuple(value.shape)} in {ckpt_path}"
                    )

        for key in float_keys:
            avg_state[key].add_(state[key].detach().float(), alpha=1.0 / n)

        del state

    assert avg_state is not None
    return avg_state


def gen_binary_score(score_file: Path, binary_file: Path, threshold: float) -> None:
    with score_file.open("r", encoding="utf-8-sig", newline="") as fin, binary_file.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["name", "predict"])
        for row in reader:
            predict = "real" if float(row["score"]) >= threshold else "fake"
            writer.writerow([row["name"].strip(), predict])


def binary_metrics(y_true: list[str], y_pred: list[str], positive: str) -> dict[str, float | int]:
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    tn = sum(t != positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "acc": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_dev_predictions(
    y_true: list[str],
    y_pred: list[str],
    threshold: float,
) -> dict[str, float | int | dict[str, float | int]]:
    real_metrics = binary_metrics(y_true, y_pred, positive="real")
    fake_metrics = binary_metrics(y_true, y_pred, positive="fake")
    macro_f1 = (float(real_metrics["f1"]) + float(fake_metrics["f1"])) / 2
    return {
        "threshold": threshold,
        "num_samples": len(y_true),
        "acc": real_metrics["acc"],
        "f1_real": real_metrics["f1"],
        "f1_fake": fake_metrics["f1"],
        "macro_f1": macro_f1,
        "real": real_metrics,
        "fake": fake_metrics,
    }


def score_eval_dataset(
    model: torch.nn.Module,
    args: SimpleNamespace,
    audio_dir: Path,
    out_dir: Path,
    prefix: str,
    threshold: float,
    num_workers: int,
) -> None:
    dataset = atadd_eval_dataset(path_to_audio=str(audio_dir), audio_length=args.audio_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
    )
    logits_path = out_dir / f"{prefix}_logits.csv"
    binary_path = out_dir / f"{prefix}_binary.csv"

    with torch.no_grad(), logits_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score"])
        for waveform, filenames in tqdm(loader, desc=f"score {prefix}"):
            waveform = waveform.to(args.device, non_blocking=True)
            _, outputs = model(waveform)
            scores = scores_from_outputs(outputs, args)
            for filename, score in zip(filenames, scores):
                writer.writerow([filename.strip(), float(score)])

    gen_binary_score(logits_path, binary_path, threshold)
    print(f"[saved] {logits_path}")
    print(f"[saved] {binary_path}")


def score_protocol_dataset(
    model: torch.nn.Module,
    args: SimpleNamespace,
    audio_dir: Path,
    label_csv: Path,
    out_dir: Path,
    prefix: str,
    filter_type: str,
    threshold: float,
    num_workers: int,
) -> dict[str, float | int | dict[str, float | int]]:
    dataset = ProtocolEvalDataset(
        audio_dir=audio_dir,
        label_csv=label_csv,
        audio_length=args.audio_len,
        filter_type=filter_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
    )
    logits_path = out_dir / f"{prefix}_logits.csv"
    binary_path = out_dir / f"{prefix}_binary.csv"
    detail_path = out_dir / f"{prefix}_detail.csv"
    y_true: list[str] = []
    y_pred: list[str] = []

    with torch.no_grad(), logits_path.open("w", encoding="utf-8", newline="") as lf, detail_path.open(
        "w", encoding="utf-8", newline=""
    ) as df:
        logits_writer = csv.writer(lf)
        detail_writer = csv.writer(df)
        logits_writer.writerow(["name", "score"])
        detail_writer.writerow(["name", "score", "predict", "type", "label"])

        for waveform, filenames, audio_types, labels in tqdm(loader, desc=f"score {prefix}"):
            waveform = waveform.to(args.device, non_blocking=True)
            _, outputs = model(waveform)
            scores = scores_from_outputs(outputs, args)
            for filename, score, audio_type, label in zip(filenames, scores, audio_types, labels):
                predict = "real" if float(score) >= threshold else "fake"
                logits_writer.writerow([filename.strip(), float(score)])
                detail_writer.writerow([filename.strip(), float(score), predict, audio_type, label])
                y_true.append(str(label).strip().lower())
                y_pred.append(predict)

    gen_binary_score(logits_path, binary_path, threshold)
    metrics = evaluate_dev_predictions(y_true, y_pred, threshold)
    metrics_path = out_dir / f"{prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[saved] {logits_path}")
    print(f"[saved] {binary_path}")
    print(f"[saved] {detail_path}")
    print(f"[saved] {metrics_path}")
    print(
        "[dev metrics] "
        f"macro_f1={float(metrics['macro_f1']):.6f} "
        f"f1_real={float(metrics['f1_real']):.6f} "
        f"f1_fake={float(metrics['f1_fake']):.6f} "
        f"acc={float(metrics['acc']):.6f}"
    )
    return metrics


def scores_from_outputs(outputs: torch.Tensor, args: SimpleNamespace):
    if getattr(args, "base_loss", "ce") == "bce":
        return torch.sigmoid(outputs[:, 0]).detach().cpu().numpy()
    return F.softmax(outputs, dim=1)[:, 0].detach().cpu().numpy()


def run_ensemble(
    name: str,
    ckpt_paths: list[Path],
    args: SimpleNamespace,
    cli_args: argparse.Namespace,
) -> dict[str, float | int | dict[str, float | int]] | None:
    out_dir = cli_args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    state = mean_checkpoints(ckpt_paths)
    model = build_model(args).to(args.device)
    model.load_state_dict(state)
    model.eval()

    meta_path = out_dir / "ckpt_mean_meta.json"
    meta = {
        "ensemble": name,
        "checkpoint_paths": [str(p) for p in ckpt_paths],
        "threshold": cli_args.threshold,
        "dev_audio": str(cli_args.dev_audio),
        "dev_label": str(cli_args.dev_label),
        "dev_filter_type": cli_args.dev_filter_type,
        "eval_audio": str(cli_args.eval_audio),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    dev_metrics = None
    if not cli_args.skip_dev:
        dev_metrics = score_protocol_dataset(
            model=model,
            args=args,
            audio_dir=cli_args.dev_audio,
            label_csv=cli_args.dev_label,
            out_dir=out_dir,
            prefix="atadd-track2_dev_speech",
            filter_type=cli_args.dev_filter_type,
            threshold=cli_args.threshold,
            num_workers=cli_args.num_workers,
        )
    if not cli_args.skip_eval:
        score_eval_dataset(
            model=model,
            args=args,
            audio_dir=cli_args.eval_audio,
            out_dir=out_dir,
            prefix="atadd-track1_eval",
            threshold=cli_args.threshold,
            num_workers=cli_args.num_workers,
        )

    del model
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dev_metrics is not None:
        meta["dev_metrics"] = dev_metrics
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    return dev_metrics


def main() -> None:
    cli_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    cli_args.out_root.mkdir(parents=True, exist_ok=True)

    args = load_train_args(cli_args.model_dir, cli_args.gpu, cli_args.batch_size)
    print("Model dir:", cli_args.model_dir)
    print("Output root:", cli_args.out_root)
    print("Model:", args.model)
    print("Device:", args.device)

    if cli_args.ckpts:
        ckpts = [resolve_checkpoint_path(cli_args.model_dir, ckpt) for ckpt in cli_args.ckpts]
        for ckpt in ckpts:
            if not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        run_ensemble(cli_args.ensemble_name, ckpts, args, cli_args)
        return

    sample_ckpts = sample_dev_checkpoints(cli_args.model_dir)
    all_plus_sample_ckpts = [all_dev_checkpoint(cli_args.model_dir), *sample_ckpts]
    run_ensemble("sample_dev_top3_mean", sample_ckpts, args, cli_args)
    run_ensemble("all_dev_plus_sample_dev_mean", all_plus_sample_ckpts, args, cli_args)


if __name__ == "__main__":
    main()
