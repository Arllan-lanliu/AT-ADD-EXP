#!/usr/bin/env python3
"""Score splitv1_stage1 speech validation data and sweep thresholds.

The script loads a trained Track-1 model directory, scores all ``speech`` rows
from ``protocol_gendisjoint_v1_stage1/val_gendisjoint.csv``, then exports
thresholded prediction files and metric summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import pad_dataset, torchaudio_load
from model.model import build_model
from utils.config import ATADDConfig

# python3 t1_3090_thredhold.py --gpu 6

DEFAULT_MODEL_DIR = ROOT / "ckpt_t1_layer/xlsr_3_11_24"
DEFAULT_PROTOCOL = "/home/new_disk/liulan/workspace/dataset/at_add_track2/labels/dev.csv"
DEFAULT_OUT_DIR = ROOT / "ckpt_t1_thredhold/default_dev"
PREFIX = "default_dev"


class SpeechProtocolDataset(Dataset):
    """Dataset backed by the speech rows in an AT-ADD protocol CSV."""

    def __init__(
        self,
        audio_dirs: list[Path],
        protocol_csv: Path,
        audio_length: int,
    ) -> None:
        self.audio_dirs = [Path(audio_dir) for audio_dir in audio_dirs]
        self.protocol_csv = Path(protocol_csv)
        self.audio_length = int(audio_length)
        self.items: list[tuple[str, str]] = []
        self.file_map: dict[str, Path] = {}

        with self.protocol_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"name", "type", "label"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.protocol_csv} missing columns: {sorted(missing)}")
            for row in reader:
                if row["type"].strip().lower() != "speech":
                    continue
                name = row["name"].strip()
                label = row["label"].strip().lower()
                if not name:
                    continue
                if label not in {"real", "fake"}:
                    raise ValueError(f"Unexpected label {label!r} for {name}")
                self.items.append((name, label))

        if not self.items:
            raise ValueError(f"No speech rows found in {self.protocol_csv}")

        missing_audio = []
        for name, _label in self.items:
            path = self._find_audio(name)
            if path is None:
                missing_audio.append(name)
            else:
                self.file_map[name] = path
        if missing_audio:
            preview = ", ".join(missing_audio[:10])
            raise FileNotFoundError(
                f"{len(missing_audio)} speech audio files are missing under "
                f"{[str(path) for path in self.audio_dirs]}. Examples: {preview}"
            )

    def _find_audio(self, name: str) -> Path | None:
        for audio_dir in self.audio_dirs:
            path = audio_dir / name
            if path.is_file():
                return path
        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        name, label = self.items[idx]
        waveform, _sr = torchaudio_load(str(self.file_map[name]))
        waveform = pad_dataset(waveform, self.audio_length)
        return waveform, name, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer splitv1_stage1 speech scores and evaluate thresholds."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Audio directory. Can be passed multiple times. Defaults to the "
            "at_add_track2/train_dev sibling when available, otherwise train and dev "
            "audio paths from the model config."
        ),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def load_train_args(model_dir: Path, gpu: str, batch_size: int) -> SimpleNamespace:
    config_yaml = model_dir / "config.yaml"
    args_json = model_dir / "args.json"
    if config_yaml.is_file():
        cfg = ATADDConfig.from_yaml(str(config_yaml))
        args = cfg.to_namespace()
    elif args_json.is_file():
        with args_json.open("r", encoding="utf-8") as handle:
            args = SimpleNamespace(**json.load(handle))
    else:
        raise FileNotFoundError(f"Neither config.yaml nor args.json found in {model_dir}")

    args.gpu = str(gpu)
    args.batch_size = int(batch_size)
    args.eval_task = "atadd-track1"
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def resolve_checkpoint_path(model_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def find_model_checkpoint(model_dir: Path) -> Path:
    all_dev_best = model_dir / "checkpoint_all_dev/best.pt"
    if all_dev_best.is_file():
        print(f"Using checkpoint_all_dev/best.pt: {all_dev_best}")
        return all_dev_best

    top3_json = model_dir / "checkpoint_sample_dev/top3.json"
    if top3_json.is_file():
        with top3_json.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if entries:
            ckpt = resolve_checkpoint_path(model_dir, entries[0]["path"])
            if ckpt.is_file():
                print(f"Using checkpoint_sample_dev top-1: {ckpt}")
                return ckpt

    legacy = model_dir / "atadd_model.pt"
    if legacy.is_file():
        print(f"Using legacy checkpoint: {legacy}")
        return legacy

    latest = model_dir / "checkpoint/latest.pt"
    if latest.is_file():
        print(f"Using checkpoint/latest.pt: {latest}")
        return latest

    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. Expected checkpoint_all_dev/best.pt, "
        "checkpoint_sample_dev/top3.json, atadd_model.pt, or checkpoint/latest.pt."
    )


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


def build_model_for_inference(args: SimpleNamespace, checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = remap_legacy_layer_fusion_keys(unwrap_state_dict(checkpoint))
    model = build_model(args).to(args.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def default_audio_dirs(args: SimpleNamespace) -> list[Path]:
    train_audio = Path(getattr(args, "atadd_t1_train_audio"))
    dev_audio = Path(getattr(args, "atadd_t1_dev_audio"))
    train_dev_audio = train_audio.parent / "train_dev"
    if train_dev_audio.is_dir():
        return [train_dev_audio]
    dirs = [train_audio]
    if dev_audio != train_audio:
        dirs.append(dev_audio)
    return dirs


def scores_from_outputs(outputs: torch.Tensor, args: SimpleNamespace) -> list[float]:
    if getattr(args, "base_loss", "ce") == "bce":
        scores = torch.sigmoid(outputs[:, 0])
    else:
        scores = F.softmax(outputs, dim=1)[:, 0]
    return [float(value) for value in scores.detach().cpu()]


def write_ground_truth(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "real_label"])
        writer.writerows(rows)


def score_speech_split(
    model: torch.nn.Module,
    args: SimpleNamespace,
    dataset: SpeechProtocolDataset,
    out_path: Path,
    num_workers: int,
) -> list[tuple[str, float, str]]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.cuda,
    )
    scored_rows: list[tuple[str, float, str]] = []

    with torch.no_grad(), out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "score"])
        for waveform, names, labels in tqdm(loader, desc="score splitv1_stage1 speech"):
            waveform = waveform.to(args.device, non_blocking=True)
            _hidden, outputs = model(waveform)
            scores = scores_from_outputs(outputs, args)
            for name, score, label in zip(names, scores, labels):
                name = name.strip()
                label = label.strip().lower()
                writer.writerow([name, f"{score:.10g}"])
                scored_rows.append((name, score, label))

    return scored_rows


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


def threshold_and_evaluate(
    rows: list[tuple[str, float, str]],
    out_dir: Path,
    prefix: str = "val_splitv1_stage1",
) -> None:
    metrics_path = out_dir / f"{prefix}_threshold_metrics.csv"
    thresholds = [i / 10 for i in range(1, 10)]
    y_true = [label for _name, _score, label in rows]

    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_handle:
        fieldnames = [
            "threshold",
            "f1",
            "acc",
            "f1_real",
            "precision_real",
            "recall_real",
            "f1_fake",
            "macro_f1",
            "tp_real",
            "tn_real",
            "fp_real",
            "fn_real",
            "num_samples",
        ]
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fieldnames)
        metrics_writer.writeheader()

        for threshold in thresholds:
            pred_rows = []
            y_pred = []
            for name, score, _label in rows:
                pred = "real" if score >= threshold else "fake"
                pred_rows.append((name, pred))
                y_pred.append(pred)

            label_path = out_dir / f"{prefix}_threshold_{threshold:.1f}.csv"
            with label_path.open("w", encoding="utf-8", newline="") as label_handle:
                writer = csv.writer(label_handle)
                writer.writerow(["name", "predict"])
                writer.writerows(pred_rows)

            real_metrics = binary_metrics(y_true, y_pred, positive="real")
            fake_metrics = binary_metrics(y_true, y_pred, positive="fake")
            metrics_writer.writerow(
                {
                    "threshold": f"{threshold:.1f}",
                    "f1": f"{real_metrics['f1']:.10g}",
                    "acc": f"{real_metrics['acc']:.10g}",
                    "f1_real": f"{real_metrics['f1']:.10g}",
                    "precision_real": f"{real_metrics['precision']:.10g}",
                    "recall_real": f"{real_metrics['recall']:.10g}",
                    "f1_fake": f"{fake_metrics['f1']:.10g}",
                    "macro_f1": f"{((real_metrics['f1'] + fake_metrics['f1']) / 2):.10g}",
                    "tp_real": real_metrics["tp"],
                    "tn_real": real_metrics["tn"],
                    "fp_real": real_metrics["fp"],
                    "fn_real": real_metrics["fn"],
                    "num_samples": len(rows),
                }
            )

    print(f"[saved] {metrics_path}")


def main() -> None:
    cli_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    cli_args.out_dir.mkdir(parents=True, exist_ok=True)

    args = load_train_args(cli_args.model_dir, cli_args.gpu, cli_args.batch_size)
    audio_dirs = cli_args.audio_dir or default_audio_dirs(args)
    checkpoint_path = find_model_checkpoint(cli_args.model_dir)

    print("Model dir:", cli_args.model_dir)
    print("Protocol:", cli_args.protocol)
    print("Audio dirs:", ", ".join(str(path) for path in audio_dirs))
    print("Output dir:", cli_args.out_dir)
    print("Model:", args.model)
    print("Device:", args.device)


    dataset = SpeechProtocolDataset(audio_dirs, cli_args.protocol, args.audio_len)
    print(f"Speech rows: {len(dataset)}")

    write_ground_truth(
        cli_args.out_dir / f"{PREFIX}_real_label.csv",
        [(name, label) for name, label in dataset.items],
    )

    model = build_model_for_inference(args, checkpoint_path)
    score_path = cli_args.out_dir / f"{PREFIX}_scores.csv"
    rows = score_speech_split(
        model=model,
        args=args,
        dataset=dataset,
        out_path=score_path,
        num_workers=cli_args.num_workers,
    )
    print(f"[saved] {score_path}")

    threshold_and_evaluate(rows, cli_args.out_dir, prefix=PREFIX)


if __name__ == "__main__":
    main()
