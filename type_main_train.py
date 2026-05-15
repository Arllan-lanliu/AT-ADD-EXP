"""
Train a **4-way audio type classifier** (speech / sound / singing / music)
using purely **command-line arguments** (no YAML).

Models (see ``model.type_classifier``):
  * ``xlsr`` — XLS-R + MLP head
  * ``logmelcnn`` — log-mel + lightweight CNN

Logs overall / per-type **accuracy** to ``out_fold/logs/`` and to **Weights & Biases**
when enabled. Intended to mirror the flow of ``main_train.py`` while staying self-contained.

Run example::

    CUDA_VISIBLE_DEVICES=0 python type_main_train.py \\
        --model xlsr --xlsr_dir /path/to/wav2vec2-xls-r-300m \\
        --train_audio /path/T2/train --train_label /path/T2/label/train.csv \\
        --dev_audio /path/T2/dev --dev_label /path/T2/label/dev.csv \\
        --out_fold ./ckpt_type/xlsr_type --wandb_project atadd-type
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler as torch_sampler
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.dataset import atadd_dataset
from model.type_classifier import (
    ATADD_AUDIO_TYPE_NAMES,
    LogMelCNNATADDTypeClassifier,
    NUM_ATADD_AUDIO_TYPES,
    XLSRATADDTypeClassifier,
)
from utils.helpers import parse_filter_types, setup_seed

torch.set_default_tensor_type(torch.FloatTensor)
try:
    torch.multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

_FINE_TYPE_NAMES = tuple(ATADD_AUDIO_TYPE_NAMES)


def _aggregate_type_accuracy(
    pred_np: np.ndarray,
    true_np: np.ndarray,
    fine_np: np.ndarray | None,
) -> dict:
    """Overall and per-class accuracy; ``fine_np`` aligns with GT type id per sample."""
    ok = pred_np == true_np
    out = {
        "overall": float(np.mean(ok)),
        "per_type": {},
    }
    if fine_np is None:
        fine_np = true_np
    for u in range(NUM_ATADD_AUDIO_TYPES):
        name = _FINE_TYPE_NAMES[u]
        m = fine_np == u
        if not np.any(m):
            out["per_type"][name] = float("nan")
        else:
            out["per_type"][name] = float(np.mean(ok[m]))
    return out


def _macro_f1(true_np: np.ndarray, pred_np: np.ndarray) -> float:
    return float(
        f1_score(true_np, pred_np, average="macro", labels=list(range(NUM_ATADD_AUDIO_TYPES)), zero_division=0)
    )


def _build_train_val_loaders(ns: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    ft = ns.filter_types_parsed
    if ft:
        print(f"Filtering train/dev to audio types: {sorted(ft)}")

    aug_probs_raw = {
        "speech": ns.aug_speech,
        "sound": ns.aug_sound,
        "music": ns.aug_music,
        "singing": ns.aug_singing,
    }
    aug_probs = {k: v for k, v in aug_probs_raw.items() if v > 0.0} or None
    if aug_probs:
        print(f"Per-type train augmentation: {aug_probs}  (music: {ns.music_aug_method})")

    train_ds = atadd_dataset(
        ns.train_audio,
        ns.train_label,
        audio_length=ns.audio_len,
        filter_types=ft,
        aug_probs=aug_probs,
        music_aug_method=ns.music_aug_method,
        dev_subsample=False,
    )
    val_ds = atadd_dataset(
        ns.dev_audio,
        ns.dev_label,
        audio_length=ns.audio_len,
        filter_types=ft,
        dev_subsample=ns.dev_subsample,
        dev_subsample_seed=ns.dev_subsample_seed,
    )

    assert len(train_ds) > 0, "Train dataset is empty — check --train_audio / --train_label."
    assert len(val_ds) > 0, "Dev dataset is empty — check --dev_audio / --dev_label."

    def _loader(ds, *, pin_memory: bool):
        return DataLoader(
            ds,
            batch_size=int(ns.batch_size),
            shuffle=False,
            num_workers=int(ns.num_workers),
            sampler=torch_sampler.SubsetRandomSampler(range(len(ds))),
            pin_memory=pin_memory,
        )

    return _loader(train_ds, pin_memory=bool(ns.cuda)), _loader(val_ds, pin_memory=bool(ns.cuda))


def _build_full_dev_loader(ns: argparse.Namespace):
    ft = ns.filter_types_parsed
    ds = atadd_dataset(
        ns.dev_audio,
        ns.dev_label,
        audio_length=ns.audio_len,
        filter_types=ft,
        dev_subsample=False,
    )
    assert len(ds) > 0, "Full dev dataset is empty."
    return DataLoader(
        ds,
        batch_size=int(ns.batch_size),
        shuffle=False,
        num_workers=int(ns.num_workers),
        sampler=torch_sampler.SubsetRandomSampler(range(len(ds))),
        pin_memory=bool(ns.cuda),
    )


def _build_model(ns: argparse.Namespace, device_torch: torch.device) -> nn.Module:
    dev_str = "cuda" if device_torch.type == "cuda" else "cpu"
    if ns.model == "xlsr":
        return XLSRATADDTypeClassifier(
            model_dir=ns.xlsr_dir,
            device=dev_str,
            freeze_frontend=ns.freeze_xlsr,
            xlsr_dim=ns.xlsr_dim,
            head_hidden_dim=ns.head_hidden,
            head_dropout=ns.head_dropout,
            sampling_rate=ns.sample_rate,
        )
    if ns.model == "logmelcnn":
        return LogMelCNNATADDTypeClassifier(
            sample_rate=ns.sample_rate,
            n_fft=ns.mel_n_fft,
            hop_length=ns.mel_hop,
            n_mels=ns.mel_bins,
            base_channels=ns.cnn_base_channels,
            head_dropout=ns.cnn_head_dropout,
        )
    raise ValueError(f"Unknown --model {ns.model!r}")


def _params_for_optimizer(model: nn.Module, ns: argparse.Namespace):
    if ns.model == "xlsr" and ns.freeze_xlsr:
        return [{"params": model.head.parameters(), "lr": ns.lr}]
    return [{"params": model.parameters(), "lr": ns.lr}]


def _wandb_safe_config(ns: argparse.Namespace) -> dict:
    d: Dict[str, Any] = {}
    for k, v in vars(ns).items():
        if k.startswith("_"):
            continue
        if callable(v):
            d[k] = str(v)
        else:
            try:
                json.dumps(v)
                d[k] = v
            except (TypeError, ValueError):
                d[k] = str(v)
    return d


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Tuple[float, dict, float]:
    """Returns (mean CE loss, accuracy dict, macro F1)."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    for feat, _, _labels, class_types, _ in tqdm(loader, leave=False, desc="eval"):
        feat = feat.to(device, non_blocking=True)
        class_types = class_types.to(device, non_blocking=True).long()

        with autocast(enabled=amp):
            logits = model(feat)
            loss = F.cross_entropy(logits, class_types, reduction="sum")

        total_loss += float(loss.item())
        total_n += int(feat.size(0))
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        trues.append(class_types.cpu().numpy())

    pred_np = np.concatenate(preds)
    true_np = np.concatenate(trues)
    mean_loss = total_loss / max(total_n, 1)
    acc_info = _aggregate_type_accuracy(pred_np, true_np, true_np)
    mf1 = _macro_f1(true_np, pred_np)
    return mean_loss, acc_info, mf1


def _fmt_acc_line(acc: dict) -> str:
    parts = [f"overall={acc['overall']:.6f}"]
    for name in _FINE_TYPE_NAMES:
        v = acc["per_type"].get(name, float("nan"))
        if isinstance(v, float) and np.isnan(v):
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}={float(v):.6f}")
    return "  ".join(parts)


def _write_eval_log(log_path: Path, tag: str, loss: float, acc: dict, mf1: float) -> None:
    line = {"tag": tag, "loss": loss, "macro_f1": mf1}
    line.update({f"acc_{k}": (None if isinstance(v, float) and np.isnan(v) else v) for k, v in acc["per_type"].items()})
    line["acc_overall"] = acc["overall"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AT-ADD 4-class type classifier training (CLI only).")

    # Model
    p.add_argument(
        "--model",
        choices=("xlsr", "logmelcnn"),
        default="logmelcnn",
        help="xlsr: XLS-R + MLP head; logmelcnn: log-mel + CNN.",
    )
    p.add_argument("--xlsr_dir", type=str, default="", help="Local XLS-R / Wav2Vec2 directory (--model xlsr).")
    p.add_argument("--freeze_xlsr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--xlsr_dim", type=int, default=1024)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--mel_n_fft", type=int, default=1024)
    p.add_argument("--mel_hop", type=int, default=256)
    p.add_argument("--mel_bins", type=int, default=128)
    p.add_argument("--cnn_base_channels", type=int, default=32)
    p.add_argument("--cnn_head_dropout", type=float, default=0.2)

    # Data
    p.add_argument("--train_audio", type=str, required=True)
    p.add_argument("--train_label", type=str, required=True)
    p.add_argument("--dev_audio", type=str, required=True)
    p.add_argument("--dev_label", type=str, required=True)
    p.add_argument("--audio_len", type=int, default=64600)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument(
        "--filter_types",
        type=str,
        default="",
        help="Comma-separated subset of speech,sound,music,singing; empty means all.",
    )
    p.add_argument("--dev_subsample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dev_subsample_seed", type=int, default=42)
    p.add_argument("--aug_speech", type=float, default=0.0)
    p.add_argument("--aug_sound", type=float, default=0.0)
    p.add_argument("--aug_music", type=float, default=0.0)
    p.add_argument("--aug_singing", type=float, default=0.0)
    p.add_argument("--music_aug_method", choices=("pitch_shift", "spec_augment"), default="spec_augment")

    # Train
    p.add_argument("--gpu", type=str, default="0", help='Sets CUDA_VISIBLE_DEVICES (same convention as main_train).')
    p.add_argument("--seed", type=int, default=6789)
    p.add_argument("--out_fold", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval_steps", type=int, default=500, help="0 disables mid-epoch eval on sampled dev.")
    p.add_argument("--full_eval_steps", type=int, default=0, help="0 disables periodic full-dev eval.")
    p.add_argument("--eval_warmup_steps", type=int, default=0)

    # Early stop / checkpoints
    p.add_argument("--patience_evals", type=int, default=0, help="0 disables early stopping (by dev overall acc).")
    p.add_argument("--continue_training", action="store_true")
    # W&B
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="5090_type_classification")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))

    ns = p.parse_args(argv)

    ns.filter_types_parsed = parse_filter_types(ns.filter_types or None)
    xd = Path(ns.xlsr_dir).expanduser()
    if ns.model == "xlsr" and not xd.is_dir():
        p.error("--model xlsr requires a valid existing --xlsr_dir directory.")

    ns.wandb_mode_effective = "disabled" if ns.no_wandb else ns.wandb_mode

    return ns


def prepare_out_dir(ns: argparse.Namespace, save_cli_path: Path) -> None:
    log_dir = Path(ns.out_fold) / "logs"
    ckpt_dir = Path(ns.out_fold) / "checkpoint"
    latest = ckpt_dir / "latest.pt"

    if ns.continue_training and latest.is_file():
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return

    if Path(ns.out_fold).exists():
        shutil.rmtree(ns.out_fold)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "train_loss.log", "w") as f:
        f.write("step\tepoch\tbatch\tloss\tbatch_acc\n")

    # Persist CLI for reproducibility
    save_cli_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_cli_path, "w") as f:
        json.dump(vars(ns), f, indent=2, default=str)


def train(ns: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ns.gpu
    setup_seed(ns.seed)

    ns.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if ns.cuda else "cpu")
    print(f"Device: {device}")

    prepare_out_dir(ns, Path(ns.out_fold) / "type_train_cli.json")

    if ns.continue_training:
        latest_chk = Path(ns.out_fold) / "checkpoint" / "latest.pt"
        if not latest_chk.is_file():
            raise FileNotFoundError(
                "--continue_training was set but no checkpoint exists at "
                f"{latest_chk}; run without --continue_training first."
            )

    train_loader, val_loader = _build_train_val_loaders(ns)
    full_dev_loader = _build_full_dev_loader(ns) if ns.full_eval_steps > 0 else None

    model = _build_model(ns, device).to(device)
    opt_groups = _params_for_optimizer(model, ns)
    for g in opt_groups:
        g["weight_decay"] = ns.weight_decay
    optimizer = torch.optim.AdamW(opt_groups, lr=ns.lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    log_dir = Path(ns.out_fold) / "logs"
    ckpt_dir = Path(ns.out_fold) / "checkpoint"
    best_path = ckpt_dir / "best.pt"
    latest_path = ckpt_dir / "latest.pt"

    start_epoch = 0
    global_step = 0
    best_dev_acc = -1.0
    no_improve_evals = 0
    amp_enabled = bool(ns.amp and ns.cuda)

    if ns.continue_training and latest_path.is_file():
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_dev_acc = float(ckpt.get("best_dev_overall_acc", -1.0))
        no_improve_evals = int(ckpt.get("no_improve_evals", 0))

    use_wandb = ns.wandb_mode_effective != "disabled"
    if use_wandb:
        run_name = ns.wandb_run_name or Path(ns.out_fold.rstrip("/")).name
        wandb.init(
            mode=ns.wandb_mode_effective,
            project=ns.wandb_project,
            name=run_name,
            config=_wandb_safe_config(ns),
            dir=ns.out_fold,
        )

    def _log_dev_to_wandb(prefix: str, step: int, loss: float, acc: dict, mf1: float) -> None:
        if not use_wandb:
            return
        wb: Dict[str, Any] = {
            f"{prefix}/loss": loss,
            f"{prefix}/overall_acc": acc["overall"],
            f"{prefix}/macro_f1": mf1,
        }
        for name, v in acc["per_type"].items():
            if isinstance(v, float) and np.isnan(v):
                continue
            wb[f"{prefix}/acc/{name}"] = float(v)
        wandb.log(wb, step=step)

    def save_checkpoint(step: int, epoch_idx: int) -> None:
        torch.save(
            {
                "epoch": epoch_idx,
                "global_step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_overall_acc": best_dev_acc,
                "no_improve_evals": no_improve_evals,
                "cli": vars(ns),
            },
            latest_path,
        )

    def maybe_eval_sample(tag_suffix: str, step: int, epoch_idx: int) -> bool:
        """Returns True if training should stop (early stopping)."""
        nonlocal best_dev_acc, no_improve_evals
        vl, vacc, vf1 = evaluate_loader(model, val_loader, device, amp_enabled)
        tag = f"{tag_suffix}"
        print(f"[Dev {tag}] loss={vl:.4f}  macro_f1={vf1:.4f}  {_fmt_acc_line(vacc)}")
        _write_eval_log(log_dir / "dev_metrics.jsonl", tag, vl, vacc, vf1)
        _log_dev_to_wandb("dev", step, vl, vacc, vf1)

        if vacc["overall"] > best_dev_acc:
            best_dev_acc = vacc["overall"]
            no_improve_evals = 0
            torch.save(model.state_dict(), best_path)
            meta = {"step": step, "overall_acc": best_dev_acc, "macro_f1": vf1, "loss": vl}
            with open(ckpt_dir / "best_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  → Saved new best.pt (overall_acc={best_dev_acc:.6f})")
        else:
            no_improve_evals += 1

        save_checkpoint(step, epoch_idx)

        stop = ns.patience_evals > 0 and no_improve_evals >= ns.patience_evals
        if stop:
            print(f"[Early stop] No dev overall_acc improvement for {no_improve_evals} evals.")
        return stop

    def maybe_eval_full(step: int) -> None:
        if full_dev_loader is None:
            return
        fl, facc, ff1 = evaluate_loader(model, full_dev_loader, device, amp_enabled)
        print(f"[Full dev @ {step}] loss={fl:.4f}  macro_f1={ff1:.4f}  {_fmt_acc_line(facc)}")
        _write_eval_log(log_dir / "full_dev_metrics.jsonl", str(step), fl, facc, ff1)
        _log_dev_to_wandb("full_dev", step, fl, facc, ff1)

    stop_training = False
    scaler = GradScaler(enabled=amp_enabled)

    for epoch in tqdm(range(start_epoch, ns.num_epochs), desc="epochs"):
        model.train()
        train_losses: List[float] = []
        train_ok: List[float] = []

        for i, (feat, _, _labels, class_types, _) in enumerate(
            tqdm(train_loader, leave=False, desc=f"epoch {epoch}")
        ):
            feat = feat.to(device, non_blocking=True)
            class_types = class_types.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(feat)
                loss = criterion(logits, class_types)

            batch_acc = float((logits.argmax(-1) == class_types).float().mean().item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.item()))
            train_ok.append(batch_acc)

            global_step += 1
            with open(log_dir / "train_loss.log", "a") as f:
                f.write(f"{global_step}\t{epoch}\t{i}\t{train_losses[-1]:.6f}\t{train_ok[-1]:.6f}\n")
            if use_wandb:
                cur_lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {"train/batch_loss": train_losses[-1], "train/batch_acc": train_ok[-1], "train/lr": cur_lr},
                    step=global_step,
                )

            if ns.eval_steps > 0 and global_step % ns.eval_steps == 0:
                if global_step >= ns.eval_warmup_steps:
                    model.eval()
                    if maybe_eval_sample(f"{epoch}.{global_step}", global_step, epoch):
                        stop_training = True
                        break
                    model.train()

            if full_dev_loader is not None and ns.full_eval_steps > 0:
                if global_step % ns.full_eval_steps == 0 and global_step >= ns.eval_warmup_steps:
                    model.eval()
                    maybe_eval_full(global_step)
                    model.train()

        if stop_training:
            break

        print(
            f"Epoch {epoch}  mean_train_loss={np.mean(train_losses):.4f}  "
            f"mean_batch_acc={np.mean(train_ok):.4f}"
        )
        if use_wandb:
            wandb.log(
                {
                    "train/epoch_mean_loss": float(np.mean(train_losses)),
                    "train/epoch_mean_acc": float(np.mean(train_ok)),
                },
                step=global_step,
            )

        if global_step >= ns.eval_warmup_steps:
            model.eval()
            if maybe_eval_sample(f"epoch_{epoch}_end", global_step, epoch):
                break
            model.train()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train(parse_args(sys.argv[1:]))
