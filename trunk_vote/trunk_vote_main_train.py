# -*- coding: utf-8 -*-
"""
Train / infer multi-head SSL (XLSR, BEATs, or XLSR+BEATs ``cat_linear`` dual).

    # Single backbone (default): ssl_backbone: xlsr | beats
    python multi_head/multi_main_train.py train --config multi_head/mult_config.yaml --gpu 0

    # Dual SSL: XLSR + BEATs fused with cat_linear (requires ssl.xlsr and ssl.beats)
    #   ssl_backbone: xlsr_beats   # or: dual
    #
    # Optional XLSR multi-layer readout (under ``ssl:``; ignored for ``ssl_backbone: beats``):
    #   xlsr_selected_layers: [2, 11, 24]
    #   xlsr_layer_fusion: cat_proj_v1   # last | cat_proj_v1 | cat_proj_v2 | cat_linear | mean | weight_sum

    python multi_head/multi_main_train.py infer \\
        --checkpoint PATH/best.pt --config multi_head/mult_config.yaml \\
        --wav_dir /path/to/wav --out_csv scores.csv [--strategies total,oracle,vote] \\
        [--protocol dev.csv]  # required if oracle is included \\
        [--score_threshold 0.5]

    Use ``--strategy total`` for a single run (deprecated alias of ``--strategies total``).
    Multiple strategies write ``<stem>_<strategy>.csv`` and ``<stem>_<strategy>_binary.csv``.

    After ``infer``, writes ``<out_csv_stem>_binary.csv`` (``name``, ``predict``) with the
    same threshold rule as ``scripts/inference.gen_binary_score`` (real if score >= threshold).

    python multi_head/multi_main_train.py analyze-dev \\
        --config multi_head/multi_base.yaml --gpu 0

``oracle`` inference needs CSV with ``name,type`` matching filenames in ``wav_dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_rs = str(_ROOT)
# Script lives under multi_head/; ensure repo root is searched first, otherwise
# ``import multi_head`` resolves to sibling ``multi_head.py`` (a module), not the package dir.
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from utils import metrics as em
from utils.helpers import parse_filter_types, setup_seed
from crop_dataset import atadd_crop_dataset, atadd_eval_crop_dataset

from analyze_dev import add_analyze_dev_parser, run_analyze_dev
from multi_head.multi_head import (
    build_mult_head_from_args,
    compute_loss,
)

try:
    import wandb
except ImportError:
    wandb = None

TYPE_NAME_TO_IDX = {"speech": 0, "sound": 1, "singing": 2, "music": 3}
IDX_TO_TYPE = {v: k for k, v in TYPE_NAME_TO_IDX.items()}


def _gen_binary_submission(score_file: str, binary_file: str, threshold: float = 0.5) -> None:
    """Convert logits CSV (name, score) to submission CSV (name, predict).

    Same semantics as ``scripts/inference.gen_binary_score``: real if score >= threshold.
    """
    with open(score_file, "r", encoding="utf-8-sig", newline="") as fin, open(
        binary_file, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["name", "predict"])
        for row in reader:
            score = float(row["score"])
            predict = "real" if score >= threshold else "fake"
            writer.writerow([row["name"].strip(), predict])


def _torch_load(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_mult_namespace(yaml_path: str) -> argparse.Namespace:
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    flat: Dict[str, Any] = {}
    for blk in ("data", "ssl", "aug"):
        node = raw.pop(blk, None)
        if isinstance(node, dict):
            flat.update(node)
        elif node is not None:
            raise TypeError(f"`{blk}` must be a mapping in {yaml_path}")
    flat.update(raw)

    ft = flat.get("filter_types")
    if ft in ("", None, "none"):
        flat["filter_types"] = None
        flat["filter_types_parsed"] = None
    elif isinstance(ft, str):
        flat["filter_types_parsed"] = parse_filter_types(ft)
    else:
        flat["filter_types_parsed"] = None

    flat.setdefault("gpu", "0")
    flat.setdefault("specialist_weight", 0.7)
    flat.setdefault("total_weight", 0.3)
    flat.setdefault("label_loss", None)
    flat.setdefault("freeze_backbone", False)
    flat.setdefault("train_task", "atadd-track2")
    flat.setdefault("num_epochs", 100)
    flat.setdefault("batch_size", 32)
    flat.setdefault("lr", 1e-6)
    flat.setdefault("lr_decay", 0.5)
    flat.setdefault("interval", 4)
    flat.setdefault("num_workers", 4)
    flat.setdefault("seed", 1234)
    flat.setdefault("audio_len", 64600)
    flat.setdefault("continue_training", False)
    flat.setdefault("save_best_by", "f1")
    flat.setdefault("eval_steps", 0)
    flat.setdefault("full_eval_steps", 0)
    flat.setdefault("eval_warmup_steps", 0)
    flat.setdefault("patience", 0)
    flat.setdefault("eval_threshold_mode", "fixed")
    flat.setdefault("score_threshold", 0.5)
    flat.setdefault("out_fold", "./ckpt_mult_head/run")
    flat.setdefault("eval_strategy", "oracle")
    flat.setdefault("ssl_backbone", "xlsr")
    flat.setdefault("backbone_dim", None)
    flat.setdefault("sound_aug_methods", ["rawboost", "musan_noise", "rir_reverb"])
    flat.setdefault("rir_path", "/data/liulan/workspace/dataset/RIRS_NOISES")
    flat.setdefault("musan_path", "/data/liulan/workspace/dataset/musan")
    return argparse.Namespace(**flat)


def _yaml_snap_value(v: Any) -> Any:
    """Make training Namespace values safe for PyYAML."""
    if isinstance(v, torch.device):
        return str(v)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _save_config_snapshot(args: argparse.Namespace) -> None:
    d = {
        k: _yaml_snap_value(v)
        for k, v in vars(args).items()
        if k != "filter_types_parsed"
    }
    Path(args.out_fold, "mult_config_used.yaml").write_text(
        yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    ft = args.filter_types_parsed
    if ft:
        print(f"[mult] filter_types: {sorted(ft)}")
    probs = dict(
        speech=args.aug_speech,
        sound=args.aug_sound,
        music=args.aug_music,
        singing=args.aug_singing,
    )
    aug_probs = {k: v for k, v in probs.items() if v > 0.0} or None

    train_ds = atadd_crop_dataset(
        args.atadd_t2_train_audio,
        args.atadd_t2_train_label,
        audio_length=args.audio_len,
        filter_types=ft,
        aug_probs=aug_probs,
        music_aug_method=args.music_aug_method,
        sound_aug_methods=args.sound_aug_methods,
        rir_path=args.rir_path,
        musan_path=args.musan_path,
    )
    sample_ds = atadd_crop_dataset(
        args.atadd_t2_dev_audio,
        args.atadd_t2_dev_label,
        audio_length=args.audio_len,
        filter_types=ft,
        dev_subsample=True,
    )

    def _loader(ds) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            sampler=torch_sampler.SubsetRandomSampler(range(len(ds))),
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )

    assert len(train_ds) and len(sample_ds)
    return _loader(train_ds), _loader(sample_ds)


def _full_dev_loader(args: argparse.Namespace) -> DataLoader:
    ft = args.filter_types_parsed
    ds = atadd_crop_dataset(
        args.atadd_t2_dev_audio,
        args.atadd_t2_dev_label,
        audio_length=args.audio_len,
        filter_types=ft,
        dev_subsample=False,
    )
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=torch_sampler.SubsetRandomSampler(range(len(ds))),
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )


def _class_weight(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    custom = getattr(args, "label_loss", None)
    if custom is None:
        custom = getattr(args, "label_loss_weight", None)

    if custom is None:
        train_task = str(getattr(args, "train_task", "atadd-track2"))
        w = [4.0, 1.0] if train_task == "atadd-track1" else [3.5, 1.0]
    elif isinstance(custom, dict):
        try:
            w = [float(custom["real"]), float(custom["fake"])]
        except KeyError as exc:
            raise ValueError("label_loss mapping must contain 'real' and 'fake'.") from exc
    elif isinstance(custom, (list, tuple)) and len(custom) == 2:
        w = [float(custom[0]), float(custom[1])]
    else:
        raise ValueError(
            "label_loss must be [real_weight, fake_weight] or {real: ..., fake: ...}."
        )
    return torch.tensor(w, dtype=torch.float32, device=device)


def _scores_to_metrics(
    scores: np.ndarray, labels_np: np.ndarray, thr_mode: str, thr_fix: float
) -> Tuple[float, float, float]:
    real_sc = scores[labels_np == 0] #"real": 0
    fake_sc = scores[labels_np == 1] #"fake": 1
    eer, eer_thr = em.compute_eer(real_sc, fake_sc)
    thr = float(eer_thr) if thr_mode == "eer" else float(thr_fix)
    preds = (scores < thr).astype(np.int64)
    f1 = f1_score(labels_np, preds, average="macro")
    return float(eer), float(f1), thr


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if y_true.size else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "real_f1": float(f1_score(y_true, y_pred, pos_label=0, labels=[0, 1], zero_division=0)),
        "fake_f1": float(f1_score(y_true, y_pred, pos_label=1, labels=[0, 1], zero_division=0)),
        "n_samples": int(y_true.size),
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
    }


def _scores_to_detailed_metrics(
    scores: np.ndarray,
    labels_np: np.ndarray,
    types_np: np.ndarray,
    thr_mode: str,
    thr_fix: float,
) -> Dict[str, Any]:
    eer, macro_f1, thr = _scores_to_metrics(scores, labels_np, thr_mode, thr_fix)
    preds = (scores < thr).astype(np.int64)
    out: Dict[str, Any] = {
        "eer": eer,
        "threshold": thr,
        "macro_f1": macro_f1,
        **_binary_metrics(labels_np, preds),
        "per_type": {},
    }
    for t_idx, t_name in IDX_TO_TYPE.items():
        m = types_np == t_idx
        out["per_type"][t_name] = _binary_metrics(labels_np[m], preds[m])
    return out


def _flatten_full_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        f"{prefix}_eer": metrics["eer"],
        f"{prefix}_threshold": metrics["threshold"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_real_f1": metrics["real_f1"],
        f"{prefix}_fake_f1": metrics["fake_f1"],
        f"{prefix}_n_samples": metrics["n_samples"],
        f"{prefix}_n_real": metrics["n_real"],
        f"{prefix}_n_fake": metrics["n_fake"],
    }
    for t_name, tm in metrics["per_type"].items():
        for key in ("accuracy", "macro_f1", "real_f1", "fake_f1", "n_samples", "n_real", "n_fake"):
            row[f"{prefix}_{t_name}_{key}"] = tm[key]
    return row


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _strategy_logits(
    outs: Dict[str, torch.Tensor],
    audio_type: torch.Tensor | None,
    strategy: str,
) -> torch.Tensor:
    strategy = strategy.strip().lower()
    if strategy == "total":
        return outs["total"]
    if strategy == "vote":
        return torch.stack(
            [outs[k] for k in ("speech", "sound", "singing", "music", "total")],
            dim=1,
        ).mean(dim=1)
    if strategy != "oracle":
        raise ValueError(f"Unknown strategy {strategy!r}")
    if audio_type is None:
        raise ValueError("oracle logits require audio_type")
    stack = torch.stack([outs[k] for k in ("speech", "sound", "singing", "music")], dim=1)
    idx = audio_type.long().to(stack.device)
    valid = (idx >= 0) & (idx <= 3)
    idx = idx.clamp(0, 3)
    ar = torch.arange(idx.numel(), device=stack.device)
    spec = stack[ar, idx]
    return torch.where(valid.unsqueeze(1), spec, outs["total"])


def _aggregate_audio_logits(
    crop_logits: List[torch.Tensor],
    crop_names: List[str],
    crop_labels: List[int] | None = None,
    crop_types: List[int] | None = None,
) -> Tuple[torch.Tensor, np.ndarray | None, np.ndarray | None, List[str]]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    labels: Dict[str, int] = {}
    types: Dict[str, int] = {}
    order: List[str] = []

    offset = 0
    for logits in crop_logits:
        cpu_logits = logits.detach().float().cpu()
        for row in cpu_logits:
            name = crop_names[offset]
            if name not in sums:
                sums[name] = row.clone()
                counts[name] = 1
                order.append(name)
                if crop_labels is not None:
                    labels[name] = int(crop_labels[offset])
                if crop_types is not None:
                    types[name] = int(crop_types[offset])
            else:
                sums[name] += row
                counts[name] += 1
            offset += 1

    avg = torch.stack([sums[n] / counts[n] for n in order], dim=0)
    label_np = (
        np.asarray([labels[n] for n in order], dtype=np.int64)
        if crop_labels is not None
        else None
    )
    type_np = (
        np.asarray([types[n] for n in order], dtype=np.int64)
        if crop_types is not None
        else None
    )
    return avg, label_np, type_np, order


def _forward_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logit_l: List[torch.Tensor] = []
    name_l: List[str] = []
    label_l: List[int] = []
    type_l: List[int] = []
    with torch.no_grad():
        for feat, filenames, labels, class_types, _generator in tqdm(
            loader, leave=False, desc=f"{strategy}_crop_logits"
        ):
            wav = feat.to(args.device)
            ctype = class_types.long().to(args.device)
            outs = model(wav)
            logit_l.append(_strategy_logits(outs, ctype, strategy))
            label_l.extend(labels.long().cpu().tolist())
            type_l.extend(class_types.long().cpu().tolist())
            name_l.extend(list(filenames))

    logits, labels_np, types_np, _names = _aggregate_audio_logits(
        logit_l, name_l, label_l, type_l
    )
    scores = F.softmax(logits, dim=1)[:, 0].numpy()
    assert labels_np is not None and types_np is not None
    return scores, labels_np, types_np


@torch.no_grad()
def _mean_dev_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    cw: torch.Tensor,
) -> float:
    model.eval()
    spec_l: List[torch.Tensor] = []
    total_l: List[torch.Tensor] = []
    name_l: List[str] = []
    label_l: List[int] = []
    type_l: List[int] = []
    for feat, filenames, labels, class_types, _generator in loader:
        wav = feat.to(args.device)
        ctype = class_types.long().to(args.device)
        outs = model(wav)
        spec_l.append(_strategy_logits(outs, ctype, "oracle"))
        total_l.append(_strategy_logits(outs, None, "total"))
        name_l.extend(list(filenames))
        label_l.extend(labels.long().cpu().tolist())
        type_l.extend(class_types.long().cpu().tolist())

    spec_logits, labels_np, _types_np, _names = _aggregate_audio_logits(
        spec_l, name_l, label_l, type_l
    )
    total_logits, _labels2, _types2, _names2 = _aggregate_audio_logits(
        total_l, name_l, label_l, type_l
    )
    assert labels_np is not None
    lbl = torch.as_tensor(labels_np, dtype=torch.long, device=args.device)
    spec_logits = spec_logits.to(args.device)
    total_logits = total_logits.to(args.device)
    ce_kw = {} if cw is None else {"weight": cw}
    loss_specialist = F.cross_entropy(spec_logits, lbl, **ce_kw)
    loss_total = F.cross_entropy(total_logits, lbl, **ce_kw)
    sw = float(args.specialist_weight)
    tw = float(args.total_weight)
    return float(((sw * loss_specialist + tw * loss_total) / (sw + tw)).item())


def _persist_latest(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    gs: int,
    out_fold: str,
    best_s: float,
    best_f: float,
    no_im: int,
    best_full_metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": gs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_sample_val": best_s,
            "best_full_val": best_f,
            "best_full_metrics": best_full_metrics,
            "no_improve": no_im,
            "xlsr_eval_locked": getattr(model, "_lock_xlsr_eval", False),
        },
        Path(out_fold, "checkpoint", "latest.pt"),
    )


def train(args: argparse.Namespace) -> torch.nn.Module:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    setup_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.log_dir = os.path.join(args.out_fold, "logs")
    ckpt_dir = os.path.join(args.out_fold, "checkpoint")

    cw = _class_weight(args, args.device)
    model = build_mult_head_from_args(args).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0005,
    )
    wors = math.inf if args.save_best_by in ("loss", "eer") else -math.inf
    best_sample_val = best_full_val = wors
    best_full_metrics = {"f1": -math.inf, "loss": math.inf, "eer": math.inf}
    no_improve = 0
    epoch0 = 0
    step = 0
    ck_latest = Path(ckpt_dir) / "latest.pt"

    if bool(args.continue_training) and ck_latest.is_file():
        blob = _torch_load(str(ck_latest), args.device)
        model.load_state_dict(blob["model_state_dict"])
        optimizer.load_state_dict(blob["optimizer_state_dict"])
        epoch0 = int(blob.get("epoch", -1)) + 1
        step = int(blob.get("global_step", 0))
        best_sample_val = float(blob.get("best_sample_val", best_sample_val))
        best_full_val = float(blob.get("best_full_val", best_full_val))
        raw_best_metrics = blob.get("best_full_metrics")
        if isinstance(raw_best_metrics, dict):
            for k in best_full_metrics:
                if k in raw_best_metrics:
                    best_full_metrics[k] = float(raw_best_metrics[k])
        no_improve = int(blob.get("no_improve", 0))
        if blob.get("xlsr_eval_locked"):
            model._lock_xlsr_eval = True
            if hasattr(model, "backbone"):
                model.backbone.eval()
            else:
                model.frontend_a.eval()
                model.frontend_b.eval()
        print(f"[mult] resume @ epoch={epoch0} global_step={step}")
    elif Path(args.out_fold).is_dir():
        shutil.rmtree(args.out_fold)

    Path(args.out_fold).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_fold, "checkpoint_all_dev").mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(args)

    with open(Path(args.log_dir, "train_loss.log"), "w", encoding="utf-8") as lf:
        lf.write("step\tepoch\tbatch\tloss\tspec\ttotal\n")

    train_ld, sample_ld = _dataloaders(args)
    full_ld = _full_dev_loader(args)

    use_wandb = (
        wandb is not None
        and not getattr(args, "no_wandb", False)
        and getattr(args, "wandb_mode", "") in ("online", "offline")
    )
    if use_wandb:
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            name=args.wandb_run_name or Path(args.out_fold).name,
            config={k: v for k, v in vars(args).items()},
            dir=args.out_fold,
        )

    def chk_metric(ml: float, eer: float, f1v: float) -> float:
        if args.save_best_by == "loss":
            return ml
        return eer if args.save_best_by == "eer" else f1v

    def better(v: float, best: float) -> bool:
        return v < best if args.save_best_by in ("loss", "eer") else v > best

    def hk(v: float) -> float:
        return -v if args.save_best_by in ("loss", "eer") else v

    def better_for_metric(metric: str, val: float, best: float) -> bool:
        return val < best if metric in ("loss", "eer") else val > best

    stop_training = False
    thr_fix = float(args.score_threshold)

    for epoch in tqdm(range(epoch0, int(args.num_epochs)), desc="epochs"):
        model.train()
        lr = args.lr * (float(args.lr_decay) ** (epoch // int(args.interval)))
        for g in optimizer.param_groups:
            g["lr"] = lr

        for i, (feat, _, labels, class_types, _) in enumerate(
            tqdm(train_ld, leave=False, desc=f"ep {epoch}")
        ):
            wav = feat.to(args.device)
            lbl = labels.long().to(args.device)
            ctype = class_types.long().to(args.device)

            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_loss(
                model,
                wav,
                lbl,
                ctype,
                specialist_weight=float(args.specialist_weight),
                total_weight=float(args.total_weight),
                class_weight=cw,
            )
            loss.backward()
            optimizer.step()

            step += 1
            gs = step
            with open(Path(args.log_dir, "train_loss.log"), "a", encoding="utf-8") as lf:
                lf.write(
                    f"{gs}\t{epoch}\t{i}\t{loss.item():.6f}\t"
                    f"{float(parts['loss_specialist']):.6f}\t"
                    f"{float(parts['loss_total']):.6f}\n"
                )
            if use_wandb:
                wandb.log(
                    {"train/loss": loss.item(), "train/lr": lr, "train/specialist_loss": parts['loss_specialist'], "train/total_loss": parts['loss_total']},
                    step=gs,
                )

            if int(args.eval_steps) <= 0 or gs % int(args.eval_steps) != 0:
                continue
            if gs < int(args.eval_warmup_steps):
                print(f"[mult] warmup: skip eval {gs}")
                continue

            mean_dev_loss = _mean_dev_loss(model, sample_ld, args, cw)
            sco_o, lab_o, _ = _forward_scores(model, sample_ld, args, "oracle")
            eer_o, f1_o, thr = _scores_to_metrics(
                sco_o, lab_o, args.eval_threshold_mode, thr_fix
            )
            sco_total, lab_total, _ = _forward_scores(model, sample_ld, args, "total")
            eer_total, f1_total, thr_total = _scores_to_metrics(
                sco_total, lab_total, args.eval_threshold_mode, thr_fix
            )

            strat = getattr(args, "eval_strategy", "oracle")  #dev时采用total策略
            mv = chk_metric(
                mean_dev_loss,
                eer_o if strat == "oracle" else eer_total,
                f1_o if strat == "oracle" else f1_total,
            )
            if better(mv, best_sample_val):
                best_sample_val, no_improve = mv, 0
            else:
                no_improve += 1

            with open(Path(args.log_dir, "dev_loss.log"), "a", encoding="utf-8") as lf:
                lf.write(
                    f"{gs}\tsample_oracle\teer:{eer_o:.6f}\tf1:{f1_o:.6f}"
                    f"\tthr:{thr:.4f}\tmult_ce:{mean_dev_loss:.6f}\n"
                )
                lf.write(
                    f"{gs}\tsample_total\teer:{eer_total:.6f}\tf1:{f1_total:.6f}"
                    f"\tthr:{thr_total:.4f}\tmult_ce:{mean_dev_loss:.6f}\n"
                )

            _persist_latest(
                model,
                optimizer,
                epoch,
                gs,
                args.out_fold,
                best_sample_val,
                best_full_val,
                no_improve,
                best_full_metrics,
            )

            if use_wandb:
                wandb.log(
                    {
                        "sample/oracle_eer": eer_o,
                        "sample/oracle_f1": f1_o,
                        "sample/total_eer": eer_total,
                        "sample/total_f1": f1_total,
                        "sample/mult_ce": mean_dev_loss,
                    },
                    step=gs,
                )

            if not getattr(model, "_lock_xlsr_eval", False):
                model.lock_xlsr_eval_after_first_dev()

            model.train()

            if int(args.patience) > 0 and no_improve >= int(args.patience):
                stop_training = True
                break

            if (
                int(args.full_eval_steps) > 0
                and gs % int(args.full_eval_steps) == 0
                and gs >= int(args.eval_warmup_steps)
            ):
                sco_f_o, lab_f_o, type_f_o = _forward_scores(model, full_ld, args, "oracle")
                eer_f_o, f1_f_o, thr_f_o = _scores_to_metrics(
                    sco_f_o, lab_f_o, args.eval_threshold_mode, thr_fix
                )
                oracle_metrics = _scores_to_detailed_metrics(
                    sco_f_o, lab_f_o, type_f_o, args.eval_threshold_mode, thr_fix
                )
                sco_f_t, lab_f_t, type_f_t = _forward_scores(model, full_ld, args, "total")
                eer_f_t, f1_f_t, thr_f_t = _scores_to_metrics(
                    sco_f_t, lab_f_t, args.eval_threshold_mode, thr_fix
                )
                total_metrics = _scores_to_detailed_metrics(
                    sco_f_t, lab_f_t, type_f_t, args.eval_threshold_mode, thr_fix
                )
                ml_full = _mean_dev_loss(model, full_ld, args, cw)
                eval_strat = getattr(args, "eval_strategy", "oracle")
                chk_f = chk_metric(
                    ml_full,
                    eer_f_o if eval_strat == "oracle" else eer_f_t,
                    f1_f_o if eval_strat == "oracle" else f1_f_t,
                )
                selected_metrics = oracle_metrics if eval_strat == "oracle" else total_metrics
                metric_values = {
                    "f1": float(selected_metrics["macro_f1"]),
                    "loss": float(ml_full),
                    "eer": float(selected_metrics["eer"]),
                }
                full_payload = {
                    "step": gs,
                    "epoch": epoch,
                    "metric": args.save_best_by,
                    "metric_val": chk_f,
                    "eval_strategy": eval_strat,
                    "full_loss": ml_full,
                    "selected": {
                        "eer": selected_metrics["eer"],
                        "f1": selected_metrics["macro_f1"],
                        "accuracy": selected_metrics["accuracy"],
                        "threshold": selected_metrics["threshold"],
                    },
                    "oracle": oracle_metrics,
                    "total": total_metrics,
                }
                step_ckpt = Path(args.out_fold, "checkpoint_all_dev", f"step_{gs}.pt")
                torch.save(model.state_dict(), step_ckpt)
                _save_json(
                    Path(args.out_fold, "checkpoint_all_dev", f"step_{gs}_meta.json"),
                    full_payload,
                )
                with open(
                    Path(args.out_fold, "checkpoint_all_dev", "full_eval_metrics.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as jf:
                    jf.write(json.dumps(full_payload) + "\n")

                full_row = {
                    "step": gs,
                    "epoch": epoch,
                    "eval_strategy": eval_strat,
                    "save_best_by": args.save_best_by,
                    "full_loss": ml_full,
                    "selected_f1": metric_values["f1"],
                    "selected_eer": metric_values["eer"],
                    "selected_accuracy": selected_metrics["accuracy"],
                    **_flatten_full_metrics("oracle", oracle_metrics),
                    **_flatten_full_metrics("total", total_metrics),
                }
                full_csv = Path(args.out_fold, "checkpoint_all_dev", "full_eval_metrics.csv")
                write_header = not full_csv.exists()
                with open(full_csv, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.DictWriter(cf, fieldnames=list(full_row.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(full_row)
                
                # 日志记录两个策略
                with open(Path(args.log_dir, "dev_loss.log"), "a") as lf:
                    lf.write(
                        f"{gs}\tfull_oracle\teer:{eer_f_o:.6f}\tf1:{f1_f_o:.6f}"
                        f"\tacc:{oracle_metrics['accuracy']:.6f}\tthr:{thr_f_o:.4f}"
                        f"\tmult_ce:{ml_full:.6f}\n"
                    )
                    lf.write(
                        f"{gs}\tfull_total\teer:{eer_f_t:.6f}\tf1:{f1_f_t:.6f}"
                        f"\tacc:{total_metrics['accuracy']:.6f}\tthr:{thr_f_t:.4f}"
                        f"\tmult_ce:{ml_full:.6f}\n"
                    )
                    for prefix, metrics in (("oracle", oracle_metrics), ("total", total_metrics)):
                        for t_name, tm in metrics["per_type"].items():
                            lf.write(
                                f"{gs}\tfull_{prefix}_{t_name}"
                                f"\tacc:{tm['accuracy']:.6f}\tf1:{tm['macro_f1']:.6f}"
                                f"\treal_f1:{tm['real_f1']:.6f}\tfake_f1:{tm['fake_f1']:.6f}"
                                f"\tn:{tm['n_samples']}\n"
                            )

                for metric_name, metric_val in metric_values.items():
                    if better_for_metric(metric_name, metric_val, best_full_metrics[metric_name]):
                        best_full_metrics[metric_name] = metric_val
                        torch.save(
                            model.state_dict(),
                            Path(args.out_fold, "checkpoint_all_dev", f"best_{metric_name}.pt"),
                        )
                        meta = dict(full_payload)
                        meta.update({"best_metric": metric_name, "best_metric_val": metric_val})
                        _save_json(
                            Path(args.out_fold, "checkpoint_all_dev", f"best_{metric_name}_meta.json"),
                            meta,
                        )

                if better(chk_f, best_full_val):
                    best_full_val = chk_f
                    torch.save(
                        model.state_dict(),
                        Path(args.out_fold, "checkpoint_all_dev", "best.pt"),
                    )
                    _save_json(Path(args.out_fold, "checkpoint_all_dev", "best_meta.json"), full_payload)
                if use_wandb:
                    wandb.log(
                        {
                            "full/oracle_eer": eer_f_o,
                            "full/oracle_f1": f1_f_o,
                            "full/total_eer": eer_f_t,
                            "full/total_f1": f1_f_t,
                            "full/mult_ce": ml_full,
                        },
                        step=gs,
                    )
                _persist_latest(
                    model,
                    optimizer,
                    epoch,
                    gs,
                    args.out_fold,
                    best_sample_val,
                    best_full_val,
                    no_improve,
                    best_full_metrics,
                )
                model.train()

        if stop_training:
            print(f"[mult] early stop patience={args.patience}")
            break

    if use_wandb:
        wandb.finish()
    return model


def _infer_resolve_strategies(ap: argparse.Namespace) -> List[str]:
    """Match ``analyze_dev`` rules: comma-separated total/oracle/vote."""
    if getattr(ap, "strategy", None) is not None:
        s0 = str(ap.strategy).strip().lower()
        if s0 not in {"total", "oracle", "vote"}:
            raise ValueError(f"Unknown --strategy {ap.strategy!r}")
        raw = ap.strategies.strip() if getattr(ap, "strategies", None) else ""
        if raw and raw != "total":
            raise ValueError("Use either --strategy or --strategies, not both.")
        return [s0]
    raw = (getattr(ap, "strategies", None) or "total").strip()
    strategies = [s.strip().lower() for s in raw.split(",") if s.strip()]
    for s in strategies:
        if s not in {"total", "oracle", "vote"}:
            raise ValueError(
                f"Unknown strategy {s!r} in --strategies (allowed: total, oracle, vote)"
            )
    if not strategies:
        strategies = ["total"]
    return strategies


def _infer_out_paths(out_csv: str, strategies: List[str]) -> Dict[str, Path]:
    """Single strategy → exact ``out_csv``; multiple → ``stem_<strat>.csv``."""
    p = Path(out_csv)
    if len(strategies) == 1:
        return {strategies[0]: p}
    stem, suf = p.stem, p.suffix
    parent = p.parent
    return {s: parent / f"{stem}_{s}{suf}" for s in strategies}


def _load_oracle_protocol(path: str) -> Dict[str, int]:
    prot: Dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            nm = row["name"].strip()
            typ = TYPE_NAME_TO_IDX[row["type"].strip().lower()]
            prot[nm] = typ
    return prot


def cmd_infer(ap: argparse.Namespace) -> None:
    args_ns = load_mult_namespace(ap.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ap.gpu)
    args_ns.cuda = torch.cuda.is_available()
    args_ns.device = torch.device("cuda" if args_ns.cuda else "cpu")

    strategies = _infer_resolve_strategies(ap)
    want = set(strategies)
    prot: Dict[str, int] | None = None
    if "oracle" in want:
        if not ap.protocol:
            raise ValueError(
                "infer with strategy 'oracle' requires --protocol CSV (name,type)"
            )
        prot = _load_oracle_protocol(ap.protocol)

    ck = _torch_load(ap.checkpoint, args_ns.device)
    state = ck.get("model_state_dict", ck)

    model = build_mult_head_from_args(args_ns).to(args_ns.device)
    model.load_state_dict(state)

    bs = ap.batch_size or args_ns.batch_size
    out_paths = _infer_out_paths(ap.out_csv, strategies)
    rows_by: Dict[str, List[Tuple[str, float]]] = {s: [] for s in strategies}

    ds = atadd_eval_crop_dataset(ap.wav_dir, audio_length=args_ns.audio_len)
    ld = DataLoader(
        ds,
        batch_size=int(bs),
        shuffle=False,
        num_workers=args_ns.num_workers,
        pin_memory=args_ns.cuda,
    )
    if "oracle" in want:
        assert prot is not None
        missing = [n for n in ds.all_files if prot.get(n.strip(), -1) < 0]
        if missing:
            raise ValueError(
                "[mult] oracle strategy: protocol missing filenames: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    crop_logits_by: Dict[str, List[torch.Tensor]] = {s: [] for s in strategies}
    crop_names: List[str] = []

    model.eval()
    with torch.no_grad():
        for wav_batch, fnames in tqdm(ld, desc="infer"):
            wav_batch = wav_batch.to(args_ns.device)
            outs = model(wav_batch)
            batch_logits: Dict[str, torch.Tensor] = {}
            if "total" in want:
                batch_logits["total"] = _strategy_logits(outs, None, "total")
            if "oracle" in want:
                assert prot is not None
                idx = [prot.get(f.strip(), -1) for f in fnames]
                idx_t = torch.tensor(idx, device=args_ns.device, dtype=torch.long)
                batch_logits["oracle"] = _strategy_logits(outs, idx_t, "oracle")
            if "vote" in want:
                batch_logits["vote"] = _strategy_logits(outs, None, "vote")

            for s in strategies:
                crop_logits_by[s].append(batch_logits[s])
            crop_names.extend([n.strip() for n in fnames])

    for s in strategies:
        logits, _labels, _types, names = _aggregate_audio_logits(
            crop_logits_by[s], crop_names
        )
        score_map = {
            n: float(v)
            for n, v in zip(names, F.softmax(logits, dim=1)[:, 0].cpu().tolist())
        }
        rows_by[s] = [(n.strip(), score_map[n.strip()]) for n in ds.all_files]

    thr = float(getattr(ap, "score_threshold", 0.5))
    for s in strategies:
        outp = out_paths[s]
        with open(outp, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(("name", "score"))
            w.writerows(rows_by[s])
        print(f"[mult] infer[{s}] {len(rows_by[s])} → {outp}")
        binary_path = outp.parent / f"{outp.stem}_binary.csv"
        _gen_binary_submission(str(outp), str(binary_path), thr)
        print(f"[mult] submission binary[{s}] (thr={thr}) → {binary_path}")


def main() -> None:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="Multi-head trainer")
    tp.add_argument("--config", required=True)
    tp.add_argument(
        "--continue-training",
        action="store_true",
        help="Resume from checkpoint/latest.pt under YAML out_fold (overrides YAML).",
    )
    tp.add_argument("--gpu", default="0")
    tp.add_argument("--wandb_mode", default="disabled", choices=("online", "offline", "disabled"))
    tp.add_argument("--wandb_project", default="AT-ADD-multi-head")
    tp.add_argument("--wandb_run_name", default=None)
    tp.add_argument("--no_wandb", action="store_true")

    ip = sub.add_parser("infer", help="Score wav folder")
    ip.add_argument("--checkpoint", required=True)
    ip.add_argument("--config", required=True)
    ip.add_argument("--wav_dir", required=True)
    ip.add_argument("--out_csv", required=True)
    ip.add_argument(
        "--strategy",
        default=None,
        choices=("oracle", "total", "vote"),
        help="Single strategy (deprecated: use --strategies). Mutually exclusive with non-default --strategies.",
    )
    ip.add_argument(
        "--strategies",
        default="total",
        help="Comma-separated subset of: total,oracle,vote (same as analyze-dev).",
    )
    ip.add_argument("--protocol", default=None)
    ip.add_argument("--batch_size", type=int, default=None)
    ip.add_argument("--gpu", default="0")
    ip.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Threshold for <out_stem>_binary.csv: real if score >= threshold (default 0.5).",
    )

    add_analyze_dev_parser(sub)

    ns = root.parse_args()
    if ns.cmd == "train":
        args = load_mult_namespace(ns.config)
        args.gpu = ns.gpu
        if getattr(ns, "continue_training", False):
            setattr(args, "continue_training", True)
        if ns.wandb_mode == "disabled" or ns.no_wandb:
            args.wandb_mode = None
        else:
            args.wandb_mode = ns.wandb_mode
            args.wandb_project = ns.wandb_project
            args.wandb_run_name = ns.wandb_run_name
        train(args)
    elif ns.cmd == "infer":
        cmd_infer(ns)
    elif ns.cmd == "analyze-dev":
        run_analyze_dev(ns)
    else:
        root.print_help()


if __name__ == "__main__":
    main()
