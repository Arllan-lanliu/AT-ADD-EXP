"""
Training setup utilities for AT-ADD.

Provides helpers for building the model / optimizer / criterion and the
DataLoaders, keeping main_train.py focused on the training loop logic.
"""

from __future__ import annotations

import json
import os
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
import torch.utils.data.sampler as torch_sampler

import random
import numpy as np

from model.model import build_model
from data.dataset import atadd_dataset
from utils.optimizer import SAM


def _dataloader_worker_seed_init(worker_id: int, base_seed: int) -> None:
    """Seed random/numpy/torch in a DataLoader worker (spawn-safe: top-level, picklable).

    Used via ``functools.partial(..., base_seed=args.seed)`` so workers stay
    deterministic under ``torch.multiprocessing`` start method ``spawn``.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def adjust_learning_rate(args, lr: float, optimizer, epoch_num: int) -> None:
    """Step-decay LR: multiply by ``args.lr_decay`` every ``args.interval`` epochs."""
    new_lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# ---------------------------------------------------------------------------
# W&B config serialisation
# ---------------------------------------------------------------------------

def args_for_wandb(args) -> dict:
    """Return a JSON-safe copy of ``vars(args)`` suitable for ``wandb.init(config=...)``."""
    d = {}
    for k, v in vars(args).items():
        if callable(v):
            d[k] = str(v)
        elif v is None:
            d[k] = None
        else:
            try:
                json.dumps(v)
                d[k] = v
            except (TypeError, ValueError):
                d[k] = str(v)
    return d


# ---------------------------------------------------------------------------
# Model + optimiser + criterion
# ---------------------------------------------------------------------------

def build_model_and_optimizer(args):
    """Construct the model, optimiser, and loss criterion.

    Also loads a checkpoint if ``args.continue_training`` is set.

    Returns
    -------
    model : nn.Module
    optimizer : torch.optim.Optimizer (or SAM wrapper)
    criterion : nn.Module
    resume_info : dict with keys ``start_epoch``, ``best_sample_val``, ``best_full_val``, ``no_improve``
    """
    model = build_model(args).to(args.device)

    if args.SAM or args.CSAM:
        optimizer = SAM(
            model.parameters(),
            torch.optim.Adam,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            eps=args.eps,
            weight_decay=0.0005,
        )

    # Sentinel values: for lower-is-better metrics (loss/eer) use +inf;
    # for higher-is-better (f1) use -inf.
    _worse_sentinel = -float("inf") if args.save_best_by == "f1" else float("inf")
    resume_info = dict(
        start_epoch=0,
        global_step=0,
        best_sample_val=_worse_sentinel,
        best_full_val=_worse_sentinel,
        no_improve=0,
    )

    if args.continue_training:
        ckpt_path = os.path.join(args.out_fold, "checkpoint", "latest.pt")
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resume_info.update(
                start_epoch=ckpt["epoch"] + 1,
                global_step=ckpt.get("global_step", 0),
                best_sample_val=ckpt.get("best_sample_val", _worse_sentinel),
                best_full_val=ckpt.get("best_full_val",   _worse_sentinel),
                no_improve=ckpt.get("no_improve", 0),
            )
            print(f"Resumed from epoch {resume_info['start_epoch']}")
        else:
            print("Checkpoint not found, training from scratch.")

    # Class-weighted loss to handle imbalanced real/fake ratio
    class_weight = torch.FloatTensor(
        [4.0, 1.0] if args.train_task == "atadd-track1" else [3.5, 1.0]
    ).to(args.device)
    print(f"Class weight: {class_weight.tolist()}  |  save_best_by: {args.save_best_by}")

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, criterion, resume_info


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    """Construct train and validation DataLoaders.

    Reads ``args.filter_types_parsed`` (frozenset or None) and all ``aug_*``
    probabilities from ``args``.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    """
    ft = args.filter_types_parsed
    if ft is not None:
        print(f"Filtering train/dev to audio types: {sorted(ft)}")

    _raw_probs = {
        "speech":  args.aug_speech,
        "sound":   args.aug_sound,
        "music":   args.aug_music,
        "singing": args.aug_singing,
    }
    aug_probs = {k: v for k, v in _raw_probs.items() if v > 0.0} or None
    if aug_probs:
        print(f"Per-type train augmentation: {aug_probs}  "
              f"(music method: {args.music_aug_method}; "
              f"speech aug: {getattr(args, 'speech_aug_method', 'none')})")

    train_kw = dict(
        audio_length=args.audio_len,
        filter_types=ft,
        aug_probs=aug_probs,
        music_aug_method=args.music_aug_method,
        speech_aug_method=getattr(args, "speech_aug_method", "none"),
        speech_rawboost_algo=int(getattr(args, "speech_rawboost_algo", 5)),
        musan_path=getattr(args, "musan_path", "") or "",
        rir_path=getattr(args, "rir_path", "") or "",
    )

    if args.train_task == "atadd-track1":
        train_ds = atadd_dataset(
            args.atadd_t1_train_audio,
            args.atadd_t1_train_label,
            **train_kw,
        )
        val_ds = atadd_dataset(
            args.atadd_t1_dev_audio, args.atadd_t1_dev_label,
            audio_length=args.audio_len, filter_types=ft,
            dev_subsample=True,
        )
    else:  # atadd-track2
        train_ds = atadd_dataset(
            args.atadd_t2_train_audio,
            args.atadd_t2_train_label,
            **train_kw,
        )
        val_ds = atadd_dataset(
            args.atadd_t2_dev_audio, args.atadd_t2_dev_label,
            audio_length=args.audio_len, filter_types=ft,
            dev_subsample=True,
        )

    assert len(train_ds) > 0, f"Train dataset is empty — check paths in your config."
    assert len(val_ds)   > 0, f"Val dataset is empty — check paths in your config."

    # Separate seeded generators so train / val data order is identical
    # across runs with the same args.seed, regardless of model architecture.
    g_train = torch.Generator()
    g_train.manual_seed(args.seed)
    g_val = torch.Generator()
    g_val.manual_seed(args.seed + 1)

    worker_init = partial(_dataloader_worker_seed_init, base_seed=args.seed)

    def _make_loader(ds, g):
        return DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            sampler=torch_sampler.SubsetRandomSampler(range(len(ds)), generator=g),
            pin_memory=args.cuda,
            worker_init_fn=worker_init,
        )

    return _make_loader(train_ds, g_train), _make_loader(val_ds, g_val)


# ---------------------------------------------------------------------------
# Full dev DataLoader (no subsampling — for periodic comprehensive eval)
# ---------------------------------------------------------------------------

def build_full_dev_loader(args):
    """Build a DataLoader for the *complete* dev set (no per-type subsampling).

    Used for the less-frequent full evaluation (``full_eval_steps``).
    """
    ft = args.filter_types_parsed
    if args.train_task == "atadd-track1":
        ds = atadd_dataset(
            args.atadd_t1_dev_audio, args.atadd_t1_dev_label,
            audio_length=args.audio_len, filter_types=ft,
            dev_subsample=False,
        )
    else:
        ds = atadd_dataset(
            args.atadd_t2_dev_audio, args.atadd_t2_dev_label,
            audio_length=args.audio_len, filter_types=ft,
            dev_subsample=False,
        )
    g_full = torch.Generator()
    g_full.manual_seed(args.seed + 2)
    worker_init = partial(_dataloader_worker_seed_init, base_seed=args.seed)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(ds)), generator=g_full),
        pin_memory=args.cuda,
        worker_init_fn=worker_init,
    )
