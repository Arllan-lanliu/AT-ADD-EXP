# -*- coding: utf-8 -*-
"""
Dev-set metrics for multi-head XLSR (Track-2): full dev (no subsample), total / oracle / vote.

Repo root examples::

    python multi_head/analyze_dev.py --config multi_head/multi_base.yaml --gpu 0

    python multi_head/multi_main_train.py analyze-dev --config multi_head/multi_base.yaml --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
_rs = str(_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from crop_dataset import atadd_crop_dataset
from multi_head.multi_head import ALL_HEAD_KEYS, build_mult_head_from_args
from utils import metrics as em

IDX_TO_TYPE = {0: "speech", 1: "sound", 2: "singing", 3: "music"}


def register_analyze_dev_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Training YAML (same as train).")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit .pt; default: under out_fold try best.pt then latest.pt (or reverse).",
    )
    p.add_argument(
        "--checkpoint_prefer",
        default="best",
        choices=("best", "latest"),
        help="When --checkpoint omitted: order to try checkpoint files under out_fold.",
    )
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader workers from config (use 0 in restricted shells).",
    )
    p.add_argument(
        "--strategies",
        default="total,oracle,vote",
        help="Comma-separated subset of: total,oracle,vote",
    )
    p.add_argument(
        "--threshold_mode",
        default="fixed",
        choices=("fixed", "eer"),
        help="fixed: use --score_threshold; eer: threshold from pooled EER on dev.",
    )
    p.add_argument("--score_threshold", type=float, default=0.5)
    p.add_argument(
        "--out_dir",
        default=None,
        help="Directory for CSV + JSON (default: <out_fold>/result).",
    )


def _torch_load(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _scores_to_metrics(
    scores: np.ndarray, labels_np: np.ndarray, thr_mode: str, thr_fix: float
) -> Tuple[float, float, float]:
    real_sc = scores[labels_np == 0]
    fake_sc = scores[labels_np == 1]
    eer, eer_thr = em.compute_eer(real_sc, fake_sc)
    thr = float(eer_thr) if thr_mode == "eer" else float(thr_fix)
    preds = (scores < thr).astype(np.int64)
    f1 = f1_score(labels_np, preds, average="macro")
    return float(eer), float(f1), float(thr)


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


def _collect_scores_one_strategy(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    strategy: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Returns scores (P real), labels 0/1, type_idx 0-3, names, all logits, selected logits."""
    return _collect_scores_many_strategies(
        model, loader, strategies=[strategy], device=device
    )[strategy]


def _selected_head_name(strategy: str, type_idx: int) -> str:
    if strategy == "total":
        return "total"
    if strategy == "oracle":
        return IDX_TO_TYPE[int(type_idx)]
    if strategy == "vote":
        return "avg_logits"
    raise ValueError(strategy)


def _collect_scores_many_strategies(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    strategies: List[str],
    device: torch.device,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
    """Collect strategies with crop logits averaged back to audio-level logits."""
    model.eval()
    wanted = list(dict.fromkeys(strategies))
    logits_l: Dict[str, List[np.ndarray]] = {k: [] for k in ALL_HEAD_KEYS}
    lb_l: List[np.ndarray] = []
    ty_l: List[np.ndarray] = []
    nm_l: List[str] = []

    with torch.no_grad():
        for feat, fnames, labels, class_types, _ in tqdm(
            loader, leave=False, desc="analyze_dev[all]"
        ):
            wav = feat.to(device)
            all_logits = model(wav)

            lb_l.append(labels.long().numpy())
            ty_l.append(class_types.long().numpy())
            nm_l.extend(f.strip() for f in fnames)
            for k in ALL_HEAD_KEYS:
                logits_l[k].append(all_logits[k].detach().float().cpu().numpy())

    crop_labels = np.concatenate(lb_l).astype(np.int64)
    crop_types = np.concatenate(ty_l).astype(np.int64)
    crop_names = np.array(nm_l)

    order: List[str] = []
    index: Dict[str, int] = {}
    for name in crop_names:
        if name not in index:
            index[name] = len(order)
            order.append(str(name))

    labels = np.zeros(len(order), dtype=np.int64)
    types = np.zeros(len(order), dtype=np.int64)
    counts = np.zeros(len(order), dtype=np.float32)
    for name, label, type_idx in zip(crop_names, crop_labels, crop_types):
        j = index[str(name)]
        labels[j] = int(label)
        types[j] = int(type_idx)
        counts[j] += 1.0

    logits: Dict[str, np.ndarray] = {}
    for head, chunks in logits_l.items():
        crop_head_logits = np.concatenate(chunks, axis=0)
        sums = np.zeros((len(order), crop_head_logits.shape[1]), dtype=np.float32)
        for row, name in zip(crop_head_logits, crop_names):
            sums[index[str(name)]] += row.astype(np.float32)
        logits[head] = sums / counts[:, None]

    names = np.array(order)
    out = {}
    for strategy in wanted:
        if strategy == "total":
            selected_logits = logits["total"]
        elif strategy == "oracle":
            stack = np.stack([logits[k] for k in ("speech", "sound", "singing", "music")], axis=1)
            selected_logits = stack[np.arange(len(names)), np.clip(types, 0, 3)]
        elif strategy == "vote":
            selected_logits = np.stack([logits[k] for k in ALL_HEAD_KEYS], axis=1).mean(axis=1)
        else:
            raise ValueError(strategy)
        scores = torch.softmax(torch.from_numpy(selected_logits), dim=1)[:, 0].numpy()
        out[strategy] = (
            scores,
            labels,
            types,
            names,
            logits,
            selected_logits,
        )
    return out


def _per_type_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, types: np.ndarray
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t_idx, t_name in IDX_TO_TYPE.items():
        m = types == t_idx
        if not np.any(m):
            continue
        rep = classification_report(
            y_true[m],
            y_pred[m],
            labels=[0, 1],
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        )
        f1_real = float(rep["real"]["f1-score"])
        f1_fake = float(rep["fake"]["f1-score"])
        out[t_name] = (f1_real + f1_fake) / 2.0
    return out


def _track2_score(type_f1: Dict[str, float]) -> float:
    order = ["speech", "sound", "singing", "music"]
    vals = [type_f1[t] for t in order if t in type_f1]
    return float(sum(vals) / max(len(vals), 1))


def resolve_checkpoint_explicit(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    return p


def resolve_checkpoint_under_out_fold(out_fold: Path, prefer: str) -> Path:
    out_fold = Path(out_fold).resolve()
    best_p = out_fold / "checkpoint_all_dev" / "best.pt"
    latest_p = out_fold / "checkpoint" / "latest.pt"
    order = (best_p, latest_p) if prefer == "best" else (latest_p, best_p)
    for p in order:
        if p.is_file():
            return p
    raise FileNotFoundError(f"No checkpoint under {out_fold} (tried {order[0]} and {order[1]}).")


def add_analyze_dev_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser("analyze-dev", help="Full dev metrics (total/oracle/vote) for multi-head")
    register_analyze_dev_args(p)
    return p


def run_analyze_dev(ns: argparse.Namespace) -> None:
    # Local import avoids circular dependency at module load.
    from multi_head.multi_main_train import load_mult_namespace

    os.environ["CUDA_VISIBLE_DEVICES"] = str(ns.gpu)
    args = load_mult_namespace(ns.config)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if ns.batch_size is not None:
        args.batch_size = int(ns.batch_size)
    if ns.num_workers is not None:
        args.num_workers = int(ns.num_workers)

    out_fold = Path(args.out_fold)
    if ns.checkpoint:
        ckpt_path = resolve_checkpoint_explicit(ns.checkpoint)
    else:
        ckpt_path = resolve_checkpoint_under_out_fold(out_fold, str(ns.checkpoint_prefer))

    out_dir = Path(ns.out_dir) if ns.out_dir else (out_fold / "result")
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = [s.strip().lower() for s in ns.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in {"total", "oracle", "vote"}:
            raise ValueError(f"Unknown strategy {s!r} (allowed: total, oracle, vote)")

    ck = _torch_load(str(ckpt_path), args.device)
    state = ck.get("model_state_dict", ck)

    model = build_mult_head_from_args(args).to(args.device)
    model.load_state_dict(state)

    loader = _full_dev_loader(args)
    print(f"[analyze-dev] samples={len(loader.dataset)} checkpoint={ckpt_path}")
    if any(s.strip().lower() == "vote" for s in strategies):
        print(
            "[analyze-dev] vote: per-crop logits are averaged across heads, then averaged "
            "across crops before scoring."
        )

    summaries: Dict[str, Any] = {}
    thr_mode = str(ns.threshold_mode)
    thr_fix = float(ns.score_threshold)
    collected = _collect_scores_many_strategies(
        model,
        loader,
        strategies=strategies,
        device=args.device,
    )

    for strategy in strategies:
        scores, labels, types, names, logits_by_head, selected_logits = collected[strategy]
        eer, macro_f1, thr = _scores_to_metrics(scores, labels, thr_mode, thr_fix)
        preds_lbl = (scores < thr).astype(np.int64)
        preds_str = np.where(scores >= thr, "real", "fake")

        csv_path = out_dir / f"multidev_{strategy}_scores.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            header = [
                "name",
                "score",
                "type",
                "label",
                "predict",
                "correct",
                "selected_head",
                "selected_logit_real",
                "selected_logit_fake",
            ]
            for head in ALL_HEAD_KEYS:
                header.extend(
                    [
                        f"{head}_logit_real",
                        f"{head}_logit_fake",
                        f"{head}_prob_real",
                        f"{head}_prob_fake",
                    ]
                )
            w.writerow(header)
            for i in range(len(names)):
                tn = IDX_TO_TYPE[int(types[i])]
                ylbl = "real" if labels[i] == 0 else "fake"
                row = [
                    names[i],
                    f"{scores[i]:.8f}",
                    tn,
                    ylbl,
                    preds_str[i],
                    int(preds_lbl[i] == labels[i]),
                    _selected_head_name(strategy, int(types[i])),
                    f"{selected_logits[i, 0]:.8f}",
                    f"{selected_logits[i, 1]:.8f}",
                ]
                for head in ALL_HEAD_KEYS:
                    head_logits = logits_by_head[head][i]
                    head_probs = torch.softmax(torch.from_numpy(head_logits), dim=0).numpy()
                    row.extend(
                        [
                            f"{head_logits[0]:.8f}",
                            f"{head_logits[1]:.8f}",
                            f"{head_probs[0]:.8f}",
                            f"{head_probs[1]:.8f}",
                        ]
                    )
                w.writerow(row)

        by_type = _per_type_macro_f1(labels, preds_lbl, types)
        t2 = _track2_score(by_type)

        print(f"\n===== strategy={strategy}  thr={thr:.6f}  threshold_mode={thr_mode} =====")
        print(f"Pooled EER: {eer:.6f}")
        print(f"Pooled Macro-F1 (real/fake @ thr): {macro_f1:.6f}")
        print(f"Track-2 avg of per-type Macro-F1: {t2:.6f}")
        for t_name in ["speech", "sound", "singing", "music"]:
            if t_name in by_type:
                print(f"  Macro-F1 [{t_name:7s}] = {by_type[t_name]:.6f}")

        report_txt = classification_report(labels, preds_lbl, target_names=["real", "fake"], digits=4)
        report_dict = classification_report(
            labels,
            preds_lbl,
            target_names=["real", "fake"],
            output_dict=True,
            zero_division=0,
        )
        print(report_txt)

        analysis_txt_path = out_dir / f"multidev_{strategy}_analysis.txt"
        with open(analysis_txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"strategy={strategy}\n")
            tf.write(f"checkpoint={ckpt_path}\n")
            tf.write(f"threshold={thr:.8f}\n")
            tf.write(f"threshold_mode={thr_mode}\n")
            tf.write(f"pooled_eer={eer:.8f}\n")
            tf.write(f"pooled_macro_f1={macro_f1:.8f}\n")
            tf.write(f"track2_avg_macro_f1={t2:.8f}\n")
            for t_name in ["speech", "sound", "singing", "music"]:
                if t_name in by_type:
                    tf.write(f"per_type_macro_f1[{t_name}]={by_type[t_name]:.8f}\n")
            tf.write("\n")
            tf.write(report_txt)

        analysis_json_path = out_dir / f"multidev_{strategy}_analysis.json"
        analysis_payload = {
            "strategy": strategy,
            "eer": eer,
            "macro_f1_pooled": macro_f1,
            "threshold": thr,
            "threshold_mode": thr_mode,
            "track2_avg_macro_f1": t2,
            "per_type_macro_f1": by_type,
            "classification_report": report_dict,
            "n_samples": int(len(names)),
            "checkpoint": str(ckpt_path),
            "scores_csv": str(csv_path.resolve()),
        }
        with open(analysis_json_path, "w", encoding="utf-8") as jf:
            json.dump(analysis_payload, jf, indent=2)

        summaries[strategy] = {
            "eer": eer,
            "macro_f1_pooled": macro_f1,
            "threshold": thr,
            "threshold_mode": thr_mode,
            "track2_avg_macro_f1": t2,
            "per_type_macro_f1": by_type,
            "n_samples": int(len(names)),
            "checkpoint": str(ckpt_path),
            "scores_csv": str(csv_path.resolve()),
            "analysis_txt": str(analysis_txt_path.resolve()),
            "analysis_json": str(analysis_json_path.resolve()),
        }

    meta_path = out_dir / "multidev_summary.json"
    payload = {"strategies": summaries, "dev_label_csv": args.atadd_t2_dev_label}
    with open(meta_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, indent=2)
    print(f"\n[analyze-dev] wrote {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-head full dev metrics")
    register_analyze_dev_args(ap)
    ns = ap.parse_args()
    run_analyze_dev(ns)


if __name__ == "__main__":
    main()
