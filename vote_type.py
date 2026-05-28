#!/usr/bin/env python3
"""
Type-routed spoof detection inference.

1. Load a **4-way type classifier** checkpoint from ``ckpt_type_classification/*``
   (``type_train_cli.json`` + ``checkpoint/best.pt`` or ``latest.pt``).
2. Predict the audio type for each utterance and pass that predicted type to the
   multi-head model's oracle strategy, selecting the matching specialist head.
3. Score **dev** and **eval** audio roots (defaults from ``conf/vote/ft_routed_ssl_aasist.yaml``
   ``data.atadd_t2_*``), write logits CSV and (for dev) an analysis CSV matching ``analyze.py``
   conventions for metrics.

Example::

    python vote_type.py \\
        --type_ckpt_dir ./ckpt_type_classification/xlsr \\
        --multi_head_model_path ./ckpt_t2_multi_head_layer/xslrbeats_beats3_6_9_xlsr3_11_24 \\
        --out_dir ./ckpt_t2_vote/type_xlsr_multiHead_oracle_beats369_xlsr31124 \\
        --vote_yaml ./conf/vote/ft_routed_ssl_aasist.yaml \\
        --gpu 1 --batch_size 160
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.type_classifier import (
    ATADD_AUDIO_TYPE_NAMES,
    LogMelCNNATADDTypeClassifier,
    XLSRATADDTypeClassifier,
)
from data.dataset import atadd_eval_dataset

TYPE_ORDER_PRINT = ["speech", "sound", "music", "singing"]


def _load_inference_module():
    path = os.path.join(_ROOT, "scripts", "inference.py")
    spec = importlib.util.spec_from_file_location("atadd_inference", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _load_torch_state(path: str, device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_model_state_dict(loaded: object) -> Dict[str, Any]:
    """Unwrap checkpoints saved like ``main_train`` (nested ``model_state_dict``)."""
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(loaded)}")
    if "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
        return loaded["model_state_dict"]
    if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        return loaded["state_dict"]
    return loaded


def _state_top_levels(sd: Dict[str, Any]) -> set:
    return {k.split(".", 1)[0] for k in sd if isinstance(k, str)}


def _rewrite_top_level(sd: Dict[str, Any], old: str, new: str) -> Dict[str, Any]:
    op, np = old + ".", new + "."
    return {np + k[len(op) :] if k.startswith(op) else k: v for k, v in sd.items()}


def _maybe_strip_outer_model_wrapper(sd: Dict[str, Any], label: str) -> Dict[str, Any]:
    """If every key is ``model.xxx``, unwrap to ``xxx``."""
    ks = list(sd.keys())
    if ks and all(isinstance(k, str) and k.startswith("model.") for k in ks):
        print(f"[{label}] stripping outer checkpoint wrapper model.*")
        return {k[len("model.") :]: v for k, v in sd.items()}
    return sd


def _is_dual_ssl_model(name: str) -> bool:
    n = (name or "").lower()
    return any(
        x in n
        for x in (
            "ft-xlsrmertaasist",
            "ft-xlsrbeatsaasist",
            "ft-xlsrwavlmaasist",
            "ft-xlsrclapaasist",
        )
    )


def _dual_ssl_legacy_key_remap(
    sd: Dict[str, Any], label: str, registry_model_name: str
) -> Dict[str, Any]:
    """
    Map historical top-level module names to current ``DualSSLModel`` fields::

        w2vxlsr / xlsr / …  → frontend_a
        w2vmert / mert / …  → frontend_b   (MERT dual models)
        beats_ssl / beats … → frontend_b   (BEATs dual models)
        fusion               → fusion_module
        fuse_proj.*          → fusion_module.proj.*  (flattened CatLinear, legacy checkpoints)
    """
    if not _is_dual_ssl_model(registry_model_name):
        return sd

    def roots() -> set:
        return _state_top_levels(sd)

    def pick_first(cands: List[str], r: set) -> str | None:
        for c in cands:
            if c in r:
                return c
        return None

    r0 = roots()
    # fusion block (cat_linear etc.)
    if "fusion_module" not in r0 and "fusion" in r0:
        print(f"[{label}] legacy ckpt: remapping fusion.* → fusion_module.*")
        sd = _rewrite_top_level(sd, "fusion", "fusion_module")
        r0 = roots()

    # XLS-R branch
    if "frontend_a" not in r0:
        leg = pick_first(
            ["w2vxlsr", "w2v_xlsr", "ssl_xlsr", "wav2vec2", "xlsr"],
            r0,
        )
        if leg:
            print(f"[{label}] legacy ckpt: remapping {leg}.* → frontend_a.*")
            sd = _rewrite_top_level(sd, leg, "frontend_a")
            r0 = roots()

    # Second encoder
    if "frontend_b" not in r0:
        nm = (registry_model_name or "").lower()
        if "mert" in nm and "beats" not in nm:
            leg = pick_first(["w2vmert", "ssl_mert", "mert_model", "mert"], r0)
        elif "beats" in nm:
            leg = pick_first(["beats_ssl", "ssl_beats", "beats_model", "beats"], r0)
        else:
            leg = pick_first(
                ["w2vmert", "ssl_mert", "beats_ssl", "ssl_beats", "mert", "beats"], r0
            )
        if leg:
            print(f"[{label}] legacy ckpt: remapping {leg}.* → frontend_b.*")
            sd = _rewrite_top_level(sd, leg, "frontend_b")
            r0 = roots()

    # CatLinear-equivalent fusion: some checkpoints store concat-proj as top-level ``fuse_proj``
    # (see ``vote_mean.py`` multi-model setups) instead of ``fusion_module.proj``.
    r0 = roots()
    has_fusion_mod = any(k.startswith("fusion_module.") for k in sd)
    if "fuse_proj" in r0 and not has_fusion_mod:
        print(f"[{label}] legacy ckpt: remapping fuse_proj.* → fusion_module.proj.*")
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("fuse_proj."):
                new_sd["fusion_module.proj." + k[len("fuse_proj.") :]] = v
            else:
                new_sd[k] = v
        sd = new_sd

    return sd


def _normalize_spoofer_state_dict(
    sd: Dict[str, Any], label: str, registry_model_name: str = ""
) -> Dict[str, Any]:
    """
    Compatibility for older checkpoints / DataParallel::

    * Strip ``module.`` prefix
    * Rename ``w2vaasist.*`` → ``backend.*`` (historical submodule name vs ``backend``)
    * Rename legacy dual-encoder tops → ``frontend_a`` / ``frontend_b`` / ``fusion_module``
    """
    out: Dict[str, Any] = dict(sd)
    out = _maybe_strip_outer_model_wrapper(out, label)
    keys = list(out.keys())

    # DataParallel wrapper
    if keys and keys[0].startswith("module."):
        print(f"[{label}] stripping DataParallel prefix module.*")
        out = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in out.items()}
        keys = list(out.keys())

    has_wb = any(k.startswith("w2vaasist.") for k in keys)
    has_backend = any(k.startswith("backend.") for k in keys)
    if has_wb and not has_backend:
        print(f"[{label}] legacy ckpt: remapping w2vaasist.* → backend.*")
        out = {
            ("backend." + k[len("w2vaasist.") :] if k.startswith("w2vaasist.") else k): v
            for k, v in out.items()
        }

    if registry_model_name:
        out = _dual_ssl_legacy_key_remap(out, label, registry_model_name)

    tops = sorted(_state_top_levels(out))
    pref = ", ".join(tops[:16])
    if len(tops) > 16:
        pref += "…"
    print(f"[{label}] checkpoint top-level groups ({len(tops)}): {pref}")

    return out


def _load_detector_checkpoint(path: str, device, label: str, registry_model_name: str = "") -> Dict[str, Any]:
    raw = _load_torch_state(path, device)
    sd = _extract_model_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError(f"{path}: expected state dict, got {type(sd)}")
    return _normalize_spoofer_state_dict(sd, label, registry_model_name)


def _load_vote_yaml_paths(yaml_path: str) -> dict:
    """Return ``atadd_t2_dev_audio``, ``atadd_t2_dev_label``, ``atadd_t2_eval_audio``."""
    import yaml

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"vote yaml not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get("data") or {}
    keys = ("atadd_t2_dev_audio", "atadd_t2_dev_label", "atadd_t2_eval_audio")
    out = {k: data.get(k) for k in keys}
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise KeyError(f"YAML {yaml_path!r} missing data fields: {missing}")
    return out  # type: ignore[return-value]


def _find_type_classifier_checkpoint(type_ckpt_dir: str) -> str:
    """Prefer ``checkpoint/best.pt``, else ``checkpoint/latest.pt``."""
    for name in ("checkpoint/best.pt", "checkpoint/latest.pt"):
        p = os.path.join(type_ckpt_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No type classifier weights under {type_ckpt_dir!r}. "
        f"Expected checkpoint/best.pt or checkpoint/latest.pt"
    )


def _load_type_model(type_ckpt_dir: str, device: torch.device) -> torch.nn.Module:
    cli_path = os.path.join(type_ckpt_dir, "type_train_cli.json")
    if not os.path.isfile(cli_path):
        raise FileNotFoundError(f"Missing {cli_path}")

    with open(cli_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    dev_str = "cuda" if device.type == "cuda" else "cpu"
    kind = cfg.get("model", "logmelcnn")
    audio_len = int(cfg.get("audio_len", 64600))

    if kind == "xlsr":
        xdir = cfg.get("xlsr_dir") or ""
        if not xdir or not os.path.isdir(os.path.expanduser(xdir)):
            raise ValueError(f"type_train_cli.json: invalid xlsr_dir {xdir!r}")
        model = XLSRATADDTypeClassifier(
            model_dir=os.path.expanduser(xdir),
            device=dev_str,
            freeze_frontend=bool(cfg.get("freeze_xlsr", True)),
            xlsr_dim=int(cfg.get("xlsr_dim", 1024)),
            head_hidden_dim=int(cfg.get("head_hidden", 256)),
            head_dropout=float(cfg.get("head_dropout", 0.1)),
            sampling_rate=int(cfg.get("sample_rate", 16000)),
        )
    elif kind == "logmelcnn":
        model = LogMelCNNATADDTypeClassifier(
            sample_rate=int(cfg.get("sample_rate", 16000)),
            n_fft=int(cfg.get("mel_n_fft", 1024)),
            hop_length=int(cfg.get("mel_hop", 256)),
            n_mels=int(cfg.get("mel_bins", 128)),
            base_channels=int(cfg.get("cnn_base_channels", 32)),
            head_dropout=float(cfg.get("cnn_head_dropout", 0.2)),
        )
    else:
        raise ValueError(f"Unsupported type model {kind!r} in type_train_cli.json")

    ckpt = _find_type_classifier_checkpoint(type_ckpt_dir)
    state = _extract_model_state_dict(_load_torch_state(ckpt, device))
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    setattr(model, "_vote_type_cli", cfg)
    setattr(model, "_vote_type_ckpt_path", ckpt)
    setattr(model, "_vote_type_audio_len", audio_len)

    print(f"[type clf] architecture={kind!r} weights={ckpt}")
    print(f"[type clf] audio_len(from cli)={audio_len}")
    return model


def _init_branch_args(inference_mod, model_path: str, gpu: str, batch_size: int):
    # ``inference._init_args`` parses the list as-is (no stripping of prog like ``sys.argv[0]``);
    # the list must start with ``--flag`` tokens only.
    argv = [
        "--model_path",
        os.path.abspath(model_path),
        "--gpu",
        gpu,
        "--eval_threshold_mode",
        "fixed",
        "--threshold",
        "0.5",
        "--batch_size",
        str(batch_size),
    ]
    return inference_mod._init_args(argv)


def _scores_from_detector(
    model: torch.nn.Module, wave_b: torch.Tensor, infer_args, use_amp: bool
) -> torch.Tensor:
    """Return P(real) as shape ``(batch,)`` float tensor on infer_args.device."""
    if wave_b.numel() == 0:
        return torch.empty(0, device=infer_args.device)
    wave_b = wave_b.to(infer_args.device, non_blocking=True)
    with torch.no_grad():
        with autocast(enabled=use_amp):
            _, outputs = model(wave_b)
    if getattr(infer_args, "base_loss", "ce") == "bce":
        return torch.sigmoid(outputs[:, 0]).detach().float().cpu()
    return F.softmax(outputs, dim=1)[:, 0].detach().float().cpu()


def _type_predict(type_model: torch.nn.Module, wave_b: torch.Tensor, device, use_amp: bool):
    wave_b = wave_b.to(device, non_blocking=True)
    with torch.no_grad():
        with autocast(enabled=use_amp):
            logits = type_model(wave_b)
        pred = logits.argmax(dim=-1).cpu().numpy()
        probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
    return pred, probs


def _speech_mask_from_pred(pred_id: np.ndarray) -> np.ndarray:
    """True where routed to MERT/XLSR+ speech branch (fine types speech, singing)."""
    return np.isin(pred_id, np.asarray([0, 2], dtype=pred_id.dtype))


def routed_scores_for_batch(
    type_model,
    speech_model,
    sound_model,
    speech_args,
    sound_args,
    wave: torch.Tensor,
    _filenames: List[str],
    device: torch.device,
    amp_type: bool,
    amp_detect: bool,
):
    """Run type head + two detectors; returns scores ndarray ``(batch,)`` aligned with filenames."""
    pred_np, probs = _type_predict(type_model, wave, device, amp_type)
    bsz = wave.size(0)
    scores_out = np.zeros(bsz, dtype=np.float64)

    speech_side = _speech_mask_from_pred(pred_np)
    idx_sp = np.flatnonzero(speech_side)
    idx_so = np.flatnonzero(~speech_side)

    if idx_sp.size > 0:
        tb = torch.as_tensor(idx_sp, dtype=torch.long)
        w_sp = wave[tb]
        s_sp = _scores_from_detector(speech_model, w_sp, speech_args, amp_detect).numpy()
        scores_out[idx_sp] = s_sp
    if idx_so.size > 0:
        tb = torch.as_tensor(idx_so, dtype=torch.long)
        w_so = wave[tb]
        s_so = _scores_from_detector(sound_model, w_so, sound_args, amp_detect).numpy()
        scores_out[idx_so] = s_so

    details = [
        {
            "pred_type": ATADD_AUDIO_TYPE_NAMES[int(pred_np[i])],
            "pred_type_id": int(pred_np[i]),
            "type_probs": {ATADD_AUDIO_TYPE_NAMES[j]: float(probs[i, j]) for j in range(4)},
        }
        for i in range(bsz)
    ]
    return scores_out.astype(np.float32), pred_np, details


def _find_multi_head_checkpoint(model_dir: str) -> str:
    """Prefer the full-dev best multi-head checkpoint, then sample-dev / latest."""
    candidates = [
        "checkpoint_all_dev/best.pt",
        "checkpoint_sample_dev/best.pt",
        "checkpoint/best.pt",
        "checkpoint/latest.pt",
    ]
    for name in candidates:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p

    top3_path = os.path.join(model_dir, "checkpoint_sample_dev", "top3.json")
    if os.path.isfile(top3_path):
        with open(top3_path, "r", encoding="utf-8") as f:
            ranked = json.load(f)
        if ranked:
            p = ranked[0].get("path")
            if p and not os.path.isabs(p):
                p = os.path.join(model_dir, p)
            if p and os.path.isfile(p):
                return p

    raise FileNotFoundError(
        f"No multi-head checkpoint found under {model_dir!r}. "
        "Expected checkpoint_all_dev/best.pt, checkpoint/latest.pt, or checkpoint_sample_dev/top3.json."
    )


def _find_multi_head_config(model_dir: str) -> str:
    for name in ("mult_config_used.yaml", "config.yaml"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No multi-head config found under {model_dir!r}. Expected mult_config_used.yaml or config.yaml."
    )


def _load_multi_head_model(
    model_dir: str,
    config_path: str | None,
    checkpoint_path: str | None,
    gpu: str,
    batch_size: int,
):
    from multi_head.multi_head import build_mult_head_from_args, inference as multi_head_inference
    from multi_head.multi_main_train import load_mult_namespace

    cfg = os.path.abspath(config_path or _find_multi_head_config(model_dir))
    ckpt = os.path.abspath(checkpoint_path or _find_multi_head_checkpoint(model_dir))

    args = load_mult_namespace(cfg)
    args.gpu = gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.batch_size = batch_size

    state = _extract_model_state_dict(_load_torch_state(ckpt, args.device))
    model = build_mult_head_from_args(args).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[multi_head] config={cfg}")
    print(f"[multi_head] weights={ckpt}")
    return model, args, multi_head_inference, cfg, ckpt


def multi_head_oracle_scores_for_batch(
    type_model,
    multi_head_model,
    multi_head_inference,
    multi_args,
    wave: torch.Tensor,
    device: torch.device,
    amp_type: bool,
    amp_detect: bool,
):
    """Predict fine type, then use the multi-head oracle specialist for that type."""
    pred_np, probs = _type_predict(type_model, wave, device, amp_type)
    pred_t = torch.as_tensor(pred_np, dtype=torch.long, device=multi_args.device)
    wave = wave.to(multi_args.device, non_blocking=True)
    with torch.no_grad():
        with autocast(enabled=amp_detect):
            scores, _ = multi_head_inference(multi_head_model, wave, audio_type=pred_t)
    sc_np = scores.detach().float().cpu().numpy()
    details = [
        {
            "pred_type": ATADD_AUDIO_TYPE_NAMES[int(pred_np[i])],
            "pred_type_id": int(pred_np[i]),
            "type_probs": {ATADD_AUDIO_TYPE_NAMES[j]: float(probs[i, j]) for j in range(4)},
        }
        for i in range(wave.size(0))
    ]
    return sc_np.astype(np.float32), pred_np, details


def load_label_meta_dev(label_path: str) -> dict:
    meta = {}
    valid_t = frozenset({"speech", "sound", "music", "singing"})
    valid_y = frozenset({"real", "fake"})
    with open(label_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()
            t = row["type"].strip().lower()
            y = row["label"].strip().lower()
            if t in valid_t and y in valid_y:
                meta[name] = (t, y)
    return meta


def analyze_dev_predictions(score_path: str) -> None:
    """Mirrors ``scripts/analyze.py`` (349–397) reporting on dev CSV."""
    import pandas as pd
    from sklearn.metrics import classification_report

    df_pred = pd.read_csv(score_path)
    if not {"predict", "label", "type"}.issubset(df_pred.columns):
        raise ValueError(f"{score_path} must have columns name,predict,type,label")

    y_true = df_pred["label"].values
    y_pred = df_pred["predict"].values

    print("\n===== Overall Performance =====")
    report = classification_report(y_true, y_pred, target_names=["real", "fake"])
    print(report)

    print("\n===== Performance by Audio Type =====")
    types = df_pred["type"].unique()
    type_macro_f1 = {}
    for t in types:
        subset = df_pred[df_pred["type"] == t]
        y_true_t = subset["label"]
        y_pred_t = subset["predict"]

        report_t = classification_report(
            y_true_t, y_pred_t, target_names=["real", "fake"], output_dict=True
        )

        f1_real = report_t["real"]["f1-score"]
        f1_fake = report_t["fake"]["f1-score"]
        macro_f1 = (f1_real + f1_fake) / 2
        type_macro_f1[t] = macro_f1

        print(f"\n--- Type: {t} ---")
        print(classification_report(y_true_t, y_pred_t, target_names=["real", "fake"]))
        print(f"Macro-F1 ({t}): {macro_f1:.4f}")

    output_order = TYPE_ORDER_PRINT
    type_macro_f1 = {t: type_macro_f1[t] for t in output_order if t in type_macro_f1}
    denom = len(type_macro_f1) if type_macro_f1 else 1
    track2_score = sum(type_macro_f1.values()) / denom
    print("\nMacro-F1\tSpeech\tSound\tSinging\tMusic")
    print(
        "{:.4f}\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
            track2_score,
            type_macro_f1.get("speech", float("nan")),
            type_macro_f1.get("sound", float("nan")),
            type_macro_f1.get("singing", float("nan")),
            type_macro_f1.get("music", float("nan")),
        )
    )


def run_dev(
    type_model,
    speech_model,
    sound_model,
    speech_args,
    sound_args,
    dev_audio: str,
    dev_label: str,
    out_dir: str,
    eval_task: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
    num_workers: int,
    amp_type: bool,
    amp_detect: bool,
    audio_len: int,
):
    label_meta = load_label_meta_dev(dev_label)
    print(f"[dev] label rows (valid): {len(label_meta)}")

    ds = atadd_eval_dataset(path_to_audio=dev_audio, audio_length=audio_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    os.makedirs(os.path.join(out_dir, "result"), exist_ok=True)
    score_path = os.path.join(out_dir, "result", f"{eval_task}_logits_dev_vote_type.csv")
    analysis_path = os.path.join(out_dir, "result", f"{eval_task}_analysis_dev_vote_type.csv")

    route_counts = {"speech_branch": 0, "sound_branch": 0}
    mismatches_vs_gt_route = []

    with open(analysis_path, "w", encoding="utf-8", newline="") as fav, open(
        score_path, "w", encoding="utf-8", newline=""
    ) as fsc:
        w_a = csv.writer(fav)
        w_s = csv.writer(fsc)
        w_a.writerow(["name", "predict", "type", "label", "pred_type", "score"])
        w_s.writerow(["name", "score"])

        for wave, filenames in tqdm(loader, desc="vote-type-dev"):
            sc_np, pred_np, _ = routed_scores_for_batch(
                type_model,
                speech_model,
                sound_model,
                speech_args,
                sound_args,
                wave,
                list(filenames),
                device,
                amp_type,
                amp_detect,
            )
            for i, fn in enumerate(filenames):
                name = fn.strip()
                meta = label_meta.get(name)
                if meta is None:
                    continue
                gt_type, gt_label = meta
                pr = ATADD_AUDIO_TYPE_NAMES[int(pred_np[i])]
                score = float(sc_np[i])
                predict = "real" if score >= threshold else "fake"
                speech_side = bool(_speech_mask_from_pred(pred_np)[i])

                route_counts["speech_branch" if speech_side else "sound_branch"] += 1
                # Compare routing implied by GT fine type vs pred type routing
                gt_speech_side = gt_type in ("speech", "singing")
                if gt_speech_side != speech_side:
                    mismatches_vs_gt_route.append(
                        {"name": name, "gt_type": gt_type, "pred_type": pr}
                    )

                w_a.writerow([name, predict, gt_type, gt_label, pr, f"{score:.8f}"])
                w_s.writerow([name, f"{score:.8f}"])

    print(f"[dev] analysis CSV       : {analysis_path}")
    print(f"[dev] logits CSV          : {score_path}")
    print(f"[dev] route counts(pred) : {route_counts}")
    print(f"[dev] route vs GT mismatches : {len(mismatches_vs_gt_route)} samples")

    analyze_dev_predictions(analysis_path)


def run_eval(
    type_model,
    speech_model,
    sound_model,
    speech_args,
    sound_args,
    eval_audio: str,
    out_dir: str,
    eval_task: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
    num_workers: int,
    amp_type: bool,
    amp_detect: bool,
    audio_len: int,
):
    ds = atadd_eval_dataset(path_to_audio=eval_audio, audio_length=audio_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    os.makedirs(os.path.join(out_dir, "result"), exist_ok=True)
    logit_path = os.path.join(out_dir, "result", f"{eval_task}_logits_eval_vote_type.csv")
    bin_path = os.path.join(out_dir, "result", f"{eval_task}_binary_eval_vote_type.csv")

    with open(logit_path, "w", encoding="utf-8", newline="") as fl, open(
        bin_path, "w", encoding="utf-8", newline=""
    ) as fb:
        wl = csv.writer(fl)
        wb = csv.writer(fb)
        wl.writerow(["name", "score"])
        wb.writerow(["name", "predict"])

        for wave, filenames in tqdm(loader, desc="vote-type-eval"):
            sc_np, pred_np, _ = routed_scores_for_batch(
                type_model,
                speech_model,
                sound_model,
                speech_args,
                sound_args,
                wave,
                list(filenames),
                device,
                amp_type,
                amp_detect,
            )
            for i, fn in enumerate(filenames):
                score = float(sc_np[i])
                pred_bin = "real" if score >= threshold else "fake"
                wl.writerow([fn.strip(), f"{score:.8f}"])
                wb.writerow([fn.strip(), pred_bin])

    meta_p = os.path.join(out_dir, "result", f"{eval_task}_vote_type_binary_meta.json")
    with open(meta_p, "w", encoding="utf-8") as mf:
        json.dump({"threshold": threshold, "logits": logit_path, "binary": bin_path}, mf, indent=2)

    print(f"[eval] logits : {logit_path}")
    print(f"[eval] binary: {bin_path}")


def run_dev_multi_head_oracle(
    type_model,
    multi_head_model,
    multi_head_inference,
    multi_args,
    dev_audio: str,
    dev_label: str,
    out_dir: str,
    eval_task: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
    num_workers: int,
    amp_type: bool,
    amp_detect: bool,
    audio_len: int,
):
    label_meta = load_label_meta_dev(dev_label)
    print(f"[dev] label rows (valid): {len(label_meta)}")

    ds = atadd_eval_dataset(path_to_audio=dev_audio, audio_length=audio_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    os.makedirs(os.path.join(out_dir, "result"), exist_ok=True)
    score_path = os.path.join(out_dir, "result", f"{eval_task}_logits_dev_vote_type.csv")
    analysis_path = os.path.join(out_dir, "result", f"{eval_task}_analysis_dev_vote_type.csv")

    route_counts = {name: 0 for name in ATADD_AUDIO_TYPE_NAMES}
    mismatches_vs_gt_route = []

    with open(analysis_path, "w", encoding="utf-8", newline="") as fav, open(
        score_path, "w", encoding="utf-8", newline=""
    ) as fsc:
        w_a = csv.writer(fav)
        w_s = csv.writer(fsc)
        w_a.writerow(["name", "predict", "type", "label", "pred_type", "score"])
        w_s.writerow(["name", "score"])

        for wave, filenames in tqdm(loader, desc="vote-type-mh-dev"):
            sc_np, pred_np, _ = multi_head_oracle_scores_for_batch(
                type_model,
                multi_head_model,
                multi_head_inference,
                multi_args,
                wave,
                device,
                amp_type,
                amp_detect,
            )
            for i, fn in enumerate(filenames):
                name = fn.strip()
                meta = label_meta.get(name)
                if meta is None:
                    continue
                gt_type, gt_label = meta
                pr = ATADD_AUDIO_TYPE_NAMES[int(pred_np[i])]
                score = float(sc_np[i])
                predict = "real" if score >= threshold else "fake"

                route_counts[pr] += 1
                if gt_type != pr:
                    mismatches_vs_gt_route.append(
                        {"name": name, "gt_type": gt_type, "pred_type": pr}
                    )

                w_a.writerow([name, predict, gt_type, gt_label, pr, f"{score:.8f}"])
                w_s.writerow([name, f"{score:.8f}"])

    print(f"[dev] analysis CSV       : {analysis_path}")
    print(f"[dev] logits CSV          : {score_path}")
    print(f"[dev] oracle head counts  : {route_counts}")
    print(f"[dev] pred type mismatches: {len(mismatches_vs_gt_route)} samples")

    analyze_dev_predictions(analysis_path)


def run_eval_multi_head_oracle(
    type_model,
    multi_head_model,
    multi_head_inference,
    multi_args,
    eval_audio: str,
    out_dir: str,
    eval_task: str,
    device: torch.device,
    batch_size: int,
    threshold: float,
    num_workers: int,
    amp_type: bool,
    amp_detect: bool,
    audio_len: int,
):
    ds = atadd_eval_dataset(path_to_audio=eval_audio, audio_length=audio_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    os.makedirs(os.path.join(out_dir, "result"), exist_ok=True)
    logit_path = os.path.join(out_dir, "result", f"{eval_task}_logits_eval_vote_type.csv")
    bin_path = os.path.join(out_dir, "result", f"{eval_task}_binary_eval_vote_type.csv")

    route_counts = {name: 0 for name in ATADD_AUDIO_TYPE_NAMES}
    with open(logit_path, "w", encoding="utf-8", newline="") as fl, open(
        bin_path, "w", encoding="utf-8", newline=""
    ) as fb:
        wl = csv.writer(fl)
        wb = csv.writer(fb)
        wl.writerow(["name", "score"])
        wb.writerow(["name", "predict"])

        for wave, filenames in tqdm(loader, desc="vote-type-mh-eval"):
            sc_np, pred_np, _ = multi_head_oracle_scores_for_batch(
                type_model,
                multi_head_model,
                multi_head_inference,
                multi_args,
                wave,
                device,
                amp_type,
                amp_detect,
            )
            for i, fn in enumerate(filenames):
                score = float(sc_np[i])
                pred_bin = "real" if score >= threshold else "fake"
                route_counts[ATADD_AUDIO_TYPE_NAMES[int(pred_np[i])]] += 1
                wl.writerow([fn.strip(), f"{score:.8f}"])
                wb.writerow([fn.strip(), pred_bin])

    meta_p = os.path.join(out_dir, "result", f"{eval_task}_vote_type_binary_meta.json")
    with open(meta_p, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "threshold": threshold,
                "strategy": "predicted_type_multi_head_oracle",
                "oracle_head_counts": route_counts,
                "logits": logit_path,
                "binary": bin_path,
            },
            mf,
            indent=2,
        )

    print(f"[eval] logits : {logit_path}")
    print(f"[eval] binary: {bin_path}")
    print(f"[eval] oracle head counts: {route_counts}")


def main():
    default_type = os.path.join(_ROOT, "ckpt_type_classification", "xlsr")
    default_multi_head = os.path.join(
        _ROOT, "ckpt_t2_multi_head_layer", "xslrbeats_beats3_6_9_xlsr3_11_24"
    )
    default_yaml = os.path.join(_ROOT, "conf", "vote", "ft_routed_ssl_aasist.yaml")
    default_out = os.path.join(
        _ROOT, "ckpt_t2_vote", "type_xlsr_multiHead_oracle_beats369_xlsr31124"
    )

    ap = argparse.ArgumentParser(description="Type-classifier routed multi-head oracle inference")
    ap.add_argument(
        "--type_ckpt_dir",
        type=str,
        default=default_type,
        help="Directory with type_train_cli.json + checkpoint/best.pt",
    )
    ap.add_argument(
        "--multi_head_model_path",
        type=str,
        default=default_multi_head,
        help="Multi-head experiment directory used for oracle-head inference.",
    )
    ap.add_argument(
        "--multi_head_config",
        type=str,
        default=None,
        help="Optional multi-head config path; defaults to <multi_head_model_path>/mult_config_used.yaml.",
    )
    ap.add_argument(
        "--multi_head_checkpoint",
        type=str,
        default=None,
        help="Optional multi-head checkpoint; defaults to checkpoint_all_dev/best.pt if present.",
    )
    ap.add_argument(
        "--vote_yaml",
        type=str,
        default=default_yaml,
        help="YAML with data.atadd_t2_dev_* and atadd_t2_eval_audio",
    )
    ap.add_argument("--dev_audio", type=str, default=None)
    ap.add_argument("--dev_label", type=str, default=None)
    ap.add_argument("--eval_audio", type=str, default=None)
    ap.add_argument(
        "--skip_dev",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip dev by default; pass --no-skip_dev to also run dev analysis.",
    )
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--out_dir", type=str, default=default_out)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--eval_task",
        type=str,
        default="atadd-track2",
        choices=["atadd-track1", "atadd-track2"],
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--amp_type",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="autocast for type classifier forward",
    )
    ap.add_argument(
        "--amp_detect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="autocast for spoof detector forwards",
    )
    ns = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ns.gpu
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print("Device:", device)

    yaml_paths = _load_vote_yaml_paths(ns.vote_yaml)
    dev_audio = ns.dev_audio or yaml_paths["atadd_t2_dev_audio"]
    dev_label = ns.dev_label or yaml_paths["atadd_t2_dev_label"]
    eval_audio = ns.eval_audio or yaml_paths["atadd_t2_eval_audio"]

    type_ckpt = os.path.abspath(ns.type_ckpt_dir)
    multi_head_mp = os.path.abspath(ns.multi_head_model_path)

    type_model = _load_type_model(type_ckpt, device)
    multi_head_model, multi_head_args, multi_head_inference, mh_cfg, mh_ckpt = _load_multi_head_model(
        multi_head_mp,
        ns.multi_head_config,
        ns.multi_head_checkpoint,
        ns.gpu,
        ns.batch_size,
    )

    cli = getattr(type_model, "_vote_type_cli", {})
    audio_len_cli = int(cli.get("audio_len", 64600))
    al_mh = int(getattr(multi_head_args, "audio_len", audio_len_cli))
    audio_len_use = audio_len_cli
    if audio_len_cli != al_mh:
        print(
            f"[warn] audio_len mismatch — type_ckpt:{audio_len_cli} "
            f"multi_head:{al_mh}; using type_ckpt {audio_len_cli}"
        )

    os.makedirs(ns.out_dir, exist_ok=True)
    meta = {
        "type_ckpt_dir": type_ckpt,
        "type_weights": getattr(type_model, "_vote_type_ckpt_path", ""),
        "strategy": "predicted_type_multi_head_oracle",
        "multi_head": {
            "path": multi_head_mp,
            "config": mh_cfg,
            "checkpoint": mh_ckpt,
            "ssl_backbone": getattr(multi_head_args, "ssl_backbone", ""),
        },
        "vote_yaml": os.path.abspath(ns.vote_yaml),
        "paths": {"dev_audio": dev_audio, "dev_label": dev_label, "eval_audio": eval_audio},
        "threshold": ns.threshold,
    }
    with open(os.path.join(ns.out_dir, "vote_type_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

    if not ns.skip_dev:
        run_dev_multi_head_oracle(
            type_model,
            multi_head_model,
            multi_head_inference,
            multi_head_args,
            dev_audio,
            dev_label,
            ns.out_dir,
            ns.eval_task,
            device,
            ns.batch_size,
            ns.threshold,
            ns.num_workers,
            ns.amp_type,
            ns.amp_detect,
            audio_len_use,
        )

    if not ns.skip_eval:
        run_eval_multi_head_oracle(
            type_model,
            multi_head_model,
            multi_head_inference,
            multi_head_args,
            eval_audio,
            ns.out_dir,
            ns.eval_task,
            device,
            ns.batch_size,
            ns.threshold,
            ns.num_workers,
            ns.amp_type,
            ns.amp_detect,
            audio_len_use,
        )

    print("[done] meta:", os.path.join(ns.out_dir, "vote_type_meta.json"))


if __name__ == "__main__":
    main()
