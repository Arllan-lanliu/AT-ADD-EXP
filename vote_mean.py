#!/usr/bin/env python3
"""
多模型融合：在 eval 上分别跑每个预训练实验，再融合为一份提交用 CSV。

* ``--model_path`` 可重复，至少 2 个。每个目录需含 ``config.yaml`` 或 ``args.json``。
* 子模型权重选择规则同 ``scripts/inference._find_model_checkpoint``。

融合（``--mode``）：

* ``mean`` (默认): 子模型对 ``score`` 取**算术平均**；二分类 ``real`` 当且仅当
  融合分 ``>= --threshold``。
* ``majority``: 子模型各用 ``--threshold`` 投 ``real``/``fake``，多数决；
  **平票**时按融合分（各模型分数的算术平均）与 ``--threshold`` 比较。
  写出 logits 时仍写**平均融合分**。

例::

    python vote.py \\
        --model_path  /data/liulan/workspace/released_models/AT-ADD-Baseline/ckpt_t2_main/ft-xlsrbeatsaasist_all_sound0.8_singing0.2_f1 \\
        --model_path  /data/liulan/workspace/released_models/AT-ADD-Baseline/ckpt_t2_prev/ft-xlsrmertaasist \\
        --out_dir     /data/liulan/workspace/released_models/AT-ADD-Baseline/ckpt_t2_main/ensemble_vote_xlsrbeats_merta \\
        --gpu 0 --batch_size 160

也可使用项目内相对路径; ``--mode majority`` 为多数票（平票再用融合均分与 ``--threshold`` 决胜负）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F
from data.dataset import atadd_eval_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_inference():
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


def _collect_scores(
    model: torch.nn.Module, args, desc: str
) -> dict[str, float]:
    dataset = atadd_eval_dataset(
        path_to_audio=args.eval_audio,
        audio_length=args.audio_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=getattr(args, "cuda", False),
    )
    out: dict[str, float] = {}
    with torch.no_grad():
        for waveform, filenames in tqdm(loader, desc=desc):
            waveform = waveform.to(args.device, non_blocking=True)
            _, outputs = model(waveform)
            if getattr(args, "base_loss", "ce") == "bce":
                scores = torch.sigmoid(outputs[:, 0]).detach().cpu().numpy()
            else:
                scores = F.softmax(outputs, dim=1)[:, 0].detach().cpu().numpy()
            for fn, score in zip(filenames, scores):
                out[fn.strip()] = float(score)
    return out


def _init_args_for_model(
    inference,
    model_path: str,
    gpu: str,
    batch_size: int | None,
    eval_audio: str | None,
    argv0: str,
):
    model_path = os.path.abspath(model_path)
    v = [argv0, "--model_path", model_path, "--gpu", gpu, "--eval_threshold_mode", "fixed"]
    if batch_size is not None:
        v.extend(["--batch_size", str(batch_size)])
    if eval_audio is not None:
        v.extend(["--eval_audio", eval_audio])
    v.extend(
        [
            "--score_file",
            os.path.join(model_path, "result", "._vote_temp_logits.csv"),
            "--threshold",
            "0.5",
        ]
    )
    return inference._init_args(v)


def _fuse_mean(
    names: list[str], per: list[dict[str, float]]
) -> dict[str, float]:
    n = len(per)
    return {k: sum(d[k] for d in per) / n for k in names}


def _binary_from_mean(
    fused: dict[str, float], names: list[str], thr: float
) -> dict[str, str]:
    return {k: ("real" if fused[k] >= thr else "fake") for k in names}


def _binary_from_majority(
    per: list[dict[str, float]], names: list[str], thr: float
) -> dict[str, str]:
    out: dict[str, str] = {}
    m = len(per)
    for k in names:
        sc = [d[k] for d in per]
        votes_r = sum(1 for s in sc if s >= thr)
        votes_f = m - votes_r
        if votes_r > votes_f:
            out[k] = "real"
        elif votes_f > votes_r:
            out[k] = "fake"
        else:
            out[k] = "real" if (sum(sc) / m) >= thr else "fake"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="多模型平均 / 投票生成提交表")
    ap.add_argument(
        "--model_path",
        action="append",
        dest="model_paths",
        metavar="DIR",
        required=True,
        help="实验目录，可写多次 (至少 2 个)。",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="融合结果保存目录 (将创建并写入 *_logits_eval_vote.csv 等).",
    )
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument(
        "--eval_audio",
        type=str,
        default=None,
        help="不指定则从第一个子模型的 config 中读取 eval 音频根目录，后续子模型复用同一路径。",
    )
    ap.add_argument(
        "--eval_task",
        type=str,
        default=None,
        choices=["atadd-track1", "atadd-track2"],
        help="子文件名前缀；不指定则与第一个子模型 config 的 train_task 一致。",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["mean", "majority"],
        default="mean",
        help="mean: 用融合分与 threshold 出 binary; majority: 用投票出 binary, logits 仍写融合平均",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="对子模型/融合分作 real 判定的阈值 (默认 0.5).",
    )
    args = ap.parse_args()

    if len(args.model_paths) < 2:
        ap.error("请至少通过两次 --model_path 指定两个不同模型目录。")

    os.makedirs(args.out_dir, exist_ok=True)
    inference = _load_inference()

    per_scores: list[dict[str, float]] = []
    metas: list[dict] = []
    eval_dir: str | None = None
    ref_task: str | None = None

    for i, mp in enumerate(args.model_paths):
        mp = os.path.abspath(mp)
        if not os.path.isdir(mp):
            raise FileNotFoundError(f"不是目录: {mp}")
        if i == 0:
            ev = args.eval_audio
        else:
            if eval_dir is None:
                raise RuntimeError("内部错误: 未从第一个子模型得到 eval 目录。")
            ev = eval_dir
        a = _init_args_for_model(
            inference, mp, args.gpu, args.batch_size, ev, "vote.py"
        )
        if i == 0:
            eval_dir = a.eval_audio
            ref_task = a.eval_task
        if args.eval_task is not None:
            a.eval_task = args.eval_task
        ckpt = inference._find_model_checkpoint(mp)
        print(f"[{i+1}/{len(args.model_paths)}] {a.model!r}  <-  {os.path.basename(ckpt)}")
        print(f"         eval_audio: {a.eval_audio}")
        state = _load_torch_state(ckpt, a.device)
        model = inference.build_model_for_inference(a)
        model.load_state_dict(state, strict=True)
        model.eval()
        sdict = _collect_scores(model, a, desc=f"model {i+1}")
        per_scores.append(sdict)
        del model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metas.append(
            {
                "model_path": mp,
                "model": a.model,
                "checkpoint": ckpt,
            }
        )

    names = list(per_scores[0].keys())
    for j, d in enumerate(per_scores):
        missing = set(names) - set(d.keys())
        if missing:
            raise ValueError(
                f"与第一个模型可打分文件数不一致: 模型 {j+1} 缺少 {len(missing)} 条; "
                f"样例: {list(sorted(missing))[:3]}"
            )

    fused = _fuse_mean(names, per_scores)
    if args.mode == "mean":
        predict = _binary_from_mean(fused, names, args.threshold)
    else:
        predict = _binary_from_majority(per_scores, names, args.threshold)

    eval_task = args.eval_task or ref_task
    if not eval_task:
        raise ValueError("无法确定 eval_task; 请设置 --eval_task。")

    logit_p = os.path.join(args.out_dir, f"{eval_task}_logits_eval_vote.csv")
    bin_p = os.path.join(args.out_dir, f"{eval_task}_binary_eval_vote.csv")
    with open(logit_p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "score"])
        for k in names:
            w.writerow([k, fused[k]])
    with open(bin_p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "predict"])
        for k in names:
            w.writerow([k, predict[k]])

    meta = {
        "mode": args.mode,
        "threshold": args.threshold,
        "eval_task": eval_task,
        "eval_audio": eval_dir,
        "logits": logit_p,
        "binary": bin_p,
        "members": metas,
    }
    with open(os.path.join(args.out_dir, "vote_meta.json"), "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2, ensure_ascii=False)

    print("融合 logits  :", logit_p)
    print("融合 binary  :", bin_p)
    print("元数据       :", os.path.join(args.out_dir, "vote_meta.json"))


if __name__ == "__main__":
    main()
