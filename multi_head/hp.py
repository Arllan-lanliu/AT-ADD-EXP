#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot train_loss.log: combined loss, specialist CE (spec), total-head CE (total).

Also writes one PNG per curve next to the combined image, e.g. train_loss_curve_loss.png
when the combined output is train_loss_curve.png.

  python multi_head/hp.py
  python multi_head/hp.py --log .../train_loss.log --out plot.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parent


def read_log(path: Path) -> Tuple[List[int], List[float], List[float], List[float]]:
    steps: List[int] = []
    loss: List[float] = []
    spec: List[float] = []
    total: List[float] = []
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if not first.lower().startswith("step"):
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            try:
                steps.append(int(parts[0]))
                loss.append(float(parts[3]))
                spec.append(float(parts[4]))
                total.append(float(parts[5]))
            except ValueError:
                continue
    return steps, loss, spec, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot train_loss.log curves.")
    ap.add_argument(
        "--log",
        type=Path,
        default="/data/liulan/workspace/released_models/AT-ADD-Baseline/ckpt_t2_multi_head/mult_head_xlsr_base/logs/train_loss.log",
        help="Path to train_loss.log (tab-separated).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image (default: same folder as log → train_loss_curve.png).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Plot every N-th point; 0 = auto (>=50k rows → stride 10).",
    )
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--height", type=float, default=5.0)
    args = ap.parse_args()

    log_path = args.log.resolve()
    if not log_path.is_file():
        raise SystemExit(f"File not found: {log_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Please install matplotlib: pip install matplotlib") from e

    steps, loss, spec, total_loss = read_log(log_path)
    if not steps:
        raise SystemExit(f"No data parsed from {log_path}")

    n = len(steps)
    stride = args.stride
    if stride <= 0:
        stride = 10 if n >= 50_000 else 5 if n >= 20_000 else 1
    if stride > 1:
        steps = steps[::stride]
        loss = loss[::stride]
        spec = spec[::stride]
        total_loss = total_loss[::stride]

    out_path = args.out
    if out_path is None:
        out_path = log_path.parent / "train_loss_curve.png"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    line_alpha = 0.7

    def save_single(path: Path, y: List[float], color: str, title: str) -> None:
        fig_s, ax_s = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
        ax_s.plot(steps, y, color=color, linewidth=0.85, alpha=line_alpha)
        ax_s.set_xlabel("step (global training step)", fontsize=10)
        ax_s.set_ylabel("value", fontsize=10)
        ax_s.set_title(
            f"{title} — {log_path.parent.parent.name}/{log_path.parent.name}\n"
            f"{log_path.name}  (n={n}, stride={stride})",
            fontsize=10,
        )
        ax_s.grid(True, linestyle="--", alpha=0.35)
        fig_s.tight_layout()
        fig_s.savefig(path, bbox_inches="tight")
        plt.close(fig_s)

    stem = out_path.stem
    single_dir = out_path.parent
    path_loss = single_dir / f"{stem}_loss.png"
    path_spec = single_dir / f"{stem}_spec.png"
    path_total = single_dir / f"{stem}_total.png"

    save_single(path_loss, loss, "#1f77b4", "loss (weighted sum)")
    save_single(path_spec, spec, "#ff7f0e", "spec_loss (specialist CE)")
    save_single(path_total, total_loss, "#2ca02c", "total_loss (total head CE)")

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    # Distinct colors: combined / specialist / total
    ax.plot(
        steps,
        loss,
        color="#1f77b4",
        label="loss (weighted sum)",
        linewidth=0.85,
        alpha=line_alpha,
    )
    ax.plot(
        steps,
        spec,
        color="#ff7f0e",
        label="spec_loss (specialist CE)",
        linewidth=0.85,
        alpha=line_alpha,
    )
    ax.plot(
        steps,
        total_loss,
        color="#2ca02c",
        label="total_loss (total head CE)",
        linewidth=0.85,
        alpha=line_alpha,
    )

    ax.set_xlabel("step (global training step)", fontsize=10)
    ax.set_ylabel("value", fontsize=10)
    ax.set_title(
        f"Training loss trends — {log_path.parent.parent.name}/{log_path.parent.name}\n"
        f"{log_path.name}  (n={n}, plotted stride={stride})",
        fontsize=10,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[hp] points={n}, stride={stride}, saved:\n"
        f"  combined → {out_path}\n"
        f"  loss     → {path_loss}\n"
        f"  spec     → {path_spec}\n"
        f"  total    → {path_total}"
    )


if __name__ == "__main__":
    main()
