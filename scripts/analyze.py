import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as a top-level script from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import build_model
from data.dataset import atadd_eval_dataset


VALID_TYPES = ("speech", "sound", "music", "singing")
VALID_LABELS = ("real", "fake")




def _load_train_config_dict(model_path: str) -> dict:
    """Load flat training hyperparameters from ``{model_path}/config.yaml``."""
    config_yaml = os.path.join(model_path, "config.yaml")
    if not os.path.isfile(config_yaml):
        raise FileNotFoundError(f"config.yaml not found in {model_path!r}")

    from utils.config import ATADDConfig

    train_cfg = ATADDConfig.from_yaml(config_yaml)
    train_ns = train_cfg.to_namespace()
    print(f"[analyze] training config: {config_yaml}")
    d = vars(train_ns)
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _find_model_checkpoint(model_path: str) -> str:
    """Same resolution as ``scripts/inference.py``."""
    all_dev_best = os.path.join(model_path, "checkpoint_all_dev", "best.pt")
    if os.path.exists(all_dev_best):
        print("Using checkpoint_all_dev/best.pt  (best F1 on full dev set)")
        return all_dev_best

    top3_json = os.path.join(model_path, "checkpoint_sample_dev", "top3.json")
    if os.path.exists(top3_json):
        with open(top3_json, encoding="utf-8") as f:
            entries = json.load(f)
        if entries:
            best_entry = entries[0]
            ckpt = best_entry["path"]
            if os.path.exists(ckpt):
                metric = best_entry.get("metric", "metric")
                val = best_entry.get("metric_val", "?")
                print(
                    f"Using checkpoint_sample_dev/{os.path.basename(ckpt)}"
                    f"  (sample-dev best {metric}={val:.4f},"
                    f" step={best_entry['step']})"
                )
                return ckpt

    legacy = os.path.join(model_path, "atadd_model.pt")
    if os.path.exists(legacy):
        print("Using legacy atadd_model.pt")
        return legacy

    raise FileNotFoundError(
        f"No model checkpoint found in {model_path!r}.\n"
        f"Expected one of:\n"
        f"  {os.path.join(model_path, 'checkpoint_all_dev', 'best.pt')}\n"
        f"  {top3_json} (with valid paths)\n"
        f"  {legacy}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze dev attention and features for a trained model."
    )
    parser.add_argument("--model_path", type=str, default="./ckpt_t2/ft-w2v2assist_baseline")
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--eval_task", type=str, default="atadd-track2", choices=["atadd-track1", "atadd-track2"])
    parser.add_argument(
        "--eval_audio", type=str, default=None,
        help="Dev audio directory (default: from saved training config)",
    )
    parser.add_argument(
        "--label_path", type=str, default=None,
        help="Dev label CSV (default: from saved training config)",
    )
    parser.add_argument("--audio_len", type=int, default=None)
    parser.add_argument("--attn_frames", type=int, default=200, help="Unified side length for attention maps.")
    parser.add_argument("--score_suffix", type=str, default="_dev")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_seed", type=int, default=1234)
    parser.add_argument(
        "--metrics_only", action="store_true",
        help="Only run inference and print metrics; skip attention heatmaps and t-SNE."
    )
    args = parser.parse_args()

    train_args = _load_train_config_dict(args.model_path)

    for k, v in train_args.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if args.batch_size is None:
        args.batch_size = int(train_args.get("batch_size", 16))
    if args.eval_task is None:
        args.eval_task = train_args.get("train_task", "atadd-track2")
    if args.eval_audio is None:
        key = "atadd_t2_dev_audio" if args.eval_task == "atadd-track2" else "atadd_t1_dev_audio"
        args.eval_audio = train_args.get(key)
    if args.label_path is None:
        key = "atadd_t2_dev_label" if args.eval_task == "atadd-track2" else "atadd_t1_dev_label"
        args.label_path = train_args.get(key)
    if args.audio_len is None:
        args.audio_len = int(train_args.get("audio_len", 64600))
    if args.out_dir is None:
        args.out_dir = os.path.join(args.model_path, "analysis_dev_attention")

    if not args.eval_audio:
        raise ValueError(
            f"eval_audio is unset. Add dev audio path to {args.model_path}/config.yaml "
            "(data.atadd_t*_dev_audio) or pass --eval_audio."
        )
    if not args.label_path:
        raise ValueError(
            f"label_path is unset. Add dev label CSV to {args.model_path}/config.yaml "
            "(data.atadd_t*_dev_label) or pass --label_path."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def load_label_meta(label_path):
    meta = {}
    with open(label_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            t = row["type"].strip().lower()
            y = row["label"].strip().lower()
            if t in VALID_TYPES and y in VALID_LABELS:
                meta[name] = (t, y)
    return meta


def to_square(mat_np, side):
    t = torch.from_numpy(mat_np.astype(np.float32)).view(1, 1, mat_np.shape[0], mat_np.shape[1])
    out = F.adaptive_avg_pool2d(t, (side, side))
    return out.squeeze(0).squeeze(0).cpu().numpy()


def last_layer_mean_head(attentions):
    # attentions[-1]: (B, heads, T, T)
    return attentions[-1].mean(dim=1).cpu().numpy()  # (B, T, T)


def save_heatmap(mat, title, out_path, cmap="magma", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_tsne_embedding(features, seed, perplexity):
    if len(features) < 5:
        return None

    x = np.asarray(features, dtype=np.float32)
    # t-SNE perplexity must be less than n_samples.
    p = min(float(perplexity), max(2.0, len(x) - 1.0))

    emb = TSNE(
        n_components=2,
        perplexity=p,
        random_state=int(seed),
        init="pca",
        learning_rate="auto",
    ).fit_transform(x)
    return emb


def save_tsne(embedding, class_names, out_path, xlim, ylim):
    y = np.asarray(class_names)
    uniq = sorted(set(y.tolist()))
    cmap = plt.cm.get_cmap("tab10", len(uniq))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(uniq):
        mask = (y == cls)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.5,
            color=cmap(i),
            label=cls,
        )
    ax.set_title("t-SNE of last_hidden features (dev)")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(markerscale=2, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_tsne_two_class(embedding, labels, out_path, title, xlim, ylim):
    """Plot a subset with fixed global embedding/axes."""
    y = np.asarray(labels)
    uniq = sorted(set(y.tolist()))

    color_map = {"real": "#1f77b4", "fake": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in uniq:
        mask = (y == cls)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.5,
            color=color_map.get(cls, None),
            label=cls,
        )
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(markerscale=2, fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    attn_dir = os.path.join(args.out_dir, "attention_aggregate")
    os.makedirs(attn_dir, exist_ok=True)
    result_dir = os.path.join(args.out_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    label_meta = load_label_meta(args.label_path)
    print(f"[meta] loaded labels: {len(label_meta)}")

    feat_model = build_model(args).to(args.device)
    ckpt_path = _find_model_checkpoint(args.model_path)
    state = torch.load(ckpt_path, map_location=args.device)
    feat_model.load_state_dict(state)

    # Enable XLSR attentions if available.
    visual = False
    if args.model in ("ft-w2v2aasist", "ft-w2v2assist_baseline"):
        feat_model.visual = True
        if hasattr(feat_model, "wav2vec2") and hasattr(feat_model.wav2vec2, "visual"):
            feat_model.wav2vec2.visual = True
            visual = True
    feat_model.eval()

    dev_set = atadd_eval_dataset(path_to_audio=args.eval_audio, audio_length=args.audio_len)
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=args.cuda,
    )

    group_mats = defaultdict(list)  # key=(type, label), value=list[(S,S)]
    tsne_features = []
    tsne_classes = []
    tsne_types = []
    tsne_labels = []

    score_path = os.path.join(result_dir, f"{args.eval_task}_logits{args.score_suffix}.csv")
    with open(score_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["name", "predict", "type", "label"])

        with torch.no_grad():
            for wave, filenames in tqdm(dev_loader, desc="dev-analysis"):
                wave = wave.to(args.device, non_blocking=True)
                out = feat_model(wave)

                if visual:
                    feats, logits, attentions = out[0], out[1], out[2]
                    ll_mean = last_layer_mean_head(attentions)  # (B,T,T)
                else:
                    feats, logits = out[0], out[1]
                    ll_mean = None
                probs_real = F.softmax(logits, dim=1)[:, 0].detach().cpu().numpy()
                feat_np = feats.detach().cpu().numpy()

                for i, fn in enumerate(filenames):
                    name = fn.strip()
                    meta = label_meta.get(name)
                    if meta is None:
                        continue
                    t, y = meta
                    predict = "real" if probs_real[i] >= 0.5 else "fake"
                    writer.writerow([name, predict, t, y])

                    if not args.metrics_only:
                        if visual:
                            sq = to_square(ll_mean[i], args.attn_frames)
                        else:
                            sq = None
                        group_mats[(t, y)].append(sq)

                        tsne_features.append(feat_np[i].reshape(-1))
                        tsne_classes.append(f"{t}_{y}")
                        tsne_types.append(t)
                        tsne_labels.append(y)

    print(f"[score] saved: {score_path}")
    import pandas as pd
    from sklearn.metrics import classification_report, f1_score

    df_pred = pd.read_csv(score_path)

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

    output_order = ["speech", "sound", "music", "singing"]
    type_macro_f1 = {t: type_macro_f1[t] for t in output_order if t in type_macro_f1}
    track2_score = sum(type_macro_f1.values()) / len(type_macro_f1)
    print("\nMacro-F1\tSpeech\tSound\tSinging\tMusic")
    print(
        "{:.4f}\t\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
            track2_score,
            type_macro_f1.get("speech", float("nan")),
            type_macro_f1.get("sound",  float("nan")),
            type_macro_f1.get("singing",float("nan")),
            type_macro_f1.get("music",  float("nan")),
        )
    )

    if args.metrics_only:
        print(f"[done] metrics_only mode — skipped attention heatmaps and t-SNE.")
        print(f"[done] score file: {score_path}")
        return

    # Prepare global color scales so all avg_* share one scale,
    # and all sum_* share one scale for direct visual comparison.
    if visual:
        avg_mats = {}
        sum_mats = {}
        for t in VALID_TYPES:
            for y in VALID_LABELS:
                mats = group_mats.get((t, y), [])
                if not mats:
                    continue
                stack = np.stack(mats, axis=0)
                avg_mats[(t, y)] = stack.mean(axis=0)
                sum_mats[(t, y)] = stack.sum(axis=0)

        if avg_mats:
            avg_vmin = min(float(np.min(m)) for m in avg_mats.values())
            avg_vmax = max(float(np.max(m)) for m in avg_mats.values())
        else:
            avg_vmin, avg_vmax = None, None
        if sum_mats:
            sum_vmin = min(float(np.min(m)) for m in sum_mats.values())
            sum_vmax = max(float(np.max(m)) for m in sum_mats.values())
        else:
            sum_vmin, sum_vmax = None, None

        # Generate 16 images: avg_* and sum_* for 4 types x 2 labels
        for t in VALID_TYPES:
            for y in VALID_LABELS:
                mats = group_mats.get((t, y), [])
                if not mats:
                    print(f"[warn] no samples for {t}_{y}")
                    continue
                avg_mat = avg_mats[(t, y)]
                sum_mat = sum_mats[(t, y)]

                save_heatmap(
                    avg_mat,
                    f"avg attention | {t} | {y} | n={len(mats)}",
                    os.path.join(attn_dir, f"avg_{t}_{y}.png"),
                    cmap="magma",
                    vmin=avg_vmin,
                    vmax=avg_vmax,
                )
                save_heatmap(
                    sum_mat,
                    f"sum attention | {t} | {y} | n={len(mats)}",
                    os.path.join(attn_dir, f"sum_{t}_{y}.png"),
                    cmap="viridis",
                    vmin=sum_vmin,
                    vmax=sum_vmax,
                )

    emb = compute_tsne_embedding(
        tsne_features,
        seed=args.tsne_seed,
        perplexity=args.tsne_perplexity,
    )
    tsne_path = os.path.join(args.out_dir, "tsne_type_label.png")
    if emb is None:
        print("[tsne] Too few samples, skip all t-SNE plots.")
        print(f"[done] outputs in: {args.out_dir}")
        return

    pad_x = 0.05 * max(1e-6, float(emb[:, 0].max() - emb[:, 0].min()))
    pad_y = 0.05 * max(1e-6, float(emb[:, 1].max() - emb[:, 1].min()))
    xlim = (float(emb[:, 0].min() - pad_x), float(emb[:, 0].max() + pad_x))
    ylim = (float(emb[:, 1].min() - pad_y), float(emb[:, 1].max() + pad_y))

    save_tsne(emb, tsne_classes, tsne_path, xlim=xlim, ylim=ylim)
    print(f"[tsne] saved: {tsne_path}")

    # Per-type views using SAME global embedding and SAME axis limits.
    for t in VALID_TYPES:
        idx = [i for i, tt in enumerate(tsne_types) if tt == t]
        if len(idx) < 5:
            print(f"[tsne] skip {t}: too few samples")
            continue
        emb_t = emb[idx, :]
        lab_t = [tsne_labels[i] for i in idx]
        out_t = os.path.join(args.out_dir, f"tsne_{t}_real_fake.png")
        save_tsne_two_class(
            emb_t,
            lab_t,
            out_t,
            title=f"t-SNE of {t} (real vs fake)",
            xlim=xlim,
            ylim=ylim,
        )
        print(f"[tsne] saved: {out_t}")
    print(f"[done] outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
