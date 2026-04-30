import heapq
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm, trange

from utils import config, metrics as em
from utils.optimizer import disable_running_stats, enable_running_stats
from utils.trainer import (
    adjust_learning_rate,
    args_for_wandb,
    build_dataloaders,
    build_full_dev_loader,
    build_model_and_optimizer,
)
from utils.helpers import parse_filter_types, setup_seed

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method("spawn", force=True)

# Dual-SSL model keys where AMP gives a meaningful speedup
_AMP_MODELS = {
    # "ft-xlsrwavlmaasist",
    # "ft-xlsrbeatsaasist",
    # "ft-xlsrmertaasist",
    # "ft-xlsrclapaasist",
}


# ---------------------------------------------------------------------------
# Per-type loss weighting
# ---------------------------------------------------------------------------
#
# This is an *optional* multiplicative weighting on top of the existing
# per-class (real/fake) CE weights produced by ``build_model_and_optimizer``.
#
# Final per-sample weight  =  class_weight[label]  *  type_weight[class_type]
#
# YAML / config field: ``ce_type_sample_weights`` (see ``utils/config.ATADDConfig``).
# When unset (None), the loss path is exactly the original
# ``criterion(outputs, labels)`` — bit-identical to baseline.
#
# When set to ``"0:1,1:1,2:1,3:1"``, the math reduces to the same value as the
# baseline (modulo fp16 round-off in the decomposed kernel), but the code path
# is the manual weighted-mean computation. Use this all-ones setting as a
# control to check baseline equivalence.
#
# Distinct from ``type_loss_weight``: that scalar weights the auxiliary
# type-classification head only.
# ---------------------------------------------------------------------------

def parse_ce_type_sample_weights(raw_value):
    """Parse "0:w0,1:w1,2:w2,3:w3" into dict[int, float], or None to disable.

    Accepts None / empty string / "none" as "feature disabled".
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return {int(k): float(v) for k, v in raw_value.items()}
    text = str(raw_value).strip()
    if text == "" or text.lower() == "none":
        return None
    parsed = {}
    for item in text.split(","):
        pair = item.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Invalid item '{pair}' in ce_type_sample_weights; expected 'type_id:weight'"
            )
        k_str, v_str = pair.split(":", 1)
        try:
            parsed[int(k_str.strip())] = float(v_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid item '{pair}' in ce_type_sample_weights; expected 'int:float'"
            ) from exc
    return parsed if parsed else None


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

def initParams():
    """Parse config, prepare output directories, set device and seed."""
    args = config.initParams()
    cfg  = getattr(args, "_config", None)   # ATADDConfig object for YAML saving

    args.filter_types_parsed = parse_filter_types(args.filter_types)
    args.log_dir = os.path.join(args.out_fold, "logs")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    ckpt_dir    = os.path.join(args.out_fold, "checkpoint")
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")

    if args.continue_training and os.path.exists(latest_ckpt):
        pass  # preserve existing output directory
    else:
        # Fresh run: (re)create the output tree
        if os.path.exists(args.out_fold):
            shutil.rmtree(args.out_fold)
        os.makedirs(args.log_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(args.out_fold, "checkpoint_sample_dev"))
        os.makedirs(os.path.join(args.out_fold, "checkpoint_all_dev"))

        # Save full config as YAML (used for --resume and reproducibility)
        if cfg is not None:
            cfg.save_to_yaml(os.path.join(args.out_fold, "config.yaml"))

        # Initialise log files
        with open(os.path.join(args.log_dir, "train_loss.log"), "w") as f:
            f.write("step\tepoch\tbatch\ttrain_loss\n")
        with open(os.path.join(args.log_dir, "dev_loss.log"), "w") as f:
            f.write("step\ttag\tval_loss\tval_eer\tval_f1\t[per-type EER/F1]\n")
        with open(os.path.join(args.log_dir, "all_dev_loss.log"), "w") as f:
            f.write("step\ttag\tval_loss\tval_eer\tval_f1\t[per-type EER/F1]\n")

    args.cuda   = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Device: {args.device}")
    return args


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    model, optimizer, criterion, resume = build_model_and_optimizer(args)
    train_loader, val_loader  = build_dataloaders(args)
    full_val_loader           = build_full_dev_loader(args)

    start_epoch      = resume["start_epoch"]
    best_sample_val  = resume["best_sample_val"]   # for early stopping + top-3 tracking
    best_full_val    = resume["best_full_val"]      # for checkpoint_all_dev/best.pt
    no_improve       = resume["no_improve"]

    # ── Per-type loss weighting setup ─────────────────────────────────────────
    # Read class_weights tensor that build_model_and_optimizer already baked
    # into ``criterion`` (e.g. [3.5, 1.0] for track2). We extract it so we can
    # apply class weighting manually when per-type weighting is also active.
    class_weight_tensor = getattr(criterion, "weight", None)
    type_weights_dict = parse_ce_type_sample_weights(
        getattr(args, "ce_type_sample_weights", None)
    )
    print(f"[loss] base_loss={args.base_loss}")
    print(f"[loss] class_weights (real, fake) = "
          f"{class_weight_tensor.tolist() if class_weight_tensor is not None else None}")
    print(f"[loss] per-type weights          = {type_weights_dict}")
    if type_weights_dict is None:
        print("[loss] using original criterion(out, labels) path "
              "(bit-identical to baseline).")
    else:
        print("[loss] using manual weighted-mean CE: "
              "w_n = class_weight[label_n] * type_weight[type_n].")

    def compute_main_loss(outputs, labels, class_types):
        """Main real/fake CE loss with optional per-class and per-type weighting.

        Behavior:
          • If ``type_weights_dict`` is None → returns ``criterion(out, labels)``
            unchanged (preserves PyTorch's fused kernel + AMP behavior).
          • If ``type_weights_dict`` is set → manual weighted mean where
            w_n = class_weight[label_n] * type_weight[type_n]. With type
            weights all = 1.0 this is mathematically identical to the
            original ``CrossEntropyLoss(weight=class_weight)``.
          • BCE path is left untouched (per-type weighting not implemented
            for BCE).
        """
        if args.base_loss == "bce":
            return criterion(outputs, labels.unsqueeze(1).float())

        if type_weights_dict is None:
            return criterion(outputs, labels)

        per_sample = F.cross_entropy(outputs, labels, reduction="none")

        if class_weight_tensor is not None:
            cw = class_weight_tensor.to(device=per_sample.device, dtype=per_sample.dtype)
            sample_class_w = cw[labels]
        else:
            sample_class_w = torch.ones_like(per_sample)

        sample_type_w = torch.ones_like(per_sample)
        for t, w in type_weights_dict.items():
            sample_type_w = torch.where(
                class_types == t,
                torch.full_like(sample_type_w, float(w)),
                sample_type_w,
            )

        final_w = sample_class_w * sample_type_w
        return (per_sample * final_w).sum() / final_w.sum().clamp(min=1e-8)

    # Helper: extract the tracked metric value from (loss, eer, f1)
    def _metric_val(val_loss, val_eer, val_f1):
        if args.save_best_by == "loss":
            return val_loss
        if args.save_best_by == "eer":
            return val_eer
        return val_f1   # "f1"

    # Helper: is new metric value better than the current best?
    def _is_better(new_val, cur_best):
        if args.save_best_by in ("loss", "eer"):
            return new_val < cur_best   # lower is better
        return new_val > cur_best       # f1: higher is better

    # Min/max heap direction: for loss/eer (lower=better) we want heap[0] to be
    # the worst (highest) → store as (-value) in a min-heap.
    # For f1 (higher=better) heap[0] should be the worst (lowest) → store as (value).
    def _heap_key(metric_val):
        return -metric_val if args.save_best_by in ("loss", "eer") else metric_val

    # ── Top-3 checkpoint tracker (sample dev, by save_best_by) ───────────────
    # Min-heap on _heap_key: heap[0] is always the worst of the top-3.
    top3_heap = []
    top3_json = os.path.join(args.out_fold, "checkpoint_sample_dev", "top3.json")
    if os.path.exists(top3_json):
        with open(top3_json) as f:
            for item in json.load(f):
                if os.path.exists(item["path"]):
                    heapq.heappush(
                        top3_heap,
                        (_heap_key(item["metric_val"]), item["step"], item["path"]),
                    )

    amp_enabled  = bool(args.amp and args.cuda and args.model in _AMP_MODELS)
    scaler       = GradScaler(enabled=amp_enabled)
    print(f"AMP: {amp_enabled}")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or os.path.basename(
            os.path.normpath(args.out_fold.rstrip("/"))
        )
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            name=run_name,
            config=args_for_wandb(args),
            dir=args.out_fold,
        )

    n_batches     = len(train_loader)
    stop_training = False

    # ── Shared inference helper ───────────────────────────────────────────────

    def _run_inference(loader):
        """Run the model over *loader* and return aggregated eval metrics.

        Uses the SAME ``compute_main_loss`` as training so dev loss is
        directly comparable to the training objective (and so save_best_by=loss
        tracks an objective-aligned signal).
        """
        model.eval()
        loss_list, score_list, label_list, type_list = [], [], [], []
        with torch.no_grad():
            for feat, _, labels, class_types, _ in tqdm(loader, leave=False, desc="eval"):
                feat        = feat.to(args.device, non_blocking=True)
                labels      = labels.to(args.device, non_blocking=True)
                class_types = class_types.to(args.device, non_blocking=True)
                with autocast(enabled=amp_enabled):
                    _, outputs = model(feat)
                if args.base_loss == "bce":
                    loss  = criterion(outputs, labels.unsqueeze(1).float())
                    score = torch.sigmoid(outputs[:, 0])
                else:
                    loss  = compute_main_loss(outputs, labels, class_types)
                    score = F.softmax(outputs, dim=1)[:, 0]
                loss_list.append(loss.item())
                score_list.append(score)
                label_list.append(labels)
                type_list.append(class_types)

        val_loss  = float(np.nanmean(loss_list))
        scores    = torch.cat(score_list).cpu().numpy()
        labels_np = torch.cat(label_list).cpu().numpy()
        types     = torch.cat(type_list).cpu().numpy()

        real_sc = scores[labels_np == 0]
        fake_sc = scores[labels_np == 1]
        val_eer, eer_thr = em.compute_eer(real_sc, fake_sc)

        if args.eval_threshold_mode == "eer":
            thr = eer_thr
        else:
            thr = float(args.score_threshold)

        # Higher score => real (label 0); fake (label 1) when score < threshold
        preds = (scores < thr).astype(np.int64)
        val_f1 = f1_score(labels_np, preds, average="macro")

        type_metrics = {}
        for t in np.unique(types):
            mask = types == t
            tl, ts = labels_np[mask], scores[mask]
            tp = (ts < thr).astype(np.int64)
            type_metrics[t] = {
                "eer": (np.nan if len(np.unique(tl)) < 2
                        else em.compute_eer(ts[tl == 0], ts[tl == 1])[0]),
                "f1":  f1_score(tl, tp, average="macro"),
            }
        return val_loss, val_eer, val_f1, type_metrics, float(thr)

    def _log_metrics(log_filename, tag, global_step, val_loss, val_eer, val_f1, type_metrics):
        with open(os.path.join(args.log_dir, log_filename), "a") as f:
            f.write(f"{global_step}\t{tag}\t{val_loss:.6f}\t{val_eer:.6f}\t{val_f1:.6f}")
            for t, m in type_metrics.items():
                f.write(f"\t{t}_EER:{m['eer']:.4f}\t{t}_F1:{m['f1']:.4f}")
            f.write("\n")

    def _update_top3_sample(val_loss, val_eer, val_f1, global_step):
        """Save to checkpoint_sample_dev/ if this eval belongs to the top-3 by save_best_by."""
        nonlocal top3_heap
        mv  = _metric_val(val_loss, val_eer, val_f1)
        hk  = _heap_key(mv)
        if len(top3_heap) < 3 or hk > top3_heap[0][0]:
            path = os.path.join(
                args.out_fold, "checkpoint_sample_dev", f"step_{global_step}.pt"
            )
            torch.save(model.state_dict(), path)
            heapq.heappush(top3_heap, (hk, global_step, path))
            if len(top3_heap) > 3:
                _, _, evicted = heapq.heappop(top3_heap)   # removes worst entry
                if os.path.exists(evicted):
                    os.remove(evicted)
            # Persist metadata — sort best-first
            reverse = args.save_best_by not in ("loss", "eer")
            top3_data = sorted(
                [{"metric_val": _heap_key(hk_), "step": s, "path": p,
                  "metric": args.save_best_by}
                 for hk_, s, p in top3_heap],
                key=lambda x: x["metric_val"],
                reverse=reverse,
            )
            with open(top3_json, "w") as f:
                json.dump(top3_data, f, indent=2)

    # ── Sample-dev evaluation (every eval_steps steps) ───────────────────────

    def do_sample_eval(epoch, global_step):
        """Evaluate on the subsampled dev set.

        - Logs to ``dev_loss.log``.
        - Maintains top-3 checkpoints by ``save_best_by`` in ``checkpoint_sample_dev/``.
        - Tracks ``save_best_by`` improvement for early stopping.
        - Returns True if early stopping should trigger.
        """
        nonlocal best_sample_val, no_improve

        tag = f"{epoch}.{global_step}"
        t0  = time.time()
        val_loss, val_eer, val_f1, type_metrics, decision_thr = _run_inference(val_loader)

        _log_metrics("dev_loss.log", tag, global_step, val_loss, val_eer, val_f1, type_metrics)

        if use_wandb:
            wb = {"sample_eval/loss": val_loss, "sample_eval/eer": val_eer,
                  "sample_eval/f1": val_f1, "sample_eval/decision_threshold": decision_thr}
            for t, m in type_metrics.items():
                wb[f"sample_eval/{t}/eer"] = m["eer"]
                wb[f"sample_eval/{t}/f1"]  = m["f1"]
            wandb.log(wb, step=global_step)

        print(f"\n[SampleEval @ {tag}]  loss={val_loss:.4f}  EER={val_eer:.4f}"
              f"  F1={val_f1:.4f}  thr={decision_thr:.4f} ({args.eval_threshold_mode})"
              f"  ({(time.time()-t0)/60:.1f} min)")
        for t, m in type_metrics.items():
            print(f"  [{t}]  EER={m['eer']:.4f}  F1={m['f1']:.4f}")

        # Top-3 by save_best_by in checkpoint_sample_dev/
        _update_top3_sample(val_loss, val_eer, val_f1, global_step)

        # Early stopping based on save_best_by improvement
        cur_val = _metric_val(val_loss, val_eer, val_f1)
        if _is_better(cur_val, best_sample_val):
            best_sample_val = cur_val
            no_improve = 0
            print(f"  → Sample-dev best {args.save_best_by} updated ({cur_val:.4f})")
        else:
            no_improve += 1

        # Always update latest.pt for clean resume
        torch.save(
            {
                "epoch":                epoch,
                "global_step":          global_step,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_sample_val":      best_sample_val,
                "best_full_val":        best_full_val,
                "no_improve":           no_improve,
            },
            os.path.join(args.out_fold, "checkpoint", "latest.pt"),
        )

        should_stop = args.patience > 0 and no_improve >= args.patience
        if should_stop:
            print(f"[Early Stop] {no_improve} evals without improvement "
                  f"(patience={args.patience}).")
        return should_stop

    # ── Full-dev evaluation (every full_eval_steps steps) ────────────────────

    def do_full_eval(epoch, global_step):
        """Evaluate on the complete dev set.

        - Logs to ``all_dev_loss.log``.
        - Saves ``checkpoint_all_dev/best.pt`` when ``save_best_by`` metric improves.
        - Does NOT affect early stopping.
        """
        nonlocal best_full_val

        tag = f"{epoch}.{global_step}"
        t0  = time.time()
        val_loss, val_eer, val_f1, type_metrics, decision_thr = _run_inference(full_val_loader)

        _log_metrics("all_dev_loss.log", tag, global_step,
                     val_loss, val_eer, val_f1, type_metrics)

        if use_wandb:
            wb = {"full_eval/loss": val_loss, "full_eval/eer": val_eer,
                  "full_eval/f1": val_f1, "full_eval/decision_threshold": decision_thr}
            for t, m in type_metrics.items():
                wb[f"full_eval/{t}/eer"] = m["eer"]
                wb[f"full_eval/{t}/f1"]  = m["f1"]
            wandb.log(wb, step=global_step)

        print(f"\n[FullEval  @ {tag}]  loss={val_loss:.4f}  EER={val_eer:.4f}"
              f"  F1={val_f1:.4f}  thr={decision_thr:.4f} ({args.eval_threshold_mode})"
              f"  ({(time.time()-t0)/60:.1f} min)")
        for t, m in type_metrics.items():
            print(f"  [{t}]  EER={m['eer']:.4f}  F1={m['f1']:.4f}")

        # Save best model by save_best_by in checkpoint_all_dev/
        cur_val = _metric_val(val_loss, val_eer, val_f1)
        if _is_better(cur_val, best_full_val):
            best_full_val = cur_val
            best_path = os.path.join(args.out_fold, "checkpoint_all_dev", "best.pt")
            torch.save(model.state_dict(), best_path)
            meta = {"f1": val_f1, "eer": val_eer, "loss": val_loss, "step": global_step,
                    "metric": args.save_best_by, "metric_val": cur_val}
            with open(os.path.join(args.out_fold, "checkpoint_all_dev", "best_meta.json"), "w") as mf:
                json.dump(meta, mf, indent=2)
            print(f"  → All-dev best model updated ({args.save_best_by}={cur_val:.4f})")

    # ── Epoch loop ────────────────────────────────────────────────────────────

    for epoch in tqdm(range(start_epoch, args.num_epochs), desc="epochs"):
        t0 = time.time()
        model.train()
        train_losses = []

        adjust_learning_rate(args, args.lr, optimizer, epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        for i, (feat, _, labels, class_types, _) in enumerate(
            tqdm(train_loader, leave=False, desc=f"epoch {epoch}")
        ):
            feat        = feat.to(args.device, non_blocking=True)
            labels      = labels.to(args.device, non_blocking=True)
            class_types = class_types.to(args.device, non_blocking=True)

            def _loss_with_type(base_loss):
                type_logits = getattr(model, "_last_type_logits", None)
                if type_logits is not None and args.type_loss_weight > 0:
                    return base_loss + args.type_loss_weight * F.cross_entropy(
                        type_logits, class_types
                    )
                return base_loss

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(model)
                with autocast(enabled=amp_enabled):
                    _, out = model(feat)
                    loss = _loss_with_type(compute_main_loss(out, labels, class_types))
                scaler.scale(loss.mean()).backward()
                if amp_enabled:
                    scaler.unscale_(optimizer)
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                with autocast(enabled=amp_enabled):
                    _, out2 = model(feat)
                    loss2 = _loss_with_type(compute_main_loss(out2, labels, class_types))
                scaler.scale(loss2.mean()).backward()
                if amp_enabled:
                    scaler.unscale_(optimizer)
                optimizer.second_step(zero_grad=True)
                scaler.update()
            else:
                optimizer.zero_grad()
                with autocast(enabled=amp_enabled):
                    _, out = model(feat)
                    loss = _loss_with_type(compute_main_loss(out, labels, class_types))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_losses.append(loss.item())
            global_step = epoch * n_batches + i
            gs = global_step + 1   # 1-indexed

            with open(os.path.join(args.log_dir, "train_loss.log"), "a") as f:
                f.write(f"{gs}\t{epoch}\t{i}\t{train_losses[-1]:.6f}\n")
            if use_wandb:
                wandb.log(
                    {"train/batch_loss": train_losses[-1],
                     "train/epoch": epoch, "train/lr": current_lr},
                    step=gs,
                )

            # Sample-dev eval every eval_steps steps
            if args.eval_steps > 0 and gs % args.eval_steps == 0:
                if gs < args.eval_warmup_steps:
                    print(f"[Warmup] skip sample eval at step {gs}")
                elif do_sample_eval(epoch, gs):
                    stop_training = True
                    break
                model.train()

            # Full-dev eval every full_eval_steps steps
            if args.full_eval_steps > 0 and gs % args.full_eval_steps == 0:
                if gs < args.eval_warmup_steps:
                    print(f"[Warmup] skip full eval at step {gs}")
                else:
                    do_full_eval(epoch, gs)
                model.train()

        print(f"Epoch {epoch}  loss={np.mean(train_losses):.4f}"
              f"  time={(time.time()-t0)/60:.1f} min")

        if use_wandb:
            wandb.log(
                {"train/epoch_mean_loss": float(np.mean(train_losses))},
                step=(epoch + 1) * n_batches,
            )

        if stop_training:
            break

        # Epoch-end sample eval
        gs_end = (epoch + 1) * n_batches
        if gs_end < args.eval_warmup_steps:
            print(f"[Warmup] skip epoch-end eval at step {gs_end}")
        elif do_sample_eval(epoch, gs_end):
            break

    if use_wandb:
        wandb.finish()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = initParams()
    train(args)