import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm, trange

from utils import config, metrics as em
from utils.optimizer import disable_running_stats, enable_running_stats
from utils.trainer import (
    adjust_learning_rate,
    args_for_wandb,
    build_model_and_optimizer,
)
from utils.helpers import parse_filter_types, setup_seed
from trunk_vote.crop_dataset import (
    build_multicrop_dataloaders as build_dataloaders,
    build_multicrop_full_dev_loader as build_full_dev_loader,
)

import wandb

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
        os.makedirs(os.path.join(args.out_fold, "checkpoint_all_dev"))
        os.makedirs(os.path.join(args.out_fold, "checkpoint_steps"))

        # Save full config as YAML (used for --resume and reproducibility)
        if cfg is not None:
            cfg.save_to_yaml(os.path.join(args.out_fold, "config.yaml"))

        # Initialise log files
        with open(os.path.join(args.log_dir, "train_loss.log"), "w") as f:
            f.write("step\tepoch\tbatch\ttrain_loss\n")
        with open(os.path.join(args.log_dir, "dev_loss.log"), "w") as f:
            f.write("step\ttag\tval_loss\tval_eer\tval_f1\t[per-type EER/F1]\t[per-generator F1]\n")
        with open(os.path.join(args.log_dir, "all_dev_loss.log"), "w") as f:
            f.write("step\ttag\tval_loss\tval_eer\tval_f1\t[per-type EER/F1]\t[per-generator F1]\n")

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
    best_sample_val  = resume["best_sample_val"]   # for early stopping
    best_full_val    = resume["best_full_val"]      # legacy: for checkpoint_all_dev/best.pt
    best_full_vals   = {
        "loss": float("inf"),
        "eer": float("inf"),
        "f1": -float("inf"),
    }
    best_full_vals.update(resume.get("best_full_vals") or {})
    if (
        args.save_best_by in ("loss", "eer")
        and best_full_vals[args.save_best_by] == float("inf")
        and best_full_val != float("inf")
    ) or (
        args.save_best_by == "f1"
        and best_full_vals["f1"] == -float("inf")
        and best_full_val != -float("inf")
    ):
        best_full_vals[args.save_best_by] = best_full_val
    no_improve       = resume["no_improve"]

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

    def _is_metric_better(metric_name, new_val, cur_best):
        if metric_name in ("loss", "eer"):
            return new_val < cur_best
        return new_val > cur_best

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
    last_completed_step = int(resume.get("global_step", 0) or 0)
    checkpoint_steps_dir = os.path.join(args.out_fold, "checkpoint_steps")
    os.makedirs(checkpoint_steps_dir, exist_ok=True)

    # ── Shared inference helper ───────────────────────────────────────────────

    def _safe_metric_key(value):
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())

    def _jsonable_metrics(metrics):
        out = {}
        for key, values in metrics.items():
            out[str(key)] = {}
            for k, v in values.items():
                if isinstance(v, (float, np.floating)):
                    out[str(key)][k] = None if np.isnan(v) else float(v)
                elif isinstance(v, (int, np.integer)):
                    out[str(key)][k] = int(v)
                else:
                    out[str(key)][k] = v
        return out

    def _run_inference(loader):
        """Run crop inference and aggregate mean logits per original audio."""
        model.eval()
        logits_by_name = {}
        labels_by_name = {}
        types_by_name = {}
        generators_by_name = {}
        with torch.no_grad():
            for feat, filenames, labels, class_types, generators in tqdm(loader, leave=False, desc="eval"):
                feat   = feat.to(args.device, non_blocking=True)
                with autocast(enabled=amp_enabled):
                    _, outputs = model(feat)
                outputs = outputs.detach().cpu()
                labels = labels.detach().cpu()
                class_types = class_types.detach().cpu()
                for i, name in enumerate(filenames):
                    name = str(name).strip()
                    logits_by_name.setdefault(name, []).append(outputs[i])
                    labels_by_name[name] = labels[i]
                    types_by_name[name] = class_types[i]
                    generators_by_name[name] = str(generators[i])

        names = sorted(logits_by_name)
        mean_logits = torch.stack([
            torch.stack(logits_by_name[name], dim=0).mean(dim=0) for name in names
        ])
        labels_tensor = torch.stack([labels_by_name[name] for name in names]).long()

        if args.base_loss == "bce":
            loss_logits = mean_logits.to(args.device)
            loss_labels = labels_tensor.to(args.device).unsqueeze(1).float()
            val_loss = float(criterion(loss_logits, loss_labels).detach().cpu().item())
            scores = torch.sigmoid(mean_logits[:, 0]).numpy()
        else:
            loss_logits = mean_logits.to(args.device)
            loss_labels = labels_tensor.to(args.device)
            val_loss = float(criterion(loss_logits, loss_labels).detach().cpu().item())
            scores = F.softmax(mean_logits, dim=1)[:, 0].numpy()

        labels_np = labels_tensor.numpy()
        types = torch.stack([types_by_name[name] for name in names]).numpy()
        generators = np.asarray([generators_by_name[name] for name in names], dtype=object)

        real_sc = scores[labels_np == 0]
        fake_sc = scores[labels_np == 1]
        val_eer, eer_thr = em.compute_eer(real_sc, fake_sc)

        if args.eval_threshold_mode == "eer":
            thr = eer_thr
        else:
            thr = float(args.score_threshold)

        # Higher score => real (label 0); fake (label 1) when score < threshold
        preds = (scores < thr).astype(np.int64)
        val_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

        type_metrics = {}
        for t in np.unique(types):
            mask = types == t
            tl, ts = labels_np[mask], scores[mask]
            tp = (ts < thr).astype(np.int64)
            type_metrics[t] = {
                "eer": (np.nan if len(np.unique(tl)) < 2
                        else em.compute_eer(ts[tl == 0], ts[tl == 1])[0]),
                "f1":  f1_score(tl, tp, average="macro", zero_division=0),
            }

        generator_metrics = {}
        for generator in sorted(np.unique(generators), key=lambda x: str(x)):
            mask = generators == generator
            gl, gs = labels_np[mask], scores[mask]
            gp = (gs < thr).astype(np.int64)
            generator_metrics[str(generator)] = {
                "eer": (np.nan if len(np.unique(gl)) < 2
                        else em.compute_eer(gs[gl == 0], gs[gl == 1])[0]),
                "f1": f1_score(gl, gp, average="macro", zero_division=0),
                "support": int(mask.sum()),
                "real": int((gl == 0).sum()),
                "fake": int((gl == 1).sum()),
            }

        return val_loss, val_eer, val_f1, type_metrics, generator_metrics, float(thr)

    def _log_metrics(
        log_filename,
        tag,
        global_step,
        val_loss,
        val_eer,
        val_f1,
        type_metrics,
        generator_metrics,
    ):
        with open(os.path.join(args.log_dir, log_filename), "a") as f:
            f.write(f"{global_step}\t{tag}\t{val_loss:.6f}\t{val_eer:.6f}\t{val_f1:.6f}")
            for t, m in type_metrics.items():
                f.write(f"\t{t}_EER:{m['eer']:.4f}\t{t}_F1:{m['f1']:.4f}")
            for generator, m in generator_metrics.items():
                gen_key = _safe_metric_key(generator)
                f.write(
                    f"\tGEN_{gen_key}_F1:{m['f1']:.4f}"
                    f"\tGEN_{gen_key}_N:{m['support']}"
                )
            f.write("\n")

    def _save_latest(epoch, global_step):
        torch.save(
            {
                "epoch":                epoch,
                "global_step":          global_step,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_sample_val":      best_sample_val,
                "best_full_val":        best_full_val,
                "best_full_vals":       best_full_vals,
                "no_improve":           no_improve,
            },
            os.path.join(args.out_fold, "checkpoint", "latest.pt"),
        )

    def _save_step_checkpoint(global_step):
        path = os.path.join(checkpoint_steps_dir, f"step_{global_step}.pt")
        torch.save(model.state_dict(), path)
        print(f"  → Saved step checkpoint: {path}")

    # ── Sample-dev evaluation (every eval_steps steps) ───────────────────────

    def do_sample_eval(epoch, global_step):
        """Evaluate on the subsampled dev set.

        - Logs to ``dev_loss.log``.
        - Tracks ``save_best_by`` improvement for early stopping.
        - Returns True if early stopping should trigger.
        """
        nonlocal best_sample_val, no_improve

        tag = f"{epoch}.{global_step}"
        t0  = time.time()
        val_loss, val_eer, val_f1, type_metrics, generator_metrics, decision_thr = _run_inference(
            val_loader
        )

        _log_metrics(
            "dev_loss.log",
            tag,
            global_step,
            val_loss,
            val_eer,
            val_f1,
            type_metrics,
            generator_metrics,
        )

        if use_wandb:
            wb = {"sample_eval/loss": val_loss, "sample_eval/eer": val_eer,
                  "sample_eval/f1": val_f1, "sample_eval/decision_threshold": decision_thr}
            for t, m in type_metrics.items():
                wb[f"sample_eval/{t}/eer"] = m["eer"]
                wb[f"sample_eval/{t}/f1"]  = m["f1"]
            for generator, m in generator_metrics.items():
                gen_key = _safe_metric_key(generator)
                wb[f"sample_eval/generator/{gen_key}/f1"] = m["f1"]
                wb[f"sample_eval/generator/{gen_key}/support"] = m["support"]
            wandb.log(wb, step=global_step)

        print(f"\n[SampleEval @ {tag}]  loss={val_loss:.4f}  EER={val_eer:.4f}"
              f"  F1={val_f1:.4f}  thr={decision_thr:.4f} ({args.eval_threshold_mode})"
              f"  ({(time.time()-t0)/60:.1f} min)")
        for t, m in type_metrics.items():
            print(f"  [{t}]  EER={m['eer']:.4f}  F1={m['f1']:.4f}")
        gen_f1 = "  ".join(
            f"{generator}_F1={m['f1']:.4f}" for generator, m in generator_metrics.items()
        )
        print(f"  [generator] {gen_f1}")

        # Early stopping based on save_best_by improvement
        cur_val = _metric_val(val_loss, val_eer, val_f1)
        if _is_better(cur_val, best_sample_val):
            best_sample_val = cur_val
            no_improve = 0
            print(f"  → Sample-dev best {args.save_best_by} updated ({cur_val:.4f})")
        else:
            no_improve += 1

        # Always update latest.pt for clean resume
        _save_latest(epoch, global_step)

        should_stop = args.patience > 0 and no_improve >= args.patience
        if should_stop:
            print(f"[Early Stop] {no_improve} evals without improvement "
                  f"(patience={args.patience}).")
        return should_stop

    # ── Full-dev evaluation (every full_eval_steps steps) ────────────────────

    def do_full_eval(epoch, global_step):
        """Evaluate on the complete dev set.

        - Logs to ``all_dev_loss.log``.
        - Saves independent best checkpoints for full-dev loss, EER, and F1.
        - Keeps ``checkpoint_all_dev/best.pt`` as a compatibility alias for
          the metric selected by ``save_best_by``.
        - Does NOT affect early stopping.
        """
        nonlocal best_full_val, best_full_vals

        tag = f"{epoch}.{global_step}"
        t0  = time.time()
        val_loss, val_eer, val_f1, type_metrics, generator_metrics, decision_thr = _run_inference(
            full_val_loader
        )

        _log_metrics("all_dev_loss.log", tag, global_step,
                     val_loss, val_eer, val_f1, type_metrics, generator_metrics)

        if use_wandb:
            wb = {"full_eval/loss": val_loss, "full_eval/eer": val_eer,
                  "full_eval/f1": val_f1, "full_eval/decision_threshold": decision_thr}
            for t, m in type_metrics.items():
                wb[f"full_eval/{t}/eer"] = m["eer"]
                wb[f"full_eval/{t}/f1"]  = m["f1"]
            for generator, m in generator_metrics.items():
                gen_key = _safe_metric_key(generator)
                wb[f"full_eval/generator/{gen_key}/f1"] = m["f1"]
                wb[f"full_eval/generator/{gen_key}/support"] = m["support"]
            wandb.log(wb, step=global_step)

        print(f"\n[FullEval  @ {tag}]  loss={val_loss:.4f}  EER={val_eer:.4f}"
              f"  F1={val_f1:.4f}  thr={decision_thr:.4f} ({args.eval_threshold_mode})"
              f"  ({(time.time()-t0)/60:.1f} min)")
        for t, m in type_metrics.items():
            print(f"  [{t}]  EER={m['eer']:.4f}  F1={m['f1']:.4f}")
        gen_f1 = "  ".join(
            f"{generator}_F1={m['f1']:.4f}" for generator, m in generator_metrics.items()
        )
        print(f"  [generator] {gen_f1}")

        metric_values = {"loss": val_loss, "eer": val_eer, "f1": val_f1}
        for metric_name, cur_val in metric_values.items():
            if not _is_metric_better(metric_name, cur_val, best_full_vals[metric_name]):
                continue

            best_full_vals[metric_name] = cur_val
            best_path = os.path.join(
                args.out_fold, "checkpoint_all_dev", f"best_{metric_name}.pt"
            )
            torch.save(model.state_dict(), best_path)
            meta = {
                "f1": val_f1,
                "eer": val_eer,
                "loss": val_loss,
                "step": global_step,
                "metric": metric_name,
                "metric_val": cur_val,
                "decision_threshold": decision_thr,
                "type_metrics": _jsonable_metrics(type_metrics),
                "generator_metrics": _jsonable_metrics(generator_metrics),
            }
            with open(
                os.path.join(
                    args.out_fold, "checkpoint_all_dev", f"best_{metric_name}_meta.json"
                ),
                "w",
            ) as mf:
                json.dump(meta, mf, indent=2)
            print(f"  → All-dev best_{metric_name} updated ({metric_name}={cur_val:.4f})")

            if metric_name == args.save_best_by:
                best_full_val = cur_val
                legacy_path = os.path.join(args.out_fold, "checkpoint_all_dev", "best.pt")
                legacy_meta = os.path.join(args.out_fold, "checkpoint_all_dev", "best_meta.json")
                torch.save(model.state_dict(), legacy_path)
                with open(legacy_meta, "w") as mf:
                    json.dump(meta, mf, indent=2)

        _save_latest(epoch, global_step)

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
                    loss = _loss_with_type(criterion(out, labels))
                scaler.scale(loss.mean()).backward()
                if amp_enabled:
                    scaler.unscale_(optimizer)
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                with autocast(enabled=amp_enabled):
                    _, out2 = model(feat)
                    loss2 = _loss_with_type(criterion(out2, labels))
                scaler.scale(loss2.mean()).backward()
                if amp_enabled:
                    scaler.unscale_(optimizer)
                optimizer.second_step(zero_grad=True)
                scaler.update()
            else:
                optimizer.zero_grad()
                with autocast(enabled=amp_enabled):
                    _, out = model(feat)
                    loss = _loss_with_type(criterion(out, labels))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_losses.append(loss.item())
            global_step = epoch * n_batches + i
            gs = global_step + 1   # 1-indexed
            last_completed_step = gs

            with open(os.path.join(args.log_dir, "train_loss.log"), "a") as f:
                f.write(f"{gs}\t{epoch}\t{i}\t{train_losses[-1]:.6f}\n")
            if use_wandb:
                wandb.log(
                    {"train/batch_loss": train_losses[-1],
                     "train/epoch": epoch, "train/lr": current_lr},
                    step=gs,
                )

            if gs % 10000 == 0:
                _save_step_checkpoint(gs)

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

    if last_completed_step > 0 and last_completed_step % 10000 != 0:
        _save_step_checkpoint(last_completed_step)

    if use_wandb:
        wandb.finish()
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = initParams()
    train(args)
