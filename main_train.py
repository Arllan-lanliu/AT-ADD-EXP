import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import json
import shutil
import numpy as np
from sklearn.metrics import f1_score

from model import *
from dataset import *
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler
from backbone.rawaasist import *
from collections import defaultdict
from tqdm import tqdm, trange
from exp.feature_extraction_exp import *
from utils import *
import eval_metrics as em
from feature_extraction import *
import config
import wandb
import time
from torch.cuda.amp import autocast, GradScaler

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

VALID_AUDIO_TYPES = ("speech", "sound", "music", "singing")


def parse_filter_types_arg(s):
    """Return None (use all types) or a frozenset of lowercase type names."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    unknown = sorted(set(parts) - set(VALID_AUDIO_TYPES))
    if unknown:
        raise ValueError(
            f"Invalid --filter_types entries: {unknown}. "
            f"Use comma-separated subset of: {', '.join(VALID_AUDIO_TYPES)}"
        )
    return frozenset(parts)


def initParams():
    parser = config.initParams()

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=4, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")

    parser.add_argument('--train_task', type=str, default="atadd-track1",
                        choices=["atadd-track1", "atadd-track2"])
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true',
                        help="continue training with trained model")

    parser.add_argument(
        '--save_best_by',
        type=str,
        default='loss',
        choices=['loss', 'eer', 'f1'],
        help='Metric used to save the best model: loss, eer, or f1'
    )

    # generalized strategy
    parser.add_argument('--SAM', type=bool, default=False, help="use SAM")
    parser.add_argument('--ASAM', type=bool, default=False, help="use ASAM")
    parser.add_argument('--CSAM', type=bool, default=False, help="use CSAM")

    #------------------------------------
    parser.add_argument('--log_dir', type=str, help="log folder", required=False, default="./models/try/logs")
    parser.add_argument('--train_class_rawboost', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=0,
                        help="Evaluate on dev set every N training steps within each epoch. "
                             "0 (default) = only evaluate at epoch end.")
    parser.add_argument('--eval_warmup_steps', type=int, default=2000,
                        help="Skip evaluation for the first N global training steps (default 2000). "
                             "Applies to both mid-epoch and epoch-end evals.")
    parser.add_argument('--patience', type=int, default=0,
                        help="Early-stopping patience: stop training if the monitored dev metric "
                             "does not improve for this many consecutive evaluations. "
                             "0 (default) = disabled.")

    # Per-type probabilistic augmentation (train set only; dev is never augmented)
    parser.add_argument('--aug_speech',  type=float, default=0.0,
                        help="Probability [0,1] of applying RawBoost(algo=5) to each speech sample")
    parser.add_argument('--aug_sound',   type=float, default=0.0,
                        help="Probability [0,1] of applying RawBoost(algo=5) to each sound sample")
    parser.add_argument('--aug_music',   type=float, default=0.0,
                        help="Probability [0,1] of applying music augmentation to each music sample")
    parser.add_argument('--aug_singing', type=float, default=0.0,
                        help="Probability [0,1] of applying RawBoost(algo=5) to each singing sample")
    parser.add_argument('--music_aug_method', type=str, default='pitch_shift',
                        choices=['pitch_shift', 'spec_augment'],
                        help="Augmentation method for music samples: "
                             "pitch_shift (PitchShiftAugment, ±1-3 semitones) or "
                             "spec_augment (SpecAugmentForAudio, frequency-band masking)")

    # Weights & Biases (set WANDB_API_KEY or run `wandb login`; WANDB_MODE=offline for no upload)
    parser.add_argument('--wandb_project', type=str, default='AT-ADD-Baseline',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (default: basename of --out_fold)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (team or username)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--amp', dest='amp', action='store_true',
                        help='Enable automatic mixed precision (AMP) when supported')
    parser.add_argument('--no_amp', dest='amp', action='store_false',
                        help='Disable automatic mixed precision (AMP)')
    parser.set_defaults(amp=True)

    parser.add_argument(
        '--filter_types',
        type=str,
        default=None,
        help='Comma-separated subset of audio types for train AND dev: speech,sound,music,singing. '
             'Omit or empty = use all types.',
    )

    args = parser.parse_args()
    args.filter_types_parsed = parse_filter_types_arg(args.filter_types)
    args.log_dir = os.path.join(args.out_fold, "logs")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training and os.path.exists(os.path.join(args.out_fold, 'checkpoint', 'latest.pt')):
        pass
    else:
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
            os.makedirs(args.log_dir)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)
            os.mkdir(args.log_dir)

        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            _args_dump = vars(args).copy()
            if _args_dump.get("filter_types_parsed") is not None:
                _args_dump["filter_types_parsed"] = sorted(_args_dump["filter_types_parsed"])
            json.dump(_args_dump, file, indent=4)

        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, 'train_loss.log'), 'w') as file:
            file.write("epoch\tstep\ttrain_loss\n")

        with open(os.path.join(args.log_dir, 'dev_loss.log'), 'w') as file:
            file.write("epoch\tval_loss\tval_eer\tval_f1\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def shuffle(feat, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    return feat, labels

def pre_model(args):
    feat_model = build_model(args).to(args.device)

    feat_optimizer = torch.optim.Adam(
        feat_model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005
    )

    if args.SAM or args.CSAM:
        base_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            base_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    start_epoch = 0
    loss = float("inf")
    eer = float("inf")
    f1 = -float("inf")
    if args.continue_training: # load from checkpoint
        ckpt_path = os.path.join(args.out_fold, 'checkpoint', 'latest.pt')
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            feat_model.load_state_dict(checkpoint["model_state_dict"])
            feat_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            loss = checkpoint["loss"]
            eer = checkpoint["eer"]
            f1 = checkpoint["f1"]
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("Checkpoint not found, training from scratch")
    
    if args.train_task == "atadd-track1":
        weight = torch.FloatTensor([4, 1]).to(args.device)

    if args.train_task == "atadd-track2":
        weight = torch.FloatTensor([3.5, 1]).to(args.device)

    print(f"Using class weight: {weight.tolist()}")
    print(f"Best model will be saved by: {args.save_best_by}")

    if args.base_loss == "ce": # default
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    return feat_model, feat_optimizer, criterion, [start_epoch, loss, eer, f1]

def pre_data(args):

    ft = args.filter_types_parsed
    if ft is not None:
        print(f"Filtering train/dev to audio types: {sorted(ft)}")

    # Build per-type augmentation probability dict for the train set.
    # Dev set always gets None (no augmentation).
    _aug_probs_raw = {
        "speech":  args.aug_speech,
        "sound":   args.aug_sound,
        "music":   args.aug_music,
        "singing": args.aug_singing,
    }
    train_aug_probs = {k: v for k, v in _aug_probs_raw.items() if v > 0.0} or None
    if train_aug_probs:
        print(f"Per-type train augmentation: {train_aug_probs}  "
              f"(music method: {args.music_aug_method})")

    if args.train_task == "atadd-track1":
        atadd_t1_trainset = atadd_dataset(
            args.atadd_t1_train_audio,
            args.atadd_t1_train_label,
            audio_length=args.audio_len,
            filter_types=ft,
            aug_probs=train_aug_probs,
            music_aug_method=args.music_aug_method,
        )
        atadd_t1_devset = atadd_dataset(
            args.atadd_t1_dev_audio,
            args.atadd_t1_dev_label,
            audio_length=args.audio_len,
            filter_types=ft,
            # aug_probs intentionally omitted: dev is never augmented
        )
        train_set = [atadd_t1_trainset]
        dev_set = [atadd_t1_devset]

    if args.train_task == "atadd-track2":
        atadd_t2_trainset = atadd_dataset(
            args.atadd_t2_train_audio,
            args.atadd_t2_train_label,
            audio_length=args.audio_len,
            class_rawboost=args.train_class_rawboost,
            filter_types=ft,
            aug_probs=train_aug_probs,
            music_aug_method=args.music_aug_method,
        )
        atadd_t2_devset = atadd_dataset(
            args.atadd_t2_dev_audio,
            args.atadd_t2_dev_label,  # protocol file
            audio_length=args.audio_len,
            filter_types=ft,
            # aug_probs intentionally omitted: dev is never augmented
        )
        train_set = [atadd_t2_trainset]
        dev_set = [atadd_t2_devset]

    for dataset in train_set:
        print(len(dataset), f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."
    for dataset in dev_set:
        print(len(dataset), f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."

    training_set = ConcatDataset(train_set)
    validation_set = ConcatDataset(dev_set)

    trainOriDataLoader = DataLoader(
        training_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))),
        pin_memory=args.cuda
    )

    valOriDataLoader = DataLoader(
        validation_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))),
        pin_memory=args.cuda
    )

    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)

    return trainOriDataLoader, valOriDataLoader, trainOri_flow, valOri_flow


def _args_for_wandb_config(args):
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


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    feat_model, feat_optimizer, criterion, infos = pre_model(args)
    start_epoch, prev_loss, prev_eer, prev_f1 = infos
    
    # data
    trainOriDataLoader, valOriDataLoader, trainOri_flow, valOri_flow = pre_data(args)

    monitor_loss = 'base_loss'
    multi_ssl_models = {
        'ft-xlsrwavlmaasist',
        'ft-xlsrbeatsaasist',
        'ft-xlsrmertaasist',
        'ft-xlsrclapaasist',
    }
    amp_enabled = bool(args.amp and args.cuda and args.model in multi_ssl_models)
    scaler = GradScaler(enabled=amp_enabled)
    print(f"AMP enabled: {amp_enabled} (model={args.model})")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.out_fold.rstrip('/')))
        wandb.init(
            mode="offline",
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=_args_for_wandb_config(args),
            dir=args.out_fold,
        )

    n_train_batches = len(trainOriDataLoader)
    eval_steps = args.eval_steps

    def do_eval(tag, global_step, is_epoch_end=False):
        """Run full dev evaluation, log metrics, and save best model.

        Returns True when early stopping should be triggered.
        """
        nonlocal prev_loss, prev_eer, prev_f1, no_improve_count
        t_eval = time.time()
        feat_model.eval()
        dev_loss_dict = defaultdict(list)
        val_flow = iter(valOriDataLoader)
        ip1_loader, idx_loader, score_loader, pred_loader, type_loader = [], [], [], [], []

        with torch.no_grad():
            for j in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    feat, filenames, labels, class_types, generators = next(val_flow)
                except StopIteration:
                    val_flow = iter(valOriDataLoader)
                    feat, filenames, labels, class_types, generators = next(val_flow)

                feat   = feat.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)

                with autocast(enabled=amp_enabled):
                    feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    with autocast(enabled=amp_enabled):
                        feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = torch.sigmoid(feat_outputs[:, 0])
                    pred  = torch.where(score >= 0.5,
                                        torch.zeros_like(labels),
                                        torch.ones_like(labels))
                else:
                    with autocast(enabled=amp_enabled):
                        feat_loss = criterion(feat_outputs, labels)
                    prob  = F.softmax(feat_outputs, dim=1)
                    score = prob[:, 0]
                    pred  = torch.where(score >= 0.5,
                                        torch.zeros_like(labels),
                                        torch.ones_like(labels))

                ip1_loader.append(feats)
                idx_loader.append(labels)
                pred_loader.append(pred)
                score_loader.append(score)
                type_loader.append(class_types)
                dev_loss_dict["base_loss"].append(feat_loss.item())

        valLoss   = np.nanmean(dev_loss_dict[monitor_loss])
        scores    = torch.cat(score_loader, 0).cpu().numpy()
        labels_np = torch.cat(idx_loader,   0).cpu().numpy()
        preds     = torch.cat(pred_loader,  0).cpu().numpy()
        types     = torch.cat(type_loader,  0).cpu().numpy()

        val_eer = em.compute_eer(scores[labels_np == 0], scores[labels_np == 1])[0]
        val_f1  = f1_score(labels_np, preds, average='macro')

        type_metrics = {}
        for t in np.unique(types):
            mask = (types == t)
            if np.sum(mask) == 0:
                continue
            t_scores = scores[mask]
            t_labels = labels_np[mask]
            t_preds  = preds[mask]
            t_eer = (np.nan if len(np.unique(t_labels)) < 2 else
                     em.compute_eer(t_scores[t_labels == 0], t_scores[t_labels == 1])[0])
            type_metrics[t] = {
                "loss": valLoss,
                "eer":  t_eer,
                "f1":   f1_score(t_labels, t_preds, average='macro'),
            }

        with open(os.path.join(args.log_dir, "dev_loss.log"), "a") as log:
            log.write(str(tag) + "\t" + str(valLoss) + "\t" + str(val_eer) + "\t" + str(val_f1))
            for t in type_metrics:
                log.write(f"\t{t}_EER:{type_metrics[t]['eer']:.4f}\t{t}_F1:{type_metrics[t]['f1']:.4f}")
            log.write("\n")

        if use_wandb:
            log_eval = {"eval/loss": valLoss, "eval/eer": val_eer, "eval/f1": val_f1}
            for t in type_metrics:
                log_eval[f"eval/{t}/eer"] = type_metrics[t]["eer"]
                log_eval[f"eval/{t}/f1"]  = type_metrics[t]["f1"]
            wandb.log(log_eval, step=global_step)

        print(f"\n[Eval @ {tag}] Loss: {valLoss:.4f}  EER: {val_eer:.4f}  F1: {val_f1:.4f}")
        print("=== Per-Type Metrics ===")
        for t in type_metrics:
            print(f"  [{t}] EER: {type_metrics[t]['eer']:.4f}  F1: {type_metrics[t]['f1']:.4f}")
        print(f"Evaluation time: {(time.time() - t_eval) / 60:.2f} min")

        # --- save best model ---
        save_flag = False
        if args.save_best_by == "loss" and valLoss < prev_loss:
            prev_loss = valLoss;  save_flag = True
        elif args.save_best_by == "eer" and val_eer < prev_eer:
            prev_eer = val_eer;   save_flag = True
        elif args.save_best_by == "f1" and val_f1 > prev_f1:
            prev_f1 = val_f1;     save_flag = True

        if save_flag:
            torch.save(feat_model.state_dict(),
                       os.path.join(args.out_fold, 'atadd_model.pt'))
            print(f"Best model updated by {args.save_best_by} at {tag}")
            no_improve_count = 0
        else:
            no_improve_count += 1

        # early-stopping check
        should_stop = args.patience > 0 and no_improve_count >= args.patience
        if should_stop:
            print(f"[Early Stop] No improvement for {no_improve_count} consecutive evals "
                  f"(patience={args.patience}). Stopping training.")

        if is_epoch_end:
            # per-epoch snapshot (only at true epoch boundaries to save disk)
            torch.save(feat_model.state_dict(),
                       os.path.join(args.out_fold, 'checkpoint',
                                    'atadd_model_%d.pt' % (tag + 1)))

        torch.save({
            "epoch":                tag,
            "model_state_dict":     feat_model.state_dict(),
            "optimizer_state_dict": feat_optimizer.state_dict(),
            "loss":                 valLoss,
            "eer":                  val_eer,
            "f1":                   val_f1,
        }, os.path.join(args.out_fold, 'checkpoint', 'latest.pt'))

        return should_stop

    no_improve_count = 0
    stop_training    = False

    for epoch_num in tqdm(range(start_epoch, args.num_epochs)):
        # Training
        t0 = time.time()
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)

        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        current_lr = feat_optimizer.param_groups[0]['lr']

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                #feat, audio_fn, labels = next(trainOri_flow)
                feat, filenames, labels, class_types, generators = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                #feat, audio_fn, labels = next(trainOri_flow)
                feat, filenames, labels, class_types, generators = next(trainOri_flow)

            feat = feat.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            class_types = class_types.to(args.device, non_blocking=True)

            def _add_type_loss(base_loss):
                """Add auxiliary type-clf loss from TypeAwareFusion side-channel."""
                type_logits = getattr(feat_model, '_last_type_logits', None)
                if type_logits is not None and args.type_loss_weight > 0:
                    return base_loss + args.type_loss_weight * F.cross_entropy(
                        type_logits, class_types
                    )
                return base_loss

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                with autocast(enabled=amp_enabled):
                    feats, feat_outputs = feat_model(feat)
                    feat_loss = _add_type_loss(criterion(feat_outputs, labels))
                scaler.scale(feat_loss.mean()).backward()
                if amp_enabled:
                    scaler.unscale_(feat_optimizer)
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                with autocast(enabled=amp_enabled):
                    feats, feat_outputs = feat_model(feat)
                    second_loss = _add_type_loss(criterion(feat_outputs, labels))
                scaler.scale(second_loss.mean()).backward()
                if amp_enabled:
                    scaler.unscale_(feat_optimizer)
                feat_optimizer.second_step(zero_grad=True)
                scaler.update()

            else:
                feat_optimizer.zero_grad()
                with autocast(enabled=amp_enabled):
                    feats, feat_outputs = feat_model(feat)
                    feat_loss = _add_type_loss(criterion(feat_outputs, labels))
                scaler.scale(feat_loss).backward()
                scaler.step(feat_optimizer)
                scaler.update()

            trainlossDict['base_loss'].append(feat_loss.item())

            with open(os.path.join(args.log_dir, "train_loss.log"), "a") as log:
                log.write(
                    str(epoch_num) + "\t" +
                    str(i) + "\t" +
                    str(trainlossDict[monitor_loss][-1]) + "\n"
                )
            if use_wandb:
                global_step = epoch_num * n_train_batches + i
                wandb.log(
                    {
                        "train/batch_loss": trainlossDict[monitor_loss][-1],
                        "train/epoch": epoch_num,
                        "train/lr": current_lr,
                    },
                    step=global_step,
                )

            # --- mid-epoch evaluation ---
            if eval_steps > 0 and (i + 1) % eval_steps == 0:
                gs = epoch_num * n_train_batches + i + 1
                if gs < args.eval_warmup_steps:
                    print(f"[Warmup] Skip eval at step {gs} (warmup={args.eval_warmup_steps})")
                else:
                    should_stop = do_eval(f"{epoch_num}.{i+1}", gs, is_epoch_end=False)
                    feat_model.train()
                    if should_stop:
                        stop_training = True
                        break

        t1 = time.time()
        print(f"Epoch {epoch_num} training time: {(t1 - t0)/60:.2f} minutes")

        if use_wandb and trainlossDict[monitor_loss]:
            wandb.log(
                {"train/epoch_mean_loss": float(np.mean(trainlossDict[monitor_loss]))},
                step=epoch_num,
            )

        if stop_training:
            break

        # --- end-of-epoch evaluation ---
        gs = (epoch_num + 1) * n_train_batches
        if gs < args.eval_warmup_steps:
            print(f"[Warmup] Skip epoch-end eval at step {gs} (warmup={args.eval_warmup_steps})")
        elif do_eval(epoch_num, gs, is_epoch_end=True):
            break


    if use_wandb:
        wandb.finish()
    return feat_model


if __name__ == "__main__":
    args = initParams()
    train(args)
