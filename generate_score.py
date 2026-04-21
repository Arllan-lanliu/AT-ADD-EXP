import os
import json
import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from backbone.rawaasist import *
from eval_dataset import atadd_eval_dataset

torch.multiprocessing.set_start_method('spawn', force=True)


def init():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved model directory")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='atadd_model.pt',
        help='Weight file under model_path (default: atadd_model.pt). '
             'Examples: atadd_model_best_f1.pt, atadd_model_best_eer.pt',
    )
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU index")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for inference")
    parser.add_argument("--eval_audio", type=str, default=None,
                        help="Path to evaluation audio directory")
    parser.add_argument("--score_file", type=str, default=None,
                        help="Path to output score csv")
    parser.add_argument("--eval_task", type=str, default=None,
                        choices=["atadd-track1", "atadd-track2"],
                        help="Evaluation task, if not set will use train_task from args.json")
    
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification")

    temp_args, _ = parser.parse_known_args()

    json_path = os.path.join(temp_args.model_path, 'args.json')
    with open(json_path, 'r') as f:
        json_args = json.load(f)

    for key, value in json_args.items():
        if key not in vars(temp_args):
            if isinstance(value, bool):
                parser.add_argument(
                    f'--{key}',
                    action='store_true' if value else 'store_false',
                    default=value
                )
            else:
                parser.add_argument(
                    f'--{key}',
                    type=type(value),
                    default=value
                )

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = json_args.get('batch_size', 1)

    if args.eval_task is None:
        args.eval_task = json_args.get("train_task", "atadd-track1")

    if args.eval_audio is None:
        if args.eval_task == "atadd-track1":
            args.eval_audio = json_args.get("atadd_t1_eval_audio")
        elif args.eval_task == "atadd-track2":
            args.eval_audio = json_args.get("atadd_t2_eval_audio")

    if args.score_file is None:
        result_dir = os.path.join(args.model_path, 'result')
        os.makedirs(result_dir, exist_ok=True)
        args.score_file = os.path.join(result_dir, f'{args.eval_task}_logits_eval.csv')
        args.binary_score_file = os.path.join(result_dir, f'{args.eval_task}_binary_eval.csv')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    print("Using GPU:", args.gpu)
    print("Eval task:", args.eval_task)
    print("Eval audio:", args.eval_audio)
    print("Threshold:", args.threshold)
    print("Score file:", args.score_file)
    print("Binary score file:", args.binary_score_file)

    return args


def load_model(args):
    """Build model from registry and move to args.device."""
    return build_model(args).to(args.device)

def gen_score(model, args):
    test_set = atadd_eval_dataset(
        path_to_audio=args.eval_audio,
        audio_length=args.audio_len
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=args.cuda
    )

    with torch.no_grad():
        with open(args.score_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "score"])

            for data_slice in tqdm(test_loader):
                waveform, filename = data_slice[0], data_slice[1]
                waveform = waveform.to(args.device, non_blocking=True)

                feats, outputs = model(waveform)
                scores = F.softmax(outputs, dim=1)[:, 0].detach().cpu().numpy()

                for fn, score in zip(filename, scores):
                    audio_fn = fn.strip()
                    writer.writerow([audio_fn, float(score)])

def gen_binary_score(score_file, binary_score_file, threshold=0.5):
    with open(score_file, "r", encoding="utf-8-sig", newline="") as fin, \
        open(binary_score_file, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin)
        writer = csv.writer(fout)

        writer.writerow(["name", "predict"])

        for row in reader:
            name = row["name"].strip()
            score = float(row["score"])
            predict = "real" if score >= threshold else "fake"
            writer.writerow([name, predict])


if __name__ == "__main__":
    args = init()

    ckpt_path = os.path.join(args.model_path, args.checkpoint)
    print("Checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    print("Model:", args.model)
    feat_model = load_model(args)
    feat_model.load_state_dict(checkpoint)
    feat_model.eval()

    gen_score(feat_model, args)
    print(f"Done. Score file saved to: {args.score_file}")

    gen_binary_score(args.score_file, args.binary_score_file, args.threshold)
    print(f"Done. Binary score file saved to: {args.binary_score_file}")
