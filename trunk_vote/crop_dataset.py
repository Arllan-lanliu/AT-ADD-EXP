from __future__ import annotations

import csv
import os
import random
from collections import Counter, defaultdict
from functools import partial
from typing import Iterable, List, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data.sampler as torch_sampler

from data.Augmentor import (
    AudioAugmentor,
    MusanSpeechAdditiveAugment,
    NoiseAugment,
    PitchShiftAugment,
    SpecAugmentForAudio,
    SpeechCodecRoundtripAugment,
)
from data.RawBoost import process_Rawboost_feature
from data.Sampler import _DEV_SUBSAMPLE_BUDGET, _stratified_sample
from data.dataset import VALID_AUDIO_TYPES
from utils.trainer import _dataloader_worker_seed_init


SR = 16000
TWO_SECONDS = 2 * SR
SIX_SECONDS = 6 * SR
TEN_POINT_FIVE_SECONDS = int(10.5 * SR)


def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=SR)
    waveform = torch.tensor(np.expand_dims(wave, axis=0), dtype=torch.float32)
    return waveform, sr


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    return (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)


def crop_starts(num_samples: int, crop_len: int) -> List[int]:
    """Return crop start offsets for the requested long-audio policy."""
    if num_samples <= crop_len:
        return [0]

    end = max(0, num_samples - crop_len)
    if num_samples <= SIX_SECONDS:
        return [0, end] if end > 0 else [0]

    if num_samples <= TEN_POINT_FIVE_SECONDS:
        starts = list(range(0, end + 1, TWO_SECONDS))
        if starts[-1] != end:
            starts.append(end)
        return starts

    return sorted({int(round(x)) for x in np.linspace(0, end, num=5)})


def crop_and_repeat_pad(wav, start: int, crop_len: int) -> torch.Tensor:
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(np.asarray(wav, dtype=np.float32), dtype=torch.float32)
    waveform = wav.squeeze(0).float()
    n = int(waveform.shape[0])
    if n <= 0:
        waveform = torch.zeros(crop_len, dtype=torch.float32)
    elif n >= crop_len:
        start = min(max(int(start), 0), n - crop_len)
        waveform = waveform[start:start + crop_len]
    else:
        num_repeats = int(crop_len / n) + 1
        waveform = torch.tile(waveform, (num_repeats,))[:crop_len]
    return normalize_waveform(waveform)


def _audio_num_samples(path: str) -> int:
    try:
        return int(librosa.get_duration(path=path) * SR)
    except Exception:
        wave, _ = librosa.load(path, sr=SR)
        return int(len(wave))


class MultiCropTrainDataset(Dataset):
    def __init__(
        self,
        path_to_audio,
        path_to_protocol,
        rawboost=False,
        musanrir=False,
        audio_length=64600,
        filter_types=None,
        aug_probs=None,
        music_aug_method="spec_augment",
        speech_aug_method="none",
        speech_rawboost_algo=5,
        musan_path="",
        rir_path="",
        dev_subsample=False,
        dev_subsample_seed=42,
    ):
        super().__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.audio_length = int(audio_length)
        self.label = {"fake": 1, "real": 0}
        self.class_type = {"speech": 0, "sound": 1, "singing": 2, "music": 3}
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor()
        self.dev_subsample = bool(dev_subsample)
        self.dev_subsample_seed = int(dev_subsample_seed)
        self.speech_aug_method = str(speech_aug_method).strip().lower()
        self.speech_rawboost_algo = int(speech_rawboost_algo)
        self._musan_aug = None
        self._speech_audio_augmentor = None
        self._speech_codec_aug = None
        self._speech_noise_aug = None

        if self.speech_aug_method == "musan" and musan_path and os.path.isdir(musan_path):
            self._musan_aug = MusanSpeechAdditiveAugment(musan_path, sr=SR)
            if not self._musan_aug.available():
                print(f"[MultiCropTrainDataset] MUSAN has no usable wavs: {musan_path!r}")
                self._musan_aug = None
        elif self.speech_aug_method == "musan" and musan_path:
            print(f"[MultiCropTrainDataset] musan_path is not a directory: {musan_path!r}")

        if self.speech_aug_method == "audio_augmentor":
            self._speech_audio_augmentor = AudioAugmentor(
                rir_path=(rir_path or "").strip() or "your_path/RIRS_NOISES",
                musan_path=(musan_path or "").strip() or "your_path/musan",
            )
            if not self._speech_audio_augmentor.usable_for_speech_aug():
                print("[MultiCropTrainDataset] audio_augmentor unavailable for speech.")
                self._speech_audio_augmentor = None

        if self.speech_aug_method == "codec":
            self._speech_codec_aug = SpeechCodecRoundtripAugment(sr=SR)
            if not self._speech_codec_aug.available():
                print("[MultiCropTrainDataset] ffmpeg codec augmentation unavailable.")
                self._speech_codec_aug = None

        if self.speech_aug_method == "noise":
            self._speech_noise_aug = NoiseAugment(snr_range=(10, 30))

        self.filter_types = None
        if filter_types is not None:
            self.filter_types = frozenset(t.lower() for t in filter_types)
            unknown = self.filter_types - VALID_AUDIO_TYPES
            if unknown:
                raise ValueError(f"Unknown filter_types {sorted(unknown)}")

        self.aug_probs = None
        self._pitch_shift_aug = None
        self._spec_aug = None
        if aug_probs is not None:
            active = {k: float(v) for k, v in aug_probs.items() if float(v) > 0.0}
            if active:
                self.aug_probs = active
                self.music_aug_method = music_aug_method
                if "music" in active:
                    if music_aug_method == "pitch_shift":
                        self._pitch_shift_aug = PitchShiftAugment(
                            sr=SR,
                            min_semitones=-3,
                            max_semitones=3,
                            exclude_zero=True,
                        )
                    elif music_aug_method == "spec_augment":
                        self._spec_aug = SpecAugmentForAudio(sr=SR)

        rows = []
        with open(self.path_to_protocol, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                filename = row["name"].strip()
                cls = row["type"].strip()
                label = row["label"].strip()
                generator = row["generator"].strip()
                if self.filter_types is not None and cls.lower() not in self.filter_types:
                    continue
                rows.append((filename, cls, label, generator))

        if self.dev_subsample:
            rows = self._apply_dev_subsample(rows, seed=self.dev_subsample_seed)

        self.audio_files = rows
        self.all_files = []
        for filename, cls, label, generator in rows:
            path = os.path.join(self.path_to_audio, filename)
            n = _audio_num_samples(path)
            starts = crop_starts(n, self.audio_length)
            for crop_idx, start in enumerate(starts):
                self.all_files.append((filename, cls, label, generator, crop_idx, start))

        self._print_stats()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, class_type, label, generator, crop_idx, start = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)

        if self.aug_probs is not None:
            aug_prob = self.aug_probs.get(class_type, 0.0)
            if aug_prob > 0.0 and random.random() < aug_prob:
                wav_np = waveform.squeeze(0).detach().cpu().numpy()
                if class_type == "speech":
                    waveform = self._augment_speech(wav_np, sr)
                elif class_type == "music":
                    waveform = self._augment_music(wav_np)
                else:
                    waveform = process_Rawboost_feature(wav_np, sr=sr, algo=5)

        waveform = crop_and_repeat_pad(waveform, start, self.audio_length)

        if self.musanrir:
            audio_length = waveform.size(0)
            waveform = self._apply_augmentation(waveform, audio_length)

        return (
            waveform,
            filename,
            self.label[label],
            self.class_type[class_type],
            generator,
        )

    def _augment_speech(self, wav_np, sr):
        sm = self.speech_aug_method
        if sm == "rawboost":
            return process_Rawboost_feature(
                np.asarray(wav_np, dtype=np.float32),
                sr=sr,
                algo=self.speech_rawboost_algo,
            )
        if sm == "musan" and self._musan_aug is not None:
            return torch.tensor(self._musan_aug.apply(wav_np), dtype=torch.float32)
        if sm == "audio_augmentor" and self._speech_audio_augmentor is not None:
            return torch.tensor(
                self._speech_audio_augmentor.apply_random_room_noise_musan(wav_np),
                dtype=torch.float32,
            )
        if sm == "codec" and self._speech_codec_aug is not None:
            return torch.tensor(self._speech_codec_aug.apply(wav_np), dtype=torch.float32)
        if sm == "noise" and self._speech_noise_aug is not None:
            return torch.tensor(
                np.asarray(self._speech_noise_aug.apply(wav_np), dtype=np.float32),
                dtype=torch.float32,
            )
        return torch.tensor(np.asarray(wav_np, dtype=np.float32), dtype=torch.float32)

    def _augment_music(self, wav_np):
        if self._pitch_shift_aug is not None:
            augmented = self._pitch_shift_aug.apply(wav_np)
        elif self._spec_aug is not None:
            augmented = self._spec_aug.apply(wav_np)
        else:
            augmented = wav_np
        if isinstance(augmented, torch.Tensor):
            return augmented
        return torch.tensor(augmented, dtype=torch.float32)

    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)
        if augtype == 0:
            return waveform
        if augtype == 1:
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            return torch.tensor(waveform).squeeze(dim=0)
        if augtype in [2, 3, 4]:
            noise_type = {2: "noise", 3: "speech", 4: "music"}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            return torch.tensor(waveform).squeeze(dim=0)
        return waveform

    @staticmethod
    def _apply_dev_subsample(all_files, seed=42):
        rng = random.Random(seed)
        by_type = defaultdict(list)
        for item in all_files:
            by_type[item[1]].append(item)

        sampled = []
        for type_name, items in sorted(by_type.items()):
            budget = _DEV_SUBSAMPLE_BUDGET.get(type_name, 2000)
            chosen = _stratified_sample(items, budget, rng)
            sampled.extend(chosen)
            print(
                f"[dev_subsample] {type_name}: {len(chosen)}/{len(items)} "
                f"samples kept (budget={budget})"
            )
        rng.shuffle(sampled)
        return sampled

    def _print_stats(self):
        role = "Dev  " if self.dev_subsample else "Train"
        sep = "=" * 68
        type_counts = Counter(item[1] for item in self.audio_files)
        crop_counts = Counter(item[1] for item in self.all_files)
        lines = [
            sep,
            f"[MultiCrop {role}] {self.path_to_protocol}",
            f"  Audio files   : {len(self.audio_files)}",
            f"  Crop samples  : {len(self.all_files)}",
            "  Type audio/crop counts:",
        ]
        for t in ["speech", "sound", "singing", "music"]:
            if type_counts.get(t, 0) or crop_counts.get(t, 0):
                lines.append(f"    {t:<10} {type_counts.get(t, 0):>6} / {crop_counts.get(t, 0):>6}")
        lines.append(sep)
        print("\n".join(lines))


class MultiCropEvalDataset(Dataset):
    def __init__(self, path_to_audio, audio_length=64600, exts=(".flac", ".wav")):
        super().__init__()
        self.path_to_audio = path_to_audio
        self.audio_length = int(audio_length)
        self.audio_files = sorted(
            f for f in os.listdir(self.path_to_audio) if f.lower().endswith(exts)
        )
        self.all_files = []
        for filename in self.audio_files:
            path = os.path.join(self.path_to_audio, filename)
            n = _audio_num_samples(path)
            for crop_idx, start in enumerate(crop_starts(n, self.audio_length)):
                self.all_files.append((filename, crop_idx, start))
        print(
            f"[MultiCrop Eval] {self.path_to_audio}: "
            f"{len(self.audio_files)} audio files -> {len(self.all_files)} crops"
        )

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, crop_idx, start = self.all_files[idx]
        waveform, _ = torchaudio_load(os.path.join(self.path_to_audio, filename))
        waveform = crop_and_repeat_pad(waveform, start, self.audio_length)
        return waveform, filename


def _train_kwargs(args):
    raw_probs = {
        "speech": args.aug_speech,
        "sound": args.aug_sound,
        "music": args.aug_music,
        "singing": args.aug_singing,
    }
    aug_probs = {k: v for k, v in raw_probs.items() if v > 0.0} or None
    return dict(
        audio_length=args.audio_len,
        filter_types=args.filter_types_parsed,
        aug_probs=aug_probs,
        music_aug_method=args.music_aug_method,
        speech_aug_method=getattr(args, "speech_aug_method", "none"),
        speech_rawboost_algo=int(getattr(args, "speech_rawboost_algo", 5)),
        musan_path=getattr(args, "musan_path", "") or "",
        rir_path=getattr(args, "rir_path", "") or "",
    )


def _loader(ds, args, seed_offset):
    generator = torch.Generator()
    generator.manual_seed(args.seed + seed_offset)
    worker_init = partial(_dataloader_worker_seed_init, base_seed=args.seed)
    return DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(ds)), generator=generator),
        pin_memory=args.cuda,
        worker_init_fn=worker_init,
    )


def build_multicrop_dataloaders(args):
    ft = args.filter_types_parsed
    if ft is not None:
        print(f"Filtering train/dev to audio types: {sorted(ft)}")

    if args.train_task == "atadd-track1":
        train_ds = MultiCropTrainDataset(
            args.atadd_t1_train_audio,
            args.atadd_t1_train_label,
            **_train_kwargs(args),
        )
        val_ds = MultiCropTrainDataset(
            args.atadd_t1_dev_audio,
            args.atadd_t1_dev_label,
            audio_length=args.audio_len,
            filter_types=ft,
            dev_subsample=True,
        )
    else:
        train_ds = MultiCropTrainDataset(
            args.atadd_t2_train_audio,
            args.atadd_t2_train_label,
            **_train_kwargs(args),
        )
        val_ds = MultiCropTrainDataset(
            args.atadd_t2_dev_audio,
            args.atadd_t2_dev_label,
            audio_length=args.audio_len,
            filter_types=ft,
            dev_subsample=True,
        )

    assert len(train_ds) > 0, "Train dataset is empty; check config paths."
    assert len(val_ds) > 0, "Val dataset is empty; check config paths."
    return _loader(train_ds, args, 0), _loader(val_ds, args, 1)


def build_multicrop_full_dev_loader(args):
    ft = args.filter_types_parsed
    if args.train_task == "atadd-track1":
        ds = MultiCropTrainDataset(
            args.atadd_t1_dev_audio,
            args.atadd_t1_dev_label,
            audio_length=args.audio_len,
            filter_types=ft,
            dev_subsample=False,
        )
    else:
        ds = MultiCropTrainDataset(
            args.atadd_t2_dev_audio,
            args.atadd_t2_dev_label,
            audio_length=args.audio_len,
            filter_types=ft,
            dev_subsample=False,
        )
    return _loader(ds, args, 2)
