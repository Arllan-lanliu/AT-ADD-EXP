import numpy as np
import torch
from torch.utils.data import Dataset
import os
import librosa
from torch.utils.data.dataloader import default_collate
import random
import csv

from data.RawBoost import process_Rawboost_feature
from data.Augmentor import AudioAugmentor, PitchShiftAugment, SpecAugmentForAudio, NoiseAugment
from data.Sampler import _DEV_SUBSAMPLE_BUDGET, _stratified_sample

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


def pad_dataset(wav, audio_length=64600):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = audio_length

    if waveform_len >= cut:
        waveform = waveform[:cut]
    else:
        num_repeats = int(cut / waveform_len) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    return waveform


VALID_AUDIO_TYPES = frozenset({"speech", "sound", "music", "singing"})


class atadd_dataset(Dataset):
    def __init__(self, path_to_audio, path_to_protocol,
                 rawboost=False, musanrir=False, audio_length=64600,
                 filter_types=None,
                 aug_probs=None,
                 music_aug_method="spec_augment",
                 dev_subsample=False,
                 dev_subsample_seed=42):
        """
        Args:
            aug_probs: dict mapping audio type → augmentation probability, e.g.
                       {"speech": 0.0, "sound": 1.0, "music": 0.7, "singing": 1.0}.
                       None (default) disables per-type augmentation entirely.
                       Pass only to *train* datasets; leave None for dev/eval.
            music_aug_method: augmentation method applied to music samples when
                              aug_probs["music"] > 0.
                              "pitch_shift"  → PitchShiftAugment (±1–3 semitones)
                              "spec_augment" → SpecAugmentForAudio (freq-band masking)
                       speech / sound / singing always use process_Rawboost_feature(algo=5).
            dev_subsample: When True, subsample the loaded rows per audio type using
                           stratified sampling over (label, generator) cells.
                           Budgets are defined in _DEV_SUBSAMPLE_BUDGET:
                             speech / sound / singing → 2000 samples each
                             music                   → 1000 samples
                           Applied after filter_types filtering.  Always pass
                           dev_subsample=True for dev datasets.
            dev_subsample_seed: Random seed for reproducible dev subsampling.
        """
        super(atadd_dataset, self).__init__()

        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"fake": 1, "real": 0}
        self.class_type = {
            "speech": 0,
            "sound": 1,
            "singing": 2,
            "music": 3,
        }
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor()
        self.dev_subsample = dev_subsample
        self.dev_subsample_seed = dev_subsample_seed
        self.filter_types = None
        if filter_types is not None:
            self.filter_types = frozenset(t.lower() for t in filter_types)
            unknown = self.filter_types - VALID_AUDIO_TYPES
            if unknown:
                raise ValueError(
                    f"Unknown filter_types {sorted(unknown)}; "
                    f"allowed: {sorted(VALID_AUDIO_TYPES)}"
                )

        # Per-type probabilistic augmentation --------------------------------
        # Normalise: keep only entries with probability > 0
        self.aug_probs = None
        self._pitch_shift_aug = None
        self._spec_aug = None
        self._noise_aug = None
        if aug_probs is not None:
            active = {k: float(v) for k, v in aug_probs.items() if float(v) > 0.0}
            if active:
                self.aug_probs = active
                self.music_aug_method = music_aug_method
                # Pre-instantiate the music augmentor if music is in active probs
                if "music" in active:
                    if music_aug_method == "pitch_shift":
                        self._pitch_shift_aug = PitchShiftAugment(
                            sr=16000,
                            min_semitones=-3,
                            max_semitones=3,
                            exclude_zero=True,
                        )
                    elif music_aug_method == "spec_augment":
                        self._spec_aug = SpecAugmentForAudio(sr=16000)
                if "speech" in active:
                    self._noise_aug = NoiseAugment()

        self.all_files = []
        with open(self.path_to_protocol, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["name"].strip()
                class_type = row["type"].strip()
                label = row["label"].strip()
                generator = row["generator"].strip()
                if self.filter_types is not None and class_type.lower() not in self.filter_types:
                    continue
                self.all_files.append((filename, class_type, label, generator))

        # Dev-set subsampling: stratified by (label, generator) per audio type.
        # Applied regardless of whether filter_types is set — when filter_types
        # is None all four types are active and each is subsampled independently.
        if self.dev_subsample:
            self.all_files = self._apply_dev_subsample(
                self.all_files, seed=self.dev_subsample_seed
            )

        self._print_stats()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, class_type, label, generator = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)

        # if self.rawboost:
        #     waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
        #     waveform = process_Rawboost_feature(waveform, sr=sr)

        # Per-type probabilistic augmentation (train only; dev passes aug_probs=None)
        if self.aug_probs is not None:
            aug_prob = self.aug_probs.get(class_type, 0.0)
            if aug_prob > 0.0 and random.random() < aug_prob:
                # Bring waveform to 1-D numpy for augmentors
                if isinstance(waveform, torch.Tensor):
                    wav_np = waveform.squeeze(0).detach().cpu().numpy()
                else:
                    wav_np = np.squeeze(waveform)

                if class_type == "speech":
                    augmented = self._noise_aug.apply(wav_np)
                    if isinstance(augmented, torch.Tensor):
                        waveform = augmented
                    else:
                        waveform = torch.tensor(augmented, dtype=torch.float32)
                elif class_type == "music":
                    if self._pitch_shift_aug is not None:
                        augmented = self._pitch_shift_aug.apply(wav_np)
                    elif self._spec_aug is not None:
                        augmented = self._spec_aug.apply(wav_np)
                    else:
                        augmented = wav_np  # fallback: no-op
                    # Both augmentors return numpy when given numpy input
                    if isinstance(augmented, torch.Tensor):
                        waveform = augmented
                    else:
                        waveform = torch.tensor(augmented, dtype=torch.float32)
                else:
                    # speech / sound / singing → RawBoost algo=5
                    waveform = process_Rawboost_feature(wav_np, sr=sr, algo=5)

        waveform = pad_dataset(waveform, self.audio_length)

        if self.musanrir:
            audio_length = waveform.size(0)
            waveform = self._apply_augmentation(waveform, audio_length)

        label = self.label[label]
        class_type = self.class_type[class_type]
        return waveform, filename, label, class_type, generator

    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)

        if augtype == 0:
            return waveform
        elif augtype == 1:
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        elif augtype in [2, 3, 4]:
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform

        return waveform

    def _print_stats(self):
        """Print dataset statistics after __init__ finishes."""
        from collections import Counter, defaultdict

        role = "Dev  " if self.dev_subsample else "Train"
        sep  = "=" * 68

        # ── per-type sample counts ────────────────────────────────────────
        type_counts = Counter(item[1] for item in self.all_files)
        total = len(self.all_files)

        lines = [sep,
                 f"[{role}] {self.path_to_protocol}",
                 f"  Total samples : {total}",
                 f"  Type counts   :"]
        TYPE_ORDER = ["speech", "sound", "singing", "music"]
        for t in TYPE_ORDER:
            cnt = type_counts.get(t, 0)
            if cnt:
                lines.append(f"    {t:<10} {cnt:>6}")
        for t in sorted(type_counts):
            if t not in TYPE_ORDER:
                lines.append(f"    {t:<10} {type_counts[t]:>6}")

        # ── train: augmentation settings ──────────────────────────────────
        if not self.dev_subsample:
            lines.append("  Augmentation  :")
            for t in TYPE_ORDER:
                if type_counts.get(t, 0) == 0:
                    continue
                if self.aug_probs and t in self.aug_probs:
                    p = self.aug_probs[t]
                    if t == "music":
                        method = getattr(self, "music_aug_method", "pitch_shift")
                        aug_desc = f"{method}  p={p}"
                    else:
                        aug_desc = f"RawBoost(algo=5)  p={p}"
                else:
                    aug_desc = "none"
                lines.append(f"    {t:<10}  {aug_desc}")

        # ── dev: label + generator distribution per type ──────────────────
        else:
            by_type = defaultdict(list)
            for item in self.all_files:
                by_type[item[1]].append(item)

            for t in TYPE_ORDER:
                items = by_type.get(t)
                if not items:
                    continue
                label_cnt = Counter(item[2] for item in items)   # real / fake
                gen_cnt   = Counter(item[3] for item in items)   # generator name

                lines.append(f"  [{t}]  {len(items)} samples")
                lines.append(f"    label     : real={label_cnt.get('real', 0)}"
                              f"  fake={label_cnt.get('fake', 0)}")
                # Sort generators: real '-' first, then by count desc
                gen_sorted = sorted(gen_cnt.items(),
                                    key=lambda kv: (kv[0] != '-', -kv[1]))
                gen_str = "  ".join(f"{g}:{n}" for g, n in gen_sorted)
                lines.append(f"    generator : {gen_str}")

        lines.append(sep)
        print("\n".join(lines))

    @staticmethod
    def _apply_dev_subsample(all_files, seed=42):
        """
        Subsample *all_files* per audio type using stratified sampling.

        Each type present in the list is sampled independently according to
        _DEV_SUBSAMPLE_BUDGET.  Types not listed in the budget fall back to
        2000 samples.  If a type has fewer rows than its budget, all its rows
        are kept without sampling.

        Returns a new shuffled list.
        """
        import random as _random
        from collections import defaultdict

        rng = _random.Random(seed)

        by_type = defaultdict(list)
        for item in all_files:
            by_type[item[1]].append(item)   # item[1] = class_type string

        sampled = []
        for type_name, items in sorted(by_type.items()):
            budget = _DEV_SUBSAMPLE_BUDGET.get(type_name, 2000)
            chosen = _stratified_sample(items, budget, rng)
            sampled.extend(chosen)
            kept = len(chosen)
            total = len(items)
            print(f"[dev_subsample] {type_name}: {kept}/{total} samples kept "
                  f"(budget={budget})")

        rng.shuffle(sampled)
        return sampled

    def collate_fn(self, samples):
        return default_collate(samples)


class atadd_eval_dataset(Dataset):
    def __init__(self, path_to_audio, audio_length=64600, exts=(".flac", ".wav")):
        super(atadd_eval_dataset, self).__init__()

        self.path_to_audio = path_to_audio
        self.audio_length = audio_length

        self.all_files = sorted([
            f for f in os.listdir(self.path_to_audio)
            if f.lower().endswith(exts)
        ])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)

        return waveform, filename

    def collate_fn(self, samples):
        return default_collate(samples)