import numpy as np
import torch
from torch.utils.data import Dataset
import os
import io
import tarfile
import librosa
from torch.utils.data.dataloader import default_collate
import random
import csv

from data.RawBoost import process_Rawboost_feature
from data.Augmentor import AudioAugmentor, PitchShiftAugment, SpecAugmentForAudio
from data.Sampler import _DEV_SUBSAMPLE_BUDGET, _stratified_sample

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath, sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]


def _is_tar_audio_source(path):
    path = os.fspath(path)
    lower = path.lower()
    return lower.endswith((".tar", ".tar.gz", ".tgz"))


def _normalize_tar_name(name):
    return name.replace("\\", "/").lstrip("./")


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


# ── Multi-crop strategy boundaries (samples at 16 kHz) ───────────────────────
_CROP_B2     = 96000   # 6 s   — boundary between 2-crop and sliding-window
_CROP_B3     = 168000  # 10.5 s — boundary between sliding-window and evenly-spaced
_CROP_STEP   = 32000   # 2 s   — sliding-window step
_CROP_N_EVN  = 5       # number of crops for very long audio (> 10.5 s)


def _audio_num_samples_from_file(filepath_or_buf, target_sr: int = 16000) -> int:
    """Return approximate sample count after resampling to *target_sr*.

    Uses soundfile header-only read when possible; falls back to full librosa
    decode for unsupported formats (e.g. mp3).
    """
    try:
        import soundfile as _sf
        info = _sf.info(filepath_or_buf)
        return int(np.ceil(info.frames * target_sr / info.samplerate))
    except Exception:
        wave, _ = librosa.load(filepath_or_buf, sr=target_sr)
        return len(wave)


def get_crop_starts(waveform_len: int, audio_length: int = 64600) -> list:
    """Return crop-start positions (in samples) for *waveform_len*.

    Crop strategy:
      ≤ audio_length (4.04 s)      → [0]              (single crop; repeat-pad)
      audio_length … 6 s           → [0, end]          (start + end)
      6 s … 10.5 s                 → sliding window every 2 s, force end crop
      > 10.5 s                     → 5 evenly-spaced crops
    """
    if waveform_len <= audio_length:
        return [0]

    end_start = waveform_len - audio_length

    if waveform_len <= _CROP_B2:                  # 4.04 s – 6 s
        return [0, end_start]

    if waveform_len <= _CROP_B3:                  # 6 s – 10.5 s  (sliding window)
        starts = list(range(0, end_start, _CROP_STEP))
        if not starts or starts[-1] != end_start:
            starts.append(end_start)
        return starts

    # > 10.5 s: evenly spaced
    return [round(i * end_start / (_CROP_N_EVN - 1)) for i in range(_CROP_N_EVN)]


def _crop_waveform(wav, start: int, audio_length: int = 64600) -> torch.Tensor:
    """Extract one crop at *start* and apply mean/var normalisation.

    Handles repeat-pad for short audio (len ≤ audio_length; start expected = 0).
    Accepts torch.Tensor (1-D or [1, T]) or numpy array.
    """
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav.astype(np.float32))
    waveform = wav.squeeze(0)
    n = waveform.shape[0]

    if n <= audio_length:
        num_repeats = int(audio_length / n) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :audio_length][0]
    else:
        waveform = waveform[start: start + audio_length]

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
                 dev_subsample_seed=42,
                 multi_crop=False):
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
            multi_crop: When True, each audio file is expanded into multiple
                        fixed-length crops according to get_crop_starts().
                        Labels are shared across all crops of the same file.
                        Requires a one-time pre-scan of audio durations at init.
        """
        super(atadd_dataset, self).__init__()

        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self._tar_audio = _is_tar_audio_source(self.path_to_audio)
        self._tar_file = None
        self._tar_pid = None
        self._tar_members = {}
        if self._tar_audio:
            self._tar_members = self._build_tar_member_index(self.path_to_audio)
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
        self.multi_crop = multi_crop
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

        # Build the flat item list used by __len__ / __getitem__.
        # multi_crop=True: pre-scan audio lengths once and expand each file into
        # multiple (file, crop_start) entries; labels are preserved per crop.
        # multi_crop=False: one entry per file, crop_start=None → pad_dataset().
        if self.multi_crop:
            self._items = self._expand_multi_crop_items()
        else:
            self._items = [(f, c, l, g, None) for f, c, l, g in self.all_files]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        filename, class_type, label, generator, crop_start = self._items[idx]
        waveform, sr = self._load_audio(filename)

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

                if class_type == "music":
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

        if self.musanrir:
            audio_length = waveform.size(0)
            waveform = self._apply_augmentation(waveform, audio_length)
            
        # crop_start=None → default single-crop / repeat-pad (multi_crop=False path)
        # crop_start=int  → extract the specific window from the multi-crop schedule
        if crop_start is None:
            waveform = pad_dataset(waveform, self.audio_length)
        else:
            waveform = _crop_waveform(waveform, crop_start, self.audio_length)

        label = self.label[label]
        class_type = self.class_type[class_type]
        return waveform, filename, label, class_type, generator

    def _load_audio(self, filename):
        if not self._tar_audio:
            filepath = os.path.join(self.path_to_audio, filename)
            return torchaudio_load(filepath)

        member_name = self._resolve_tar_member(filename)
        tar = self._get_tar_file()
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(
                f"Tar member is not a regular file: {member_name!r} in {self.path_to_audio!r}"
            )
        with extracted:
            return torchaudio_load(io.BytesIO(extracted.read()))

    def _get_tar_file(self):
        pid = os.getpid()
        if self._tar_file is None or self._tar_pid != pid:
            self._tar_file = tarfile.open(self.path_to_audio, "r:*")
            self._tar_pid = pid
        return self._tar_file

    def _resolve_tar_member(self, filename):
        key = _normalize_tar_name(filename)
        member_name = self._tar_members.get(key)
        if member_name is None:
            member_name = self._tar_members.get(os.path.basename(key))
        if member_name is None:
            raise FileNotFoundError(
                f"Audio file {filename!r} was not found in tar archive {self.path_to_audio!r}"
            )
        return member_name

    @staticmethod
    def _build_tar_member_index(path_to_audio):
        members = {}
        collisions = set()

        def add_key(key, member_name):
            if not key or key in collisions:
                return
            existing = members.get(key)
            if existing is None:
                members[key] = member_name
            elif existing != member_name:
                members.pop(key, None)
                collisions.add(key)

        with tarfile.open(path_to_audio, "r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                member_name = member.name
                norm = _normalize_tar_name(member_name)
                parts = norm.split("/")
                add_key(norm, member_name)
                add_key(parts[-1], member_name)
                for start in range(1, len(parts) - 1):
                    add_key("/".join(parts[start:]), member_name)

        return members

    def _get_audio_num_samples(self, filename: str) -> int:
        """Return audio length in samples at 16 kHz (header-only read when possible)."""
        if not self._tar_audio:
            filepath = os.path.join(self.path_to_audio, filename)
            return _audio_num_samples_from_file(filepath)
        member_name = self._resolve_tar_member(filename)
        tar = self._get_tar_file()
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(
                f"Tar member is not a regular file: {member_name!r}"
            )
        with extracted:
            data = extracted.read()
        return _audio_num_samples_from_file(io.BytesIO(data))

    def _expand_multi_crop_items(self):
        """Pre-scan audio lengths and return expanded (file, type, label, gen, crop_start) list."""
        items = []
        n_files = len(self.all_files)
        print(f"[multi_crop] pre-scanning {n_files} files for crop expansion …", flush=True)
        for k, (filename, class_type, label, generator) in enumerate(self.all_files):
            if k > 0 and k % 2000 == 0:
                print(f"[multi_crop]  … {k}/{n_files}", flush=True)
            try:
                n_samples = self._get_audio_num_samples(filename)
            except Exception as exc:
                print(f"[multi_crop] warn: {filename}: {exc} → single crop")
                n_samples = 0
            for start in get_crop_starts(n_samples, self.audio_length):
                items.append((filename, class_type, label, generator, start))
        print(f"[multi_crop] {n_files} files → {len(items)} crops "
              f"(×{len(items)/max(n_files,1):.2f} avg)", flush=True)
        return items

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

        if self.multi_crop:
            lines.append(f"  Multi-crop    : enabled (pre-scan pending …)")
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
    def __init__(
        self,
        path_to_audio,
        audio_length=64600,
        exts=(".flac", ".wav"),
        multi_crop=False,
    ):
        super(atadd_eval_dataset, self).__init__()

        self.path_to_audio = path_to_audio
        self.audio_length = audio_length
        self.multi_crop = multi_crop

        self.all_files = sorted([
            f for f in os.listdir(self.path_to_audio)
            if f.lower().endswith(exts)
        ])
        if self.multi_crop:
            self._items = self._expand_multi_crop_items()
        else:
            self._items = [(f, None) for f in self.all_files]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        filename, crop_start = self._items[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)
        if crop_start is None:
            waveform = pad_dataset(waveform, self.audio_length)
        else:
            waveform = _crop_waveform(waveform, crop_start, self.audio_length)

        return waveform, filename

    def _expand_multi_crop_items(self):
        items = []
        for filename in self.all_files:
            filepath = os.path.join(self.path_to_audio, filename)
            try:
                n_samples = _audio_num_samples_from_file(filepath)
            except Exception as exc:
                print(f"[eval_multi_crop] warn: {filename}: {exc} -> single crop")
                n_samples = 0
            for start in get_crop_starts(n_samples, self.audio_length):
                items.append((filename, start))
        print(f"[eval_multi_crop] {len(self.all_files)} files -> {len(items)} crops "
              f"(x{len(items)/max(len(self.all_files), 1):.2f} avg)", flush=True)
        return items

    def collate_fn(self, samples):
        return default_collate(samples)