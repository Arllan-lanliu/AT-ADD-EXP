#!/usr/bin/python3

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import librosa
from torch.utils.data.dataloader import default_collate
import glob
import random
import numpy
import soundfile
import csv
from scipy import signal
from RawBoost import process_Rawboost_feature, PitchShiftAugment, SpecAugmentForAudio


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


class AudioAugmentor:
    def __init__(self, rir_path='/data/liulan/workspace/dataset/RIRS_NOISES', musan_path='/data/liulan/workspace/dataset/musan'):
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = self._load_noiselist(musan_path)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))

    def _load_noiselist(self, musan_path):
        noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            category = file.split('/')[-3]
            if category not in noiselist:
                noiselist[category] = []
            noiselist[category].append(file)
        return noiselist

    def add_rev(self, audio, audio_length):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :audio_length]

    def add_noise(self, audio, noisecat, audio_length):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []

        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = audio_length
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)

        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


VALID_AUDIO_TYPES = frozenset({"speech", "sound", "music", "singing"})


class atadd_dataset(Dataset):
    def __init__(self, path_to_audio, path_to_protocol,
                 rawboost=False, musanrir=False, audio_length=64600, class_rawboost=False,
                 filter_types=None,
                 aug_probs=None,
                 music_aug_method="pitch_shift"):
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
        self.class_rawboost = class_rawboost
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

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, class_type, label, generator = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)

        # if self.rawboost:
        #     waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
        #     waveform = process_Rawboost_feature(waveform, sr=sr)

        if self.class_rawboost and (class_type == "sound" or class_type == "singing"):
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr=sr)

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

    def collate_fn(self, samples):
        return default_collate(samples)


if __name__ == "__main__":
    dataset = atadd_dataset(
        path_to_audio="/nas5_heyuan/xieyuankun/atadd/data/T2/train",
        path_to_protocol="/nas5_heyuan/xieyuankun/atadd/data/T2/label/train.csv"
    )

    print("dataset size:", len(dataset))

    real_count = sum(1 for _, label in dataset.all_files if label == "real")
    fake_count = sum(1 for _, label in dataset.all_files if label == "fake")

    print(f"real count: {real_count}")
    print(f"fake count: {fake_count}")

    if real_count > 0 and fake_count > 0:
        print(f"real:fake = {real_count}:{fake_count}")
        print(f"real/fake = {real_count / fake_count:.4f}")
        print(f"fake/real = {fake_count / real_count:.4f}")

        max_count = max(real_count, fake_count)
        weight_real = max_count / real_count   # label 0
        weight_fake = max_count / fake_count   # label 1

        print(f"class weight for real(label=0): {weight_real:.4f}")
        print(f"class weight for fake(label=1): {weight_fake:.4f}")

