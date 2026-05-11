import os
import glob
import random
import shutil
import subprocess
import tempfile

import numpy
import numpy as np
import torch
import librosa
import soundfile
from scipy import signal
from scipy.signal import butter, sosfilt


class SpecAugmentForAudio:
    """
    针对音频波形的SpecAugment
    在频域进行掩码，专为Deepfake检测优化
    """
    def __init__(self,
                 sr=16000,
                 num_freq_masks=2,        # 频率掩码数量
                 freq_mask_ratio=0.15,    # 最大掩码比例（相对总频率范围）
                 focus_high_freq=True,    # 是否偏向高频掩码
                 high_freq_threshold=4000, # 高频区域起始（Hz）
                 high_freq_prob=0.7):     # 掩码落在高频区域的概率
        self.sr = sr
        self.num_freq_masks = num_freq_masks
        self.freq_mask_ratio = freq_mask_ratio
        self.focus_high_freq = focus_high_freq
        self.high_freq_threshold = high_freq_threshold
        self.high_freq_prob = high_freq_prob
        self.nyquist = sr // 2

    def _butter_bandstop(self, lowcut, highcut, order=4):
        """
        设计巴特沃斯带阻滤波器
        用于遮盖特定频率范围
        """
        # 保证频率范围合法
        lowcut = max(lowcut, 20)          # 最低20Hz
        highcut = min(highcut, self.nyquist - 50)  # 最高不超过奈奎斯特

        if lowcut >= highcut:
            return None

        low = lowcut / self.nyquist
        high = highcut / self.nyquist

        # 边界保护
        low = np.clip(low, 1e-4, 0.999)
        high = np.clip(high, 1e-4, 0.999)

        if low >= high:
            return None

        try:
            sos = butter(order, [low, high], btype='bandstop', output='sos')
            return sos
        except Exception:
            return None

    def _get_freq_range(self):
        """
        根据focus_high_freq策略决定掩码的频率范围
        """
        max_mask_width = int(self.nyquist * self.freq_mask_ratio)

        if self.focus_high_freq and np.random.random() < self.high_freq_prob:
            # 高频区域掩码（4000Hz ~ Nyquist）
            freq_start_range = (self.high_freq_threshold, self.nyquist - max_mask_width)
        else:
            # 全频段随机掩码
            freq_start_range = (200, self.nyquist - max_mask_width)

        if freq_start_range[0] >= freq_start_range[1]:
            # 范围无效，回退到全频段
            freq_start_range = (200, max(300, self.nyquist - max_mask_width))

        f_start = np.random.randint(freq_start_range[0], freq_start_range[1])
        f_width = np.random.randint(
            max(50, max_mask_width // 4),
            max(100, max_mask_width)
        )
        f_end = min(f_start + f_width, self.nyquist - 50)

        return f_start, f_end

    def apply(self, waveform):
        """
        对波形应用频率掩码

        Args:
            waveform: numpy array [T] 或 torch.Tensor [T]
        Returns:
            同输入类型，同形状
        """
        is_tensor = isinstance(waveform, torch.Tensor)
        if is_tensor:
            wav_np = waveform.detach().cpu().numpy()
        else:
            wav_np = waveform.copy()

        wav_np = wav_np.astype(np.float32)

        # 应用多个频率掩码
        for _ in range(self.num_freq_masks):
            f_start, f_end = self._get_freq_range()
            sos = self._butter_bandstop(f_start, f_end)

            if sos is not None:
                try:
                    wav_np = sosfilt(sos, wav_np).astype(np.float32)
                except Exception:
                    pass  # 滤波失败则跳过此次掩码

        # 重新归一化（滤波可能改变幅度）
        std = np.std(wav_np)
        if std > 1e-7:
            wav_np = wav_np / (std + 1e-7) * np.std(
                waveform.numpy() if is_tensor else waveform
            )

        if is_tensor:
            return torch.tensor(wav_np, dtype=torch.float32)
        return wav_np


class PitchShiftAugment:
    """
    音调偏移增强
    专为音乐类Deepfake检测设计
    """
    def __init__(self,
                 sr=16000,
                 min_semitones=-3,   # 最多降低3个半音
                 max_semitones=3,    # 最多升高3个半音
                 exclude_zero=False, # 是否排除0（不偏移）
                 n_steps_choices=None):  # 指定可选半音数列表
        """
        Args:
            min_semitones: 最小偏移量（负数=降调）
            max_semitones: 最大偏移量（正数=升调）
            n_steps_choices: 如果指定，从列表中随机选择例：[-2, -1, 1, 2] 排除0
        """
        self.sr = sr
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.exclude_zero = exclude_zero

        if n_steps_choices is not None:
            self.n_steps_choices = n_steps_choices
        else:
            choices = list(range(min_semitones, max_semitones + 1))
            if exclude_zero and 0 in choices:
                choices.remove(0)
            self.n_steps_choices = choices

    def apply(self, waveform, n_steps=None):
        """
        对波形应用音调偏移

        Args:
            waveform: numpy [T] 或 torch.Tensor [T]
            n_steps: 指定偏移半音数，None则随机选择
        Returns:
            同输入类型，同形状
        """
        is_tensor = isinstance(waveform, torch.Tensor)
        if is_tensor:
            wav_np = waveform.detach().cpu().numpy().astype(np.float32)
        else:
            wav_np = waveform.astype(np.float32)

        original_len = len(wav_np)

        # 选择偏移量
        if n_steps is None:
            n_steps = np.random.choice(self.n_steps_choices)

        if n_steps == 0:
            return waveform  # 不偏移，直接返回

        try:
            # librosa的pitch_shift
            # bins_per_octave=12 → 以半音为单位
            shifted = librosa.effects.pitch_shift(
                wav_np,
                sr=self.sr,
                n_steps=n_steps,
                bins_per_octave=12
            )

            # 保证输出长度与输入一致
            if len(shifted) > original_len:
                shifted = shifted[:original_len]
            elif len(shifted) < original_len:
                shifted = np.pad(
                    shifted,
                    (0, original_len - len(shifted)),
                    mode='constant'
                )

            shifted = shifted.astype(np.float32) 
        except Exception as e:
            print(f"PitchShift failed (n_steps={n_steps}): {e}")
            shifted = wav_np  # 失败则返回原始

        if is_tensor:
            return torch.tensor(shifted, dtype=torch.float32)
        return shifted


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

    def apply_random_room_noise_musan(self, wav_1d):
        """Speech training augmentation: same random mix as legacy ``musanrir`` policy.

        Each call picks uniformly among: **no-op** | **RIR** | **MUSAN noise** |
        **MUSAN speech (babble)** | **MUSAN music**. Falls back to no-op if the
        chosen branch has no data (empty RIR list or empty noise category).
        """
        wav = np.asarray(wav_1d, dtype=np.float32).reshape(-1)
        L = int(wav.shape[0])
        if L <= 0:
            return wav
        audio = np.expand_dims(wav, axis=0)
        augtype = random.randint(0, 4)

        def _safe_noise(cat):
            lst = self.noiselist.get(cat) if isinstance(self.noiselist, dict) else None
            if not lst:
                return None
            try:
                out = self.add_noise(audio, cat, L)
                return np.asarray(out, dtype=np.float32).reshape(-1)
            except Exception:
                return None

        try:
            if augtype == 0:
                return wav
            if augtype == 1:
                if not self.rir_files:
                    return wav
                out = self.add_rev(audio, L)
                return np.asarray(out, dtype=np.float32).reshape(-1)
            if augtype == 2:
                o = _safe_noise("noise")
                return o if o is not None else wav
            if augtype == 3:
                o = _safe_noise("speech")
                return o if o is not None else wav
            if augtype == 4:
                o = _safe_noise("music")
                return o if o is not None else wav
        except Exception:
            pass
        return wav

    def usable_for_speech_aug(self) -> bool:
        if self.rir_files:
            return True
        if not isinstance(self.noiselist, dict):
            return False
        return any(len(v) > 0 for v in self.noiselist.values())


class NoiseAugment:
    """白噪声 + 环境噪声"""
    def __init__(self, snr_range=(10, 30)):
        self.snr_range = snr_range

    def apply(self, waveform):
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return wav
        snr_db = random.uniform(*self.snr_range)
        noise = np.random.randn(wav.shape[0]).astype(np.float32)

        signal_power = np.mean(wav ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_m = np.mean(noise ** 2)
        if noise_m <= 1e-12:
            return wav
        noise = noise * np.sqrt(noise_power / noise_m)

        return (wav + noise).astype(np.float32)


class MusanSpeechAdditiveAugment:
    """MUSAN additive mixing for speech (train only).

    Each apply: uniformly pick ``noise`` / ``music`` / ``speech`` (babble).
    SNR is uniform in **[0, 20] dB** for ``noise`` and ``music``, and **[10, 20]
    dB** for ``speech`` (weaker babble / less semantic collision vs clean speech).
    Expects MUSAN layout: ``<root>/<noise|music|speech>/.../*.wav``.
    """

    _CATEGORIES = ("noise", "music", "speech")

    def __init__(self, musan_path: str, sr: int = 16000):
        self.sr = int(sr)
        self.noiselist = self._load_noiselist(musan_path)
        self._warned_empty = False

    def _load_noiselist(self, musan_path: str) -> dict:
        buckets = {c: [] for c in self._CATEGORIES}
        if not musan_path or not os.path.isdir(musan_path):
            return buckets
        pattern = os.path.join(musan_path, "*", "*", "*.wav")
        for fp in glob.glob(pattern):
            norm = fp.replace("\\", "/")
            parts = norm.split("/")
            category = parts[-3].lower() if len(parts) >= 3 else ""
            if category in buckets:
                buckets[category].append(fp)
        return buckets

    def available(self) -> bool:
        return any(len(v) > 0 for v in self.noiselist.values())

    def apply(self, waveform) -> np.ndarray:
        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            wav_np = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if wav_np.size == 0:
            return wav_np

        usable = [c for c in self._CATEGORIES if self.noiselist[c]]
        if not usable:
            if not self._warned_empty:
                print("[MusanSpeechAdditiveAugment] No wav under MUSAN tree; skipping.")
                self._warned_empty = True
            return wav_np

        cat = random.choice(usable)
        snr_lo, snr_hi = (10.0, 20.0) if cat == "speech" else (0.0, 20.0)
        snr_db = random.uniform(snr_lo, snr_hi)

        noise_path = random.choice(self.noiselist[cat])
        noiseaudio, file_sr = soundfile.read(noise_path, dtype="float32", always_2d=False)
        noiseaudio = np.asarray(noiseaudio, dtype=np.float32).reshape(-1)
        if file_sr != self.sr and noiseaudio.size > 0:
            noiseaudio = librosa.resample(noiseaudio, orig_sr=file_sr, target_sr=self.sr)

        length = wav_np.shape[0]
        if noiseaudio.shape[0] <= length:
            shortage = length - noiseaudio.shape[0]
            noiseaudio = np.pad(noiseaudio, (0, shortage), mode="wrap")
        max_start = noiseaudio.shape[0] - length
        start = random.randint(0, max_start) if max_start > 0 else 0
        noise_seg = noiseaudio[start : start + length]

        clean_db = 10 * np.log10(np.mean(wav_np ** 2) + 1e-8)
        noise_db = 10 * np.log10(np.mean(noise_seg ** 2) + 1e-8)
        scale = np.sqrt(10 ** ((clean_db - noise_db - snr_db) / 10.0))
        mixed = wav_np + scale * noise_seg
        return mixed.astype(np.float32)


class SpeechCodecRoundtripAugment:
    """Speech codec encode→decode round-trip via **ffmpeg** (train only).

    Each ``apply`` uniformly chooses one of:

    - **MP3** (``libmp3lame``): bitrate ∈ {32, 64, 96, 128} kbps
    - **Opus** (``libopus``): bitrate ∈ {16, 24, 32, 48} kbps (``.ogg``)
    - **AAC** (``aac``): bitrate ∈ {32, 64, 96} kbps (``.m4a``)
    - **GSM-FR** (``libgsm``): telephone-grade path — force **8 kHz mono** for the
      lossy leg, then decode back to ``sr`` (default 16 kHz).

    Requires ``ffmpeg`` on ``PATH`` with the listed encoders.
    """

    _MP3_BR = (32, 64, 96, 128)
    _OPUS_BR = (16, 24, 32, 48)
    _AAC_BR = (32, 64, 96)

    def __init__(self, sr: int = 16000):
        self.sr = int(sr)
        self._ffmpeg_ok = shutil.which("ffmpeg") is not None
        self._warned_no_ffmpeg = False

    def available(self) -> bool:
        return bool(self._ffmpeg_ok)

    @staticmethod
    def _run(cmd):
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, check=False
            )
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def apply(self, waveform) -> np.ndarray:
        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            wav_np = np.asarray(waveform, dtype=np.float32).reshape(-1)

        if wav_np.size == 0:
            return wav_np

        if not self._ffmpeg_ok:
            if not self._warned_no_ffmpeg:
                print("[SpeechCodecRoundtripAugment] ffmpeg not found; skipping codec aug.")
                self._warned_no_ffmpeg = True
            return wav_np

        original_len = wav_np.shape[0]
        choice = random.choice(("mp3", "opus", "aac", "gsm"))

        path_in = path_enc = path_out = None
        try:
            fd, path_in = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            soundfile.write(path_in, wav_np, self.sr, subtype="PCM_16")

            if choice == "mp3":
                br = random.choice(self._MP3_BR)
                fd, path_enc = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                enc = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path_in,
                    "-codec:a", "libmp3lame", "-b:a", f"{br}k", path_enc,
                ]
            elif choice == "opus":
                br = random.choice(self._OPUS_BR)
                fd, path_enc = tempfile.mkstemp(suffix=".ogg")
                os.close(fd)
                enc = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path_in,
                    "-c:a", "libopus", "-b:a", f"{br}k", "-f", "ogg", path_enc,
                ]
            elif choice == "aac":
                br = random.choice(self._AAC_BR)
                fd, path_enc = tempfile.mkstemp(suffix=".m4a")
                os.close(fd)
                enc = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path_in,
                    "-c:a", "aac", "-b:a", f"{br}k", path_enc,
                ]
            else:  # gsm-fr telephony path
                fd, path_enc = tempfile.mkstemp(suffix=".gsm")
                os.close(fd)
                enc = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path_in,
                    "-ar", "8000", "-ac", "1", "-codec:a", "libgsm", path_enc,
                ]

            if not self._run(enc):
                return wav_np

            fd, path_out = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            dec = [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path_enc,
                "-f", "wav", "-ac", "1", "-ar", str(self.sr),
                "-acodec", "pcm_f32le", path_out,
            ]
            if not self._run(dec):
                return wav_np

            out, sr_out = soundfile.read(path_out, dtype="float32", always_2d=False)
            out = np.asarray(out, dtype=np.float32).reshape(-1)
            if sr_out != self.sr and out.size > 0:
                out = librosa.resample(out, orig_sr=sr_out, target_sr=self.sr)

            if out.shape[0] > original_len:
                out = out[:original_len]
            elif out.shape[0] < original_len:
                out = np.pad(out, (0, original_len - out.shape[0]), mode="constant")
            return out.astype(np.float32)
        finally:
            for p in (path_in, path_enc, path_out):
                if p and os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass