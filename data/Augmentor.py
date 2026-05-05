import os
import glob
import random

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
    def __init__(self, rir_path='your_path/RIRS_NOISES', musan_path='your_path/musan'):
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


class NoiseAugment:
    """白噪声 + 环境噪声"""
    def __init__(self, snr_range=(10, 30)):
        self.snr_range = snr_range

    def apply(self, waveform):
        snr_db = random.uniform(*self.snr_range)
        noise = np.random.randn(len(waveform))
        
        signal_power = np.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
        
        return waveform + noise