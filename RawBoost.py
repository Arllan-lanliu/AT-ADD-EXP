#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import copy
import argparse
import torch
'''
   Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
   RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.
   In Proc. ICASSP 2022, pp:6382--6386.
'''
# parser = argparse.ArgumentParser(description='rawboost')
# parser.add_argument('--algo', type=int, default=0,
#                     help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
#                           5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

# # LnL_convolutive_noise parameters
# parser.add_argument('--nBands', type=int, default=5,
#                 help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
# parser.add_argument('--minF', type=int, default=20,
#                 help='minimum centre frequency [Hz] of notch filter.[default=20] ')
# parser.add_argument('--maxF', type=int, default=8000,
#                 help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
# parser.add_argument('--minBW', type=int, default=100,
#                 help='minimum width [Hz] of filter.[default=100] ')
# parser.add_argument('--maxBW', type=int, default=1000,
#                 help='maximum width [Hz] of filter.[default=1000] ')
# parser.add_argument('--minCoeff', type=int, default=10,
#                 help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
# parser.add_argument('--maxCoeff', type=int, default=100,
#                 help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
# parser.add_argument('--minG', type=int, default=0,
#                 help='minimum gain factor of linear component.[default=0]')
# parser.add_argument('--maxG', type=int, default=0,
#                 help='maximum gain factor of linear component.[default=0]')
# parser.add_argument('--minBiasLinNonLin', type=int, default=5,
#                 help=' minimum gain difference between linear and non-linear components.[default=5]')
# parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
#                 help=' maximum gain difference between linear and non-linear components.[default=20]')
# parser.add_argument('--N_f', type=int, default=5,
#                 help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

# # ISD_additive_noise parameters
# parser.add_argument('--P', type=int, default=10,
#                 help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
# parser.add_argument('--g_sd', type=int, default=2,
#                 help='gain parameters > 0. [default=2]')

# # SSI_additive_noise parameters
# parser.add_argument('--SNRmin', type=int, default=10,
#                 help='Minimum SNR value for coloured additive noise.[defaul=10]')
# parser.add_argument('--SNRmax', type=int, default=40,
#                 help='Maximum SNR value for coloured additive noise.[defaul=40]')
# args = parser.parse_args()  


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x


def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)

        
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise

def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x


def process_Rawboost_feature(feature, sr, algo=4, N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                             minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, 
                             maxBiasLinNonLin=20, P=10, g_sd=2, SNRmin=10, SNRmax=40):
    # Data process by Convolutive noise (1st algo)

    if algo == 1:
        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        print("ISD_additive_noise")
        feature = ISD_additive_noise(feature, P, g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF,
                                     maxF, minBW, maxBW, minCoeff, maxCoeff, minG,
                                     maxG, sr)

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, P, g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:
        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                         minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                         maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, P, g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # Data process by original data without Rawboost processing           
    else:
        feature = feature

    return torch.Tensor(feature)


#===================================================================================================================================================================
import torch
import numpy as np
import librosa
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


import librosa
import numpy as np
import torch

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
