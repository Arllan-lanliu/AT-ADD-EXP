"""
Audio **type** classifiers for AT-ADD fine labels: speech / sound / singing / music.

This module provides two standalone models (full waveform → 4-way logits):

1. **XLSRATADDTypeClassifier** — frozen or finetunable XLS-R (Wav2Vec2) frame features,
   mean-pooled over time, plus a small MLP head.

2. **LogMelCNNATADDTypeClassifier** — on-the-fly log-mel spectrogram + compact CNN;
   no HuggingFace checkpoint, light and fast for type-only training or baselines.

Index order matches the rest of the repo (e.g. ``main_train._FINE_TYPE_NAMES``):
``0=speech, 1=sound, 2=singing, 3=music``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from model.SSL import XLSR

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ATADD_AUDIO_TYPE_NAMES: Tuple[str, ...] = ("speech", "sound", "singing", "music")
NUM_ATADD_AUDIO_TYPES: int = 4


def masked_mean_pool(
    feats: torch.Tensor, feat_lens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean-pool frame features ``(B, T, D)``. If ``feat_lens`` is None, pool all frames."""
    if feat_lens is None:
        return feats.mean(dim=1)
    bsz, max_t, _ = feats.shape
    feat_lens_t = feat_lens.to(device=feats.device).long().clamp(min=1, max=max_t)
    mask = torch.arange(max_t, device=feats.device).unsqueeze(0) < feat_lens_t.unsqueeze(1)
    mask = mask.unsqueeze(-1).to(dtype=feats.dtype)
    summed = (feats * mask).sum(dim=1)
    denom = feat_lens_t.unsqueeze(1).to(dtype=feats.dtype)
    return summed / denom


class SimpleTypeMLPHead(nn.Module):
    """LayerNorm + two-layer MLP classifier (same spirit as ``vote_model.TypeClassifier``)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = NUM_ATADD_AUDIO_TYPES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XLSRATADDTypeClassifier(nn.Module):
    """
    Pretrained XLS-R frontend + mean-pooled frame embedding + MLP → 4 type logits.

    Parameters
    ----------
    model_dir:
        Local HuggingFace-style Wav2Vec2 / XLS-R directory (see ``args.xlsr`` in config).
    freeze_frontend:
        If True, XLS-R weights are frozen and run under ``no_grad`` (head-only training).
    xlsr_dim:
        Frame dimension (1024 for common XLS-R 300M checkpoints).
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        freeze_frontend: bool = True,
        xlsr_dim: int = 1024,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        sampling_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.xlsr = XLSR(
            model_dir=model_dir,
            device=device,
            sampling_rate=sampling_rate,
            freeze=freeze_frontend,
            visual=False,
        )
        self.head = SimpleTypeMLPHead(
            input_dim=xlsr_dim,
            hidden_dim=head_hidden_dim,
            num_classes=NUM_ATADD_AUDIO_TYPES,
            dropout=head_dropout,
        )

    def train(self, mode: bool = True) -> "XLSRATADDTypeClassifier":
        super().train(mode)
        if mode and getattr(self.xlsr, "freeze", False):
            self.xlsr.eval()
        return self

    def forward(self, wav: torch.Tensor, wav_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        wav : (B, T) waveform, 16 kHz mono (same convention as the rest of AT-ADD-Baseline).
        wav_lens : optional (B,) valid sample lengths for pooling (rarely used with fixed crops).

        Returns
        -------
        type_logits : (B, 4)
        """
        feat = self.xlsr.extract_features(wav)
        if isinstance(feat, tuple):
            feat = feat[0]
        if feat.dim() == 2:
            pooled = feat
        else:
            pooled = masked_mean_pool(feat, wav_lens)
        return self.head(pooled)


class LogMelCNNATADDTypeClassifier(nn.Module):
    """
    Log-mel spectrogram + shallow CNN (no SSL): simple, data-efficient type baseline.

    The CNN sees ``(B, 1, n_mels, time)``; strides downsample mostly along time while
    keeping full mel resolution early, then global average-pools to a fixed vector.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: Optional[float] = 7_600.0,
        base_channels: int = 32,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.melspec = mel
        self.ampto_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(c * 4, NUM_ATADD_AUDIO_TYPES),
        )

    def _log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T)
        if wav.dim() != 2:
            raise ValueError(f"Expected wav shape (B, T), got {tuple(wav.shape)}")
        x = self.melspec(wav)
        x = self.ampto_db(x)
        x = x - x.mean(dim=(1, 2), keepdim=True)
        x = x / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        return x.unsqueeze(1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        wav : (B, T) mono waveform at ``self.sample_rate``.

        Returns
        -------
        type_logits : (B, 4)
        """
        x = self._log_mel(wav)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.fc(x)
