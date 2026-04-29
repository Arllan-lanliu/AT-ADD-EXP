"""
XLSR/BEATs + AASIST routed variant (registered name: ``ft-routed-ssl-aasist``).

* **ft-routed-ssl-aasist** — **mask** subset forward per branch, shared AASIST;
  ``route_mode_train``: ``oracle`` (GT) vs ``pred`` (argmax type).

Returns ``(last_hidden, spoof_logits)``, sets ``_last_type_logits`` (2-way coarse),
and ``use_class_types=True`` for ``main_train`` (dev type CE + type-clf metrics).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SSL import BEATs, XLSR
from model.backbone.ASSIST import AASIST

from model.model import _dev, register_model

# AASIST readout dim (``5 * gat_dims[1]`` with default gat_dims[1]=32 in ``ASSIST.py``).
_AASIST_READOUT_DIM = 160


def _coarse_type_from_4class(class_types: torch.Tensor) -> torch.Tensor:
    """Map AT-ADD types speech(0), sound(1), singing(2), music(3) -> 0/1.

    0  -> speech or singing, 1 -> sound or music.
    """
    return ((class_types == 1) | (class_types == 3)).long()


# =============================================================================
# Reference-style: mask routing, subset forward, shared AASIST (see module doc).
# =============================================================================


def _unpack_ssl_output(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``SSL.extract_features`` → ``(tensor[, ...])``; XLSR/BEATs in this repo return features only."""
    if isinstance(out, tuple):
        return (out[0], out[1] if len(out) > 1 else None)
    return out, None


def _masked_mean_pool(
    feats: torch.Tensor, feat_lens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean pool over time. ``feats``: [B, T, D]."""
    if feat_lens is None:
        return feats.mean(dim=1)
    bsz, max_t, _ = feats.shape
    feat_lens_t = feat_lens.to(device=feats.device).long().clamp(min=1, max=max_t)
    mask = torch.arange(max_t, device=feats.device).unsqueeze(0) < feat_lens_t.unsqueeze(1)
    mask = mask.unsqueeze(-1).to(dtype=feats.dtype)
    summed = (feats * mask).sum(dim=1)
    denom = feat_lens_t.unsqueeze(1).to(dtype=feats.dtype)
    return summed / denom


def _call_assist_full(
    assist: nn.Module, feats: torch.Tensor, feat_lens: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``AASIST`` returns ``(last_hidden, logits)``; this repo only uses the ``feats`` argument."""
    try:
        out = assist(feats, feat_lens)
    except TypeError:
        out = assist(feats)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError("AASIST forward must return (last_hidden, logits).")
    return out[0], out[1]


def _maybe_freeze(module: nn.Module, freeze: bool) -> None:
    if not freeze:
        return
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


class TypeClassifier(nn.Module):
    """Binary type classifier: 0 = speech/singing, 1 = sound/music."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
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

    def forward(self, pooled_ssl_feat: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_ssl_feat)


class RoutedSSLASSIST(nn.Module):
    """
    Router XLSR -> type MLP -> mask-routed ``speech_xlsr`` | ``sound_beats`` -> shared AASIST.

    ``route_mode_train``:
        - ``\"oracle\"``: while training, use GT coarse type (from ``class_types`` or ``type_labels``).
        - ``\"pred\"``: always route by argmax of ``type_logits``.
    """

    SPEECH_SINGING = 0
    SOUND_MUSIC = 1

    def __init__(
        self,
        *,
        xlsr_router_dir: str,
        xlsr_speech_dir: str,
        beats_dir: str,
        device: str = "cuda",
        router_dim: int = 1024,
        type_head_hidden: int = 256,
        type_head_dropout: float = 0.1,
        num_spoof_classes: int = 2,
        route_mode_train: str = "oracle",
        freeze_router: bool = False,
        freeze_speech: bool = False,
        freeze_sound: bool = False,
    ) -> None:
        super().__init__()
        self.route_mode_train = route_mode_train
        assert self.route_mode_train in ("oracle", "pred")
        self.num_spoof_classes = num_spoof_classes

        self.router_xlsr = XLSR(
            model_dir=xlsr_router_dir, device=device, freeze=True, visual=False
        )
        self.xlsr_ssl = XLSR(
            model_dir=xlsr_speech_dir, device=device, freeze=freeze_speech, visual=False
        )
        self.beats_ssl = BEATs(model_dir=beats_dir, device=device, freeze=freeze_sound)
        self.xlsr_assist = AASIST(in_dim=1024)
        self.beats_assist = AASIST(in_dim=768)
        _maybe_freeze(self.router_xlsr, True)
        _maybe_freeze(self.xlsr_ssl, freeze_speech)
        _maybe_freeze(self.beats_ssl, freeze_sound)

        self.type_classifier = TypeClassifier(
            input_dim=router_dim,
            hidden_dim=type_head_hidden,
            num_classes=2,
            dropout=type_head_dropout,
        )

        # 检查是否意外共享参数
        router_params = set(id(p) for p in self.router_xlsr.parameters())
        speech_params = set(id(p) for p in self.xlsr_ssl.parameters())
        shared = router_params & speech_params
        if shared:
            print(f"⚠️ WARNING: {len(shared)} parameters shared between router and speech XLSR!")
        
        # 检查初始化是否相同
        r_first = next(self.router_xlsr.parameters())
        s_first = next(self.xlsr_ssl.parameters())
        if torch.equal(r_first, s_first):
            print("⚠️ WARNING: Router and Speech XLSR have identical initial weights!")

    @staticmethod
    def _run_ssl(
        ssl_mod: nn.Module, wav: torch.Tensor, wav_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Fixed-length AT-ADD crop: ignore ``wav_lens``; subset batches still valid.
        _ = wav_lens
        return _unpack_ssl_output(ssl_mod.extract_features(wav))

    @staticmethod
    def _coerce_binary_type_labels(
        class_types: Optional[torch.Tensor],
        type_labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if type_labels is not None:
            return type_labels.long()
        if class_types is not None:
            return _coarse_type_from_4class(class_types)
        return None

    def _select_routes(
        self,
        type_logits: torch.Tensor,
        oracle_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.training and self.route_mode_train == "oracle" and oracle_labels is not None:
            return oracle_labels.long()
        return type_logits.argmax(dim=-1).long()

    def forward(
        self,
        wav: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        type_labels: Optional[torch.Tensor] = None,
        class_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = wav.size(0)
        device = wav.device
        oracle_type = self._coerce_binary_type_labels(class_types, type_labels)

        router_feats, router_feat_lens = self._run_ssl(self.router_xlsr, wav, wav_lens)
        if isinstance(router_feats, torch.Tensor) and router_feats.dim() == 2:
            router_pooled = router_feats
        else:
            router_pooled = _masked_mean_pool(router_feats, router_feat_lens)
        type_logits = self.type_classifier(router_pooled)

        route_ids = self._select_routes(type_logits, oracle_type)
        speech_mask = route_ids.eq(self.SPEECH_SINGING)
        sound_mask = route_ids.eq(self.SOUND_MUSIC)

        last_hidden = torch.zeros(
            bsz, _AASIST_READOUT_DIM, device=device, dtype=type_logits.dtype
        )
        spoof_logits = torch.zeros(
            bsz, self.num_spoof_classes, device=device, dtype=type_logits.dtype
        )

        if speech_mask.any():
            wav_s = wav[speech_mask]
            lens_s = wav_lens[speech_mask] if wav_lens is not None else None
            feats_s, feat_lens_s = self._run_ssl(self.xlsr_ssl, wav_s, lens_s)
            h_s, lg_s = _call_assist_full(self.xlsr_assist, feats_s, feat_lens_s)
            last_hidden[speech_mask] = h_s
            spoof_logits[speech_mask] = lg_s

        if sound_mask.any():
            wav_m = wav[sound_mask]
            lens_m = wav_lens[sound_mask] if wav_lens is not None else None
            feats_m, feat_lens_m = self._run_ssl(self.beats_ssl, wav_m, lens_m)
            h_m, lg_m = _call_assist_full(self.beats_assist, feats_m, feat_lens_m)
            last_hidden[sound_mask] = h_m
            spoof_logits[sound_mask] = lg_m

        return {
            "last_hidden": last_hidden,
            "spoof_logits": spoof_logits,
            "type_logits": type_logits,
            "route_ids": route_ids,
        }


class RoutedMaskATADDAasistModel(nn.Module):
    """AT-ADD API: ``forward(audio, class_types) -> (last_hidden, logits)``; sets ``_last_type_logits``."""

    use_class_types: bool = True

    def __init__(
        self,
        xlsr_router_dir: str,
        xlsr_speech_dir: str,
        beats_dir: str,
        device: str = "cuda",
        route_mode_train: str = "oracle",
        type_head_hidden: int = 256,
        type_head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.routed = RoutedSSLASSIST(
            xlsr_router_dir=xlsr_router_dir,
            xlsr_speech_dir=xlsr_speech_dir,
            beats_dir=beats_dir,
            device=device,
            route_mode_train=route_mode_train,
            type_head_hidden=type_head_hidden,
            type_head_dropout=type_head_dropout,
        )
        self._last_type_logits: torch.Tensor | None = None

    def train(self, mode: bool = True) -> "RoutedMaskATADDAasistModel":
        super().train(mode)
        for m in (self.routed.router_xlsr, self.routed.xlsr_ssl, self.routed.beats_ssl):
            if mode and hasattr(m, "freeze") and m.freeze:
                m.eval()
        return self

    def forward(
        self, audio_data: torch.Tensor, class_types: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.routed(
            audio_data, wav_lens=None, class_types=class_types, type_labels=None
        )
        self._last_type_logits = out["type_logits"]
        return out["last_hidden"], out["spoof_logits"]


@register_model("ft-routed-ssl-aasist")
def _build_ft_routed_ssl_aasist(args) -> RoutedMaskATADDAasistModel:
    r_dir = getattr(args, "xlsr_type", None) or args.xlsr
    s_dir = getattr(args, "xlsr_ss", None) or args.xlsr
    return RoutedMaskATADDAasistModel(
        xlsr_router_dir=r_dir,
        xlsr_speech_dir=s_dir,
        beats_dir=args.beats,
        device=_dev(args),
        route_mode_train=str(
            getattr(args, "route_mode_train", "oracle")
        ),
        type_head_hidden=int(getattr(args, "type_head_hidden", 256)),
        type_head_dropout=float(getattr(args, "type_head_dropout", 0.1)),
    )


