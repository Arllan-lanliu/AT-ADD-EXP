# -*- coding: utf-8 -*-
"""
Multi-head SSL + five AASIST experts (speech / sound / singing / music / total).

- ``MultiHeadXLSR``: one backbone (XLSR or BEATs).
- ``MultiHeadDualSSL``: XLSR + BEATs, ``cat_linear`` fusion (same as ``DualSSLModel`` in ``model.model``), then the same five experts.

Frame features are ``(B, T, D)`` before each ``AASIST(in_dim=D)``.

AT-ADD type IDs: speech=0, sound=1, singing=2, music=3 (same as ``data.dataset``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.ASSIST import AASIST
from model.fusion import build_fusion_module

MULT_HEAD_KEYS: Tuple[str, ...] = ("speech", "sound", "singing", "music")
ALL_HEAD_KEYS: Tuple[str, ...] = ("speech", "sound", "singing", "music", "total")


class MultiHeadXLSR(nn.Module):
    """
    ``wav → SSL backbone (XLSR or BEATs) → 5 × AASIST(D) → logits (B, 2) each``.

    Training policy (controlled by trainer): the shared encoder follows ``train()`` until the
    first dev evaluation completes, then ``lock_xlsr_eval_after_first_dev()`` keeps the
    **backbone in eval()** during later ``model.train()`` (stable BatchNorm / dropout) while
    **AASIST experts** still follow the global train/eval mode. If ``freeze_backbone=True``,
    the backbone stays in eval during training anyway.
    """

    def __init__(
        self,
        xlsr_model_dir: Optional[str] = None,
        device: str = "cuda",
        backbone_dim: Optional[int] = None,
        freeze_backbone: bool = False,
        *,
        beats_model_dir: Optional[str] = None,
        ssl_backbone: str = "xlsr",
    ) -> None:
        super().__init__()
        ssl_backbone = (ssl_backbone or "xlsr").strip().lower()
        self.ssl_backbone = ssl_backbone

        if ssl_backbone == "xlsr":
            from model.SSL import XLSR

            if not xlsr_model_dir:
                raise ValueError("MultiHeadXLSR(ssl_backbone='xlsr') requires xlsr_model_dir")
            self.backbone = XLSR(
                model_dir=xlsr_model_dir,
                device=device,
                freeze=freeze_backbone,
                visual=False,
            )
            dim = 1024 if backbone_dim is None else int(backbone_dim)
        elif ssl_backbone == "beats":
            from model.SSL import BEATs

            if not beats_model_dir:
                raise ValueError("MultiHeadXLSR(ssl_backbone='beats') requires beats_model_dir")
            self.backbone = BEATs(
                model_dir=beats_model_dir,
                device=device,
                freeze=freeze_backbone,
            )
            dim = 768 if backbone_dim is None else int(backbone_dim)
        else:
            raise ValueError(f"ssl_backbone must be 'xlsr' or 'beats', got {ssl_backbone!r}")

        self.backbone_dim = dim
        self.experts = nn.ModuleDict(
            {name: AASIST(in_dim=dim) for name in ALL_HEAD_KEYS}
        )
        # Trainer sets True after first full sample-dev evaluation; persists in checkpoint.
        self._lock_xlsr_eval = False

    @classmethod
    def from_mult_config(
        cls,
        *,
        cuda: bool,
        ssl_backbone: str = "xlsr",
        xlsr_model_dir: Optional[str] = None,
        beats_model_dir: Optional[str] = None,
        freeze_backbone: bool = False,
        backbone_dim: Optional[int] = None,
    ) -> "MultiHeadXLSR":
        """Build from multi-head YAML fields: ``ssl_backbone``, ``ssl.xlsr`` / ``ssl.beats``."""
        ssl_backbone = (ssl_backbone or "xlsr").strip().lower()
        dev = "cuda" if cuda else "cpu"
        if ssl_backbone == "beats":
            if not beats_model_dir:
                raise ValueError(
                    "ssl_backbone='beats' requires beats_model_dir (set ssl.beats in YAML)."
                )
            print(
                f"[MultiHeadXLSR] backbone=BEATs dim={768 if backbone_dim is None else backbone_dim} "
                f"path={beats_model_dir}",
                flush=True,
            )
            return cls(
                xlsr_model_dir=None,
                device=dev,
                backbone_dim=backbone_dim,
                freeze_backbone=freeze_backbone,
                beats_model_dir=beats_model_dir,
                ssl_backbone="beats",
            )
        if not xlsr_model_dir:
            raise ValueError(
                "ssl_backbone='xlsr' requires xlsr_model_dir (set ssl.xlsr in YAML)."
            )
        print(
            f"[MultiHeadXLSR] backbone=XLSR dim={1024 if backbone_dim is None else backbone_dim} "
            f"path={xlsr_model_dir}",
            flush=True,
        )
        return cls(
            xlsr_model_dir=xlsr_model_dir,
            device=dev,
            backbone_dim=backbone_dim,
            freeze_backbone=freeze_backbone,
            ssl_backbone="xlsr",
        )

    def encode_frames(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` wave → ``(B, T, D)`` SSL frame features."""
        h = self.backbone.extract_features(wav)
        if isinstance(h, tuple):
            h = h[0]
        return h

    def forward(self, wav: torch.Tensor, audio_type: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Return logits dict. ``audio_type`` ignored (routing in ``compute_loss`` / ``inference``)."""
        del audio_type
        frames = self.encode_frames(wav)
        out: Dict[str, torch.Tensor] = {}
        for name in ALL_HEAD_KEYS:
            _hidden, logits = self.experts[name](frames)
            out[name] = logits
        return out

    def lock_xlsr_eval_after_first_dev(self) -> None:
        """After first dev metrics: keep the shared SSL backbone in ``eval()`` even during ``train()``."""
        if self._lock_xlsr_eval:
            return
        self._lock_xlsr_eval = True
        self.backbone.eval()
        print(
            "[MultiHeadXLSR] Backbone locked to eval mode; "
            "AASIST experts still follow global train()/eval().",
            flush=True,
        )

    def train(self, mode: bool = True) -> "MultiHeadXLSR":
        super().train(mode)
        if not mode:
            self.backbone.eval()
            return self
        if getattr(self.backbone, "freeze", False) or self._lock_xlsr_eval:
            self.backbone.eval()
        return self

    def eval(self) -> "MultiHeadXLSR":
        super().eval()
        self.backbone.eval()
        return self


class MultiHeadDualSSL(nn.Module):
    """
    ``wav → XLSR (1024) + BEATs (768) → cat_linear → D_fused → 5 × AASIST(D_fused)``.

    Mirrors ``DualSSLModel`` time alignment (min length) + ``build_fusion_module('cat_linear', ...)``.
    ``forward`` / expert outputs match :class:`MultiHeadXLSR` so ``compute_loss`` / ``inference`` are unchanged.
    """

    _FUSION_NAME = "cat_linear"
    _DIM_XLSR = 1024
    _DIM_BEATS = 768

    def __init__(
        self,
        xlsr_model_dir: str,
        beats_model_dir: str,
        device: str = "cuda",
        freeze_backbone: bool = False,
        fused_expert_dim: int = 1024,
    ) -> None:
        super().__init__()
        from model.SSL import BEATs, XLSR

        self.frontend_a = XLSR(
            model_dir=xlsr_model_dir,
            device=device,
            freeze=freeze_backbone,
            visual=False,
        )
        self.frontend_b = BEATs(
            model_dir=beats_model_dir,
            device=device,
            freeze=freeze_backbone,
        )
        self.fusion = build_fusion_module(
            self._FUSION_NAME,
            self._DIM_XLSR,
            self._DIM_BEATS,
            fused_expert_dim,
        )
        self.backbone_dim = int(fused_expert_dim)
        self.experts = nn.ModuleDict(
            {name: AASIST(in_dim=fused_expert_dim) for name in ALL_HEAD_KEYS}
        )
        self._lock_xlsr_eval = False

    @classmethod
    def from_mult_config(
        cls,
        *,
        cuda: bool,
        xlsr_model_dir: Optional[str],
        beats_model_dir: Optional[str],
        freeze_backbone: bool,
        fused_dim: Optional[int] = None,
    ) -> "MultiHeadDualSSL":
        """XLSR + BEATs with ``cat_linear``; ``fused_dim`` is expert ``in_dim`` (default 1024)."""
        if not xlsr_model_dir:
            raise ValueError("MultiHeadDualSSL requires xlsr_model_dir (ssl.xlsr).")
        if not beats_model_dir:
            raise ValueError("MultiHeadDualSSL requires beats_model_dir (ssl.beats).")
        out_d = 1024 if fused_dim is None else int(fused_dim)
        dev = "cuda" if cuda else "cpu"
        print(
            f"[MultiHeadDualSSL] XLSR ({cls._DIM_XLSR}) + BEATs ({cls._DIM_BEATS}) "
            f"cat_linear → {out_d}d experts",
            flush=True,
        )
        print(f"  xlsr:  {xlsr_model_dir}", flush=True)
        print(f"  beats: {beats_model_dir}", flush=True)
        return cls(
            xlsr_model_dir=xlsr_model_dir,
            beats_model_dir=beats_model_dir,
            device=dev,
            freeze_backbone=freeze_backbone,
            fused_expert_dim=out_d,
        )

    @staticmethod
    def _extract_frames(frontend: nn.Module, wav: torch.Tensor) -> torch.Tensor:
        h = frontend.extract_features(wav)
        if isinstance(h, tuple):
            h = h[0]
        return h

    def encode_frames(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` → aligned XLSR/BEATs frames → fused ``(B, T', D)``."""
        feat_a = self._extract_frames(self.frontend_a, wav)
        feat_b = self._extract_frames(self.frontend_b, wav)
        t = min(feat_a.size(1), feat_b.size(1))
        feat_a = feat_a[:, :t, :]
        feat_b = feat_b[:, :t, :]
        return self.fusion(feat_a, feat_b)

    def forward(self, wav: torch.Tensor, audio_type: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        del audio_type
        frames = self.encode_frames(wav)
        out: Dict[str, torch.Tensor] = {}
        for name in ALL_HEAD_KEYS:
            _hidden, logits = self.experts[name](frames)
            out[name] = logits
        return out

    def lock_xlsr_eval_after_first_dev(self) -> None:
        if self._lock_xlsr_eval:
            return
        self._lock_xlsr_eval = True
        self.frontend_a.eval()
        self.frontend_b.eval()
        print(
            "[MultiHeadDualSSL] Both SSL frontends locked to eval; "
            "AASIST experts still follow global train()/eval().",
            flush=True,
        )

    def train(self, mode: bool = True) -> "MultiHeadDualSSL":
        super().train(mode)
        if not mode:
            self.frontend_a.eval()
            self.frontend_b.eval()
            return self
        for fe in (self.frontend_a, self.frontend_b):
            if getattr(fe, "freeze", False) or self._lock_xlsr_eval:
                fe.eval()
        return self

    def eval(self) -> "MultiHeadDualSSL":
        super().eval()
        self.frontend_a.eval()
        self.frontend_b.eval()
        return self


def build_mult_head_from_args(args: Any) -> Union[MultiHeadXLSR, MultiHeadDualSSL]:
    """Dispatch multi-head model from ``ssl_backbone`` (``xlsr`` / ``beats`` / ``xlsr_beats``)."""
    ssl_bb = str(getattr(args, "ssl_backbone", "xlsr") or "xlsr").strip().lower()
    cuda = bool(getattr(args, "cuda"))
    if ssl_bb in ("xlsr_beats", "dual"):
        return MultiHeadDualSSL.from_mult_config(
            cuda=cuda,
            xlsr_model_dir=getattr(args, "xlsr", None),
            beats_model_dir=getattr(args, "beats", None),
            freeze_backbone=bool(getattr(args, "freeze_backbone", False)),
            fused_dim=getattr(args, "backbone_dim", None),
        )
    return MultiHeadXLSR.from_mult_config(
        cuda=cuda,
        ssl_backbone=ssl_bb,
        xlsr_model_dir=getattr(args, "xlsr", None),
        beats_model_dir=getattr(args, "beats", None),
        freeze_backbone=bool(getattr(args, "freeze_backbone", False)),
        backbone_dim=getattr(args, "backbone_dim", None),
    )


def compute_loss(
    model: nn.Module,
    wav: torch.Tensor,
    labels: torch.Tensor,
    audio_type: torch.Tensor,
    specialist_weight: float = 0.2,
    total_weight: float = 0.8,
    class_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Specialist CE on the head indexed by ``audio_type`` for each sample; total CE on all.

    ``loss = specialist_weight * loss_specialist + total_weight * loss_total`` (not auto-normalised).
    """
    outs = model(wav)
    bsz = labels.size(0)
    device = labels.device

    stack = torch.stack(
        [outs["speech"], outs["sound"], outs["singing"], outs["music"]],
        dim=1,
    )  # (B, 4, 2)
    idx = audio_type.long().clamp(0, 3)
    ar = torch.arange(bsz, device=device)
    spec_logits = stack[ar, idx]

    ce_kw = {} if class_weight is None else {"weight": class_weight}
    loss_specialist = F.cross_entropy(spec_logits, labels.long(), **ce_kw)
    loss_total = F.cross_entropy(outs["total"], labels.long(), **ce_kw)

    w_sum = specialist_weight + total_weight
    loss = (specialist_weight * loss_specialist + total_weight * loss_total) / w_sum
    extras = {
        "loss_specialist": loss_specialist.detach(),
        "loss_total": loss_total.detach(),
    }
    return loss, extras


@torch.no_grad()
def inference(
    model: nn.Module,
    wav: torch.Tensor,
    audio_type: Optional[torch.Tensor] = None,  #dev和eval时均不提供type
    *,
    score_real_class: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Known type → specialist head; unknown ``audio_type`` → ``total``.
    
    If audio_type contains -1 or values > 3, those samples will use the total head.

    Returns:
        score: ``(B,)`` P(real)=softmax[:,0] by default.
        all_logits: dict of all heads.
    """
    model.eval()
    all_logits = model(wav)

    def _scores_from_logits(lg: torch.Tensor) -> torch.Tensor:
        p = F.softmax(lg, dim=1)
        return p[:, 0] if score_real_class else p[:, 1]   # {"fake": 1, "real": 0}

    if audio_type is None:
        return _scores_from_logits(all_logits["total"]), all_logits

    bsz = wav.size(0)
    device = wav.device
    stacks = torch.stack([all_logits[k] for k in MULT_HEAD_KEYS], dim=1)  # (B, 4, 2)
    
    idx = audio_type.long().to(device)
    valid_mask = (idx >= 0) & (idx <= 3)
    idx_clamped = idx.clamp(0, 3)
    
    ar = torch.arange(bsz, device=device)
    specialist_logits = stacks[ar, idx_clamped]
    
    final_logits = torch.where(
        valid_mask.unsqueeze(1),
        specialist_logits,
        all_logits["total"]
    )
    
    return _scores_from_logits(final_logits), all_logits


@torch.no_grad()
def inference_vote(
    model: nn.Module,
    wav: torch.Tensor,
    *,
    score_real_class: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_dict = model(wav)
    
    # 所有头的概率
    probs = torch.stack([F.softmax(logits_dict[k], dim=1) for k in ALL_HEAD_KEYS], dim=1)  # (B, 5, 2)
    
    # 平均概率
    avg_prob = probs.mean(dim=1)  # (B, 2)
    scores = avg_prob[:, 0] if score_real_class else avg_prob[:, 1]
    
    # 投票预测
    votes = probs.argmax(dim=2)  # (B, 5)
    fake_ct = (votes == 1).sum(dim=1)
    pred = (fake_ct * 2 > votes.size(1)).long()
    
    return scores, pred