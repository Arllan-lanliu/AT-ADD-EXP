import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

from model.SSL import (
    XLSR, WAVLM, MERT, BEATs, CLAP, _clap_preprocess,
    PT_XLSR, PT_WAVLM, PT_MERT,
)
from model.backbone.ASSIST import AASIST
from model.backbone.ResNet18ForAudio import ResNet18ForAudio
from model.fusion import build_fusion_module
from model.backbone.Rawassist import Rawaasist

# WPT (wavelet-prompt) models are optional — they live in model/SSL.py once
# implemented.  Registration is skipped silently if the classes are absent.
try:
    from model.SSL import WPT_XLSR, WPT_WAVLM, WPT_MERT
    _WPT_AVAILABLE = True
except ImportError:
    _WPT_AVAILABLE = False



# =============================================================================
# Generic frontend-backend wrappers
# =============================================================================

class SingleSSLModel(nn.Module):
    """
    Composable single-frontend + backend detector.

    Any feature extractor with an ``extract_features(audio) -> Tensor`` method
    can be paired with any backend accepting ``(B, T, D)`` and returning
    ``(hidden, logits)``.

    The ``train()`` override ensures that a frozen frontend stays in eval mode
    even while the rest of the model trains — consistent with the fine-tuning
    and prompt-tuning strategies used throughout this project.

    Args:
        frontend (nn.Module): Waveform-to-feature encoder.  Must have a
            boolean ``freeze`` attribute for correct train/eval management.
        backend (nn.Module): Sequence classifier, e.g. ``AASIST``.
        feat_preprocess (callable, optional): Applied to frontend output
            *before* the backend, e.g. ``_clap_preprocess`` for CLAP.
        visual (bool): When True, ``forward`` returns
            ``(hidden, logits, attn_weights)`` for analysis.

    Example::

        model = SingleSSLModel(
            frontend=XLSR(model_dir="...", freeze=False),
            backend=AASIST(in_dim=1024),
        )
    """

    def __init__(
        self,
        frontend: nn.Module,
        backend: nn.Module,
        feat_preprocess: Optional[Callable] = None,
        visual: bool = False,
    ):
        super().__init__()
        self.frontend = frontend
        self.backend = backend
        self.feat_preprocess = feat_preprocess
        self.visual = visual

    def forward(self, audio_data):
        if self.visual:
            extracted = self.frontend.extract_features(audio_data)
            if isinstance(extracted, tuple):
                # (feat, attn)  or  (first_hidden, feat, attn) for PT models
                feat = extracted[1] if len(extracted) >= 3 else extracted[0]
                attn = extracted[-1]
            else:
                feat, attn = extracted, None
        else:
            extracted = self.frontend.extract_features(audio_data)
            # Some frontends may return (feat, hidden_states) tuples
            feat = extracted[0] if isinstance(extracted, tuple) else extracted

        if self.feat_preprocess is not None:
            feat = self.feat_preprocess(feat)
        #feat [B, T, C]
        hidden, out = self.backend(feat)
        if self.visual:
            return hidden, out, attn
        return hidden, out

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen frontends in eval mode during training
        if mode and hasattr(self.frontend, 'freeze') and self.frontend.freeze:
            self.frontend.eval()
        return self


class DualSSLModel(nn.Module):
    """
    Composable dual-frontend + fusion + backend detector.

    Two encoders independently extract features from the same waveform; their
    frame sequences are aligned in time, fused by ``fusion_module``, and then
    passed to ``backend``.  All fusion strategies in ``build_fusion_module``
    are supported.

    The side-channel ``model._last_type_logits`` is populated when
    ``TypeAwareFusion`` is active, allowing the caller to compute the auxiliary
    type-classification loss.

    Args:
        frontend_a, frontend_b (nn.Module): Feature extractors with
            ``extract_features`` and boolean ``freeze`` attributes.
        fusion_module (nn.Module): Combines the two feature sequences, e.g.
            ``build_fusion_module('cat_linear', 1024, 1024, 1024)``.
        backend (nn.Module): Sequence classifier, e.g. ``AASIST``.
        feat_align_fn (callable, optional):
            ``(feat_a, feat_b) -> (feat_a, feat_b)`` applied *before* fusion.
        visual (bool): When True, ``forward`` returns
            ``(hidden, logits, attn_weights)`` for analysis.

    Example::

        model = DualSSLModel(
            frontend_a=XLSR(model_dir="...", freeze=False),
            frontend_b=MERT(model_dir="...", freeze=False),
            fusion_module=build_fusion_module('cat_linear', 1024, 1024, 1024),
            backend=AASIST(in_dim=1024),
        )
    """

    def __init__(
        self,
        frontend_a: nn.Module,
        frontend_b: nn.Module,
        fusion_module: nn.Module,
        backend: nn.Module,
        visual: bool = False,
    ):
        super().__init__()
        self.frontend_a = frontend_a
        self.frontend_b = frontend_b
        self.fusion_module = fusion_module
        self.backend = backend
        self.visual = visual
        self._last_type_logits = None

    def _align_and_fuse(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        # truncate to the shorter sequence length
        t = min(feat_a.size(1), feat_b.size(1))
        feat_a = feat_a[:, :t, :]
        feat_b = feat_b[:, :t, :]
        result = self.fusion_module(feat_a, feat_b)
        if isinstance(result, tuple):
            fused, self._last_type_logits = result
        else:
            fused = result
            self._last_type_logits = None
        return fused

    def forward(self, audio_data):
        if self.visual:
            extracted_a = self.frontend_a.extract_features(audio_data)
            if isinstance(extracted_a, tuple):
                feat_a, attn = extracted_a[0], extracted_a[-1]
            else:
                feat_a, attn = extracted_a, None
        else:
            feat_a = self.frontend_a.extract_features(audio_data)

        feat_b = self.frontend_b.extract_features(audio_data)
        fused = self._align_and_fuse(feat_a, feat_b)
        hidden, out = self.backend(fused)

        if self.visual:
            return hidden, out, attn
        return hidden, out

    def train(self, mode: bool = True):
        super().train(mode)
        for fe in (self.frontend_a, self.frontend_b):
            if mode and hasattr(fe, 'freeze') and fe.freeze:
                fe.eval()
        return self


# =============================================================================
# Model registry and factory
# =============================================================================

_MODEL_REGISTRY: dict = {}


def register_model(name: str):
    """
    Decorator that registers a model factory function under ``name``.

    The decorated function must have signature ``(args) -> nn.Module`` and
    return the model on CPU (the caller moves it to the target device).

    Example::

        @register_model('my-new-model')
        def _build_my_model(args):
            return SingleSSLModel(
                frontend=XLSR(model_dir=args.xlsr, freeze=False),
                backend=AASIST(in_dim=1024),
            )
    """
    def _decorator(fn):
        _MODEL_REGISTRY[name] = fn
        return fn
    return _decorator


def build_model(args) -> nn.Module:
    """
    Build and return a model from ``args.model``.

    The returned model is on CPU; move it to the target device afterwards::

        model = build_model(args).to(args.device)

    Args:
        args: Parsed argument namespace.  Must contain at minimum ``args.model``
              and any model-specific fields (``args.xlsr``, ``args.wavlm``, …).

    Raises:
        ValueError: If ``args.model`` is not in the registry.
    """
    name = args.model
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](args)


# ── private helpers ───────────────────────────────────────────────────────────

def _dev(args) -> str:
    """Return device string from args."""
    return str(getattr(args, 'device', 'cuda'))


def _fusion(args) -> str:
    """Return fusion method name from args (default: cat_linear)."""
    return getattr(args, 'fusion', 'cat_linear')


def _beats_frontend_kw(args) -> dict:
    """Kwargs for BEATs hidden-layer readout, mirroring XLSR's options."""
    return {
        "selected_layers": getattr(args, "beats_selected_layers", None),
        "layer_fusion": getattr(args, "beats_layer_fusion", "last"),
    }


# ── Conventional CM ───────────────────────────────────────────────────────────

@register_model('aasist')
def _build_aasist(args):
    return Rawaasist()


@register_model('specresnet')
def _build_specresnet(args):
    return ResNet18ForAudio()


# ── Frozen (FR) SSL + AASIST ─────────────────────────────────────────────────

@register_model('fr-w2v2aasist')
def _build_fr_w2v2aasist(args):
    return SingleSSLModel(
        frontend=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=True),
        backend=AASIST(in_dim=1024),
    )


@register_model('fr-wavlmaasist')
def _build_fr_wavlmaasist(args):
    return SingleSSLModel(
        frontend=WAVLM(model_dir=args.wavlm, device=_dev(args), freeze=True),
        backend=AASIST(in_dim=1024),
    )


@register_model('fr-mertaasist')
def _build_fr_mertaasist(args):
    return SingleSSLModel(
        frontend=MERT(model_dir=args.mert, device=_dev(args), freeze=True),
        backend=AASIST(in_dim=1024),
    )


# ── Fine-tuned (FT) single-SSL + AASIST ──────────────────────────────────────

@register_model('ft-w2v2aasist')
def _build_ft_w2v2aasist(args):
    return SingleSSLModel(
        frontend=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        backend=AASIST(in_dim=1024),
    )


@register_model('ft-wavlmaasist')
def _build_ft_wavlmaasist(args):
    return SingleSSLModel(
        frontend=WAVLM(model_dir=args.wavlm, device=_dev(args), freeze=False),
        backend=AASIST(in_dim=1024),
    )


@register_model('ft-mertaasist')
def _build_ft_mertaasist(args):
    return SingleSSLModel(
        frontend=MERT(model_dir=args.mert, device=_dev(args), freeze=False),
        backend=AASIST(in_dim=1024),
    )


@register_model('ft-beatsaasist')
def _build_ft_beatsaasist(args):
    # BEATs encoder stays frozen; only the AASIST head is fine-tuned
    return SingleSSLModel(
        frontend=BEATs(
            model_dir=args.beats,
            device=_dev(args),
            freeze=False,
            **_beats_frontend_kw(args),
        ),
        backend=AASIST(in_dim=768),
    )


@register_model('ft-clapaasist')
def _build_ft_clapaasist(args):
    return SingleSSLModel(
        frontend=CLAP(model_dir=args.clap, device=_dev(args), freeze=False),
        backend=AASIST(in_dim=1024),
        feat_preprocess=_clap_preprocess,
    )


# ── Fine-tuned (FT) dual-SSL + AASIST ────────────────────────────────────────

@register_model('ft-xlsrwavlmaasist')
def _build_ft_xlsrwavlmaasist(args):
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=WAVLM(model_dir=args.wavlm, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 1024, 1024),
        backend=AASIST(in_dim=1024),
    )


@register_model('ft-xlsrbeatsaasist')
def _build_ft_xlsrbeatsaasist(args):
    # BEATs output is 768-d; CatLinear(1024+768→1024) is the only valid fusion here.
    # The --fusion argument is not applicable for this combination.
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=BEATs(
            model_dir=args.beats,
            device=_dev(args),
            freeze=False,
            **_beats_frontend_kw(args),
        ),
        fusion_module=build_fusion_module(_fusion(args), 1024, 768, 1024),
        backend=AASIST(in_dim=1024),
    )


@register_model('ft-xlsrmertaasist')
def _build_ft_xlsrmertaasist(args):
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=MERT(model_dir=args.mert, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 1024, 1024),
        backend=AASIST(in_dim=1024),
    )


@register_model('ft-xlsrclapaasist')
def _build_ft_xlsrclapaasist(args):
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=CLAP(model_dir=args.clap, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 1024, 1024),
        backend=AASIST(in_dim=1024),
    )


# ── Prompt-tuned (PT) SSL + AASIST ───────────────────────────────────────────

@register_model('pt-w2v2aasist')
def _build_pt_w2v2aasist(args):
    return SingleSSLModel(
        frontend=PT_XLSR(
            model_dir=args.xlsr,
            prompt_dim=args.prompt_dim,
            device=_dev(args),
            num_prompt_tokens=args.num_prompt_tokens,
            dropout=args.pt_dropout,
        ),
        backend=AASIST(in_dim=1024),
    )


@register_model('pt-wavlmaasist')
def _build_pt_wavlmaasist(args):
    return SingleSSLModel(
        frontend=PT_WAVLM(
            model_dir=args.wavlm,
            prompt_dim=args.prompt_dim,
            device=_dev(args),
            num_prompt_tokens=args.num_prompt_tokens,
            dropout=args.pt_dropout,
        ),
        backend=AASIST(in_dim=1024),
    )


@register_model('pt-mertaasist')
def _build_pt_mertaasist(args):
    return SingleSSLModel(
        frontend=PT_MERT(
            model_dir=args.mert,
            prompt_dim=args.prompt_dim,
            device=_dev(args),
            num_prompt_tokens=args.num_prompt_tokens,
            dropout=args.pt_dropout,
        ),
        backend=AASIST(in_dim=1024),
    )


# ── Wavelet Prompt-tuned (WPT) SSL + AASIST ──────────────────────────────────
# These models are registered only when WPT_XLSR / WPT_WAVLM / WPT_MERT are
# implemented in model/SSL.py and importable.

if _WPT_AVAILABLE:

    @register_model('wpt-w2v2aasist')
    def _build_wpt_w2v2aasist(args):
        return SingleSSLModel(
            frontend=WPT_XLSR(
                model_dir=args.xlsr,
                prompt_dim=args.prompt_dim,
                device=_dev(args),
                num_prompt_tokens=args.num_prompt_tokens,
                num_wavelet_tokens=args.num_wavelet_tokens,
                dropout=args.pt_dropout,
            ),
            backend=AASIST(in_dim=1024),
        )

    @register_model('wpt-wavlmaasist')
    def _build_wpt_wavlmaasist(args):
        return SingleSSLModel(
            frontend=WPT_WAVLM(
                model_dir=args.wavlm,
                prompt_dim=args.prompt_dim,
                device=_dev(args),
                num_prompt_tokens=args.num_prompt_tokens,
                num_wavelet_tokens=args.num_wavelet_tokens,
                dropout=args.pt_dropout,
            ),
            backend=AASIST(in_dim=1024),
        )

    @register_model('wpt-mertaasist')
    def _build_wpt_mertaasist(args):
        return SingleSSLModel(
            frontend=WPT_MERT(
                model_dir=args.mert,
                prompt_dim=args.prompt_dim,
                device=_dev(args),
                num_prompt_tokens=args.num_prompt_tokens,
                num_wavelet_tokens=args.num_wavelet_tokens,
                dropout=args.pt_dropout,
            ),
            backend=AASIST(in_dim=1024),
        )
