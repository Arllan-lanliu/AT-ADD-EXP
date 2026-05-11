"""
AT-ADD training configuration.

Design
------
All training parameters are captured in the ``ATADDConfig`` dataclass and its
four sub-configs.  The dataclass owns type annotations, default values, and
``__post_init__`` validation.

W&B settings (and the ``--config`` / ``--gpu`` meta flags) are the only things
handled by argparse, so the CLI stays thin while experiment parameters live
entirely in YAML files.

Priority (highest → lowest)
----------------------------
    1. ``--gpu`` CLI flag (overrides YAML gpu field)
    2. YAML config file supplied via ``--config PATH``
    3. ATADDConfig / sub-config field defaults

Usage
-----
    # YAML-based (recommended)
    python main_train.py --config conf/experiments/xlsr_mert_add.yaml

    # Override GPU on the fly
    python main_train.py --config conf/experiments/xlsr_mert_add.yaml --gpu 2

    # Enable W&B (online sync to wandb.ai; run ``wandb login`` once first)
    python main_train.py --config conf/experiments/xlsr_mert_add.yaml \\
        --wandb_project my-project --wandb_run_name run-01 --wandb_mode online

    # Local-only W&B logs
    python main_train.py --config conf/experiments/xlsr_mert_add.yaml --wandb_mode offline

    # Default is ``disabled`` (no W&B); same spirit as ``multi_main_train.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_range(value: float, lo: float, hi: float, name: str) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value!r}")


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _validate_choice(value: str, choices: tuple[str, ...], name: str) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """AT-ADD Track-1 and Track-2 data paths."""

    # Track 1
    atadd_t1_train_audio: str = "your_path_to_dataset/atadd/T1/train"
    atadd_t1_train_label: str = "your_path_to_dataset/atadd/T1/label/train.csv"
    atadd_t1_dev_audio:   str = "your_path_to_dataset/atadd/T1/dev"
    atadd_t1_dev_label:   str = "your_path_to_dataset/atadd/T1/label/dev.csv"
    atadd_t1_eval_audio:  str = "your_path_to_dataset/atadd/T1/eval"

    # Track 2
    atadd_t2_train_audio: str = "your_path_to_dataset/atadd/T2/train"
    atadd_t2_train_label: str = "your_path_to_dataset/atadd/T2/label/train.csv"
    atadd_t2_dev_audio:   str = "your_path_to_dataset/atadd/T2/dev"
    atadd_t2_dev_label:   str = "your_path_to_dataset/atadd/T2/label/dev.csv"
    atadd_t2_eval_audio:  str = "your_path_to_dataset/atadd/T2/eval"


@dataclass
class SSLConfig:
    """Paths to pre-trained SSL model directories / checkpoints."""

    xlsr:  str = "your_path_to_huggingface/wav2vec2-xls-r-300m"
    wavlm: str = "your_path_to_huggingface/wavlm-large"
    mert:  str = "your_path_to_huggingface/MERT-v1-330M"
    beats: str = "your_path_to_huggingface/BEATs_iter3"
    clap:  str = "your_path_to_huggingface/larger_clap_music_and_speech"


@dataclass
class AugConfig:
    """Per-type audio augmentation settings (training set only)."""

    aug_speech:  float = 0.0   # P(apply speech aug | speech sample)
    aug_sound:   float = 0.0   # P(RawBoost algo-5 | sound sample)
    aug_singing: float = 0.0   # P(RawBoost algo-5 | singing sample)
    aug_music:   float = 0.0   # P(music aug | music sample)

    speech_aug_method: str = "none"
    """Speech branch when ``aug_speech`` > 0: ``none`` | ``rawboost`` | ``musan``
    | ``audio_augmentor`` | ``noise`` (AWGN vs SNR) | ``codec`` (ffmpeg round-trip)."""

    speech_rawboost_algo: int = 5
    """RawBoost ``process_Rawboost_feature`` algo id (1–9) for speech when method is ``rawboost``."""

    musan_path: str = "your_path/musan"
    """MUSAN root for ``musan`` or ``audio_augmentor`` speech augmentation."""

    rir_path: str = "your_path/RIRS_NOISES"
    """OpenSR/RIRS-style tree for ``audio_augmentor`` (room impulse responses)."""

    music_aug_method: str = "spec_augment"
    """``pitch_shift``: ±1–3 semitones;  ``spec_augment``: frequency-band masking."""

    def __post_init__(self) -> None:
        for name in ("aug_speech", "aug_sound", "aug_singing", "aug_music"):
            _validate_range(getattr(self, name), 0.0, 1.0, name)
        sm = self.speech_aug_method.strip().lower()
        _validate_choice(
            sm,
            ("none", "rawboost", "musan", "audio_augmentor", "noise", "codec"),
            "speech_aug_method",
        )
        object.__setattr__(self, "speech_aug_method", sm)
        if not (1 <= int(self.speech_rawboost_algo) <= 9):
            raise ValueError(
                f"speech_rawboost_algo must be in [1, 9], got {self.speech_rawboost_algo!r}"
            )
        _validate_choice(
            self.music_aug_method,
            ("pitch_shift", "spec_augment"),
            "music_aug_method",
        )


@dataclass
class PromptConfig:
    """Prompt-tuning settings (``pt-*`` / ``wpt-*`` model families only)."""

    prompt_dim:         int   = 1024
    num_prompt_tokens:  int   = 10
    pt_dropout:         float = 0.1
    num_wavelet_tokens: int   = 4   # extra wavelet-domain tokens for wpt-* models

    def __post_init__(self) -> None:
        _validate_positive(self.prompt_dim, "prompt_dim")
        _validate_positive(self.num_prompt_tokens, "num_prompt_tokens")
        _validate_range(self.pt_dropout, 0.0, 1.0, "pt_dropout")
        _validate_positive(self.num_wavelet_tokens, "num_wavelet_tokens")


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

_VALID_TRAIN_TASKS = ("atadd-track1", "atadd-track2")
_VALID_LOSSES      = ("ce", "bce")
_VALID_SAVE_BY     = ("loss", "eer", "f1")
_VALID_EVAL_THRESHOLD_MODES = ("fixed", "eer")
_VALID_FUSIONS     = (
    "cat_linear", "gated", "cross_attn", "film", "type_aware", "proj512_cat", "add"
)


@dataclass
class ATADDConfig:
    """
    Full training configuration for the AT-ADD countermeasure system.

    Instantiate directly for defaults, or use :meth:`from_yaml` to load from a
    YAML file.  Call :meth:`to_namespace` to obtain the flat
    ``argparse.Namespace`` expected by the rest of the codebase.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    seed: int = 1234
    gpu:  str = "0"

    # ── Task & output ────────────────────────────────────────────────────────
    train_task: str = "atadd-track2"
    out_fold:   str = "./ckpt_t2/try"

    # ── Model ────────────────────────────────────────────────────────────────
    model:     str   = "ft-xlsrmertaasist"
    fusion:    str   = "cat_linear"
    audio_len: int   = 64600   # 4 s at 16 kHz
    filter_types:     Optional[str] = None
    """Comma-separated audio types for train AND dev (speech,sound,music,singing).
    ``None`` means all types."""
    type_loss_weight: float = 0.1
    """Weight of the auxiliary type-classification loss (fusion=type_aware only)."""

    assist_project_choice: int = 0
    """AASIST input projector: ``0`` Linear (default), ``1`` MLP, ``2`` GRKAN."""

    # ── Training hyperparameters ─────────────────────────────────────────────
    num_epochs:        int   = 20
    batch_size:        int   = 24
    lr:                float = 0.000001
    lr_decay:          float = 0.5    # LR multiplied by lr_decay every `interval` epochs
    interval:          int   = 4      # epoch interval for LR decay
    beta_1:            float = 0.9
    beta_2:            float = 0.999
    eps:               float = 1e-8
    num_workers:       int   = 8
    base_loss:         str   = "ce"
    save_best_by:      str   = "f1"
    """Metric used to rank checkpoints: ``loss`` (lower is better), ``eer`` (lower), ``f1`` (higher)."""
    continue_training: bool  = False

    eval_threshold_mode: str = "fixed"
    """How to turn scores into hard predictions for F1: ``fixed`` uses ``score_threshold``; ``eer`` uses the EER operating threshold on the current eval split."""
    score_threshold: float = 0.5
    """Score cutoff when ``eval_threshold_mode`` is ``fixed`` (real if score >= threshold, else fake)."""

    # ── Sharpness-aware optimisation ─────────────────────────────────────────
    SAM:  bool = False
    ASAM: bool = False
    CSAM: bool = False

    # ── AMP ──────────────────────────────────────────────────────────────────
    amp: bool = True

    # ── Evaluation schedule ──────────────────────────────────────────────────
    log_dir:            Optional[str] = None
    """Override log directory.  Defaults to ``<out_fold>/logs`` at runtime."""
    eval_steps:         int = 200     # sample-dev eval every N steps (0 = epoch-end only)
    full_eval_steps:    int = 2000    # full-dev eval every N steps (0 = disabled)
    eval_warmup_steps:  int = 2000    # skip all evals for the first N global steps
    patience:           int = 0       # early-stopping patience (sample-dev evals); 0 = off

    # ── Sub-configs ──────────────────────────────────────────────────────────
    data:   DataConfig   = field(default_factory=DataConfig)
    ssl:    SSLConfig    = field(default_factory=SSLConfig)
    aug:    AugConfig    = field(default_factory=AugConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    # ── Validation ───────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        _validate_choice(self.train_task,   _VALID_TRAIN_TASKS, "train_task")
        _validate_choice(self.base_loss,    _VALID_LOSSES,      "base_loss")
        _validate_choice(self.save_best_by, _VALID_SAVE_BY,     "save_best_by")
        _validate_choice(self.fusion,       _VALID_FUSIONS,     "fusion")
        _validate_choice(
            self.eval_threshold_mode, _VALID_EVAL_THRESHOLD_MODES, "eval_threshold_mode"
        )
        if self.assist_project_choice not in (0, 1, 2):
            raise ValueError(
                "assist_project_choice must be 0, 1, or 2, "
                f"got {self.assist_project_choice!r}"
            )

        _validate_positive(self.audio_len,  "audio_len")
        _validate_positive(self.num_epochs, "num_epochs")
        _validate_positive(self.batch_size, "batch_size")
        _validate_positive(self.num_workers, "num_workers")
        _validate_range(self.lr,            1e-9, 1.0,  "lr")
        _validate_range(self.lr_decay,      0.0,  1.0,  "lr_decay")
        _validate_range(self.beta_1,        0.0,  1.0,  "beta_1")
        _validate_range(self.beta_2,        0.0,  1.0,  "beta_2")
        _validate_range(self.eps,           0.0,  1e-4, "eps")
        _validate_range(self.patience,      0,    10000, "patience")
        _validate_range(self.eval_steps,      0, 10**7, "eval_steps")
        _validate_range(self.full_eval_steps, 0, 10**7, "full_eval_steps")
        _validate_range(self.score_threshold, 0.0, 1.0, "score_threshold")

    # ── I/O ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "ATADDConfig":
        """Load an ``ATADDConfig`` from a YAML file.

        The YAML may use a nested layout::

            model: ft-xlsrmertaasist
            data:
              atadd_t2_train_audio: /path/train
            ssl:
              xlsr: /path/xlsr
            aug:
              aug_speech: 0.2
            prompt:
              prompt_dim: 512

        Top-level keys that correspond to W&B settings (``no_wandb``,
        ``wandb_project``, etc.) are silently ignored—those belong on the CLI.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for --config support.  "
                "Install with:  pip install pyyaml"
            ) from exc

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path!r}")

        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML config must be a top-level mapping, got {type(raw).__name__!r}"
            )

        # Extract sub-config sections
        data_cfg   = DataConfig(**raw.pop("data",   {}))
        ssl_cfg    = SSLConfig(**raw.pop("ssl",    {}))
        aug_cfg    = AugConfig(**raw.pop("aug",    {}))
        prompt_cfg = PromptConfig(**raw.pop("prompt", {}))

        # Drop W&B / meta keys that were valid in the old flat YAML format
        _wb_keys = {
            "no_wandb", "wandb_project", "wandb_run_name", "wandb_entity", "wandb_mode",
        }
        ignored = _wb_keys & raw.keys()
        if ignored:
            warnings.warn(
                f"W&B keys in YAML are ignored (use CLI flags instead): {sorted(ignored)}",
                stacklevel=2,
            )
            for k in ignored:
                raw.pop(k)

        # Warn about any truly unknown keys
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            warnings.warn(
                f"Unknown keys in YAML config (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
            for k in unknown:
                raw.pop(k)

        return cls(**raw, data=data_cfg, ssl=ssl_cfg, aug=aug_cfg, prompt=prompt_cfg)

    def to_dict(self) -> Dict[str, Any]:
        """Return a nested dict representation (suitable for YAML serialisation)."""
        return dataclasses.asdict(self)

    def save_to_yaml(self, path: str) -> None:
        """Serialise this config to a YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pip install pyyaml") from exc

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(), f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                sort_keys=False,
            )

    def to_namespace(self) -> argparse.Namespace:
        """
        Return a flat ``argparse.Namespace`` with all fields from this config
        and its sub-configs.

        Nested sub-config dicts are unpacked one level so that callers can do
        ``args.xlsr``, ``args.aug_speech``, etc. as before.
        """
        d: Dict[str, Any] = {}
        for key, value in dataclasses.asdict(self).items():
            if isinstance(value, dict):
                d.update(value)  # flatten DataConfig, SSLConfig, AugConfig, PromptConfig
            else:
                d[key] = value
        return argparse.Namespace(**d)

    def __str__(self) -> str:
        lines = ["ATADDConfig:"]
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if dataclasses.is_dataclass(val):
                lines.append(f"  {f.name}:")
                for sf in dataclasses.fields(val):
                    lines.append(f"    {sf.name}: {getattr(val, sf.name)}")
            else:
                lines.append(f"  {f.name}: {val}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (W&B + meta flags only)
# ---------------------------------------------------------------------------

def _wb_parser() -> argparse.ArgumentParser:
    """Argparse for W&B settings and a few meta flags (gpu, config path, resume)."""
    p = argparse.ArgumentParser(
        description="AT-ADD countermeasure training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Path to a YAML config file (all non-W&B training settings).",
    )
    p.add_argument(
        "--resume", type=str, default=None, metavar="OUT_FOLD",
        help=(
            "Resume training from a previous run directory.  "
            "Loads <OUT_FOLD>/config.yaml for the full config and "
            "<OUT_FOLD>/checkpoint/latest.pt for model + optimiser state.  "
            "Implies --continue_training."
        ),
    )
    p.add_argument(
        "--gpu", type=str, default=None,
        help="Override the gpu field from YAML / default.",
    )
    p.add_argument("--wandb_project",  type=str, default="AT-ADD-Baseline")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="W&B run name (default: basename of out_fold).")
    p.add_argument("--wandb_entity",   type=str, default=None)
    p.add_argument(
        "--wandb_mode", type=str, default="disabled",
        choices=("online", "offline", "disabled"),
        help=(
            "W&B mode: ``online`` syncs to wandb.ai; ``offline`` writes local runs only; "
            "``disabled`` skips ``wandb.init`` entirely (default)."
        ),
    )
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging.")
    return p


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def initParams(argv=None) -> argparse.Namespace:
    """
    Load training configuration and return a flat ``argparse.Namespace``.

    Sources (highest priority first):

    1. ``--gpu`` CLI flag
    2. YAML config file (``--config PATH``)
    3. ``ATADDConfig`` field defaults

    W&B settings (``--wandb_project``, ``--wandb_run_name``, ``--wandb_entity``,
    ``--wandb_mode``, ``--no_wandb``) are always taken from the CLI.

    Parameters
    ----------
    argv:
        Argument list to parse.  ``None`` reads from ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Flat namespace containing all ``ATADDConfig`` fields (sub-config fields
        unpacked to the top level) plus the four W&B fields.
    """
    cli = _wb_parser().parse_args(argv)

    if cli.resume is not None:
        # Resume mode: load config from the previous run directory.
        config_yaml = os.path.join(cli.resume, "config.yaml")
        if not os.path.isfile(config_yaml):
            raise FileNotFoundError(
                f"--resume: config.yaml not found in {cli.resume!r}"
            )
        cfg = ATADDConfig.from_yaml(config_yaml)
        cfg.continue_training = True
        cfg.out_fold = cli.resume   # handle directories that were moved
    elif cli.config is not None:
        cfg = ATADDConfig.from_yaml(cli.config)
    else:
        cfg = ATADDConfig()

    # CLI gpu flag overrides whatever is in the YAML / default
    if cli.gpu is not None:
        cfg.gpu = cli.gpu

    # Flatten config to namespace and inject W&B CLI fields
    args = cfg.to_namespace()
    args._config = cfg   # preserved for YAML saving in main_train.initParams()
    args.wandb_project  = cli.wandb_project
    args.wandb_run_name = cli.wandb_run_name
    args.wandb_entity   = cli.wandb_entity
    args.wandb_mode     = cli.wandb_mode
    args.no_wandb       = cli.no_wandb

    return args
