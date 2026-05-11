import os
import random

import numpy as np
import torch
from distutils import util


_VALID_AUDIO_TYPES = ("speech", "sound", "music", "singing")


def str2bool(v: str) -> bool:
    return bool(util.strtobool(v))


def parse_filter_types(s) -> "frozenset | None":
    """Parse a comma-separated audio-type filter string.

    Returns ``None`` (= use all types) or a frozenset of lowercase type names.
    Raises ``ValueError`` on unrecognised type names.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    unknown = sorted(set(parts) - set(_VALID_AUDIO_TYPES))
    if unknown:
        raise ValueError(
            f"Invalid filter_types entries: {unknown}. "
            f"Use comma-separated subset of: {', '.join(_VALID_AUDIO_TYPES)}"
        )
    return frozenset(parts)


def setup_seed(random_seed: int, cudnn_deterministic: bool = True) -> None:
    """Set all random seeds for reproducible experiments."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # Must seed both CPU and CUDA generators; CPU covers model init, dropout,
    # SubsetRandomSampler, etc.
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False
