import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Feature-fusion modules for dual-SSL models
# =============================================================================

class CatLinearFusion(nn.Module):
    """Concatenate both feature streams then project: [x ; y] -> Linear(out_dim)."""

    def __init__(self, dim_x: int, dim_y: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim_x + dim_y, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([x, y], dim=-1))


class GatedFusion(nn.Module):
    """
    Soft-gating fusion.

    A gate vector g ∈ (0,1)^out_dim is derived from the concatenation of both
    streams and used to interpolate their individual projections:
        g      = σ( W_gate · [x ; y] )
        output = g ⊙ proj_x(x)  +  (1−g) ⊙ proj_y(y)
    """

    def __init__(self, dim_x: int, dim_y: int, out_dim: int):
        super().__init__()
        self.proj_x = nn.Linear(dim_x, out_dim)
        self.proj_y = nn.Linear(dim_y, out_dim)
        self.gate   = nn.Linear(dim_x + dim_y, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x, y], dim=-1)))
        return g * self.proj_x(x) + (1.0 - g) * self.proj_y(y)


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion.

    Each stream attends to the other via a separate MultiheadAttention layer,
    the attended representations are combined with a residual connection, and
    the two normalised streams are concatenated then projected:
        x' = LayerNorm( x_proj  +  MHA(Q=x_proj,  K=y_proj,  V=y_proj) )
        y' = LayerNorm( y_proj  +  MHA(Q=y_proj,  K=x_proj,  V=x_proj) )
        output = Linear( [x' ; y'] )
    """

    def __init__(self, dim_x: int, dim_y: int, out_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj_x    = nn.Linear(dim_x, out_dim) if dim_x != out_dim else nn.Identity()
        self.proj_y    = nn.Linear(dim_y, out_dim) if dim_y != out_dim else nn.Identity()
        self.cross_x2y = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_y2x = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_x    = nn.LayerNorm(out_dim)
        self.norm_y    = nn.LayerNorm(out_dim)
        self.proj_out  = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.proj_x(x)
        y = self.proj_y(y)
        attended_x, _ = self.cross_x2y(query=x, key=y, value=y)
        attended_y, _ = self.cross_y2x(query=y, key=x, value=x)
        x = self.norm_x(x + attended_x)
        y = self.norm_y(y + attended_y)
        return self.proj_out(torch.cat([x, y], dim=-1))


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) fusion.

    Stream y acts as the conditioning signal that generates per-channel scale
    (γ) and shift (β) parameters to modulate stream x:
        px          = proj_x(x)
        γ, β        = chunk( W_film · y )
        modulated   = sigmoid(γ) ⊙ px  +  β
        output      = LayerNorm( modulated + proj_y(y) )
    """

    def __init__(self, dim_x: int, dim_y: int, out_dim: int):
        super().__init__()
        self.proj_x   = nn.Linear(dim_x, out_dim)
        self.film_gen = nn.Linear(dim_y, out_dim * 2)
        self.proj_y   = nn.Linear(dim_y, out_dim)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        px = self.proj_x(x)
        gamma, beta = self.film_gen(y).chunk(2, dim=-1)
        modulated = torch.sigmoid(gamma) * px + beta
        return self.norm(modulated + self.proj_y(y))


class TypeAwareFusion(nn.Module):
    """
    Type-aware dynamic fusion with auxiliary classification loss.

    An utterance-level type classifier predicts the audio category
    (speech / sound / music / singing) from global-pooled features of both
    streams.  Per-type learnable weights then produce a soft convex
    combination of the two individually projected streams:

        xlsr_g, mert_g  = mean-pool over T
        type_logits      = MLP( [xlsr_g ; mert_g] )     (B, num_types)
        type_probs        = softmax(type_logits)          (B, num_types)
        weights           = type_probs @ softmax(W_type)  (B, 2)
        fused             = w0·proj_x(x)  +  w1·proj_y(y) (B, T, out_dim)

    Returns ``(fused, type_logits)`` so the caller can compute an auxiliary
    cross-entropy loss on the type prediction side-task.

    Type index mapping (must match dataset.py):
        0 = speech,  1 = sound,  2 = music,  3 = singing
    """

    NUM_TYPES = 4

    def __init__(self, dim_x: int, dim_y: int, out_dim: int):
        super().__init__()
        self.proj_x = nn.Linear(dim_x, out_dim)
        self.proj_y = nn.Linear(dim_y, out_dim)

        self.type_classifier = nn.Sequential(
            nn.Linear(dim_x + dim_y, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.NUM_TYPES),
        )

        # Per-type raw fusion logits (before softmax): shape (num_types, 2).
        # Initialised to zero so both streams are weighted equally at the start.
        self.type_fusion_weights = nn.Parameter(torch.zeros(self.NUM_TYPES, 2))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xlsr_g  = x.mean(dim=1)   # (B, dim_x)
        mert_g  = y.mean(dim=1)   # (B, dim_y)

        type_logits = self.type_classifier(
            torch.cat([xlsr_g, mert_g], dim=-1)
        )  # (B, num_types)
        type_probs = F.softmax(type_logits, dim=-1)  # (B, num_types)

        # (B, num_types) @ (num_types, 2) -> (B, 2),  rows sum to 1.
        weights = torch.matmul(
            type_probs,
            F.softmax(self.type_fusion_weights, dim=-1),
        )  # (B, 2)

        w_x = weights[:, 0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        w_y = weights[:, 1].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

        fused = w_x * self.proj_x(x) + w_y * self.proj_y(y)  # (B, T, out_dim)
        return fused, type_logits


class ProjCatFusion(nn.Module):
    """
    Project-then-concatenate fusion (ablation baseline).

    Each stream is projected to out_dim//2 independently, then the two
    half-dim representations are concatenated to recover out_dim:
        output = [ Linear(dim_x, out_dim//2)(x) ; Linear(dim_y, out_dim//2)(y) ]

    Unlike CatLinearFusion there is NO joint linear after the concat, so the
    two encoder spaces are kept strictly separated up to the AASIST head.
    """

    def __init__(self, dim_x: int, dim_y: int, out_dim: int):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even for ProjCatFusion"
        half = out_dim // 2
        self.proj_x = nn.Linear(dim_x, half)
        self.proj_y = nn.Linear(dim_y, half)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.proj_x(x), self.proj_y(y)], dim=-1)


class AddFusion(nn.Module):
    """
    Element-wise addition fusion (ablation baseline).

    Both streams are summed directly without any learned projection:
        output = x + y

    Zero additional parameters; requires dim_x == dim_y == out_dim.
    """

    def __init__(self, dim_x: int, dim_y: int, out_dim: int):
        super().__init__()
        assert dim_x == dim_y == out_dim, (
            f"AddFusion requires dim_x == dim_y == out_dim, "
            f"got {dim_x}, {dim_y}, {out_dim}"
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


# Registry — must be defined AFTER all fusion classes above.
_FUSION_REGISTRY: dict = {
    "cat_linear":  CatLinearFusion,
    "gated":       GatedFusion,
    "cross_attn":  CrossAttentionFusion,
    "film":        FiLMFusion,
    "type_aware":  TypeAwareFusion,
    "proj512_cat": ProjCatFusion,
    "add":         AddFusion,
}


def build_fusion_module(name: str, dim_x: int, dim_y: int, out_dim: int) -> nn.Module:
    """
    Instantiate a fusion module by name.

    Args:
        name:    One of the keys in ``_FUSION_REGISTRY``.
        dim_x:   Feature dimension of the first (primary) stream.
        dim_y:   Feature dimension of the second stream.
        out_dim: Output feature dimension passed to the backend.

    Raises:
        ValueError: If ``name`` is not a registered fusion method.
    """
    if name not in _FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion method '{name}'. "
            f"Choose from: {list(_FUSION_REGISTRY.keys())}"
        )
    return _FUSION_REGISTRY[name](dim_x, dim_y, out_dim)
