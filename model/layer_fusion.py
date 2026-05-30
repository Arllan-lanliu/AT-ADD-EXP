"""
Shared multi-layer intermediate feature fusion for SSL front-ends (e.g. XLSR, BEATs).

Supports: last, cat_linear, cat_proj_v1, cat_proj_v2, mean, weight_sum, mhfa_fuse, mhfa_pool.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_layer_fusion(layer_fusion: str) -> str:
    lf = str(layer_fusion).strip().lower()
    if lf == "cat":
        lf = "cat_proj_v2"
    elif lf == "cat_proj":
        lf = "cat_proj_v2"
    return lf


MULTI_LAYER_FUSIONS = frozenset(
    {
        "cat_linear",
        "cat_proj_v1",
        "cat_proj_v2",
        "mean",
        "weight_sum",
        "mhfa_fuse",
        "mhfa_pool",
    }
)


def validate_layer_fusion_config(
    layer_fusion: str,
    selected_layers: Optional[Sequence[int]],
    *,
    backend_name: str,
) -> None:
    """Raise ValueError if layer_fusion / selected_layers are inconsistent."""
    lf = layer_fusion
    if lf in MULTI_LAYER_FUSIONS:
        if not selected_layers:
            raise ValueError(
                f"{backend_name}: selected_layers must be a non-empty sequence when "
                f"layer_fusion={lf!r}"
            )
    elif lf == "last" and selected_layers is not None:
        if len(selected_layers) > 1:
            raise ValueError(
                f"{backend_name}: layer_fusion='last' accepts at most one "
                f"selected_layers index; got {tuple(selected_layers)!r}"
            )
    elif lf != "last":
        raise ValueError(
            f"Unsupported {backend_name} layer_fusion={lf!r}. Choose from "
            "['last', 'cat_linear', 'cat_proj_v1', 'cat_proj_v2', 'mean', "
            "'weight_sum', 'mhfa_fuse', 'mhfa_pool']."
        )


def _mhfa_layer_weighted_sums(
    stack: torch.Tensor, w_k: nn.Parameter, w_v: nn.Parameter
) -> Tuple[torch.Tensor, torch.Tensor]:
    """stack: (B, T, D, L); returns K_feat, V_feat (B, T, D) each."""
    wk = F.softmax(w_k, dim=0)
    wv = F.softmax(w_v, dim=0)
    K_feat = (stack * wk).sum(dim=-1)
    V_feat = (stack * wv).sum(dim=-1)
    return K_feat, V_feat


class IntermediateLayerFusion(nn.Module):
    """Learnable fusion head over a list of per-layer tensors (B, T, D)."""

    def __init__(
        self,
        hidden_size: int,
        n_sel: int,
        layer_fusion: str,
        *,
        mhfa_compression_dim: Optional[int] = None,
        mhfa_num_heads: int = 8,
        mhfa_output_dim: Optional[int] = None,
        mhfa_dropout: float = 0.1,
    ):
        super().__init__()
        lf = normalize_layer_fusion(layer_fusion)
        self.layer_fusion = lf
        self.hidden_size = hidden_size
        self.n_sel = int(n_sel)

        self.per_layer_ln = None
        self.layer_proj = None
        self.cat_linear_head = None
        self.layer_weights = None

        self.mhfa_w_k = None
        self.mhfa_w_v = None
        self.mhfa_proj_k = None
        self.mhfa_proj_v = None
        self.mhfa_fuse_out = None
        self.mhfa_query = None
        self.mhfa_head_proj = None
        self.mhfa_num_heads = int(mhfa_num_heads)
        self.mhfa_compression_dim = None
        self.mhfa_output_dim = None

        ns = self.n_sel
        if lf == "cat_linear" and ns > 0:
            self.cat_linear_head = nn.Linear(hidden_size * ns, hidden_size)
        elif lf == "cat_proj_v1" and ns > 0:
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(hidden_size * ns),
                nn.Linear(hidden_size * ns, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        elif lf == "cat_proj_v2" and ns > 0:
            self.per_layer_ln = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in range(ns)]
            )
            self.layer_proj = nn.Sequential(
                nn.LayerNorm(hidden_size * ns),
                nn.Linear(hidden_size * ns, hidden_size),
                nn.Dropout(0.1),
            )
        elif lf == "weight_sum" and ns > 0:
            self.layer_weights = nn.Parameter(torch.zeros(ns))

        elif lf in ("mhfa_fuse", "mhfa_pool") and ns > 0:
            D = hidden_size
            D_cmp = mhfa_compression_dim if mhfa_compression_dim is not None else D // 4

            self.mhfa_w_k = nn.Parameter(torch.zeros(ns))
            self.mhfa_w_v = nn.Parameter(torch.zeros(ns))

            self.mhfa_proj_k = nn.Linear(D, D_cmp, bias=False)
            self.mhfa_proj_v = nn.Linear(D, D_cmp, bias=False)
            self.mhfa_dropout_layer = nn.Dropout(mhfa_dropout)

            if lf == "mhfa_fuse":
                self.mhfa_fuse_out = nn.Sequential(
                    nn.LayerNorm(D_cmp),
                    nn.Linear(D_cmp, D),
                )
            else:
                H = int(mhfa_num_heads)
                D_out = mhfa_output_dim if mhfa_output_dim is not None else D
                self.mhfa_compression_dim = D_cmp
                self.mhfa_output_dim = D_out

                self.mhfa_query = nn.Parameter(torch.randn(H, D_cmp) * (D_cmp ** -0.5))
                self.mhfa_head_proj = nn.Linear(H * D_cmp, D_out)

    def _mhfa_fuse_forward(self, selected: List[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(selected, dim=-1)
        K_feat, V_feat = _mhfa_layer_weighted_sums(stack, self.mhfa_w_k, self.mhfa_w_v)
        K = self.mhfa_proj_k(K_feat)
        V = self.mhfa_proj_v(V_feat)
        gate = torch.sigmoid(K)
        gated = gate * V
        gated = self.mhfa_dropout_layer(gated)
        return self.mhfa_fuse_out(gated)

    def _mhfa_pool_forward(self, selected: List[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(selected, dim=-1)
        K_feat, V_feat = _mhfa_layer_weighted_sums(stack, self.mhfa_w_k, self.mhfa_w_v)
        K = self.mhfa_proj_k(K_feat)
        V = self.mhfa_proj_v(V_feat)

        D_cmp = K.size(-1)
        scores = torch.einsum("btd,hd->bht", K, self.mhfa_query) / (D_cmp ** 0.5)
        attn = F.softmax(scores, dim=-1)

        head_out = torch.einsum("bht,btd->bhd", attn, V)
        head_out = self.mhfa_dropout_layer(head_out)

        B, H, Dc = head_out.shape
        head_out = head_out.reshape(B, H * Dc)
        return self.mhfa_head_proj(head_out)

    def fuse(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        selected_layers: Optional[Tuple[int, ...]],
        *,
        backend_name: str,
    ) -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]]:
        """
        Returns fused representation and the original ``hidden_states`` sequence
        (same container type as passed in; BEATs may pass tuples from the model).

        Notes:
          - ``mhfa_pool`` returns utterance-level (B, D_out), not (B, T, D).
        """
        n_h = len(hidden_states)

        if self.layer_fusion == "last":
            if not selected_layers:
                return last_hidden_state, hidden_states
            if len(selected_layers) > 1:
                raise ValueError(
                    f"{backend_name}: layer_fusion='last' accepts at most one index in "
                    "selected_layers; for multiple layers use cat_linear, cat_proj_v1, "
                    f"cat_proj_v2, mean, weight_sum, mhfa_fuse, or mhfa_pool. Got "
                    f"selected_layers={selected_layers!r}."
                )
            idx = selected_layers[0]
            if idx < 0 or idx >= n_h:
                raise IndexError(
                    f"{backend_name}: selected_layers index {idx} out of range for "
                    f"{n_h} hidden_states (valid 0..{n_h - 1})."
                )
            return hidden_states[idx], hidden_states

        if not selected_layers:
            # Should not validate as valid config; callers guard this.
            return last_hidden_state, hidden_states

        for i in selected_layers:
            if i < 0 or i >= n_h:
                raise IndexError(
                    f"{backend_name}: selected_layers index {i} out of range for "
                    f"{n_h} hidden_states (valid 0..{n_h - 1})."
                )

        selected = [hidden_states[i] for i in selected_layers]
        lf = self.layer_fusion

        if lf == "cat_linear":
            fused = torch.cat(selected, dim=-1)
            fused = self.cat_linear_head(fused)
        elif lf == "cat_proj_v1":
            fused = torch.cat(selected, dim=-1)
            fused = self.layer_proj(fused)
        elif lf == "cat_proj_v2":
            normed = [ln(h) for ln, h in zip(self.per_layer_ln, selected)]
            fused = torch.cat(normed, dim=-1)
            fused = self.layer_proj(fused)
        elif lf == "mean":
            fused = torch.stack(selected, dim=0).mean(dim=0)
        elif lf == "weight_sum":
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * h for w, h in zip(weights, selected))
        elif lf == "mhfa_fuse":
            fused = self._mhfa_fuse_forward(selected)
        elif lf == "mhfa_pool":
            fused = self._mhfa_pool_forward(selected)
        else:
            raise ValueError(
                f"{backend_name}: unsupported layer_fusion={lf!r}. Choose from "
                "['last', 'cat_linear', 'cat_proj_v1', 'cat_proj_v2', 'mean', "
                "'weight_sum', 'mhfa_fuse', 'mhfa_pool']."
            )

        return fused, hidden_states
