import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
from pytorch_model_summary import summary
from typing import Union, Optional, Callable
from feature_extraction import *
import torchvision.models as models
import torchaudio
from dataclasses import dataclass
from exp.feature_extraction_exp import *


# =============================================================================
# AASIST building blocks
# =============================================================================

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # directional edge for master node
        master = self._update_master(x, master)
        # projection
        x = self._project(x, att_map)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


# =============================================================================
# SSLAASIST backend  
# =============================================================================

class SSLAASIST(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()

        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.LL = nn.Linear(in_dim, 128)

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):

        x = x.squeeze(dim=1)

        x = self.LL(x)
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output


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

        xlsr_g, wavlm_g  = mean-pool over T
        type_logits       = MLP( [xlsr_g ; wavlm_g] )      (B, num_types)
        type_probs         = softmax(type_logits)            (B, num_types)
        weights            = type_probs @ softmax(W_type)    (B, 2)
        fused              = w0·proj_x(x)  +  w1·proj_y(y)  (B, T, out_dim)

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
        # Utterance-level summary for type prediction.
        xlsr_g  = x.mean(dim=1)   # (B, dim_x)
        wavlm_g = y.mean(dim=1)   # (B, dim_y)

        type_logits = self.type_classifier(
            torch.cat([xlsr_g, wavlm_g], dim=-1)
        )  # (B, num_types)
        type_probs = F.softmax(type_logits, dim=-1)  # (B, num_types)

        # Blend weights per sample via soft type assignment.
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


_FUSION_CLASSES = {
    "cat_linear":  CatLinearFusion,
    "gated":       GatedFusion,
    "cross_attn":  CrossAttentionFusion,
    "film":        FiLMFusion,
    "type_aware":  TypeAwareFusion,
    "proj512_cat": ProjCatFusion,
    "add":         AddFusion,
}


def build_fusion_module(name: str, dim_x: int, dim_y: int, out_dim: int) -> nn.Module:
    if name not in _FUSION_CLASSES:
        raise ValueError(
            f"Unknown fusion method '{name}'. "
            f"Choose from: {list(_FUSION_CLASSES.keys())}"
        )
    return _FUSION_CLASSES[name](dim_x, dim_y, out_dim)


# =============================================================================
# Feature-alignment helpers for dual-SSL models
# =============================================================================

def _clap_preprocess(features: torch.Tensor) -> torch.Tensor:
    """CLAP raw output [B, C, F, T] → frame-level sequence [B, T, C]."""
    return features.mean(dim=2).permute(0, 2, 1)


def _align_clap_to_xlsr(xlsr_feat: torch.Tensor, clap_raw: torch.Tensor):
    """
    Preprocess CLAP [B, C, F, T] → [B, T_clap, C], then interpolate
    T_clap → T_xlsr to match the XLSR frame sequence.

    Returns (xlsr_feat, clap_aligned), both [B, T_xlsr, 1024].
    """
    clap_feat = clap_raw.mean(dim=2).permute(0, 2, 1)           # [B, T_clap, 1024]
    T = xlsr_feat.size(1)
    clap_aligned = F.interpolate(
        clap_feat.permute(0, 2, 1),                              # [B, 1024, T_clap]
        size=T, mode='linear', align_corners=False,
    ).permute(0, 2, 1)                                           # [B, T_xlsr, 1024]
    return xlsr_feat, clap_aligned


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
        backend (nn.Module): Sequence classifier, e.g. ``SSLAASIST``.
        feat_preprocess (callable, optional): Applied to frontend output
            *before* the backend, e.g. ``_clap_preprocess`` for CLAP.
        visual (bool): When True, ``forward`` returns
            ``(hidden, logits, attn_weights)`` for analysis.

    Example::

        model = SingleSSLModel(
            frontend=XLSR(model_dir="...", freeze=False),
            backend=SSLAASIST(in_dim=1024),
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
        backend (nn.Module): Sequence classifier, e.g. ``SSLAASIST``.
        feat_align_fn (callable, optional):
            ``(feat_a, feat_b) -> (feat_a, feat_b)`` applied *before* fusion.
            Use ``_align_clap_to_xlsr`` when one encoder outputs a 4-D map.
        visual (bool): When True, ``forward`` returns
            ``(hidden, logits, attn_weights)`` for analysis.

    Example::

        model = DualSSLModel(
            frontend_a=XLSR(model_dir="...", freeze=False),
            frontend_b=MERT(model_dir="...", freeze=False),
            fusion_module=build_fusion_module('cat_linear', 1024, 1024, 1024),
            backend=SSLAASIST(in_dim=1024),
        )
    """

    def __init__(
        self,
        frontend_a: nn.Module,
        frontend_b: nn.Module,
        fusion_module: nn.Module,
        backend: nn.Module,
        feat_align_fn: Optional[Callable] = None,
        visual: bool = False,
    ):
        super().__init__()
        self.frontend_a = frontend_a
        self.frontend_b = frontend_b
        self.fusion_module = fusion_module
        self.backend = backend
        self.feat_align_fn = feat_align_fn
        self.visual = visual
        self._last_type_logits = None

    def _align_and_fuse(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        if self.feat_align_fn is not None:
            feat_a, feat_b = self.feat_align_fn(feat_a, feat_b)
        else:
            # Default: truncate to the shorter sequence length
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
# Standalone models with non-standard interfaces
# =============================================================================

class ResNet18ForAudio(nn.Module):
    """Mel-spectrogram + ResNet-18 baseline (conventional CM, no SSL)."""

    def __init__(self, enc_dim=256, nclasses=2):
        super(ResNet18ForAudio, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.resnet18.fc = nn.Identity()

        self.fc = nn.Linear(512, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=512, power=2, normalized=True)

        self.initialize_params()

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.spec(x.cuda().float()).unsqueeze(dim=1)
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        feat = self.fc(x)
        mu = self.fc_mu(feat)
        return feat, mu


class SLS(nn.Module):
    """Layer-weighted SSL backend (requires the full hidden-state sequence)."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def getAttenF(self, layerResult):  # layerresult = [24] (B, Frame, Dim)
        poollayerResult = []
        fullf = []
        for layer in layerResult:
            layery = layer.transpose(1, 2)  # (B, Frame, Dim) -> (B, Dim, Frame)
            layery = F.adaptive_avg_pool1d(layery, 1)
            layery = layery.transpose(1, 2)
            poollayerResult.append(layery)

            x = layer  # (B, Frame, Dim)
            x = x.view(x.size(0), -1, x.size(1), x.size(2))
            fullf.append(x)

        layery = torch.cat(poollayerResult, dim=1)
        fullfeature = torch.cat(fullf, dim=1)
        return layery, fullfeature

    def forward(self, layerResult):  # layerresult = [25] (B, Frame, Dim)
        layerResult = layerResult[1:]  # skip embedding_output
        y0, fullfeature = self.getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)
        return x, output


class XLSR_SLS(nn.Module):
    """
    XLSR with layer-wise SLS backend.

    This model uses the full hidden-state sequence from all transformer layers
    (not just the last-layer output), so it cannot use the standard
    ``SingleSSLModel`` interface.
    """

    def __init__(self, model_dir, device='cuda', freeze=True, visual=False):
        super().__init__()
        self.wav2vec2 = XLSR(
            model_dir=model_dir,
            device=device,
            freeze=freeze,
            visual=visual,
            return_hidden_states=True,
        )
        self.sls = SLS(device=device)
        self.visual = visual

    def forward(self, audio_data):
        if self.visual:
            features, attention_weights, hidden_states = self.wav2vec2.extract_features(audio_data)
            last_hidden, output = self.sls(hidden_states)
            return last_hidden, output, attention_weights

        features, hidden_states = self.wav2vec2.extract_features(audio_data)
        last_hidden, output = self.sls(hidden_states)
        return last_hidden, output

    def train(self, mode=True):
        if mode:
            self.sls.train(mode)
        else:
            self.sls.eval()

    def eval(self):
        self.sls.eval()
        self.wav2vec2.eval()


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
                backend=SSLAASIST(in_dim=1024),
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
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('fr-wavlmaasist')
def _build_fr_wavlmaasist(args):
    return SingleSSLModel(
        frontend=WAVLM(model_dir=args.wavlm, device=_dev(args), freeze=True),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('fr-mertaasist')
def _build_fr_mertaasist(args):
    return SingleSSLModel(
        frontend=MERT(model_dir=args.mert, device=_dev(args), freeze=True),
        backend=SSLAASIST(in_dim=1024),
    )


# ── Fine-tuned (FT) single-SSL + AASIST ──────────────────────────────────────

@register_model('ft-w2v2aasist')
def _build_ft_w2v2aasist(args):
    return SingleSSLModel(
        frontend=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('ft-wavlmaasist')
def _build_ft_wavlmaasist(args):
    return SingleSSLModel(
        frontend=WAVLM(model_dir=args.wavlm, device=_dev(args), freeze=False),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('ft-mertaasist')
def _build_ft_mertaasist(args):
    return SingleSSLModel(
        frontend=MERT(model_dir=args.mert, device=_dev(args), freeze=False),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('ft-beats_aasist')
def _build_ft_beats_aasist(args):
    # BEATs encoder stays frozen; only the AASIST head is fine-tuned
    return SingleSSLModel(
        frontend=BEATs(model_dir=args.beats, device=_dev(args), freeze=False),
        backend=SSLAASIST(in_dim=768),
    )


@register_model('ft-clap_aasist')
def _build_ft_clap_aasist(args):
    return SingleSSLModel(
        frontend=CLAP(model_dir=args.clap, device=_dev(args), freeze=False),
        backend=SSLAASIST(in_dim=1024),
        feat_preprocess=_clap_preprocess,
    )


@register_model('ft-xlsr_sls')
def _build_ft_xlsr_sls(args):
    return XLSR_SLS(model_dir=args.xlsr, device=_dev(args), freeze=False)


# ── Fine-tuned (FT) dual-SSL + AASIST ────────────────────────────────────────

@register_model('ft-xlsrwavlmaasist')
def _build_ft_xlsrwavlmaasist(args):
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=WAVLM(model_dir=args.wavlm, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 1024, 1024),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('ft-xlsrbeats_aasist')
def _build_ft_xlsrbeats_aasist(args):
    # BEATs output is 768-d; CatLinear(1024+768→1024) is the only valid fusion here.
    # The --fusion argument is not applicable for this combination.
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=BEATs(model_dir=args.beats, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 768, 1024),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('ft-xlsrmertaasist')
def _build_ft_xlsrmertaasist(args):
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=MERT(model_dir=args.mert, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 1024, 1024),
        backend=SSLAASIST(in_dim=1024),
    )


@register_model('ft-xlsrclapaasist')
def _build_ft_xlsrclapaasist(args):
    # CLAP outputs [B,C,F,T]; _align_clap_to_xlsr preprocesses + interpolates.
    return DualSSLModel(
        frontend_a=XLSR(model_dir=args.xlsr, device=_dev(args), freeze=False),
        frontend_b=CLAP(model_dir=args.clap, device=_dev(args), freeze=False),
        fusion_module=build_fusion_module(_fusion(args), 1024, 1024, 1024),
        backend=SSLAASIST(in_dim=1024),
        feat_align_fn=_align_clap_to_xlsr,
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
        backend=SSLAASIST(in_dim=1024),
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
        backend=SSLAASIST(in_dim=1024),
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
        backend=SSLAASIST(in_dim=1024),
    )


# ── Wavelet Prompt-tuned (WPT) SSL + AASIST ──────────────────────────────────

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
        backend=SSLAASIST(in_dim=1024),
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
        backend=SSLAASIST(in_dim=1024),
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
        backend=SSLAASIST(in_dim=1024),
    )
