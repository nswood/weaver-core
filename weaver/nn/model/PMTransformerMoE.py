''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from weaver.utils.logger import _logger


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y

# Import PM layers
import os
os.environ['PYTHONPATH'] = '/n/home11/nswood/weaver-core/weaver/nn/model'

from weaver.nn.model.PM_utils import *
from weaver.nn.model.riemannian_batch_norm import *

def two_point_mid(x1,x2, man, w1,w2):
    if man.name == 'Euclidean':
        mid = (x1 * w1 + x2 * w2)/(w1 + w2)
    else:
        lam_x1 = man.lambda_x(x1).unsqueeze(-1)
        lam_x2 = man.lambda_x(x2).unsqueeze(-1)
        t1 = (x1 * lam_x1 *w1 + x2 *lam_x2 * w2)/(lam_x1*w1 + lam_x2*w2 -2)
        mid = man.mobius_scalar_mul(torch.tensor(0.5),t1)
    return mid


import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import copy
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from timm.models.layers import trunc_normal_
import logging

_logger = logging.getLogger(__name__)

# Assuming custom modules are defined elsewhere or imported appropriately
# from your_custom_modules import Manifold_Linear, Mob_Act, Mob_Res_Midpoint, two_point_mid, ManifoldMHA, Embed, PairEmbed, SequenceTrimmer, build_sparse_tensor

# Define the ParticleAttentionGatingNetwork class
class ParticleAttentionGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, num_heads=8):
        super().__init__()
        self.num_experts = num_experts
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x, padding_mask=None):
        # x: (seq_len, batch_size, input_dim)
        # padding_mask: (batch_size, seq_len)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=padding_mask)  # (seq_len, batch_size, input_dim)
        # Aggregate over sequence length
        attn_output = attn_output.mean(dim=0)  # (batch_size, input_dim)
        gating_logits = self.fc(attn_output)   # (batch_size, num_experts)
        gating_weights = F.softmax(gating_logits, dim=-1)  # (batch_size, num_experts)
        return gating_weights

# Define the JetGatingNetwork class
class JetGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        gating_logits = self.fc(x)  # (batch_size, num_experts)
        gating_weights = F.softmax(gating_logits, dim=-1)  # (batch_size, num_experts)
        return gating_weights

# Updated PMBlock class
class PMBlock(nn.Module):
    def __init__(self,
                 manifolds, 
                 n_experts,
                 embed_dim=128, 
                 num_heads=8, 
                 ffn_ratio=4,
                 dropout=0.1, 
                 attn_dropout=0.1, 
                 activation_dropout=0.1,
                 add_bias_kv=False, 
                 activation='gelu',
                 scale_fc=True, 
                 scale_attn=True, 
                 scale_heads=False, 
                 scale_resids=True,
                 man_att=False,
                 weight_init_ratio=1,
                 att_metric='dist',
                 inter_man_att_method='v2',
                 base_resid_agg=False,
                 base_activations='act',
                 remove_pm_norm_layers=False):
        super().__init__()
        
        self.manifolds = manifolds
        self.n_man = len(manifolds)
        self.n_experts = n_experts  # Fixed number of experts to use
        self.man_att = man_att
        self.man_att_dim = 2 * embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        
        self.inter_man_att_method = inter_man_att_method
        self.base_resid_agg = base_resid_agg
        self.base_activations = base_activations
        self.remove_pm_norm_layers = remove_pm_norm_layers
    
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        
        # Initialize lists to hold modules for each manifold
        self.pre_attn_norm = nn.ModuleList()
        self.pre_fc_norm = nn.ModuleList()
        self.res_agg = nn.ModuleList()
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        self.attn = nn.ModuleList()
        
        if man_att and self.n_experts > 1:
            self.w_man_att = nn.ModuleList()
            self.theta_man_att = nn.ModuleList()
            
            if inter_man_att_method == 'v3':
                # Initialize midpoint_weighting with n_experts
                self.midpoint_weighting = nn.Sequential(
                    nn.Linear(self.n_experts * embed_dim, int(4 * self.n_experts)), 
                    nn.ReLU(),
                    nn.Linear(int(4 * self.n_experts), self.n_experts)
                )
        
        for man in manifolds:
            # Activation function per manifold
            if self.base_activations == 'act' or man.name == 'Euclidean':
                act = nn.ReLU()
            elif self.base_activations == 'mob_act':
                act = Mob_Act(nn.ReLU(), man)
            elif self.base_activations == 'None':
                act = None
            
            # Residual aggregation per manifold
            self.res_agg.append(Mob_Res_Midpoint(man))
            
            # Feedforward layers per manifold
            if act is not None:
                self.fc1.append(nn.Sequential(
                    Manifold_Linear(embed_dim, self.ffn_dim, ball=man, weight_init_ratio=weight_init_ratio),
                    act
                ))
                self.fc2.append(nn.Sequential(
                    Manifold_Linear(self.ffn_dim, embed_dim, ball=man, weight_init_ratio=weight_init_ratio),
                    act
                ))
            else:
                self.fc1.append(nn.Sequential(
                    Manifold_Linear(embed_dim, self.ffn_dim, ball=man, weight_init_ratio=weight_init_ratio)
                ))
                self.fc2.append(nn.Sequential(
                    Manifold_Linear(self.ffn_dim, embed_dim, ball=man, weight_init_ratio=weight_init_ratio)
                ))
            
            # Attention layers per manifold
            self.attn.append(ManifoldMHA(
                embed_dim, num_heads, dropout=attn_dropout, ball=man, weight_init_ratio=weight_init_ratio, att_metric=att_metric
            ))
            
            # Inter-manifold attention parameters
            if self.man_att and self.n_experts > 1:
                if inter_man_att_method == 'v1':
                    self.w_man_att.append(Manifold_Linear(embed_dim, self.man_att_dim, ball=man))
                    self.theta_man_att.append(nn.Linear(self.man_att_dim, 1))
                elif inter_man_att_method in ['v2', 'v3']:
                    self.w_man_att.append(Manifold_Linear(embed_dim, embed_dim, ball=man))
            
            # Pre-attention and pre-FFN layer normalization per manifold
            self.pre_attn_norm.append(nn.LayerNorm(embed_dim))
            self.pre_fc_norm.append(nn.LayerNorm(embed_dim))

    def forward(self, pm_x, pm_x_cls=None, padding_mask=None, attn_mask=None, selected_indices=None):
        """
        Args:
            pm_x (List[Tensor]): list of input tensors, one per manifold, each of shape `(seq_len, batch, embed_dim)`
            pm_x_cls (List[Tensor], optional): class token inputs, one per manifold, each of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary mask of shape `(batch, seq_len)`
            attn_mask (ByteTensor, optional): attention mask
            selected_indices (List[int], optional): indices of manifolds (experts) to use
        Returns:
            List[Tensor]: outputs from the selected experts
        """
        # If selected_indices is None, select the first n_experts manifolds
        if selected_indices is None:
            selected_indices = list(range(self.n_experts))
        
        output = [None] * self.n_man  # Placeholder for outputs
        
        for idx in selected_indices:
            man = self.manifolds[idx]
            x = pm_x[idx]
            
            if pm_x_cls is not None:
                x_cls = pm_x_cls[idx]
                with torch.no_grad():
                    # Prepend one element for x_cls: -> (batch, 1 + seq_len)
                    padding_mask_cur = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
                    
                # Class attention
                residual = x_cls
                u = torch.cat((x_cls, x), dim=0)  # (seq_len + 1, batch, embed_dim)
                u = man.projx(u)
                if man.name == 'Euclidean':
                    u = self.pre_attn_norm[idx](u)
                elif not self.remove_pm_norm_layers:
                    u = man.expmap0(self.pre_attn_norm[idx](man.logmap0(u)))
                    
                x = self.attn[idx](x_cls, u, u, key_padding_mask=padding_mask_cur)[0]
                x = man.projx(x)
            else:
                residual = x
                if man.name == 'Euclidean':
                    x = self.pre_attn_norm[idx](x)
                elif not self.remove_pm_norm_layers:
                    x = man.expmap0(self.pre_attn_norm[idx](man.logmap0(x)))
                
                x = self.attn[idx](x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]
                x = man.projx(x)
            
            # Residual connection
            if man.name == 'Euclidean':
                x = x + residual
            elif self.base_resid_agg:
                x = man.mobius_add(x, residual)
            else:
                x = self.res_agg[idx](x, residual)
            
            residual = x
            if man.name == 'Euclidean':
                x = self.pre_fc_norm[idx](x)
            elif not self.remove_pm_norm_layers:
                x = man.expmap0(self.pre_fc_norm[idx](man.logmap0(x)))
                
            x = man.projx(x)
            x = self.fc1[idx](x)
            x = self.act_dropout(x)
            x = self.fc2[idx](x)
            
            # Residual connection
            if man.name == 'Euclidean':
                x = x + residual
            elif self.base_resid_agg:
                x = man.mobius_add(x, residual)
            else:
                x = self.res_agg[idx](x, residual)
            x = man.projx(x)
            output[idx] = x
        
        # Inter-manifold attention (only among selected experts)
        if self.man_att and self.n_experts > 1:
            # Prepare variables for selected experts
            selected_output = [output[idx] for idx in selected_indices]
            selected_manifolds = [self.manifolds[idx] for idx in selected_indices]
            # Implement inter-manifold attention among selected experts
            if self.inter_man_att_method == 'v3':
                tan_output = [selected_manifolds[i].logmap0(selected_output[i]) for i in range(self.n_experts)]
                mu_stacked = torch.stack(tan_output)  # Shape: (n_experts, seq_len, batch_size, embed_dim)
                # Reshape for midpoint_weighting
                mu_stacked = mu_stacked.permute(2, 1, 0, 3)  # (batch_size, seq_len, n_experts, embed_dim)
                mu_flat = mu_stacked.reshape(mu_stacked.shape[0], mu_stacked.shape[1], -1)  # (batch_size, seq_len, n_experts * embed_dim)
                # Pass through midpoint_weighting
                weights = self.midpoint_weighting(mu_flat)  # (batch_size, seq_len, n_experts)
                weights = F.softmax(weights, dim=-1)  # Softmax over n_experts
                weights = weights.unsqueeze(-1)  # (batch_size, seq_len, n_experts, 1)
                # Compute weighted sum
                mu = (weights * mu_stacked).sum(dim=2)  # (batch_size, seq_len, embed_dim)
                mu = mu.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
                # Projected means for each expert
                proj_mu = [self.w_man_att[selected_indices[i]](selected_manifolds[i].expmap0(mu)) for i in range(self.n_experts)]
                # Compute distances and weights
                d_i = []
                for i in range(self.n_experts):
                    man = selected_manifolds[i]
                    if man.name == 'Euclidean':
                        d_i.append(torch.norm(selected_output[i] - proj_mu[i], dim=-1))
                    else:
                        d_i.append(man.dist(selected_output[i], proj_mu[i]))
                d_i_tensor = torch.stack(d_i, dim=0)  # Shape: (n_experts, seq_len, batch_size)
                softmax_d_i = torch.softmax(d_i_tensor, dim=0).unsqueeze(-1)  # Shape: (n_experts, seq_len, batch_size, 1)
                # Compute final outputs
                proc_jets = []
                for i in range(self.n_experts):
                    man = selected_manifolds[i]
                    proc_jets.append(two_point_mid(
                        selected_output[i], proj_mu[i], man,
                        torch.ones_like(softmax_d_i[i]), softmax_d_i[i]
                    ))
                # Update outputs for selected experts
                for idx, expert_idx in enumerate(selected_indices):
                    output[expert_idx] = proc_jets[idx]
        
        return output
    
    
# updated PMTransformer class
class PMTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 pair_embed_dims=[64, 64, 64],
                 part_geom='R',
                 part_dim=64,
                 jet_geom='R',
                 jet_dim=64,
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 PM_weight_initialization_factor=1, 
                 att_metric='dist', 
                 inter_man_att_method='v2',
                 inter_man_att=-1,
                 equal_heads=False,
                 base_resid_agg=False,
                 base_activations='act',
                 remove_pm_norm_layers=False,
                 topk=1,  # Number of experts to select per sample
                 n_experts=None,  # Fixed number of experts for particle-level
                 jet_topk=1,  # Number of experts to select at jet-level
                 gating_hidden_dim=128,  # Hidden dimension for gating networks
                 **kwargs):
        super().__init__(**kwargs)
        
        # Initialize particle-level manifolds
        self.part_manifolds = nn.ModuleList()
        parts = part_geom.split('x') if 'x' in part_geom else [part_geom]
        
        for m in parts:
            if m == 'R':
                self.part_manifolds.append(geoopt.Euclidean())
            elif m == 'H':
                self.part_manifolds.append(geoopt.PoincareBall(c=1.2, learnable=True))
            elif m == 'S':
                self.part_manifolds.append(geoopt.SphereProjection(k=1, learnable=True))
        
        self.n_man = len(self.part_manifolds)
        self.n_experts = n_experts if n_experts is not None else self.n_man  # Default to using all manifolds
        
        total_part_dim = part_dim * self.n_experts  # Adjusted for the number of experts
        
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        
        self.base_resid_agg = base_resid_agg
        self.base_activations = base_activations
        self.remove_pm_norm_layers = remove_pm_norm_layers

        embed_dim = part_dim
    
        if equal_heads and self.n_experts > 1: 
            num_heads = int(num_heads / self.n_experts)
            
        default_cfg = dict(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            ffn_ratio=1,
            dropout=0.1, 
            attn_dropout=0.1, 
            activation_dropout=0.1,
            add_bias_kv=False, 
            activation=activation,
            scale_fc=True, 
            scale_attn=True, 
            scale_heads=False, 
            scale_resids=True, 
            weight_init_ratio=PM_weight_initialization_factor,
            att_metric=att_metric,
            manifolds=self.part_manifolds,
            man_att=False,
            inter_man_att_method=inter_man_att_method,
            base_resid_agg=base_resid_agg,
            base_activations=base_activations,
            remove_pm_norm_layers=remove_pm_norm_layers,
            n_experts=self.n_experts  # Pass n_experts to PMBlock
        )
        
        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        cfg_cls_block['man_att'] = False
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, [embed_dim], activation=activation) if len([embed_dim]) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        
        self.part_embedding = nn.ModuleList()
        for man in self.part_manifolds:
            self.part_embedding.append(nn.Sequential(
                Manifold_Linear(input_dim, part_dim, ball=man, weight_init_ratio=PM_weight_initialization_factor)
            ))
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if inter_man_att > 0:
                cfg_block['man_att'] = (i != 0 and i % inter_man_att == 0)
            elif inter_man_att == 0:
                cfg_block['man_att'] = (i in range(1, num_layers - 1))
            else: 
                cfg_block['man_att'] = False
            self.blocks.append(PMBlock(**cfg_block))
        
        self.cls_blocks = nn.ModuleList([PMBlock(**cfg_cls_block) for _ in range(num_cls_layers)])
        
        dim_dif = jet_dim - total_part_dim
        if fc_params is not None:
            self.jet_fc = nn.ModuleList()
            self.jet_man_fc = nn.ModuleList()
            self.jet_manifolds = nn.ModuleList()
            jets = jet_geom.split('x') if 'x' in jet_geom else [jet_geom]
            for m in jets:
                if m == 'R':
                    self.jet_manifolds.append(geoopt.Euclidean())
                elif m == 'H':
                    self.jet_manifolds.append(geoopt.PoincareBall(c=1.2, learnable=True))
                elif m == 'S':
                    self.jet_manifolds.append(geoopt.SphereProjection(k=1.0, learnable=True))
            self.jet_num_experts = len(self.jet_manifolds)
            self.jet_topk = jet_topk  # Number of jet-level experts to select
            for man in self.jet_manifolds:
                if self.base_activations == 'act' or man.name == 'Euclidean':
                    act = nn.ReLU()
                elif self.base_activations == 'mob_act':
                    act = Mob_Act(nn.ReLU(), man)
                elif self.base_activations == 'None':
                    act = None
                self.jet_fc.append(nn.Sequential(
                    nn.Linear(total_part_dim, total_part_dim + int(dim_dif * 0.5)), nn.ReLU(),
                    nn.Linear(total_part_dim + int(dim_dif * 0.5), total_part_dim + int(dim_dif * 0.75)), nn.ReLU(),
                    nn.Linear(total_part_dim + int(dim_dif * 0.75), jet_dim), nn.ReLU()))
                
                if act is not None:
                    self.jet_man_fc.append(nn.Sequential(
                        Manifold_Linear(jet_dim, jet_dim, ball=man, weight_init_ratio=PM_weight_initialization_factor), 
                        act,
                        Manifold_Linear(jet_dim, jet_dim, ball=man, weight_init_ratio=PM_weight_initialization_factor)
                    ))
                else:
                    self.jet_man_fc.append(nn.Sequential(
                        Manifold_Linear(jet_dim, jet_dim, ball=man, weight_init_ratio=PM_weight_initialization_factor),
                        Manifold_Linear(jet_dim, jet_dim, ball=man, weight_init_ratio=PM_weight_initialization_factor)
                    ))
            post_jet_dim = jet_dim * self.jet_topk  # Adjusted for selected jet-level experts
            self.final_fc = nn.Sequential(
                nn.Linear(post_jet_dim, post_jet_dim), nn.ReLU(),
                nn.Linear(post_jet_dim, post_jet_dim), nn.ReLU(),
                nn.Linear(post_jet_dim, num_classes))
        else:
            self.jet_fc = None

        # Initialize class tokens
        self.cls_token = nn.ParameterList()
        for man in self.part_manifolds:
            cur_token = geoopt.ManifoldParameter(torch.zeros(1, 1, embed_dim), requires_grad=True, manifold=man)
            trunc_normal_(cur_token, std=.02)
            self.cls_token.append(cur_token)

        # Initialize particle-level gating network
        self.num_experts = self.n_man
        self.topk = self.n_experts  # Number of experts to select per sample
        self.gating_network = ParticleAttentionGatingNetwork(input_dim, self.num_experts, num_heads=num_heads)

        # Initialize jet-level gating network
        self.jet_gating_network = JetGatingNetwork(embed_dim, self.jet_num_experts, hidden_dim=gating_hidden_dim)

        # Initialize jet-level inter-manifold attention components if needed
        if self.jet_num_experts > 1:
            self.jet_w_man_att = nn.ModuleList()
            for man in self.jet_manifolds:
                self.jet_w_man_att.append(Manifold_Linear(jet_dim, jet_dim, ball=man))
            self.jet_midpoint_weighting = nn.Sequential(
                nn.Linear(self.jet_num_experts * jet_dim, int(4 * self.jet_num_experts)),
                nn.ReLU(),
                nn.Linear(int(4 * self.jet_num_experts), self.jet_num_experts)
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, embed=False):
        # x: (N, C, P)
        # v: (N, 4, P) [px, py, pz, energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Input embedding
            x = x.permute(2, 0, 1)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # Particle-level gating network
            gating_weights = self.gating_network(x, padding_mask=padding_mask)  # (N, num_experts)
            # Select top-k experts per sample
            batch_size = x.size(1)
            topk = self.n_experts
            _, top_expert_indices = gating_weights.topk(topk, dim=-1)  # (N, topk)
            selected_indices = torch.unique(top_expert_indices)  # Unique experts selected across the batch
            selected_indices = selected_indices.tolist()

            # Prepare inputs for selected experts
            pm_x = [None] * self.n_man
            pm_x_cls = [None] * self.n_man
            for idx in selected_indices:
                man = self.part_manifolds[idx]
                cls_tokens = self.cls_token[idx].expand(1, x.size(1), -1)
                x_embedded = self.part_embedding[idx](man.expmap0(x))
                pm_x[idx] = x_embedded
                pm_x_cls[idx] = cls_tokens

            # Pass through blocks
            for block in self.blocks:
                pm_x = block(pm_x, pm_x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask, selected_indices=selected_indices)

            # Pass through class blocks
            for block in self.cls_blocks:
                pm_x_cls = block(pm_x, pm_x_cls=pm_x_cls, padding_mask=padding_mask, selected_indices=selected_indices)

            # Collect outputs
            cls_tokens_outputs = []
            for idx in selected_indices:
                man = self.part_manifolds[idx]
                cls_token_output = man.logmap0(pm_x_cls[idx])  # (1, batch_size, embed_dim)
                cls_tokens_outputs.append(cls_token_output)

            # Aggregate outputs from multiple experts
            if len(cls_tokens_outputs) > 1:
                x_cls = torch.cat(cls_tokens_outputs, dim=-1).squeeze(0)  # (batch_size, total_embed_dim)
            else:
                x_cls = cls_tokens_outputs[0].squeeze(0)  # (batch_size, embed_dim)

            # Jet-level gating network
            jet_gating_weights = self.jet_gating_network(x_cls)  # (batch_size, jet_num_experts)
            # Select top-k jet-level experts per sample
            _, jet_top_expert_indices = jet_gating_weights.topk(self.jet_topk, dim=-1)  # (batch_size, jet_topk)
            jet_selected_indices = torch.unique(jet_top_expert_indices)  # Unique jet-level experts selected across the batch
            jet_selected_indices = jet_selected_indices.tolist()

            # Process x_cls through selected jet-level experts
            x_jets_list = [None] * self.jet_num_experts
            for idx in jet_selected_indices:
                idx = idx
                man = self.jet_manifolds[idx]
                x_jets_fc = self.jet_fc[idx](x_cls)  # (batch_size, jet_dim)
                x_jets_manifold = man.expmap0(x_jets_fc)
                x_jets_manifold = self.jet_man_fc[idx](x_jets_manifold)
                x_jets_logmap0 = man.logmap0(x_jets_manifold)  # (batch_size, jet_dim)
                x_jets_list[idx] = x_jets_logmap0

            # Jet-level inter-manifold attention (method 'v3')
            if self.jet_num_experts > 1 and len(jet_selected_indices) > 1:
                # Collect outputs from selected experts
                tan_output = [x_jets_list[idx] for idx in jet_selected_indices]  # List of tensors
                tan_output_stacked = torch.stack(tan_output, dim=0)  # (num_selected_experts, batch_size, jet_dim)

                # Reshape for midpoint_weighting
                mu_flat = tan_output_stacked.permute(1, 0, 2).reshape(batch_size, -1)  # (batch_size, num_selected_experts * jet_dim)

                # Pass through midpoint_weighting
                weights = self.jet_midpoint_weighting(mu_flat)  # (batch_size, num_selected_experts)
                weights = F.softmax(weights, dim=-1)  # (batch_size, num_selected_experts)
                weights = weights.unsqueeze(-1)  # (batch_size, num_selected_experts, 1)

                # Compute weighted sum
                mu = torch.sum(weights * tan_output_stacked.permute(1, 0, 2), dim=1)  # (batch_size, jet_dim)

                # Projected means for each expert
                proj_mu = []
                for idx_i, idx in enumerate(jet_selected_indices):
                    man = self.jet_manifolds[idx]
                    proj_mu_i = self.jet_w_man_att[idx](man.expmap0(mu))
                    proj_mu.append(proj_mu_i)  # (batch_size, jet_dim)

                # Compute distances and weights
                d_i = []
                for idx_i, idx in enumerate(jet_selected_indices):
                    man = self.jet_manifolds[idx]
                    x_jets = x_jets_list[idx]
                    proj_mu_i = proj_mu[idx_i]
                    if man.name == 'Euclidean':
                        d = torch.norm(x_jets - proj_mu_i, dim=-1)
                    else:
                        d = man.dist(x_jets, proj_mu_i)
                    d_i.append(d)  # (batch_size,)
                d_i_tensor = torch.stack(d_i, dim=0)  # (num_selected_experts, batch_size)
                softmax_d_i = torch.softmax(d_i_tensor, dim=0).unsqueeze(-1)  # (num_selected_experts, batch_size, 1)

                # Compute final outputs
                proc_jets = []
                for idx_i, idx in enumerate(jet_selected_indices):
                    man = self.jet_manifolds[idx]
                    x_jets = x_jets_list[idx]
                    proj_mu_i = proj_mu[idx_i]
                    weight = softmax_d_i[idx_i]  # (batch_size, 1)
                    proc_jet = two_point_mid(x_jets, proj_mu_i, man, torch.ones_like(weight), weight)
                    proc_jets.append(proc_jet)  # (batch_size, jet_dim)

                # Combine proc_jets
                x_out = torch.cat(proc_jets, dim=-1)  # (batch_size, num_selected_experts * jet_dim)
            else:
                # If only one jet-level expert is selected
                idx = jet_selected_indices[0]
                x_out = x_jets_list[idx]  # (batch_size, jet_dim)

            # Final classification FC
            output = self.final_fc(x_out)  # (batch_size, num_classes)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output