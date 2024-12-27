''' Product Manifold Transformer (PM-Transformer)

Paper: "Product Manifold Machine Learning for Physics" - https://arxiv.org/abs/2412.07033
'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from weaver.utils.logger import _logger

import os
os.environ['PYTHONPATH'] = '/n/home11/nswood/weaver-core/weaver/nn/model'

from weaver.nn.model.PM_utils import *

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



def two_point_mid(x1,x2, man, w1,w2):
    if man.name == 'Euclidean':
        mid = (x1 * w1 + x2 * w2)/(w1 + w2)
    else:
        lam_x1 = man.lambda_x(x1).unsqueeze(-1)
        lam_x2 = man.lambda_x(x2).unsqueeze(-1)
        t1 = (x1 * lam_x1 *w1 + x2 *lam_x2 * w2)/(lam_x1*w1 + lam_x2*w2 -2)
        mid = man.mobius_scalar_mul(torch.tensor(0.5),t1)
    return mid




class PM_Attention_Expert(nn.Module):
    def __init__(self,
                 man, 
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
                 man_att = False,
                 weight_init_ratio =1,
                 att_metric = 'tan_space',
                 inter_man_att_method = 'v3',
                 base_resid_agg= False,
                 base_activations = 'act',
                 remove_pm_norm_layers=False):
        
        super().__init__()
        
        self.man = man
        self.man_att = man_att
        self.man_att_dim = 2*embed_dim
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
        
                                  
        if self.base_activations == 'act' or self.man.name == 'Euclidean':
            act = nn.ReLU()
        elif self.base_activations == 'mob_act':
            act = Mob_Act(nn.ReLU(), self.man)
        elif self.base_activations == 'None':
            act = None
        
        self.res_agg.append(Mob_Res_Midpoint(self.man))
        
        if act is not None:
            self.fc1 = nn.Sequential(Manifold_Linear(embed_dim, self.ffn_dim,ball = self.man, weight_init_ratio = weight_init_ratio), act)
            self.fc2 = nn.Sequential(Manifold_Linear(self.ffn_dim, embed_dim,ball = self.man, weight_init_ratio = weight_init_ratio), act)


        self.attn = ManifoldMHA(embed_dim,num_heads,dropout=attn_dropout, ball = self.man, weight_init_ratio = weight_init_ratio,att_metric = att_metric)                                   
        
        
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.pre_fc_norm = nn.LayerNorm(embed_dim)
            
    def forward(self, x, x_cls, padding_mask=None, attn_mask=None):
        
       
        if x_cls is not None:
            
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask_cur = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
                
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            if self.man.name == 'Euclidean':
                u = self.pre_attn_norm(u)
            elif not self.remove_pm_norm_layers:
                u = self.man.expmap0(self.pre_attn_norm(self.man.logmap0(u)))
            
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask_cur)[0]  # (1, batch, embed_dim)
            x = self.man.projx(x)
                
                
        else:
            residual = x
            if self.man.name == 'Euclidean':
                x = self.pre_attn_norm(x)
            elif not self.remove_pm_norm_layers:
                x = self.man.expmap0(self.pre_attn_norm(self.man.logmap0(x)))
            
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                            attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
            x = self.man.projx(x)
                
        
        if self.man.name == 'Euclidean':
            x = x + residual
        elif self.base_resid_agg:
            x = self.man.mobius_add(x,residual)
        else:
            x = self.res_agg(x,residual)
        
        x = self.man.projx(x)
        
        residual = x
        if self.man.name == 'Euclidean':
            x = self.pre_fc_norm(x)
        elif not self.remove_pm_norm_layers:
            x = self.man.expmap0(self.pre_fc_norm(self.man.logmap0(x)))
        
        x = self.man.projx(x)
        x = self.fc1(x)
        x = self.act_dropout(x)
        x = self.man.projx(x)
        x = self.fc2(x)
            
        x = self.man.mobius_add(x,residual)
        
        x = self.man.projx(x)
        
    
        return x


class PM_MoE_Att_Block(nn.Module):
    def __init__(self,
                 man, 
                 top_k_part = 2,
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
                 man_att = False,
                 weight_init_ratio =1,
                 att_metric = 'tan_space',
                 inter_man_att_method = 'v3',
                 base_resid_agg= False,
                 base_activations = 'act',
                 remove_pm_norm_layers=False):
        super(PM_MoE_Att_Block, self).__init__()
        self.part_experts = nn.ModuleList([
            PM_Attention_Expert(manifold, embed_dim, num_heads, ffn_ratio, dropout, attn_dropout, activation_dropout,
                     add_bias_kv, activation, scale_fc, scale_attn, scale_heads, scale_resids, man_att,
                     weight_init_ratio, att_metric, inter_man_att_method, base_resid_agg, base_activations,
                     remove_pm_norm_layers)
                    for manifold in part_manifolds])

    def forward(self, features, expert_indices, x_cls = None):
        # Initialize a list to store the outputs for each sample
        outputs = [[] for _ in range(features.size(0))]

        # Iterate over each expert
        for expert_idx, expert in enumerate(self.part_experts):
            
            # Collect all elements from the batch that need to pass through the current expert
            batch_elements = []
            batch_x_cls = []
            batch_indices = []
            
            for i in range(features.size(0)):
                if expert_idx in expert_indices[i]:
                    j = expert_indices[i].index(expert_idx)
                    batch_elements.append(features[i][j].unsqueeze(0))
                    if x_cls is not None:
                        batch_x_cls.append(x_cls[i][j].unsqueeze(0))
                    batch_indices.append(i)
            
            if batch_elements:
                # Stack the collected elements to form a batch
                batch_elements = torch.cat(batch_elements, dim=0)
                if x_cls is not None:
                    batch_x_cls = torch.cat(batch_x_cls, dim=0)
                
                # Pass the batch through the expert
                if x_cls is not None:
                    expert_outputs = expert(batch_elements, batch_x_cls)
                else:
                    expert_outputs = expert(batch_elements)
                
                # Recombine the outputs
                for idx, output in zip(batch_indices, expert_outputs):
                    outputs[idx].append(output)

        # Now outputs contain the aggregated outputs for each sample
        return outputs


class PM_MLP_Expert(nn.Module):
    def __init__(self, input_dim, output_dim,man, activation='relu'):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim + int((output_dim - input_dim) * 0.5)), nn.ReLU(),
            nn.Linear(input_dim + int((output_dim - input_dim) * 0.5), input_dim + int((output_dim - input_dim) * 0.75)), nn.ReLU(),
            nn.Linear(input_dim + int((output_dim - input_dim) * 0.75), output_dim), nn.ReLU()
        )
        self.man = man
        self.man_fc = nn.Sequential(
                Manifold_Linear(output_dim, output_dim, ball=man), 
                nn.ReLU(),
                Manifold_Linear(output_dim, output_dim, ball=man)
            )

    def forward(self, x):
        x = self.fc(x)
        return self.man_fc(self.man.expmap0(x))
        

class PM_MoE_MLP_Block(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k, manifolds, activation='relu'):
        super().__init__()
        self.experts = nn.ModuleList([PM_MLP_Expert(input_dim, output_dim, manifolds[i], activation) for i in range(num_experts)])
        self.top_k = top_k

    def forward(self, x, selected_experts):
        outputs = [[] for _ in range(len(x))]
        
        for expert_idx, expert in enumerate(self.experts):
            batch_elements = []
            batch_indices = []
            
            for i in range(len(x)):
                if expert_idx in selected_experts[i]:
                    batch_elements.append(x[i].unsqueeze(0))
                    batch_indices.append(i)
            
            if batch_elements:
                batch_elements = torch.cat(batch_elements, dim=0)
                expert_outputs = expert(batch_elements)
                
                for idx, output in zip(batch_indices, expert_outputs):
                    outputs[idx].append(output)
        
        return [torch.cat(output, dim=0) for output in outputs]


class MoG(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 pair_embed_dims=[64, 64, 64],
                 part_experts = 4,
                 part_experts_dim = 64,
                 part_router_n_parts = 16,
                 top_k_part = 2,
                 jet_experts = 4,
                 jet_experts_dim= 64,
                 top_k_jet = 2,
                 shared_expert = True, 
                 shared_expert_ratio = 1,
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
                 PM_weight_initialization_factor = 1, 
                 att_metric = 'tanspace', 
                 inter_man_att_method = 'v3',
                 inter_man_att = -1,
                 equal_heads = False,
                 base_resid_agg= False,
                 base_activations = 'act',
                 remove_pm_norm_layers=False,
                 dropout_rate = 0.1,
                 curvature_init = 1.2,
                 conv_embed = 'True',
                 clamp = -1,
                 learnable = True, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.part_manifolds = nn.ModuleList()
        self.jet_manifolds = nn.ModuleList()
        total_part_dim = 0
        total_jet_dim = 0
        if shared_expert:
            self.part_manifolds.append(geoopt.Euclidean())
        
            self.jet_manifolds.append(geoopt.Euclidean())
            total_part_dim += part_experts_dim*shared_expert_ratio
            total_jet_dim += jet_experts_dim*shared_expert_ratio


        embed_dims = [64, 64, part_experts_dim]
        fc_params = [[jet_experts_dim,0.1], [jet_experts_dim,0.1], [jet_experts_dim,0.1]]

        self.conv_embed = conv_embed  =='True'

        for i in part_experts:
            self.part_manifolds.append(geoopt.Stereographic(learnable=learnable))
            total_part_dim += part_experts_dim
        
        for i in jet_experts:
            self.jet_manifolds.append(geoopt.Stereographic(learnable=learnable))
            total_jet_dim += jet_experts_dim
            
        self.n_part_man = len(self.part_manifolds)
        self.n_jet_man = len(self.jet_manifolds)

        self.n_part_experts = part_experts if shared_expert else part_experts + 1
        self.n_jet_experts = jet_experts if shared_expert else jet_experts + 1

        # Router for particle experts
        # Takes in the concatenated particle features for top part_router_n_parts particles based on pT
        # and outputs the weights for each expert
        self.part_router_n_parts = part_router_n_parts
        part_router_input= part_router_n_parts*part_experts_dim
        self.part_router = nn.Sequential(
                                    nn.Linearpart_router_input,int(part_router_input*0.25), 
                                    nn.ReLU(),
                                    nn.Linear(int(part_router_input*0.25), self.n_part_man),
                                    nn.Softmax(dim = -1))

        # Router for jet experts
        # Takes jet-level latent vector output from particle-level processing
        jet_router_input = total_part_dim
        self.jet_router = nn.Sequential(nn.Linear(jet_router_input, int(jet_router_input*0.25)),
                                        nn.ReLU(),
                                        nn.Linear(int(jet_router_input*0.25), self.n_jet_man),
                                        nn.Softmax(dim = -1))
        
        self.norm = nn.LayerNorm(total_jet_dim)
        
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        
        self.base_resid_agg = base_resid_agg
        self.base_activations = base_activations
        self.remove_pm_norm_layers = remove_pm_norm_layers

        embed_dim = part_dim
    
        if equal_heads and self.n_part_man > 1: 
            num_heads = int(num_heads/self.n_part_man)
            
        default_cfg = dict(embed_dim=embed_dim, 
                           num_heads=num_heads, 
                           ffn_ratio=1,
                           dropout=dropout_rate, 
                           attn_dropout=dropout_rate, 
                           activation_dropout=dropout_rate,
                           add_bias_kv=False, 
                           activation=activation,
                           scale_fc=True, 
                           scale_attn=True, 
                           scale_heads=False, 
                           scale_resids=True, 
                           weight_init_ratio = PM_weight_initialization_factor,
                           att_metric =att_metric,
                           manifolds = self.part_manifolds,
                           man_att = False,
                           inter_man_att_method = inter_man_att_method,
                           base_resid_agg = base_resid_agg,
                           base_activations = base_activations,
                           remove_pm_norm_layers = remove_pm_norm_layers)
        
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

        if conv_embed:
            self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        
        self.part_embedding = nn.ModuleList()
        for man in self.part_manifolds:
            self.part_embedding.append(nn.Sequential(Manifold_Linear(input_dim, part_dim, ball = man,weight_init_ratio = PM_weight_initialization_factor)))
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            
            cfg_block['man_att'] = False

            self.blocks.append(PM_MoE_Att_Block(**cfg_block))
            
        self.cls_blocks = nn.ModuleList([PM_MoE_Att_Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        

        # Update correct dims here
        if shared_expert and i == 0:
            jet_dim = jet_experts_dim*shared_expert_ratio
        else:
            jet_dim = jet_experts_dim
        dim_dif = jet_dim - total_part_dim

        # Initialize PM_MoE_MLP_Block for jet_fc
        self.jet_experts = PM_MoE_MLP_Block(input_dim=total_part_dim, output_dim=jet_dim, num_experts=jet_experts, top_k=self.top_k_jet)
            
        post_jet_dim = self.n_jet_man * jet_dim

        self.final_fc = nn.Sequential(nn.Linear(post_jet_dim, post_jet_dim), nn.ReLU(),
                                        nn.Linear(post_jet_dim, post_jet_dim), nn.ReLU(),
                                        nn.Linear(post_jet_dim, num_classes))
        
        # init
        self.cls_token = nn.ParameterList()
        for man in self.part_manifolds:
            cur_token = geoopt.ManifoldParameter(torch.zeros(1, 1, embed_dim), requires_grad=True, manifold = man)
            trunc_normal_(cur_token, std=.02)
            self.cls_token.append(cur_token)
        
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, embed = False):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
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
            if self.conv_embed:
                x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            x =x.permute(2,0, 1)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)
            
            router_output = self.part_router(x[:,:self.part_router_n_parts].reshape(x.size(0),-1))
            selected_part_experts = torch.topk(router_output, self.top_k_part, dim = -1).indices
            
            # If shared expert, always route to index 0 and select remaining experts from 1 to n
            if self.shared_expert:
                selected_part_experts = torch.cat((torch.zeros_like(selected_part_experts[:,0]).unsqueeze(-1),selected_part_experts+1),dim=-1)
            
            x_parts = []
            cls_tokens_parts = []
            for i in self.n_part_experts:
                cls_tokens = self.cls_token[i].expand(1, x.size(1), -1)
                x_parts.append(self.part_embedding[i](man.expmap0(x)))
                cls_tokens_parts.append(cls_tokens)
           
            del cls_tokens
            del x
            
            # transform
            for block in self.blocks:
                x_parts = block(x_parts, pm_x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
                
            for block in self.cls_blocks:
                cls_tokens_parts = block(x_parts, x_cls=cls_tokens_parts, padding_mask=padding_mask)
            
            
            cls_tokens_parts = [man.logmap0(cls_tokens_parts[i]) for i,man in enumerate(self.part_manifolds)]
            
            # Concatenate particle man outputs
            if self.n_part_man > 1:
                x_cls = torch.cat(cls_tokens_parts,dim=-1)
            else:
                x_cls = cls_tokens_parts[0]
            del cls_tokens_parts
             
            # fc
            if self.jet_fc is None:
                return x_cls
            
            router_output = self.jet_router(x_cls)

            selected_jet_experts = torch.topk(router_output, self.top_k_jet, dim=-1).indices
            
            # If shared expert, always route to index 0 and select remaining experts from 1 to n
            if self.shared_expert:
                selected_part_experts = torch.cat((torch.zeros_like(selected_part_experts[:,0]).unsqueeze(-1),selected_jet_experts+1),dim=-1)

            # Map to correct Jet dim in Euclidean space using MoE
            x_jets = self.jet_fc[0](x_cls, selected_jet_experts)

            # Map to Jet Manifold
            x_jets = [man.expmap0(x_jets[i]) for i, man in enumerate(self.jet_manifolds)]

            # Map to final Jet Manifold using MoE
            x_jets = [self.jet_man_fc[i](x_jets[i]) for i in range(self.n_jet_man)]
            
            del x_cls
            proc_jets = x_jets
            
            x_jets_tan = [man.logmap0(proc_jets[i]) for i,man in enumerate(self.jet_manifolds)]
            
                    
            if embed:
                proc_jets = [torch.squeeze(a,0) for a in proc_jets]
                x_jets_tan = [torch.squeeze(a,0) for a in x_jets_tan]
                
                return proc_jets, x_jets_tan,list(self.jet_manifolds)
            del x_jets
            
            if self.n_jet_man > 1:
                x_out = torch.cat(x_jets_tan,dim=-1)
            else:
                x_out = x_jets_tan[0]
            
            # Regular LayerNorm    
            x_out = self.norm(x_out).squeeze(0)
            
            # Final classification FC
            output = self.final_fc(x_out).squeeze(0)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output



