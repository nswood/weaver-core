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



# Import PM layers
import os
os.environ['PYTHONPATH'] = '/n/home11/nswood/weaver-core/weaver/nn/model'

from .PM_utils import *



class PMBlock(nn.Module):
    def __init__(self,manifolds, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='relu',
                 scale_fc=True, scale_attn=True, scale_heads=False, scale_resids=True,man_att = False):
        super().__init__()
        
        self.part_manifolds = manifolds
        self.n_man = len(manifolds)
        self.man_att = man_att
        self.man_att_dim = embed_dim
        self.embed_dim = embed_dim
    
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        
        
        self.fc1 = nn.ModuleList()
        if self.n_man > 1:
            self.w_man_att = nn.ModuleList() if man_att else None
            self.theta_man_att = nn.ModuleList() if man_att else None
        
        for man in manifolds:
            self.fc1.append(Manifold_Linear(embed_dim, self.ffn_dim,ball = man))
            self.fc2.append(Manifold_Linear(self.ffn_dim, embed_dim,ball = man))
            self.attn.append(ManifoldMHA(embed_dim,num_heads,dropout=attn_dropout, ball = man))
            if scale_heads:
                self.c_attn.append(nn.Parameter(torch.ones(num_heads), requires_grad=True))
            if scale_resids:
                self.w_resid.append(nn.Parameter(torch.ones(embed_dim), requires_grad=True))
            if self.n_man > 1:
                if self.man_att:
                    self.w_man_att.append(Manifold_Linear(embed_dim, self.man_att_dim ,ball = man))
                    self.theta_man_att.append(nn.Linear(self.man_att_dim, 1))
                

    def forward(self,  pm_x, pm_x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(n_man, seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(n_man, 1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(n_man, seq_len, batch, embed_dim)`
        """
#         print('Padding mask')
#         print(padding_mask.shape)
        output = []
        for i, man in enumerate(self.part_manifolds):
            x = pm_x[i]
            x = self.act(self.fc1[i](x))

            output.append(x)
        
        if self.man_att and self.n_man > 1:
            tan_output = [self.part_manifolds[i].logmap0(self.w_man_att[i](output[i])) for i in range(self.n_man)]
            mu = torch.stack(tan_output)
            mu = torch.mean(mu, dim=0)
            inter_att = [self.theta_man_att[i](tan_output[i]-mu) for i in range(self.n_man)]
            w_i = nn.Softmax(dim=0)(torch.stack(inter_att,dim =0))
            proc_jets = []
            for i in range(self.n_man):
                proc_jets.append(self.part_manifolds[i].mobius_scalar_mul(w_i[i], output[i]))
        else:
            proc_jets = output
        flat_inputs =[torch.mean(self.part_manifolds[i].logmap0(proc_jets[i]),dim = 1) for i in range(self.n_man)]
        
        h = torch.cat(flat_inputs,dim=-1)
        
        h = self.final_embedder(h)
        
        if self.softmax:
            h = nn.Softmax(dim=1)(h)
        
        return h
        
        

    
class PMNN(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 part_geom = 'R',
                 part_dim = 16,
                 jet_geom = None,
                 jet_dim = None,
                 activation='relu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        print('Classes')
        print(num_classes)
        
        
        self.r = 0.6
        self.part_manifolds = nn.ModuleList()
        parts = part_geom.split('x') if 'x' in part_geom else [part_geom]
        embed_dims = [part_dim, part_dim, part_dim] #embed_dims=[128, 512, 128]
        self.act = nn.ReLU()
        
        self.n_man = len(parts)
        for i, m in enumerate(parts):
            if m == 'R':
                self.part_manifolds.append(geoopt.Euclidean())
            elif m == 'H':
                self.part_manifolds.append(geoopt.PoincareBall(c=1.2, learnable=True))
            elif m == 'S':
                self.part_manifolds.append(geoopt.SphereProjection(k=1.0, learnable=True))
        
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
#         self.fc1 = nn.ModuleList()
#         self.fc2 = nn.ModuleList()
        
#         if self.n_man > 1:
#             self.w_man_att = nn.ModuleList()
#             self.theta_man_att = nn.ModuleList()
#         dim_dif = part_dim - input_dim
#         for man in self.part_manifolds:
#             self.fc1.append(nn.Sequential(
#                 nn.Linear(input_dim, int(input_dim + dim_dif*0.5)),
#                 nn.ReLU(),
#                 nn.Linear(int(input_dim + dim_dif*0.5), int(input_dim + dim_dif*0.75)),
#                 nn.ReLU(),
#                 nn.Linear(int(input_dim + dim_dif*0.75), part_dim)))
#             if man.name =='Euclidean':
#                 self.fc2.append(nn.Sequential(
#                     Manifold_Linear(part_dim, part_dim,ball = man),
#                     nn.ReLU(),
#                     Manifold_Linear(part_dim, part_dim,ball = man),
#                     nn.ReLU(),
#                     Manifold_Linear(part_dim, part_dim,ball = man)))
#             else:
#                 self.fc2.append(nn.Sequential(
#                     Manifold_Linear(part_dim, part_dim,ball = man),
#                     Mob_Act(nn.ReLU(), man),
#                     Manifold_Linear(part_dim, part_dim,ball = man),
#                     Mob_Act(nn.ReLU(), man),
#                     Manifold_Linear(part_dim, part_dim,ball = man)
#                                    ))
        if self.n_man > 1:
            self.w_man_att = nn.ModuleList()
            self.theta_man_att = nn.ModuleList()
        dim_dif = part_dim - input_dim
        for man in self.part_manifolds:
            if man.name =='Euclidean':
                self.fc1.append(nn.Sequential(
                    Manifold_Linear(input_dim, int(input_dim + dim_dif*0.5),ball = man),
                    nn.ReLU(),
                    Manifold_Linear(int(input_dim + dim_dif*0.5), int(input_dim + dim_dif*0.75),ball = man),
                    nn.ReLU(),
                    Manifold_Linear(int(input_dim + dim_dif*0.75), part_dim,ball = man)))
            else:
                 
                self.fc1.append(nn.Sequential(
                    Manifold_Linear(input_dim, int(input_dim + dim_dif*0.5),ball = man),
                    Mob_Act(nn.ReLU(), man),
                    Manifold_Linear(int(input_dim + dim_dif*0.5), int(input_dim + dim_dif*0.75), ball = man),
                    Mob_Act(nn.ReLU(), man),
                    Manifold_Linear(int(input_dim + dim_dif*0.75), part_dim,ball = man)
                                   ))
                                   
#                 self.fc1.append(nn.Sequential(Manifold_Linear(input_dim, part_dim,ball = man),
#                                              Manifold_Linear(part_dim, part_dim,ball = man),
#                                              Manifold_Linear(part_dim, part_dim,ball = man)))
                
            if self.n_man > 1:
                self.w_man_att.append(Manifold_Linear(part_dim, part_dim ,ball = man))
                self.theta_man_att.append(nn.Linear(part_dim, 1))
        
        

        self.n_part_man = len(self.part_manifolds)
        
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        
        self.final_fc = nn.Sequential(nn.Linear(part_dim*self.n_man, num_classes),nn.ReLU(),nn.Linear(num_classes, num_classes))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None
        
        with torch.no_grad():
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            x = x.permute(0,2,1)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pm_x = []
            for i,man in enumerate(self.part_manifolds):
#                 if 'Poincare' in self.part_manifolds[i].name:
                    #Clamp
#                     x = torch.clamp(self.r/x.norm(dim=2), max = -1).unsqueeze(-1)*x
                pm_x.append(man.expmap0(x))
                
            output = []
            for i, man in enumerate(self.part_manifolds):
                x = pm_x[i]
                x = self.fc1[i](x)
                if man.name == 'Euclidean':
                    x = self.act(x)
                output.append(x)
        
            if self.n_man > 1:
                tan_output = [self.part_manifolds[i].logmap0(self.w_man_att[i](output[i])) for i in range(self.n_man)]
                mu = torch.stack(tan_output)
                mu = torch.mean(mu, dim=0)
                inter_att = [self.theta_man_att[i](tan_output[i]-mu) for i in range(self.n_man)]
                w_i = nn.Softmax(dim=0)(torch.stack(inter_att,dim =0))
                proc_jets = []
                for i in range(self.n_man):
                    proc_jets.append(self.part_manifolds[i].mobius_scalar_mul(w_i[i], output[i]))
            else:
                proc_jets = output
            output =[torch.mean(self.part_manifolds[i].logmap0(proc_jets[i]),dim = 1) for i in range(self.n_man)]

            output = torch.cat(output,dim=-1)

            output = self.final_fc(output)
            
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


      