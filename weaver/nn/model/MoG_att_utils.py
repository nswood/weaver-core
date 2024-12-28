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
                 weight_init_ratio =1):
        
        super().__init__()
        
        self.man = man
        self.man_att = man_att
        self.man_att_dim = 2*embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        
    
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        
                                  
        act = Mob_Act(nn.ReLU(), self.man)
        if act is not None:
            self.fc1 = nn.Sequential(Manifold_Linear(embed_dim, self.ffn_dim,ball = self.man, weight_init_ratio = weight_init_ratio), act)
            self.fc2 = nn.Sequential(Manifold_Linear(self.ffn_dim, embed_dim,ball = self.man, weight_init_ratio = weight_init_ratio), act)


        self.attn = ManifoldMHA(embed_dim,num_heads,dropout=attn_dropout, ball = self.man, weight_init_ratio = weight_init_ratio,att_metric = 'tan_space')                                   
        
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
            
            u = self.man.expmap0(self.pre_attn_norm(self.man.logmap0(u)))
            
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask_cur)[0]  # (1, batch, embed_dim)
            x = self.man.projx(x)
                
                
        else:
            residual = x
            x = self.man.expmap0(self.pre_attn_norm(self.man.logmap0(x)))
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                            attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
            x = self.man.projx(x)
                
        
        # Residual Aggregation
        x = self.man.mobius_add(x,residual)
        
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
        
        # Residual Aggregation
        x = self.man.mobius_add(x,residual)
        x = self.man.projx(x)
        
    
        return x


class PM_MoE_Att_Block(nn.Module):
    def __init__(self,
                 manifolds, 
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
                    for manifold in manifolds])

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
