import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from weaver.utils.logger import _logger
import os
# os.environ['PYTHONPATH'] = '/n/home11/nswood/weaver-core/weaver/nn/model'

from weaver.nn.model.PM_utils import *


class PM_MLP_Expert(nn.Module):
    def __init__(self, input_dim, output_dim,man, activation='relu',ffn_ratio = 4):
        super().__init__()
        
        self.man = man
        self.swiglu = SwiGLU(input_dim, ffn_ratio*output_dim)
        self.Euclidean = self.man.name == 'Euclidean'
        if self.Euclidean:
            self.man_fc = nn.Sequential(nn.Linear(ffn_ratio*output_dim, ffn_ratio*output_dim), 
                nn.ReLU(),
                nn.Linear(ffn_ratio*output_dim, output_dim)
            )
        else:
            self.man_fc = nn.Sequential(
                    Manifold_Linear(ffn_ratio*output_dim, ffn_ratio*output_dim, ball=man), 
                    Mob_Act(nn.ReLU(),man),
                    Manifold_Linear(ffn_ratio*output_dim, output_dim, ball=man)
                )

    def forward(self, x):
        if self.Euclidean:
            x = self.swiglu(x)
        else:
            x = self.man.expmap0(self.swiglu(self.man.logmap0(x)))
        x = self.man_fc(x)
        # print('MLP expert output', x.shape)
        return x

class PM_MoE_part_lvl_MLP_Block(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 num_experts, 
                 top_k, 
                 manifolds, 
                 activation='relu', 
                 shared_expert=False,
                 shared_expert_ratio=1,
                 ffn_ratio=4):
        super().__init__()
        self.experts = nn.ModuleList()
        if shared_expert:
            num_experts += 1
        
        for i in range(num_experts):
            if shared_expert and i == 0:
                self.experts.append(PM_MLP_Expert(input_dim, output_dim * shared_expert_ratio, manifolds[i], activation, ffn_ratio))
            else:
                self.experts.append(PM_MLP_Expert(input_dim, output_dim, manifolds[i], activation, ffn_ratio))
        self.top_k = top_k

    def forward(self, x_parts, selected_experts):
        # x_parts is P x N x K x F
        P, N, K, F = x_parts.size()
        outputs = [[] for _ in range(P)]
        
        for expert_idx, expert in enumerate(self.experts):
            batch_elements = []
            batch_indices = []
            
            for p in range(P):
                for i in range(N):
                    for k in range(K):
                        if expert_idx == selected_experts[p, i, k]:
                            batch_elements.append(x_parts[p, i, k].unsqueeze(0))
                            batch_indices.append((p, i))
            
            if batch_elements:
                batch_elements = torch.cat(batch_elements, dim=0)
                expert_outputs = expert(batch_elements)
                
                for (p, i), output in zip(batch_indices, expert_outputs):
                    outputs[p].append(output)
        
        return outputs


