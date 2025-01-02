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
    def __init__(self, input_dim, output_dim,man, activation='relu'):
        super().__init__()
        
        self.man = man
        self.swiglu = SwiGLU(input_dim, 4*output_dim)
        self.Euclidean = self.man.name == 'Euclidean'
        if self.Euclidean:
            self.man_fc = nn.Sequential(nn.Linear(4*output_dim, 4*output_dim), 
                nn.ReLU(),
                nn.Linear(4*output_dim, output_dim)
            )
        else:
            self.man_fc = nn.Sequential(
                    Manifold_Linear(4*output_dim, 4*output_dim, ball=man), 
                    Mob_Act(nn.ReLU(),man),
                    Manifold_Linear(4*output_dim, output_dim, ball=man)
                )

    def forward(self, x):
        if self.Euclidean:
            x = self.swiglu(x)
        else:
            x = self.man.expmap0(self.swiglu(self.man.logmap0(x)))
        x = self.man_fc(x)
        # print('MLP expert output', x.shape)
        return x


class PM_MoE_MLP_Block(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k, manifolds, activation='relu', shared_expert=False,shared_expert_ratio = 1):
        super().__init__()
        self.experts = nn.ModuleList()
        if shared_expert:
            num_experts += 1
        
        for i in range(num_experts):
            if shared_expert and i ==0:
                self.experts.append(PM_MLP_Expert(input_dim, output_dim*shared_expert_ratio, manifolds[i], activation))
            else:
                self.experts.append(PM_MLP_Expert(input_dim,output_dim, manifolds[i], activation))
        self.top_k = top_k

    def forward(self, x, selected_experts):
        outputs = [[] for _ in range(len(x))]
        
        for expert_idx, expert in enumerate(self.experts):
            batch_elements = []
            batch_indices = []
            
            for i in range(len(x)):
                if expert_idx in selected_experts[i]:
                    cur_exp_id = torch.nonzero(selected_experts[i] == expert_idx, as_tuple=True)[0].item()                    
                    batch_elements.append(x[i][cur_exp_id].unsqueeze(0))
                    batch_indices.append(i)
            
            if batch_elements:
                batch_elements = torch.cat(batch_elements, dim=0)
                expert_outputs = expert(batch_elements)
                # print('expert_outputs', expert_outputs.shape)
                for idx, output in zip(batch_indices, expert_outputs):
                    # print('output', output.shape)
                    outputs[idx].append(output)
        
        return outputs

