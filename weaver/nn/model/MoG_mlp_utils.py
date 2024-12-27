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

