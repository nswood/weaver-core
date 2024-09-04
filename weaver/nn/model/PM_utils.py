import sys
import matplotlib.pyplot as plt
import geoopt
from geoopt.layers.stereographic import Distance2StereographicHyperplanes
from geoopt.manifolds.stereographic.math import arsinh, artanh,artan_k
from typing import List, Optional, Tuple, Union

    
import numpy as np
import torch
# torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.nn.init as init
import itertools
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




import torch.nn.functional as F
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfAttention#, prune_linear_layer#, gelu_new
import tqdm
import math
from scipy.special import beta

# Naive Manifold_Linear
class Manifold_Linear(nn.Module):
    def __init__(self, in_features, out_features, ball, bias=True):
        super(Manifold_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball
        self.weight = geoopt.ManifoldParameter(torch.Tensor(out_features,in_features,),manifold=self.ball)
        self.__params__ = self.in_features* self.out_features
        if bias: 
            self.__params__ += self.out_features
        
        if bias:
            self.bias = geoopt.ManifoldParameter(torch.Tensor(out_features),manifold=self.ball)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        if self.ball.name == 'Euclidean':
            init.kaiming_uniform_(self.weight, a=0.001)
        else:
            init.kaiming_uniform_(self.weight, a=0.0001)
            
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        
        mv = self.ball.mobius_matvec(self.weight,x)
#         if self.ball.name != 'Euclidean':
#             print(self.ball.k)
        
        if not self.bias is None:
            mv = self.ball.mobius_add(mv, self.bias)
        return self.ball.projx(mv)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.ball
        )


class ManifoldMHA(nn.Module):
    def __init__(self,hidden_size, num_attention_heads,  dropout,ball):
        super().__init__()
        self.ball = ball
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = Manifold_Linear(hidden_size, hidden_size, ball=self.ball)
        self.key = Manifold_Linear(hidden_size, hidden_size, ball=self.ball)
        self.value = Manifold_Linear(hidden_size, hidden_size, ball=self.ball)

        self.dropout = nn.Dropout(dropout)
        self.sigmoid_fn = nn.Sigmoid()
        
        self.is_flat = ball.name == 'Euclidean'
   


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key=None, value=None, attn_mask=None, key_padding_mask=None):
        VERBOSE = False
        
        if key is None:
            key = query
        if value is None:
            value = query
        #Shape: # Parts x Batch x Embed
        query = query.permute(1,0,2)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        
        
        #Shape: # Batch x Parts x Embed
        query_parts = query.size(1)
        nparts = key.size(1)  # Dynamically determine nparts based on the input size
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)
        
        query_layer = self.ball.logmap0(mixed_query_layer)
        query_layer = query_layer.view(-1, self.num_attention_heads, query_parts, self.attention_head_size)
        query_layer = self.ball.expmap0(query_layer)

        key_layer = self.ball.logmap0(mixed_key_layer)
        key_layer = key_layer.view(-1, self.num_attention_heads, nparts, self.attention_head_size)
        key_layer = self.ball.expmap0(key_layer)

        value_layer = self.ball.logmap0(mixed_value_layer)
        value_layer = value_layer.view(-1, self.num_attention_heads, nparts, self.attention_head_size)
        value_layer = self.ball.expmap0(value_layer)
        
       
        
        
        # Distance pairwise distance calculation for attention scores using hyperbolic distance
        if self.is_flat:
            key_layer_transposed = key_layer.transpose(-1, -2)
            
#             print('Inf In att: Q layer')
#             print(torch.isinf(query_layer).any())
#             print(' In att: Q layer')
#             print(query_layer.isnan().any())
            
#             print('Inf In att: K layer')
#             print(torch.isinf(key_layer_transposed).any())
#             print('In att: K layer')
#             print(key_layer_transposed.isnan().any())
            
#             print('Max/Min in Q layer:', query_layer.max().item(), query_layer.min().item())
#             print('Max/Min in K layer:', key_layer_transposed.max().item(), key_layer_transposed.min().item())

            
            attention_scores = torch.matmul(query_layer, key_layer_transposed)
            
#             print('Inf In att: pre scaling attention_scores')
#             print(torch.isinf(attention_scores).any())
            
            scaling_factor = self.attention_head_size ** 0.5
            attention_scores =  attention_scores / scaling_factor

        else:
#             print('Max/Min in Q layer:', query_layer.max().item(), query_layer.min().item())
#             print('Max/Min in K layer:', key_layer.max().item(), key_layer.min().item())

            t1 = self.ball.mobius_add(-query_layer.unsqueeze(-2), key_layer.unsqueeze(-2).transpose(2, 3)).norm(dim=-1, p=-2)
            dist = 2.0 * artan_k(t1, k=self.ball.k)
            attention_scores = -1 * dist
        
#         print('In att: attention_scores pre masking')
#         print(attention_scores.isnan().any())
#         print('Inf In att: pre masking attention_scores')
#         print(torch.isinf(attention_scores).any())
#         if attention_scores.isnan().any():
#             print('In Att: attention_scores 1')
        # Apply the key_padding_mask if provided
        if key_padding_mask is not None:
            # Expand mask to [batch_size, num_heads, 1, seq_len] and then subtract a large value where the mask is True
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(key_padding_mask, float('-inf'))
            
#         if attention_scores.isnan().any():
#             print('In Att: attention_scores 2')
        # Apply the attn_mask if provided
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(query.shape[0], self.num_attention_heads, nparts, nparts)
#             print(attn_mask.shape)
            attention_scores += attn_mask
#         if attention_scores.isnan().any():
#             print('In Att: attention_scores 3')
#         print('In att: attention_scores after masking')
#         print(attention_scores.isnan().any())
#         print('Inf In att: attention_scores')
#         print(torch.isinf(attention_scores).any())
        attention_scores = torch.clamp(attention_scores, min=-1e10, max=1e10)
        attention_probs = self.sigmoid_fn(attention_scores)
#         if attention_scores.isnan().any():
#             print('In Att: attention_scores 4')
        
# #         print('Inf In att: att probs')
# #         print(torch.isinf(attention_probs).any())
# #         print('In att: attention_probs')
#         print(attention_probs.isnan().any())
        
        attention_probs = self.dropout(attention_probs)

        if VERBOSE:
            print(torch.max(attention_probs), torch.min(attention_probs))
        if self.is_flat:
#             print('IN FLAT PROCESSING')
#             print('Inf In att: value layer')
#             print(torch.isinf(value_layer).any())
#             print('In att: value layer')
#             print(value_layer.isnan().any())
            context_layer = torch.matmul(attention_probs, value_layer)
#             print('Inf In att: context layer')
#             print(torch.isinf(context_layer).any())
#             print('In att: context layer')
#             print(context_layer.isnan().any())
        else:
#             attention_probs = attention_probs.permute(0,1,3,2)
#             print('Inf In att: value layer')
#             print(torch.isinf(value_layer).any())
#             print(value_layer.shape)
#             print(attention_probs.shape)
            context_layer = self.ball.weighted_midpoint(value_layer, weights=attention_probs, reducedim=[-1], parts=query_parts, dim =-1)
#             if context_layer.isnan().any():
#                 print('In Att: context_layer 1')
            
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = self.ball.logmap0(context_layer)
        context_layer = context_layer.view(new_context_layer_shape)
        context_layer = self.ball.expmap0(context_layer)
        
#         if context_layer.isnan().any():
#             print('In Att: context_layer 2')
        
        context_layer = context_layer.permute(1,0,2)
#         print('Inf In att: context layer')
#         print(torch.isinf(context_layer).any())
#         print('In att: context layer')
#         print(context_layer.isnan().any())
        return context_layer, attention_probs

