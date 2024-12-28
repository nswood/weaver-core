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
from weaver.nn.model.utils import *
from weaver.nn.model.MoG_att_utils import *
from weaver.nn.model.MoG_mlp_utils import *


def two_point_mid(x1,x2, man, w1,w2):
    if man.name == 'Euclidean':
        mid = (x1 * w1 + x2 * w2)/(w1 + w2)
    else:
        lam_x1 = man.lambda_x(x1).unsqueeze(-1)
        lam_x2 = man.lambda_x(x2).unsqueeze(-1)
        t1 = (x1 * lam_x1 *w1 + x2 *lam_x2 * w2)/(lam_x1*w1 + lam_x2*w2 -2)
        mid = man.mobius_scalar_mul(torch.tensor(0.5),t1)
    return mid


class MoG(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 pair_embed_dims=[32, 32, 32],
                 part_experts = 4,
                 part_experts_dim = 32,
                 part_router_n_parts = 16,
                 top_k_part = 2,
                 jet_experts = 4,
                 jet_experts_dim= 32,
                 top_k_jet = 2,
                 shared_expert = True, 
                 shared_expert_ratio = 1,
                 num_heads=4,
                 num_layers=4,
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
                 inter_man_att = -1,
                 dropout_rate = 0.1,
                 learnable = True, 
                 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.part_manifolds = nn.ModuleList()
        self.jet_manifolds = nn.ModuleList()
        total_part_dim = 0
        total_jet_dim = 0
        self.shared_expert = shared_expert
        if shared_expert:
            self.part_manifolds.append(geoopt.Euclidean())
        
            self.jet_manifolds.append(geoopt.Euclidean())
            total_part_dim += int(part_experts_dim*shared_expert_ratio)
            total_jet_dim += int(jet_experts_dim*shared_expert_ratio)


        embed_dims = [64, 64, part_experts_dim]
        fc_params = [[jet_experts_dim,0.1], [jet_experts_dim,0.1], [jet_experts_dim,0.1]]

        for i in range(part_experts):
            self.part_manifolds.append(geoopt.Stereographic(learnable=learnable))
        
        total_part_dim += top_k_part*part_experts_dim
        
        for i in range(jet_experts):
            self.jet_manifolds.append(geoopt.Stereographic(learnable=learnable))
        
        total_jet_dim += top_k_jet*jet_experts_dim
            

        self.top_k_part = top_k_part
        self.top_k_jet = top_k_jet
        self.n_part_man = len(self.part_manifolds)
        self.n_jet_man = len(self.jet_manifolds)

        self.n_part_experts = part_experts if shared_expert else part_experts + 1
        self.n_jet_experts = jet_experts if shared_expert else jet_experts + 1

        # Router for particle experts
        # Takes in the concatenated particle features for top part_router_n_parts particles based on pT
        # and outputs the weights for each expert
        embed_dim = part_experts_dim
        self.part_router_n_parts = part_router_n_parts
        part_router_input= part_router_n_parts*part_experts_dim
        self.part_router = nn.Sequential(
                                    nn.Linear(part_router_input,int(part_router_input*0.25)), 
                                    nn.ReLU(),
                                    nn.Linear(int(part_router_input*0.25), part_experts),
                                    nn.Softmax(dim = -1))

        # Router for jet experts
        # Takes jet-level latent vector output from particle-level processing
        jet_router_input = total_part_dim
        self.jet_router = nn.Sequential(nn.Linear(jet_router_input, int(jet_router_input*0.25)),
                                        nn.ReLU(),
                                        nn.Linear(int(jet_router_input*0.25), jet_experts),
                                        nn.Softmax(dim = -1))
        
        self.norm = nn.LayerNorm(total_jet_dim)
        
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        
        
            
        # (self,
        # manifolds, 
        # top_k_part = 2,
        # embed_dim=128, 
        # num_heads=8, 
        # ffn_ratio=4,
        # dropout=0.1, 
        # attn_dropout=0.1, 
        # activation_dropout=0.1,
        # add_bias_kv=False, 
        # activation='gelu',
        # scale_fc=True, 
        # scale_attn=True, 
        # scale_heads=False, 
        # scale_resids=True,
        # man_att = False,
        # weight_init_ratio =1):
        default_cfg = dict(manifolds=self.part_manifolds,
                           embed_dim=embed_dim, 
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
                           man_att = False)        
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

        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        
        self.part_embedding = nn.ModuleList()
        for i, man in enumerate(self.part_manifolds):
            if i == 0 and shared_expert:
                self.part_embedding.append(nn.Sequential(Manifold_Linear(input_dim, int(part_experts_dim*shared_expert_ratio), ball = man,weight_init_ratio = PM_weight_initialization_factor)))
            else:
                self.part_embedding.append(nn.Sequential(Manifold_Linear(input_dim, part_experts_dim, ball = man,weight_init_ratio = PM_weight_initialization_factor)))
        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            
            cfg_block['man_att'] = False

            self.blocks.append(PM_MoE_Att_Block(**cfg_block))
            
        self.cls_blocks = nn.ModuleList([PM_MoE_Att_Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        

        # Update correct dims here
        if shared_expert and i == 0:
            jet_dim = int(jet_experts_dim*shared_expert_ratio)
        else:
            jet_dim = jet_experts_dim
            

        # Initialize PM_MoE_MLP_Block for jet_fc
        self.jet_experts = PM_MoE_MLP_Block(manifolds = self.jet_manifolds,
                                            input_dim=total_part_dim, 
                                            output_dim=jet_dim, 
                                            num_experts=jet_experts, 
                                            top_k=top_k_jet, 
                                            shared_expert_ratio=shared_expert_ratio)
            
        post_jet_dim = top_k_jet * jet_dim

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
        # N is batch size, P is particles, C is channels
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None
        print(x.shape)
        
        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            
            # want x in (seq_len, batch, embed_dim)

            print(x.shape)
            # x =x.permute(2,0, 1)
            # x =x.permute(1,0, 2)
            
            # print(x.shape)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)
            
            print(x[:self.part_router_n_parts].shape)
            
            x_for_router = x[:self.part_router_n_parts]
            
            print(x_for_router[:self.part_router_n_parts].reshape(x_for_router.size(1),-1).shape)

            router_output = self.part_router(x_for_router[:self.part_router_n_parts].reshape(x_for_router.size(1),-1))
            selected_part_experts = torch.topk(router_output, self.top_k_part, dim = -1).indices
            print(selected_part_experts[0:10])
            # If shared expert, always route to index 0 and select remaining experts from 1 to n
            if self.shared_expert:
                selected_part_experts = torch.cat((torch.zeros_like(selected_part_experts[:,0]).unsqueeze(-1),selected_part_experts+1),dim=-1)
            print(selected_part_experts[0:10])
            
            # Map input onto selected particle manifolds
            # x shape: (P, N, C)

            x_parts = [] # N x K x P x F
            cls_tokens_parts = [] # N x K x F
            for i in range(x.size(1)):
                cur_x = []
                cur_token = []
                for k in selected_part_experts[i]:
                    print(x[:,i].shape)
                    print(self.cls_token[k])
                    cls_tokens = self.cls_token[k].expand(1, x.size(1), -1)
                    print(cls_tokens.shape)
                    cur_x.append(self.part_manifolds[k].expmap0(x[:,i]))
                    cur_token.append(cls_tokens)
                x_parts.append(cur_x)
                cls_tokens_parts.append(cur_token)

            
            del cls_tokens
            del x

            print(len(x_parts))
            
            # transform
            for block in self.blocks:
                print('Att block')
                x_parts = block(x_parts,selected_part_experts, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
                print('Batch', len(x_parts))
                print('Number of experts',len(x_parts[0]))
                for i in range(len(x_parts[0])):
                    print('Expert',i)
                    print('Data shape',x_parts[0][i].shape)


            print('Original cls tokens')
            print('Batch', len(cls_tokens_parts))
            print('Number of experts',len(cls_tokens_parts[0]))
            for i in range(len(cls_tokens_parts[0])):
                print('Expert',i)
                print('Data shape',cls_tokens_parts[0][i].shape)


            for block in self.cls_blocks:
                print('cls block')
                cls_tokens_parts = block(x_parts,selected_part_experts, x_cls=cls_tokens_parts, padding_mask=padding_mask)
                print('Batch', len(cls_tokens_parts))
                print('Number of experts',len(cls_tokens_parts[0]))
                
                for i in range(len(cls_tokens_parts[0])):
                    print('Expert',i)
                    print('Data shape',cls_tokens_parts[0][i].shape)
            # Map to tangent space from particle-representation space
            

            
            tan_cls_tokens_parts = []
            for i in range(len(cls_tokens_parts)):
                cur = []
                for j,k in enumerate(selected_part_experts[i]):
                    cur.append(self.part_manifolds[k].logmap0(cls_tokens_parts[i][j]))
                tan_cls_tokens_parts.append(cur)
                    

            # Concatenate particle man outputs
            if self.top_k_part > 1:
                x_cls = [torch.cat(a,dim=-1) for a in tan_cls_tokens_parts]
            else:
                x_cls = tan_cls_tokens_parts[0]
            
            print('Len x_cls',len(x_cls))
            print('Len x_cls[0]',len(x_cls[0]))
            print('x_cls[0][0]',x_cls[0][0].shape)
            x_cls = torch.cat(x_cls,dim=0)
            print('x_cls',x_cls.shape)
            x_cls = x_cls.squeeze(1)
            del cls_tokens_parts
                
            router_output = self.jet_router(x_cls)

            selected_jet_experts = torch.topk(router_output, self.top_k_jet, dim=-1).indices
            
            # If shared expert, always route to index 0 and select remaining experts from 1 to n
            if self.shared_expert:
                selected_part_experts = torch.cat((torch.zeros_like(selected_part_experts[:,0]).unsqueeze(-1),selected_jet_experts+1),dim=-1)

            x_jets = [] # Batch x K x N x F
            for i in range(x_cls.size(0)):
                cur_x = []
                
                for k in selected_part_experts[i]:
                    cur_x.append(self.jet_manifolds[k].expmap0(x_cls[i]))
                    
                x_jets.append(cur_x)
            del x_cls

            proc_jets = self.jet_experts(x_jets, selected_jet_experts)
            del x_jets
            
            x_jets_tan = [man.logmap0(proc_jets[i]) for i,man in enumerate(self.jet_manifolds)]
            
                    
            if embed:
                proc_jets = [torch.squeeze(a,0) for a in proc_jets]
                x_jets_tan = [torch.squeeze(a,0) for a in x_jets_tan]
                
                return proc_jets, x_jets_tan,list(self.jet_manifolds),selected_jet_experts
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



