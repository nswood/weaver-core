import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from weaver.utils.logger import _logger

import os
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


def find_neighbors_knn(four_momentum_tensor, dists, index=None, k=5):
    # Randomly select a point
    num_points = four_momentum_tensor.shape[0]
    if index is None:
        index = random.randint(0, num_points - 1)
    selected_point = four_momentum_tensor[index]
    
    # Find the indices of the k-nearest neighbors
    neighbors_indices = torch.topk(dists[index], k, largest=False).indices

    neighbors = four_momentum_tensor[neighbors_indices]

    return selected_point, neighbors, neighbors_indices


class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(CrossAttentionPooling, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, padding_mask=None):
        """
        x: Tensor of shape (seq_len, batch_size, embed_dim)
        padding_mask: Tensor of shape (batch_size, seq_len)
                      True indicates positions to be masked.
        """
        batch_size = x.size(1)
        query = self.query.repeat(1, batch_size, 1)
        
        # Apply attention with padding mask
        output, _ = self.attention(query, x, x, key_padding_mask=padding_mask)
        
        return output.squeeze(0)

class MoG_part_lvl_MLP(nn.Module):
    
    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 pair_embed_dims=[64, 64, 64],
                 ffn_ratio=2,
                 part_experts = 6,
                 part_expert_curvature_init = [],
                 part_experts_dim = 32,
                 part_router_n_parts = 32,
                 top_k_part = 2,
                 jet_experts = 6,
                 jet_expert_curvature_init =[],
                 jet_experts_dim= 32,
                 top_k_jet = 2,
                 shared_expert = True, 
                 shared_expert_ratio = 2,
                 all_k = [4, 8, 16],
                 num_heads=1,
                 num_layers=2,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='relu',
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

        self.all_k = all_k
        self.local_geom_pooling = nn.ModuleList(CrossAttentionPooling(part_experts_dim, 1) for _ in all_k)
        
            
        
        



        if activation == 'relu':
            self.activation = nn.ReLU()

        if shared_expert:
            self.part_manifolds.append(geoopt.Euclidean())
            self.jet_manifolds.append(geoopt.Euclidean())

            self.part_shared_expert_dim = int(part_experts_dim*shared_expert_ratio)
            self.jet_shared_expert_dim = int(jet_experts_dim*shared_expert_ratio)

            total_part_dim += self.part_shared_expert_dim
            total_jet_dim += self.jet_shared_expert_dim
        
        embed_dims = [64, 64, part_experts_dim]
        fc_params = [[jet_experts_dim,0.1], [jet_experts_dim,0.1], [jet_experts_dim,0.1]]
        
        if part_expert_curvature_init == []:
            part_expert_curvature_init = np.linspace(-int(part_experts/2), int(part_experts/2), part_experts)

        if jet_expert_curvature_init == []:
            jet_expert_curvature_init = np.linspace(-int(jet_experts/2), int(jet_experts/2), jet_experts)
        
        for i in range(part_experts):
            self.part_manifolds.append(geoopt.StereographicExact(learnable=learnable, k = part_expert_curvature_init[i]))

        total_part_dim += top_k_part*part_experts_dim
        
        for i in range(jet_experts):
            self.jet_manifolds.append(geoopt.StereographicExact(learnable=learnable, k = jet_expert_curvature_init[i]))

        total_jet_dim += top_k_jet*jet_experts_dim
        
        print('Particle Manifolds:')
        print('====================')
        print('====================')
        for i,man in enumerate(self.part_manifolds):
            if man.name == 'Euclidean':
                print('Euclidean Manifold',i)
            else:
                print('Stereographic Manifold:',i, 'Curvature:',man.k)
        print('====================')
        print('==================== \n\n')

        
        
        print('Jet Manifolds:')
        print('====================')
        print('====================')
        for i,man in enumerate(self.jet_manifolds):
            if man.name == 'Euclidean':
                print('Euclidean Manifold',i)
            else:
                print('Stereographic Manifold:',i, 'Curvature:',man.k)
        print('====================')
        print('====================')

        self.top_k_part = top_k_part
        self.top_k_jet = top_k_jet
        self.n_part_man = len(self.part_manifolds)
        self.n_jet_man = len(self.jet_manifolds)

        self.n_part_experts = part_experts + 1 if shared_expert else part_experts
        self.n_jet_experts = jet_experts + 1 if shared_expert else jet_experts
        

        # Particle Router
        embed_dim = part_experts_dim        
        part_router_input= input_dim

        dim_dif = part_router_input - part_experts_dim
        d1 = part_router_input - int(dim_dif*0.5)
        self.part_router = nn.Sequential(
                                    nn.Linear(part_router_input,d1), 
                                    self.activation,
                                    nn.Linear(d1, part_experts),
                                    nn.Softmax(dim = -1))
        
        # Jet Router
        jet_router_input = total_part_dim
        dim_dif = jet_router_input - jet_experts_dim
        d1 = jet_router_input - int(dim_dif*0.5)
        self.jet_router = nn.Sequential(nn.Linear(jet_router_input, d1),
                                        self.activation,
                                        nn.Linear(d1, jet_experts),
                                        nn.Softmax(dim = -1))
        
        # Normalization Layers
        self.part_experts_norms = nn.ModuleList(nn.RMSNorm(input_dim) for i in range(self.n_part_experts))
        self.jet_experts_norms = nn.ModuleList(nn.RMSNorm(total_part_dim) for i in range(self.n_jet_experts))


        self.norm1 = nn.RMSNorm(total_part_dim)
        if shared_expert:
            self.norm2 = nn.RMSNorm(self.jet_shared_expert_dim)
            self.norm = nn.RMSNorm(self.jet_shared_expert_dim)
        else:
            self.norm2 = nn.RMSNorm(jet_experts_dim)
            self.norm = nn.RMSNorm(jet_experts_dim)

        self.for_inference = for_inference
        self.use_amp = use_amp
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        # Particle MLP Experts
        mlp_input_dim = part_experts_dim
        self.part_experts = PM_MoE_MLP_Block(manifolds = self.jet_manifolds,
                                            input_dim=input_dim, 
                                            output_dim=part_experts_dim, 
                                            num_experts=part_experts, 
                                            top_k=top_k_part, 
                                            shared_expert_ratio=shared_expert_ratio,
                                            shared_expert=shared_expert,
                                            ffn_ratio = ffn_ratio)
        
        # Jet MLP Experts
        mlp_input_dim = part_experts_dim
        self.jet_experts = PM_MoE_MLP_Block(manifolds = self.jet_manifolds,
                                            input_dim=total_part_dim, 
                                            output_dim=jet_experts_dim, 
                                            num_experts=jet_experts, 
                                            top_k=top_k_jet, 
                                            shared_expert_ratio=shared_expert_ratio,
                                            shared_expert=shared_expert,
                                            ffn_ratio = ffn_ratio)
        
        if shared_expert:
            post_jet_dim = self.jet_shared_expert_dim
        else:
            post_jet_dim = jet_experts_dim
        dim_dif = post_jet_dim - num_classes
        d1 = post_jet_dim - int(dim_dif*0.5)
        d2 = d1 - int(dim_dif*0.25)
        self.final_fc = nn.Sequential(nn.Linear(post_jet_dim, d1), 
                                        self.activation,
                                        nn.Linear(d1, d2), 
                                        self.activation,
                                        nn.Linear(d2, num_classes))

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
        # print(x.shape)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp): 

            with torch.no_grad():
                if not self.for_inference:
                    if uu_idx is not None:
                        uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
                x, v, mask, uu = self.trimmer(x, v, mask, uu)
                padding_mask = ~mask.squeeze(1)  # (N, P)    
            x = x.permute(2,0,1) # (N, C, P) -> (P, N, C)
            v = v.permute(2,0,1) if v is not None else None  # (P, N, C)
            local_geom_features = []
            # Iterate over jets
            for i in range(x.size(1)):
                # Calculating interaction features for particle EMD
                cur_v = v[:,i] if v is not None else None
                cur_x = x[:,i]

                part_energy = cur_v[:, 3] if v is not None else None  # (P, N)
                part_pT = torch.norm(cur_v[:,:2], dim=-1) if v is not None else None  # (P, N)
                part_eta = 0.5 * torch.log((part_pT + v[:, 2]) / (part_pT - cur_v[:, 2])) if v is not None else None  # (P, N)
                part_phi = torch.atan2(cur_v[:, 1], cur_v[:, 0]) if v is not None else None  # (P, N)

                # Compute pairwise energy differences |E_i - E_j|
                energy_diffs = torch.abs(part_energy[:, None] - part_energy[None, :])

                # Compute pairwise angular distances ΔR_ij = sqrt((η_i - η_j)^2 + (φ_i - φ_j)^2)
                delta_eta = part_eta[:, None] - part_eta[None, :]
                delta_phi = part_phi[:, None] - part_phi[None, :]

                # Ensure phi differences wrap around correctly (handle 2pi periodicity)
                delta_phi = torch.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi

                # Calculate ΔR
                delta_R = torch.sqrt(delta_eta**2 + delta_phi**2)

                # Calculate EMD-inspired distance: E_diff * ΔR / R (where R is a scale factor, e.g., jet radius)
                R = 1  # Example jet radius
                dists = (energy_diffs * delta_R) / R

                
                output = []
                # Loop over all particles in the jet
                for j in range(cur_v.size(0)):
                    cur_output =[]
                    cur_mask = padding_mask[j]
                    is_zero_pad = cur_mask[j,j] == 0
                    if is_zero_pad:
                        output.append(torch.zeros(len(self.all_k)*self.part_experts_dim))
                        continue
                    else:
                        for k_i, k in enumerate(self.all_k):
                            # need to do something here for zero padding
                            selected_point, neighbors, neighbors_indices = find_neighbors_knn(cur_v, dists, index=j, k=k)
                            neighbor_mask = cur_mask[neighbors_indices,neighbors_indices]

                            # Aggregate KNN using attention
                            attention_aggregated_neighbors = self.local_geom_pooling[k_i](neighbors, padding_mask=neighbor_mask)
                            cur_output.append(attention_aggregated_neighbors)
                        # Stack aggregated local geometry features
                        output.append(torch.cat(cur_output, dim=-1))
                
                local_geom_features.append(torch.stack(output, dim=0))
            local_geom_features = torch.stack(local_geom_features, dim=1)
                
            # Router per particle
            part_router_output = self.part_router(local_geom_features)
            selected_part_experts = torch.topk(part_router_output, self.top_k_part, dim = -1).indices

            # If shared expert, always route to index 0 and select remaining experts from 1 to n
            if self.shared_expert:
                selected_part_experts = torch.cat((torch.zeros_like(selected_part_experts[:,:,0]).unsqueeze(-1),selected_part_experts+1),dim=-1)
            
            print('selected_part_experts.shape:',selected_part_experts.shape)
            # Map input onto selected particle manifolds
            # x shape: (P, N, C)

            x_parts = [] # N x K x P x F
            for i in range(x.size(1)):
                cur_x = []
                for k in selected_part_experts[i]:
                    if self.part_manifolds[k].name == 'Euclidean':
                        cur_x.append(self.part_experts_norms[k](x[:,i]))
                    else:
                        cur_x.append(self.part_manifolds[k].expmap0(self.part_experts_norms[k](x[:,i])))
                x_parts.append(cur_x)

            del x

            proc_parts = self.part_experts(x_parts, selected_part_experts)
            
            # Add + Norm aggregation
            tan_cls_tokens_parts = []
            for i in range(len(proc_parts)):
                cur = []
                for j, k in enumerate(selected_part_experts[i]):
                    if self.part_manifolds[k].name == 'Euclidean':
                        cur.append(proc_parts[i][j])
                    else:
                        cur.append(self.part_manifolds[k].logmap0(proc_parts[i][j]))
                    
                tan_cls_tokens_parts.append(torch.cat(cur, dim=-1))

            # Convert list to tensor
            tan_cls_tokens_parts = torch.stack(tan_cls_tokens_parts, dim=0)  # Batch x N x F

            # Add and norm over N
            tan_cls_tokens_parts = torch.sum(tan_cls_tokens_parts, dim=1)  # Batch x F
            x_cls = self.norm1(tan_cls_tokens_parts)  # Batch x F
            x_cls = x_cls.reshape(x_cls.size(0),-1)
                
            jet_router_output = self.jet_router(x_cls)

            selected_jet_experts = torch.topk(jet_router_output, self.top_k_jet, dim=-1).indices
            
            # If shared expert, always route to index 0 and select remaining experts from 1 to n
            if self.shared_expert:
                selected_jet_experts = torch.cat((torch.zeros_like(selected_jet_experts[:,0]).unsqueeze(-1),selected_jet_experts+1),dim=-1)

            x_jets = [] # Batch x K x N x F
            for i in range(x_cls.size(0)):
                cur_x = []
                
                for k in selected_jet_experts[i]:
                    if self.jet_manifolds[k].name == 'Euclidean':
                        cur_x.append(self.jet_experts_norms[k](x_cls[i]))
                    else:
                        cur_x.append(self.jet_manifolds[k].expmap0(self.jet_experts_norms[k](x_cls[i])))
                    
                x_jets.append(cur_x)
            del x_cls

            proc_jets = self.jet_experts(x_jets, selected_jet_experts)
              
            del x_jets

            # Add + Norm Aggregation
            x_jets_tan = [] # N x K x P x F
            
            for i in range(len(proc_jets)):
                cur_x = []
                for j, k in enumerate(selected_jet_experts[i]):
                    if self.jet_manifolds[k].name == 'Euclidean':
                        temp_log = proc_jets[i][j]
                    else:
                        temp_log = self.jet_manifolds[k].logmap0(proc_jets[i][j])
                    
                    if self.shared_expert and k != 0:
                        temp_log = torch.nn.functional.pad(temp_log, (0, self.jet_shared_expert_dim - temp_log.size(-1)))   
                    cur_x.append(temp_log)

                # Stack along the second dimension (K) and sum over it
                cur_x = torch.stack(cur_x, dim=1)  # N x K x F
                cur_x = torch.sum(cur_x, dim=1)  # N x F
                cur_x = self.norm2(cur_x)  # Apply normalization
                
                x_jets_tan.append(cur_x)
                    
            if embed:
                proc_jets = [torch.squeeze(a,0) for a in proc_jets]
                x_jets_tan = [torch.squeeze(a,0) for a in x_jets_tan]
                
                return proc_jets, x_jets_tan,list(self.jet_manifolds),selected_jet_experts
            
            
            x_out = torch.vstack(x_jets_tan)
            
            # Regular LayerNorm    
            x_out = self.norm(x_out).squeeze(0)
            
            # Final classification FC
            output = self.final_fc(x_out).squeeze(0)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output, part_router_output, jet_router_output



