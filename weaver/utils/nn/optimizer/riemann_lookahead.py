import math
import torch
import itertools as it
from torch.optim import Optimizer
from collections import defaultdict

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Euclidean

# https://github.com/lonePatient/lookahead_pytorch/blob/1055128057408fe8533ffa30654551a317f07f0a/optimizer.py
class Riemann_Lookahead(Optimizer):
    '''
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''

    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        '''
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        '''
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.defaults = optimizer.defaults
        self._default_manifold = Euclidean()
        self.reset()

    def reset(self):
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self.reset()
                     
    def man_midpoint(self,man, p,cached,alpha):
        p_lam_x = man.lambda_x(p).unsqueeze(-1)
        cached_lam_x = man.lambda_x(cached).unsqueeze(-1)
        denom = (alpha*(p_lam_x-1) + (1-alpha)*(cached_lam_x-1))
        t1 = (p * p_lam_x*alpha + cached *cached_lam_x*(1-alpha))/denom
        mid = man.mobius_scalar_mul(torch.tensor(0.5),t1)
        return mid
                     
    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    
                    param_state = self.state[p]
                    
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                        if manifold.name == 'Euclidean':
                            p.data.mul_(self.alpha).add_(param_state['cached_params'], alpha=1.0 - self.alpha)
                        else:
                            p.data = self.man_midpoint(manifold, p.data,param_state['cached_params'],self.alpha)
                        
                    else:
                        p.data.mul_(self.alpha).add_(param_state['cached_params'], alpha=1.0 - self.alpha)  # crucial line

                    param_state['cached_params'].copy_(p.data)
#                     if self.pullback_momentum == "pullback":
#                         internal_momentum = self.optimizer.state[p]["momentum_buffer"]
#                         self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
#                             param_state["cached_mom"], alpha=1.0 - self.alpha)
#                         param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
#                     elif self.pullback_momentum == "reset":
#                         self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
