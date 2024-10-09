import torch
import torch.nn as nn
import math

from weaver.nn.model.frechetmean.manifolds import Poincare, Lorentz
from weaver.nn.model.frechetmean.frechet import frechet_mean

# Modified
class RiemannianBatchNorm(nn.Module):
    def __init__(self, dim):
        super(RiemannianBatchNorm, self).__init__()
        self.man = Poincare(K = -1.2)
        self.mean = nn.Parameter(self.man.zero_tan(self.man.dim_to_sh(dim)))
        self.var = nn.Parameter(torch.tensor(1.0))

        # statistics
        self.first = nn.Parameter(torch.tensor(1.0))
        self.running_mean = nn.Parameter(torch.tensor(0.0),requires_grad = False)
        self.running_var = nn.Parameter(torch.tensor(1.0),requires_grad = False)
        self.updates = nn.Parameter(torch.tensor(0.0),requires_grad = False)

    def forward(self,x, K,training=True, momentum=0.9):
        self.man = Poincare(K)
        shapes = x.shape
        if len(shapes) > 2:
            x=torch.reshape(x,(shapes[0]*shapes[1],shapes[2]))
        on_manifold = self.man.exp0(self.mean)
        if training:
            # frechet mean, use iterative and don't batch (only need to compute one mean)
            input_mean = frechet_mean(x, self.man)
            
            input_var = self.man.frechet_variance(x, input_mean)
            
            # transport input from current mean to learned mean
            input_logm = self.man.transp(
                input_mean,
                on_manifold,
                self.man.log(input_mean, x),
            )

            # re-scaling
            input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

            # project back
            output = self.man.exp(on_manifold.unsqueeze(-2), input_logm)

            self.updates += 1
            if self.first.item():
                self.running_mean.data = input_mean.clone().detach() 
                self.running_var.data = input_var.clone().detach()
                self.first.data  = torch.tensor(0.0)
            else:
                self.running_mean.data = self.man.exp(
                    self.running_mean,
                    (1 - momentum) * self.man.log(self.running_mean, input_mean)
                )
                self.running_var.data = (
                    1 - 1 / self.updates
                ) * self.running_var + input_var / self.updates
        else:
#             if self.updates == 0:
#                 raise ValueError("must run training at least once")

            input_mean = frechet_mean(x, self.man)
            input_var = self.man.frechet_variance(x, input_mean)

            input_logm = self.man.transp(
                input_mean,
                on_manifold,#self.running_mean,
                self.man.log(input_mean, x),
            )

#             assert not torch.any(torch.isnan(input_logm))
            if x.shape[0] > 1:
            # re-scaling
                input_logm = (
                    self.running_var / (x.shape[0] / (x.shape[0] - 1) * input_var + 1e-6)
                ).sqrt() * input_logm

            # project back
            output = self.man.exp(on_manifold, input_logm)
        del x
        if len(shapes) > 2:
            output=torch.reshape(output,(shapes[0],shapes[1],shapes[2]))
        
        return output
