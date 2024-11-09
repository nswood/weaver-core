# from .radam import RAdam
from .lookahead import Lookahead
from .riemann_lookahead import Riemann_Lookahead
import geoopt
from geoopt.optim.radam import RiemannianAdam 

def Riemiann_ranger(params,
           lr=1e-3,  # lr
           betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # RAdam options
           alpha=0.5, k=6,  # LookAhead options
           ):
    radam = RiemannianAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Riemann_Lookahead(radam, alpha, k)
