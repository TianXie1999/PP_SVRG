from torch.optim import Optimizer

# adapted from pytorch official implementation. 
class SGD_Simple(Optimizer):
    r"""Implements stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    """
    def __init__(self, params, lr, weight_decay=0):
        print("Using optimizer: SGD_Simple")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SGD_Simple, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)    
                p.data.add_(d_p, alpha=-lr)