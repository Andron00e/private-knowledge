import torch
from torch.optim import Optimizer
from typing import List, Optional

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class NoisySGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                weight_decay=0, nesterov=False, is_clipped=False, 
                clipping_level=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.is_clipped = is_clipped
        self.clipping_level = clipping_level
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        is_clipped=is_clipped, clipping_level=clipping_level)
        super(NoisySGD, self).__init__(params, defaults)
        
        
    def __setstate__(self, state):
        super(NoisySGD, self).__setstate__(state)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            is_clipped = self.is_clipped
            clipping_level = self.clipping_level

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                #noise = torch.randn(p.data.shape)
                #d_p.add_(noise, p.data)

                if is_clipped:
                    torch.nn.utils.clip_grad_norm_(d_p, clipping_level)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)

                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
    
