import os
from typing import Optional
import torch
from . import util
import math
import hashlib
import os
import wandb

wandb.login()


class InputWrapper(torch.nn.Module):
    def __init__(self, v):
        super(InputWrapper, self).__init__()
        self.v = v

    def net_inp(
        self,
        t: torch.Tensor,  # [1]
        x: torch.Tensor   # [batch x dim]
    ) -> torch.Tensor:    # [batch x (1 + dim)]
        """Concatenate time over the batch dimension."""
        inp = torch.cat((t.repeat(x.shape[0]).unsqueeze(1), x), dim=1)
        return inp

    def forward(self, x, t):
        tx = self.net_inp(t, x)
        return self.v(tx)


def make_fc_net(hidden_sizes, in_size, out_size, inner_act, final_act, **config):
    sizes = [in_size] + hidden_sizes + [out_size]
    net = []
    for i in range(len(sizes) - 1):
        net.append(torch.nn.Linear(
            sizes[i], sizes[i+1]))
        if i != len(sizes) - 2:
            net.append(make_activation(inner_act))
            continue
        else:
            if make_activation(final_act):
                net.append(make_activation(final_act))

    v_net = torch.nn.Sequential(*net)
    return InputWrapper(v_net)


def make_It(path='linear', gamma=None, gamma_dot=None, gg_dot=None):
    """gamma function must be specified if using the trigonometric interpolant"""

    if path == 'linear':

        def a(t): return (1-t)
        def adot(t): return -1.0
        def b(t): return t
        def bdot(t): return 1.0
        def It(t, x0, x1): return a(t)*x0 + b(t)*x1
        def dtIt(t, x0, x1): return adot(t)*x0 + bdot(t)*x1

    elif path == 'trig':
        if gamma == None:
            raise TypeError(
                "Gamma function must be provided for trigonometric interpolant!")

        def a(t): return torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)
        def b(t): return torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)

        def adot(t): return -gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t) \
            - 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        def bdot(t): return -gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t) \
            + 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)

        def It(t, x0, x1): return a(t)*x0 + b(t)*x1
        def dtIt(t, x0, x1): return adot(t)*x0 + bdot(t)*x1

    elif path == 'encoding-decoding':

        def a(t):
             return torch.where(t <= 0.5, torch.cos(math.pi*t)**2, torch.zeros_like(t))
 
        def adot(t):
            return torch.where(t <= 0.5, -2*math.pi * torch.cos(math.pi*t)*torch.sin(math.pi*t), torch.zeros_like(t))

        def b(t):
            return torch.where(t > 0.5,  torch.cos(math.pi*t)**2, torch.zeros_like(t))

        def bdot(t):
            return torch.where(t > 0.5,  -2*math.pi * torch.cos(math.pi*t)*torch.sin(math.pi*t), torch.zeros_like(t))

        def It(t, x0, x1): return a(t)*x0 + b(t)*x1
        def dtIt(t, x0, x1): return adot(t)*x0 + bdot(t)*x1

    elif path == 'one-sided-linear':

        def a(t): return (1-t)
        def adot(t): return -1.0
        def b(t): return t
        def bdot(t): return 1.0

        def It(t, x0, x1): return a(t)*x0 + b(t)*x1
        def dtIt(t, x0, x1): return adot(t)*x0 + bdot(t)*x1

    elif path == 'one-sided-trig':

        def a(t): return torch.cos(0.5*math.pi*t)
        def adot(t): return -0.5*math.pi*torch.sin(0.5*math.pi*t)
        def b(t): return torch.sin(0.5*math.pi*t)
        def bdot(t): return 0.5*math.pi*torch.cos(0.5*math.pi*t)

        def It(t, x0, x1): return a(t)*x0 + b(t)*x1
        def dtIt(t, x0, x1): return adot(t)*x0 + bdot(t)*x1

    elif path == 'mirror':
        if gamma == None:
            raise TypeError(
                "Gamma function must be provided for mirror interpolant!")

        def a(t): return gamma(t)
        def adot(t): return gamma_dot(t)
        def b(t): return torch.ones_like(t)
        def bdot(t): return torch.zeros_like(t)
 
        def It(t, x0, x1): return b(t)*x1 + a(t)*x0
        def dtIt(t, x0, x1): return adot(t)*x0

    elif path == 'custom':
        return None, None, None

    else:
        raise NotImplementedError(
            "The interpolant you specified is not implemented.")

    return It, dtIt, (a, adot, b, bdot)


def make_gamma(gamma_type: Optional[str] = 'brownian', aval=None):
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    if gamma_type == 'brownian':
        def gamma(t): return torch.sqrt(t*(1-t))
        def gamma_dot(t): return (1/(2*torch.sqrt(t*(1-t)))) * (1 - 2*t)
        def gg_dot(t): return (1/2)*(1-2*t)

    elif gamma_type == 'a-brownian':
        def gamma(t): return torch.sqrt(aval*t*(1-t))
        def gamma_dot(t): return (
            1/(2*torch.sqrt(aval*t*(1-t)))) * aval*(1 - 2*t)

        def gg_dot(t): return (aval/2)*(1-2*t)

    elif gamma_type == 'zero':
        gamma = gamma_dot = gg_dot = lambda t: torch.zeros_like(t)

    elif gamma_type == 'bsquared':
        def gamma(t): return t*(1-t)
        def gamma_dot(t): return 1 - 2*t
        def gg_dot(t): return gamma(t)*gamma_dot(t)

    elif gamma_type == 'sinesquared':
        def gamma(t): return torch.sin(math.pi * t)**2
        def gamma_dot(t): return 2*math.pi * \
            torch.sin(math.pi * t)*torch.cos(math.pi*t)

        def gg_dot(t): return gamma(t)*gamma_dot(t)

    elif gamma_type == 'sigmoid':
        f_val = 10.0
 
        def gamma(t):
            f = torch.as_tensor(f_val).to(t)
            return torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f *
                                                                 (t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
 
        def gamma_dot(t):
            f = torch.as_tensor(f_val).to(t)
            return (-f)*(1 - torch.sigmoid(-1 + f*(t - (1/2))))*torch.sigmoid(-1 +
                                                                             f*(t - (1/2))) + f*(1 - torch.sigmoid(1 + f*(t - (1/2))))*torch.sigmoid(1 + f*(t - (1/2)))
 
        def gg_dot(t): return gamma(t)*gamma_dot(t)
 
    elif gamma_type == None:
        def gamma(t): return torch.zeros_like(t)         # no gamma
        def gamma_dot(t): return torch.zeros_like(t)     # no gamma
        def gg_dot(t): return torch.zeros_like(t)        # no gamma

    else:
        raise NotImplementedError(
            "The gamma you specified is not implemented.")

    return gamma, gamma_dot, gg_dot


def make_activation(act):
    if act == 'elu':
        return torch.nn.ELU()
    if act == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act == 'elu':
        return torch.nn.ELU()
    elif act == 'relu':
        return torch.nn.ReLU()
    elif act == 'tanh':
        return torch.nn.Tanh()
    elif act == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act == 'softplus':
        return torch.nn.Softplus()
    elif act == 'silu':
        return torch.nn.SiLU()
    elif act == 'Sigmoid2Pi':
        class Sigmoid2Pi(torch.nn.Sigmoid):
            def forward(self, input):
                return 2*np.pi*super().forward(input) - np.pi
        return Sigmoid2Pi()
    elif act == 'none' or act is None:
        return None
    else:
        raise NotImplementedError(f'Unknown activation function {act}')
