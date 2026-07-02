from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import vmap
from torchdiffeq import odeint_adjoint as odeint

from . import fabrics

Time = torch.Tensor
Sample = torch.Tensor
Velocity = torch.nn.Module
Score = torch.nn.Module
SingleModel = torch.nn.Module


def compute_div(
    f: Callable[[Time, Sample], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor  # [batch x dim]
):
    """Compute the divergence of f(x,t) with respect to x, assuming that x is batched. Assumes data is [bs, d]"""
    bs = x.shape[0]
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        t.requires_grad_(True)
        f_val = f(x, t)
        divergence = 0.0
        for i in range(x.shape[1]):
            divergence += \
                torch.autograd.grad(
                    f_val[:, i].sum(), x, create_graph=True
                )[0][:, i]

    return divergence.view(bs)


class SFromEta(torch.nn.Module):
    """Class for turning a noise model into a score model."""

    def __init__(
        self,
        gamma: Callable[[Time], torch.Tensor],
    ) -> None:
        super(SFromEta, self).__init__()
        self.gamma = gamma

    # Eq. 3.17
    def forward(self, eta, t):

        # val = (eta / self.gamma(t))               # Original version
        val = -(eta / self.gamma(t))                # IMO: Correct version

        return val


class BFromVS(torch.nn.Module):

    """
    Class for turning a velocity model $v$ and a score model $s$ into a drift model $b$.
    If one-sided interpolation, gg_dot should be replaced with alpha*alpha_dot.
    """

    def __init__(
        self,
        gamma: Callable[[Time], torch.Tensor],
        gamma_dot: Callable[[Time], torch.Tensor],
    ) -> None:
        super(BFromVS, self).__init__()
        self.gamma = gamma
        self.gamma_dot = gamma_dot

    def forward(self, v, s, t):
        return v - self.gamma(t) * self.gamma_dot(t) * s


class Interpolant(torch.nn.Module):
    """
    Class for all things interpoalnt $x_t = I_t(x_0, x_1) + \gamma(t)z.
    If path is one-sided, then interpolant constructs x_t = a(t) x_0 + b(t) x_1 with x_0 ~ N(0,1).

    path: str,    what type of interpolant to use, e.g. 'linear' for linear interpolant. see fabrics for options
    gamma_type:   what type of gamma function to use, e.g. 'brownian' for $\gamma(t) = \sqrt{t(1-t)}
    """

    def __init__(
        self,
        path: str,
        gamma_type: str | None,
        gamma: Callable[[Time], torch.Tensor] | None = None,
        gamma_dot: Callable[[Time], torch.Tensor] | None = None,
        gg_dot: Callable[[Time], torch.Tensor] | None = None,
        It: Callable[[Time, Sample, Sample], Sample] | None = None,
        dtIt: Callable[[Time, Sample, Sample], Sample] | None = None
    ):
        super(Interpolant, self).__init__()

        self.path = path
        if gamma == None:
            if self.path == 'one-sided-linear' or self.path == 'one-sided-trig':
                gamma_type = None
            self.gamma, self.gamma_dot, self.gg_dot = fabrics.make_gamma(
                gamma_type=gamma_type)
        else:
            self.gamma, self.gamma_dot, self.gg_dot = gamma, gamma_dot, gg_dot
        if self.path == 'custom':
            print('Assuming interpolant was passed in directly...')
            self.It = It
            self.dtIt = dtIt
            assert self.It != None
            assert self.dtIt != None

        self.It, self.dtIt, ab = fabrics.make_It(
            path, self.gamma, self.gamma_dot, self.gg_dot)
        self.a, self.adot, self.b, self.bdot = ab[0], ab[1], ab[2], ab[3]

    def calc_xt(self, t: Time, x0: Sample, x1: Sample, z: Sample = None):
        
        if self.path == 'one-sided-linear' or self.path == 'mirror' or self.path == 'one-sided-trig':
            return self.It(t, x0, x1), x0
        else:
            if z is None:
                z = torch.randn(x0.shape).to(t)

            return self.It(t, x0, x1) + self.gamma(t)*z, z

    def calc_antithetic_xts(self, t: Time, x0: Sample, x1: Sample, z: Sample = None):
        """
        Used if estimating the score and not the noise (eta). 
        """
        if self.path == 'one-sided-linear' or self.path == 'one-sided-trig':
            It_p = self.b(t)*x1 + self.a(t)*x0
            It_m = self.b(t)*x1 - self.a(t)*x0
            return It_p, It_m, x0
        else:
            if z is None:
                z = torch.randn(x0.shape).to(t)

            gam = self.gamma(t)
            It = self.It(t, x0, x1)
            return It + gam*z, It - gam*z, z

    def forward(self, _):
        raise NotImplementedError("No forward pass for interpolant.")


class PFlowRHS(torch.nn.Module):
    def __init__(self, b: Velocity, interpolant: Interpolant, sample_only=False):
        super(PFlowRHS, self).__init__()
        self.b = b
        self.interpolant = interpolant
        self.sample_only = sample_only

    def setup_rhs(self):
        def rhs(x: torch.Tensor, t: torch.Tensor):
            self.b.to(x)

            t = t.unsqueeze(0)
            return self.b(x, t)

        self.rhs = rhs

    def forward(self, t: torch.Tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t), -compute_div(self.rhs, x, t))

    def reverse(self, t: torch.Tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (-self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (-self.rhs(x, t), compute_div(self.rhs, x, t))


class MirrorPFlowRHS(torch.nn.Module):
    def __init__(self, s: Velocity, interpolant: Interpolant, sample_only=False):
        super(MirrorPFlowRHS, self).__init__()
        self.s = s
        self.interpolant = interpolant
        self.sample_only = sample_only

    def setup_rhs(self):
        def rhs(x: torch.Tensor, t: torch.Tensor):
            # tx = net_inp(t, x)
            self.s.to(x)

            t = t.unsqueeze(0)
            return self.interpolant.gg_dot(t)*self.s(x, t)

        self.rhs = rhs

    def forward(self, t: torch.Tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t), -compute_div(self.rhs, x, t))

    def reverse(self, t: torch.Tensor, states: Tuple):
        x = states[0]
        if self.sample_only:
            return (-self.rhs(x, t), torch.zeros(x.shape[0]).to(x))
        else:
            return (-self.rhs(x, t), compute_div(self.rhs, x, t))


@dataclass
class PFlowIntegrator:
    b: Velocity
    method: str
    interpolant: Interpolant
    start_end: tuple = (0.0, 1.0)
    n_step: int = 5
    atol: float = 1e-5
    rtol: float = 1e-5
    sample_only: bool = False
    mirror:      bool = False

    def __post_init__(self) -> None:
        if self.mirror:
            self.rhs = MirrorPFlowRHS(
                s=self.b, interpolant=self.interpolant, sample_only=self.sample_only)
        else:
            self.rhs = PFlowRHS(
                b=self.b, interpolant=self.interpolant, sample_only=self.sample_only)
        self.rhs.setup_rhs()

        self.start, self.end = self.start_end[0], self.start_end[1]

    def rollout(self, x0: Sample, reverse=False):
        
        if reverse:
            integration_times = torch.linspace(self.end, self.start, self.n_step).to(x0)
        else:
            integration_times = torch.linspace(self.start, self.end, self.n_step).to(x0)
            
        dlogp = torch.zeros(x0.shape[0]).to(x0)

        state = odeint(
            self.rhs,
            (x0, dlogp),
            integration_times,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol
        )

        x, dlogp = state

        return x, dlogp


@dataclass
class SDEIntegrator:
    model_A: torch.nn.Module
    model_B: torch.nn.Module | None
    single_model: bool
    type_of_learning: str
    eps: float
    interpolant: Interpolant
    n_save: int = 4
    start_end: tuple = (0, 1)
    n_step: int = 100
    n_likelihood: int = 1

    def __post_init__(self) -> None:
        """Initialize forward dynamics, reverse dynamics, and likelihood."""

        # Velocity - Score
        def bf(x: torch.Tensor, t: torch.Tensor):
            """Forward drift. Assume x is batched but t is not."""
            t_model = t.expand(x.shape[0]) if t.numel() == 1 else t

            output = self.model_A(x, t_model).to(x)
            c = x.shape[1]
            output_A = output[:, :c] if self.single_model else self.model_A(
                x, t_model).to(x)
            output_B = output[:, c:] if self.single_model else self.model_B(
                x, t_model).to(x)

            s = SFromEta(self.interpolant.a)(
                output_B, t) if 'denoiser' in self.type_of_learning else output_B
            b = BFromVS(
                self.interpolant.a, self.interpolant.a_dot)(output_A, s, t) if "vector" in self.type_of_learning else output_A

            return b + self.eps * s

        def br(x: torch.Tensor, t: torch.Tensor):
            """Backwards drift. Assume x is batched but t is not."""
            t_model = t.expand(x.shape[0]) if t.numel() == 1 else t

            output = self.model_A(x, t_model).to(x)
            c = x.shape[1]
            output_A = output[:, :c] if self.single_model else self.model_A(
                x, t_model).to(x)
            output_B = output[:, c:] if self.single_model else self.model_B(
                x, t_model).to(x)

            s = SFromEta(self.interpolant.a)(
                output_B, t) if "denoiser" in self.type_of_learning else output_B
            b = BFromVS(
                self.interpolant.a, self.interpolant.a_dot)(output_A, s, t) if "vector" in self.type_of_learning else output_A

            with torch.no_grad():
                return b - self.eps*s

        def dt_logp(x: torch.Tensor, t: torch.Tensor):
            """Time derivative of the log-likelihood, assumed integrating from 1 to 0.
            Assume x is batched but t is not.
            """
            score = self.s(x, t)
            s_norm = torch.linalg.norm(score, axis=-1)**2
            return -(compute_div(self.bf, x, t) + self.eps*s_norm)

        self.bf = bf
        self.br = br
        self.dt_logp = dt_logp
        self.start, self.end = self.start_end[0], self.start_end[1]
        self.ts = torch.linspace(self.start, self.end, self.n_step)
        self.dt = (self.ts[1] - self.ts[0])

    def step_forward_heun(self, x: Sample, t: torch.Tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + (2*self.eps)**0.5*dW
        K1 = self.bf(xhat, t + self.dt)
        xp = xhat + self.dt*K1
        K2 = self.bf(xp, t + self.dt)
        return xhat + 0.5*self.dt*(K1 + K2)

    def step_forward(self, x: Sample, t: torch.Tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x + self.bf(x, t)*self.dt + (2*self.eps)**0.5*dW

    def step_reverse(self, x: Sample, t: torch.Tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x - self.br(x, t)*self.dt + (2*self.eps)**0.5*dW

    def step_reverse_heun(self, x: Sample, t: torch.Tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + (2*self.eps)**0.5*dW
        K1 = self.br(xhat, t - self.dt)
        xp = xhat - self.dt*K1
        K2 = self.br(xp, t - self.dt)
        return xhat - 0.5*self.dt*(K1 + K2)

    def step_likelihood(self, like: torch.Tensor, x: Sample, t: torch.Tensor) -> Sample:
        """Forward-Euler."""
        return like - self.dt_logp(x, t)*self.dt

    def rollout_likelihood(
        self,
        init: Sample  # [batch x dim]
    ):
        """Solve the reverse-time SDE to generate a likelihood estimate."""
        bs, d = init.shape
        likes = torch.zeros((self.n_likelihood, bs)).to(init)
        xs = torch.zeros((self.n_likelihood, bs, d)).to(init)

        # TODO: for more general dimensions, need to replace these 1's by something else.
        x = init.repeat((self.n_likelihood, 1, 1)).reshape(
            (self.n_likelihood*bs, d))
        like = torch.zeros(self.n_likelihood*bs).to(x)
        save_counter = 0

        for ii, t in enumerate(self.ts[:-1]):
            t = self.end - t.to(x)
            x = self.step_reverse_heun(x, t)
            # semi-implicit discretization?
            like = self.step_likelihood(like, x, t-self.dt)

        xs, likes = x.reshape((self.n_likelihood, bs, d)
                              ), like.reshape((self.n_likelihood, bs))

        # only output mean
        return xs, torch.mean(likes, axis=0)

    def rollout_forward(
        self,
        init: Sample,  # [batch x dim]
        method: str = 'heun'
    ):
        """Solve the forward-time SDE to generate a batch of samples."""
        save_every = int(self.n_step/self.n_save)
        xs = torch.zeros((self.n_save, *init.shape)).to(init)
        x = init
        self.dt = self.dt.to(x)

        save_counter = 0

        for ii, t in enumerate(self.ts[:-1]):
            t = t.to(x)
            t = t.unsqueeze(0)
            if method == 'heun':
                x = self.step_forward_heun(x, t)
            else:
                x = self.step_forward(x, t)

            if ((ii+1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1

        xs[save_counter] = x

        return xs


@dataclass
class MirrorSDEIntegrator:
    s: Score
    eps: torch.Tensor
    interpolant: Interpolant
    n_save: int = 4
    start_end: tuple = (0, 1)
    n_step: int = 100
    n_likelihood: int = 1

    def __post_init__(self) -> None:
        """Initialize forward dynamics, reverse dynamics, and likelihood."""

        def bf(x: torch.Tensor, t: torch.Tensor):
            """Forward drift. Assume x is batched but t is not."""
            self.s.to(
                x)  # needed to make lightning work. arises because using __post_init__

            return -self.interpolant.gg_dot(t)*self.s(x, t) + self.eps*self.s(x, t)

        def br(x: torch.Tensor, t: torch.Tensor):
            """Backwards drift. Assume x is batched but t is not."""
            self.s.to(
                x)  # needed to make lightning work. arises because using __post_init__
            return (-self.interpolant.gg_dot(t) - self.eps)*self.s(x, t)

        def dt_logp(x: torch.Tensor, t: torch.Tensor):
            """Time derivative of the log-likelihood, assumed integrating from 1 to 0.
            Assume x is batched but t is not.
            """
            # tx     = net_inp(t, x)
            score = self.s(x, t)
            s_norm = torch.linalg.norm(score, axis=-1)**2
            return -(compute_div(self.bf, x, t) + self.eps*s_norm)

        def eps_fn(eps0: torch.Tensor, t: torch.Tensor):
            # return eps0*torch.sqrt((1-t))
            # return torch.sqrt(eps0*t*(1-t))
            # return 4*eps0*(t-1/2)**2
            return eps0

        self.bf = bf
        self.br = br
        self.eps_fn = eps_fn
        self.dt_logp = dt_logp
        self.start, self.end = self.start_end[0], self.start_end[1]
        self.ts = torch.linspace(self.start, self.end, self.n_step)
        self.dt = (self.ts[1] - self.ts[0])

    def step_forward_heun(self, x: Sample, t: torch.Tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + (2*self.eps)**0.5*dW
        K1 = self.bf(xhat, t + self.dt)
        xp = xhat + self.dt*K1
        K2 = self.bf(xp, t + self.dt)
        return xhat + 0.5*self.dt*(K1 + K2)

    def step_forward(self, x: Sample, t: torch.Tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x + self.bf(x, t)*self.dt + (2*self.eps)**0.5*dW

    def step_reverse(self, x: Sample, t: torch.Tensor) -> Sample:
        """Euler-Maruyama."""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        return x - self.br(x, t)*self.dt + (2*self.eps)**0.5*dW

    def step_reverse_heun(self, x: Sample, t: torch.Tensor) -> Sample:
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW = torch.sqrt(self.dt)*torch.randn(size=x.shape).to(x)
        xhat = x + (2*self.eps)**0.5*dW
        K1 = self.br(xhat, t - self.dt)
        xp = xhat - self.dt*K1
        K2 = self.br(xp, t - self.dt)
        return xhat - 0.5*self.dt*(K1 + K2)

    def step_likelihood(self, like: torch.Tensor, x: Sample, t: torch.Tensor) -> Sample:
        """Forward-Euler."""
        return like - self.dt_logp(x, t)*self.dt

    def rollout_likelihood(
        self,
        init: Sample  # [batch x dim]
    ):
        """Solve the reverse-time SDE to generate a likelihood estimate."""
        bs, d = init.shape
        likes = torch.zeros((self.n_likelihood, bs)).to(init)
        xs = torch.zeros((self.n_likelihood, bs, d)).to(init)

        # TODO: for more general dimensions, need to replace these 1's by something else.
        x = init.repeat((self.n_likelihood, 1, 1)).reshape(
            (self.n_likelihood*bs, d))
        like = torch.zeros(self.n_likelihood*bs).to(x)
        save_counter = 0

        for ii, t in enumerate(self.ts[:-1]):
            t = self.end - t.to(x)
            x = self.step_reverse_heun(x, t)
            # semi-implicit discretization?
            like = self.step_likelihood(like, x, t-self.dt)

        xs, likes = x.reshape((self.n_likelihood, bs, d)
                              ), like.reshape((self.n_likelihood, bs))

        # only output mean
        return xs, torch.mean(likes, axis=0)

    def rollout_forward(
        self,
        init: Sample,  # [batch x dim]
        method: str = 'heun'
    ):
        """Solve the forward-time SDE to generate a batch of samples."""
        save_every = int(self.n_step/self.n_save)
        xs = torch.zeros((self.n_save, *init.shape)).to(init)
        x = init
        self.dt = self.dt.to(x)

        save_counter = 0
        for ii, t in enumerate(self.ts[:-1]):
            t = t.to(x)
            t = t.unsqueeze(0)
            if method == 'heun':
                x = self.step_forward_heun(x, t)
            else:
                x = self.step_forward(x, t)

            if ((ii+1) % save_every) == 0:
                xs[save_counter] = x
                save_counter += 1

        xs[save_counter] = x

        return xs


# here ye we define all the possible losses! For b, v, s, eta

def loss_per_sample_b_s(
    b_s: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_tp = b_s(xtp, t)
    output_tm = b_s(xtm, t)

    btp = output_tp[:, 0:out_channels]
    btm = output_tm[:, 0:out_channels]

    stp = output_tp[:, out_channels:]
    stm = output_tm[:, out_channels:]

    # Compute the velocity loss
    loss_b = 0.5*(btp**2) - ((dtIt + gamma_dot*z) * btp)
    loss_b += 0.5*(btm**2) - ((dtIt - gamma_dot*z) * btm)

    # Compute the score loss
    loss_s = 0.5*(stp**2) + \
        (1 / interpolant.gamma(t))*(stp*z)
    loss_s += 0.5*(stm**2) - \
        (1 / interpolant.gamma(t))*(stm*z)

    return loss_b.mean(), loss_s.mean()

def loss_per_sample_b_eta(
    b_eta: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1, z)

    xtp, xtm, xt, t = xtp.unsqueeze(0), xtm.unsqueeze(0), xt.unsqueeze(0), t.unsqueeze(0)

    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_tp = b_eta(xtp, t)
    output_tm = b_eta(xtm, t)

    btp = output_tp[:, 0:out_channels]
    btm = output_tm[:, 0:out_channels]

    # Compute the velocity loss
    loss_b = 0.5*(btp**2) - ((dtIt + gamma_dot*z) * btp)
    loss_b += 0.5*(btm**2) - ((dtIt - gamma_dot*z) * btm)

    # Compute the eta loss
    output_t = b_eta(xt, t)
    eta_val = output_t[:, out_channels:]

    loss_eta = 0.5*(eta_val**2) + (eta_val*z)

    return loss_b.mean(), loss_eta.mean()

def loss_per_sample_v_s(
    v_s: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1, z)

    xtp, xtm, xt, t = xtp.unsqueeze(0), xtm.unsqueeze(0), xt.unsqueeze(0), t.unsqueeze(0)
    
    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_t = v_s(xt, t)
    output_tp = v_s(xtp, t)
    output_tm = v_s(xtm, t)

    v_val = output_t[:, 0:out_channels]

    stp = output_tp[:, out_channels:]
    stm = output_tm[:, out_channels:]

    # Compute the vector loss
    loss_v = 0.5*(v_val**2) - (dtIt * v_val)

    # Compute the score loss
    loss_s = 0.5*(stp**2) + \
        (1 / interpolant.gamma(t))*(stp*z)
    loss_s += 0.5*(stm**2) - \
        (1 / interpolant.gamma(t))*(stm*z)

    return loss_v.mean(), loss_s.mean()

def loss_per_sample_v_eta(
    v_eta: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)

    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_t = v_eta(xt, t)

    v_val = output_t[:, 0:out_channels]
    eta_val = output_t[:, out_channels:]

    # Compute the vector loss
    loss_v = 0.5*(v_val**2) - (dtIt * v_val)

    # Compute the eta loss
    loss_eta = 0.5*(eta_val**2) + (eta_val*z)

    return loss_v.mean(), loss_eta.mean()

def loss_per_sample_one_sided_b_s(
    b_s: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,

):
    """Compute the loss on an individual sample."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)

    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_tp = b_s(xtp, t)
    output_tm = b_s(xtm, t)

    btp = output_tp[:, 0:out_channels]
    btm = output_tm[:, 0:out_channels]

    stp = output_tp[:, out_channels:]
    stm = output_tm[:, out_channels:]

    # Compute the velocity loss
    loss_b = 0.5*(btp**2) - ((dtIt) * btp)
    loss_b += 0.5*(btm**2) - ((dtIt) * btm)

    # Compute the score loss
    alpha = interpolant.a(t)

    loss_s = 0.5*(stp**2) + (1 / (alpha))*(stp*x0)
    loss_s += 0.5*(stm**2) - (1 / (alpha))*(stm*x0)

    return loss_b.mean(), loss_s.mean()

def loss_per_sample_one_sided_b_eta(
    b_eta: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,

):
    """Compute the loss on an individual sample."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1, z)

    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_t = b_eta(xt, t)
    output_tp = b_eta(xtp, t)
    output_tm = b_eta(xtm, t)

    btp = output_tp[:, 0:out_channels]
    btm = output_tm[:, 0:out_channels]

    eta_val = output_t[:, out_channels:]

    # Compute the velocity loss
    loss_b = 0.5*(btp**2) - ((dtIt) * btp)
    loss_b += 0.5*(btm**2) - ((dtIt) * btm)

    # Compute the eta loss
    loss_eta = 0.5*(eta_val**2) + (eta_val*z)
    
    return loss_b.mean(), loss_eta.mean()

def loss_per_sample_one_sided_v_s(
    v_s: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the loss on an individual sample."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1, z)

    xtp, xtm, xt, t = xtp.unsqueeze(0), xtm.unsqueeze(0), xt.unsqueeze(0), t.unsqueeze(0)

    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]

    output_tp = v_s(xtp, t)
    output_tm = v_s(xtm, t)

    vtp = output_tp[:, 0:out_channels]
    vtm = output_tm[:, 0:out_channels]

    stp = output_tp[:, out_channels:]
    stm = output_tm[:, out_channels:]

    # Compute the vector loss
    loss_v = 0.5*(vtp**2) - ((dtIt) * vtp)
    loss_v += 0.5*(vtm**2) - ((dtIt) * vtm)

    # Compute the score loss
    alpha = interpolant.a(t)

    loss_s = 0.5*(stp**2) + (1 / (alpha))*(stp*x0)
    loss_s += 0.5*(stm**2) - (1 / (alpha))*(stm*x0)

    return loss_v.mean(), loss_s.mean()

def loss_per_sample_one_sided_v_eta(
    v_eta: SingleModel,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,

):
    """Compute the loss on an individual sample."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)

    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    out_channels = xt.shape[1]
    
    output_t = v_eta(xt, t)
    output_tp = v_eta(xtp, t)
    output_tm = v_eta(xtm, t)

    vtp = output_tp[:, 0:out_channels]
    vtm = output_tm[:, 0:out_channels]

    eta_val = output_t[:, out_channels:]

    # Compute the vector loss
    loss_v = 0.5*(vtp**2) - ((dtIt) * vtp)
    loss_v += 0.5*(vtm**2) - ((dtIt) * vtm)

    # Compute the eta loss
    loss_eta = 0.5*(eta_val**2) + (eta_val*z)
    
    return loss_v.mean(), loss_eta.mean()

def loss_per_sample_b(
    b: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    dtIt = interpolant.dtIt(t, x0, x1)
    gamma_dot = interpolant.gamma_dot(t)

    btp = b(xtp, t)
    btm = b(xtm, t)

    loss = 0.5*(btp**2) - ((dtIt + gamma_dot*z) * btp)
    loss += 0.5*(btm**2) - ((dtIt - gamma_dot*z) * btm)

    return loss.mean()

def loss_per_sample_s(
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    stp = s(xtp, t)
    stm = s(xtm, t)
    loss = 0.5*(stp**2) + (1 / interpolant.gamma(t))*(stp*z)
    loss += 0.5*(stm**2) - (1 / interpolant.gamma(t))*(stm*z)

    return loss.mean()

def loss_per_sample_eta(
    eta: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):

    xt, z = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    eta_val = eta(xt, t)

    loss = 0.5*(eta_val**2) + (eta_val*z)
    
    return loss.mean()

def loss_per_sample_v(
    v: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt = interpolant.dtIt(t, x0, x1)
    v_val = v(xt, t)

    loss = 0.5*(v_val**2) - (dtIt * v_val)

    return loss.mean()

def loss_per_sample_one_sided_b(
    b: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,

):
    """Compute the loss on an individual sample."""
    xt = interpolant.calc_xt(t, x0, x1)
    dtIt = interpolant.dtIt(t, x0, x1)
    # gamma_dot   = interpolant.gamma_dot(t)
    bt = b(xt, t)

    loss = 0.5*(bt**2) - ((dtIt) * bt)

    return loss.mean()


def loss_per_sample_one_sided_v(
    v: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant
):
    """Compute the loss on an individual sample."""
    xt = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt = interpolant.dtIt(t, x0, x1)
    vt = v(xt, t)
    loss = 0.5*(vt**2) - ((dtIt) * vt)

    return loss.mean()

def loss_per_sample_one_sided_s(
    s: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the loss on an individual sample via antithetic samples for x_t = sqrt(1-t)z + sqrt(t) x1 where z=x0.
    """
    xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    xtp, xtm, t = xtp.unsqueeze(0), xtm.unsqueeze(0), t.unsqueeze(0)
    stp = s(xtp, t)
    stm = s(xtm, t)
    alpha = interpolant.a(t)

    loss = 0.5*(stp**2) + (1 / (alpha))*(stp*x0)
    loss += 0.5*(stm**2) - (1 / (alpha))*(stm*x0)

    return loss.mean()

def loss_per_sample_one_sided_eta(
    eta: Velocity,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the loss on an individual sample via samples for x_t = alpha(t)z + beta(t) x1 where z=x0.
    """
    xt = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    etat = eta(xt, t)
    loss = 0.5*(etat**2) + (etat*x0)

    return loss.mean()

def loss_per_sample_mirror(
    s: Score,
    x0: Sample,
    x1: Sample,
    t: torch.Tensor,
    interpolant: Interpolant,
):
    """Compute the loss on an individual sample via antithetic sampling."""
    xt = interpolant.calc_xt(t, x0, x1)
    xt, t = xt.unsqueeze(0), t.unsqueeze(0)
    dtIt = interpolant.dtIt(t, x0, x1)
    st = s(xt, t)

    loss = 0.5*(st**2) + (1 / interpolant.gamma(t))*(st*x0)

    return loss.mean()

def make_batch_loss(loss_per_sample: Callable, method: str = 'shared') -> Callable:
    """Convert a sample loss into a batched loss."""
    if method == 'shared':

        # Share the batch dimension i for x0, x1, t
        in_dims_set = (None, 0, 0, 0, None)
        batched_loss = vmap(
            loss_per_sample, in_dims=in_dims_set, randomness='different')

        return batched_loss


# global variable for the available losses
losses = {
    'b': loss_per_sample_b, 's': loss_per_sample_s, 'eta': loss_per_sample_eta,
    'v': loss_per_sample_v, 'one-sided-b': loss_per_sample_one_sided_b, 'one-sided-s': loss_per_sample_one_sided_s,
    'one-sided-eta': loss_per_sample_one_sided_eta, 'one-sided-v': loss_per_sample_one_sided_v,
    'b-s': loss_per_sample_b_s, 'b-eta': loss_per_sample_b_eta, 'v-s': loss_per_sample_v_s, 'v-eta': loss_per_sample_v_eta,
    'one-sided-b-s': loss_per_sample_one_sided_b_s, 'one-sided-b-eta': loss_per_sample_one_sided_b_eta,
    'one-sided-v-s': loss_per_sample_one_sided_v_s, 'one-sided-v-eta': loss_per_sample_one_sided_v_eta,
    'mirror': loss_per_sample_mirror
}


def make_loss(
    method: str,
    interpolant: Interpolant,
    loss_type: str,
) -> Callable:

    loss_fn_unbatched = losses[loss_type]

    # batchify the loss
    def loss(
        bvseta: Velocity,
        x0s: torch.Tensor,
        x1s: torch.Tensor,
        ts: torch.Tensor,
        interpolant: Interpolant,
    ) -> torch.Tensor:

        # loss_fn = make_batch_loss(loss_fn_unbatched, method)
        ts = ts[:, None, None, None]
        loss_val = loss_fn_unbatched(bvseta, x0s, x1s, ts, interpolant)
        return loss_val

    return loss
