import torch

from .paths import GaussianConditionalPath


def euler_conditional(start: torch.Tensor, end: torch.Tensor, conditional_path: GaussianConditionalPath, n_steps):
    h = 1.0 / n_steps

    time_steps = []
    positions = []
    velocities = []

    t = torch.Tensor([0.0])
    xt = start

    for i in range(0, n_steps):
        velocity = conditional_path.get_conditional_velocity_field(t, xt, end)
        time_steps.append(t)
        positions.append(xt)
        velocities.append(h * velocity)

        xt = xt + h * velocity
        t = t+h

    time_steps.append(torch.Tensor([1.0]))
    positions.append(xt)
    velocities.append(torch.zeros_like(xt))

    return (time_steps, positions, velocities)
