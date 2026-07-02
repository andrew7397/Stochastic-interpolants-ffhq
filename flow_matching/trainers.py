import torch

from .components.noise_schedulers import LinearNoiseScheduler
from .components.paths import GaussianConditionalPath


def flow_matching_guided_train_step(data_batch, model):
    noise_scheduler = LinearNoiseScheduler()

    gaussian_path = GaussianConditionalPath(noise_scheduler)

    n_samples = data_batch.shape[0]

    flow_time_steps = torch.rand(n_samples)

    xt = gaussian_path.sample_conditional_distribution(
        flow_time_steps, data_batch)

    utTarget = gaussian_path.get_conditional_velocity_field(
        flow_time_steps, xt, data_batch)

    ut = model(data_batch)

    raise NotImplementedError()
