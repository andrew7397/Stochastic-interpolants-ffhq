from abc import ABC, abstractmethod

import torch
from torch.distributions import MultivariateNormal

from .noise_schedulers import NoiseScheduler


class ConditionalPath(ABC):
    @abstractmethod
    def sample_conditional_distribution(self, time_steps: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_conditional_velocity_field(self, time_steps: torch.Tensor, curr_samples: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class GaussianConditionalPath(ConditionalPath):
    def __init__(self, noise_scheduler: NoiseScheduler):
        self.noise_scheduler = noise_scheduler

    def sample_conditional_distribution(self, time_steps: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        nSamples = data_samples.shape[0]
        dataDimensionality = data_samples.shape[1]

        normalDist = MultivariateNormal(torch.zeros(
            dataDimensionality), torch.eye(dataDimensionality))

        noise_schedules = self.noise_scheduler(time_steps)

        return noise_schedules.alphas * data_samples + noise_schedules.betas * normalDist.sample((nSamples,))

    def get_conditional_velocity_field(self, time_steps: torch.Tensor, curr_samples: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        noise_schedule = self.noise_scheduler(time_steps)

        return (noise_schedule.d_alphas - (noise_schedule.d_betas / noise_schedule.betas) * noise_schedule.alphas) * data_samples + (noise_schedule.d_betas / noise_schedule.betas) * curr_samples
