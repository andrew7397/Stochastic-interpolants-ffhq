from abc import ABC, abstractmethod

import torch
from torch.distributions import MultivariateNormal

from .noise_schedulers import NoiseScheduler


class ConditionalProbabilityPath(ABC):
    @abstractmethod
    def sample(self, time_steps: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):

    def __init__(self, noise_scheduler: NoiseScheduler):
        self.noise_scheduler = noise_scheduler

    def sample(self, time_steps: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        n_samples = data_samples.shape[0]
        data_dim = data_samples.shape[1]

        normal_dist = MultivariateNormal(
            torch.zeros(data_dim), torch.eye(data_dim))

        noise_scheduler = self.noise_scheduler(time_steps)

        return noise_scheduler.alphas * data_samples + noise_scheduler.betas * normal_dist.sample((n_samples,))
