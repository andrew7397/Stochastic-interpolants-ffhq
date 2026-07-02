from abc import ABC, abstractmethod

import torch

from .noise_schedulers import NoiseSchedule, NoiseScheduler


class ConditionalVectorField(ABC):

    @abstractmethod
    def __call__(self, time_steps: torch.Tensor, curr_samples: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class GaussianConditionalVectorField(ConditionalVectorField):

    def __init__(self, noise_scheduler: NoiseScheduler):
        self.noise_scheduler = noise_scheduler

    def __call__(self, time_steps: torch.Tensor, curr_samples: torch.Tensor, data_samples: torch.Tensor) -> torch.Tensor:
        noise_scheduler = self.noise_scheduler(time_steps)

        return (noise_scheduler.d_alphas - noise_scheduler.d_betas / noise_scheduler.betas * noise_scheduler.alphas) * data_samples + noise_scheduler.d_betas / noise_scheduler.betas * curr_samples
