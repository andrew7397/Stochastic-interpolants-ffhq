from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class NoiseSchedule:
    alphas: torch.Tensor
    betas: torch.Tensor
    d_alphas: torch.Tensor
    d_betas: torch.Tensor


class NoiseScheduler(ABC):
    @abstractmethod
    def __call__(self, timeSteps: torch.Tensor) -> NoiseSchedule:
        raise NotImplementedError()


class LinearNoiseScheduler(NoiseScheduler):
    def __call__(self, timeSteps: torch.Tensor) -> NoiseSchedule:
        return NoiseSchedule(
            alphas=timeSteps,
            betas=1.0 - timeSteps,
            d_alphas=torch.ones_like(timeSteps),
            d_betas=-1.0 * torch.ones_like(timeSteps)
        )
