from dataclasses import dataclass
from typing import Any, Callable

import torch

from satflow.common.interflow.stochastic_interpolant import (
    Interpolant, Velocity)


@dataclass
class ModelEntry:
    name: str
    model: torch.nn.Module
    opt: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    ema: torch.optim.swa_utils.AveragedModel
    loss: Callable[[Velocity, Any, Any, Any, Interpolant], Any]
    ckpt_best: str
    ckpt_latest: str
    stochastic: bool = True
