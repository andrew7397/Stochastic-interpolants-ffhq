import os
import shutil
from typing import cast

import torch

import satflow.common.config_helpers as helpers
from satflow.common.model_entry import ModelEntry
from satflow.common.utils import (ckpt_paths, is_deterministic_learning,
                                  load_checkpoint, print_param_counts)

from .base import BaseTraining


class SingleModelTraining(BaseTraining):
    """
    Training pipeline for a single-model setup.

    This class is responsible for:
    - Building the neural network model.
    - Configuring the optimizer.
    - Selecting the appropriate training strategy.
    - Initializing checkpoints and restoring previous state if available.

    It produces all the components required by the training loop, including the
    model list, interpolant, initial distribution, and checkpoint state.
    """

    def build(self):

        model = cast(torch.nn.Module, helpers.config_model(
            self.model_config, self.interpolant_config, self.device))

        print_param_counts(model)

        optimizer = helpers.config_optimizer(self.training_config, model)

        if self.training_config.cosine_scheduler_with_warmup:
            scheduler = helpers.config_scheduler(
                self.training_config, optimizer)
        else:
            scheduler = None

        if self.training_config.ema:
            ema = helpers.config_ema(self.training_config, model)
        else:
            ema = None

        interpolant, loss, _ = helpers.config_interpolant(
            self.interpolant_config,
            self.model_config,
        )

        initial_distr = helpers.config_initial_distribution(
            self.interpolant_config,
            self.model_config,
            self.device
        )

        self.training_config.ckpt_path = os.path.join(
            self.training_config.ckpt_path, self.model_config.model_name)

        if self.training_config.resume == False and os.path.exists(self.training_config.ckpt_path):
            print(f"⚠️ Warning: Checkpoint directory {self.training_config.ckpt_path} exists but 'resume' is set to False. Training will start from scratch and might overwrite existing files.")

        ckpt = ckpt_paths(
            self.training_config.ckpt_path,
            self.model_config.model_name,
        )

        stochastic = not is_deterministic_learning(
            self.interpolant_config.type_of_learning
        )

        model_list = [
            ModelEntry(
                name="single_model",
                stochastic=stochastic,
                model=model,
                loss=loss,
                opt=optimizer,
                scheduler=scheduler,
                ema=ema,
                ckpt_latest=ckpt.latest,
                ckpt_best=ckpt.best,
            )
        ]

        initial_epoch = 0
        initial_step = 0
        initial_best = None

        _, initial_epoch, initial_step = load_checkpoint(
            ckpt.latest, model, optimizer, scheduler, ema)

        if os.path.exists(ckpt.best):
            best_state = torch.load(
                ckpt.best
            )
            initial_best = best_state.get("best_loss")
            print(f"Loaded initial best loss: {initial_best}")

        return (
            model_list,
            interpolant,
            initial_distr,
            initial_epoch,
            initial_step,
            initial_best,
        )
