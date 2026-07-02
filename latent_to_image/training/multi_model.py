import os
import shutil
from typing import cast

import torch
from torchinfo import summary

import satflow.common.config_helpers as helpers
from satflow.common.model_entry import ModelEntry
from satflow.common.utils import (ckpt_paths, load_checkpoint,
                                  print_param_counts)

from .base import BaseTraining


class MultiModelTraining(BaseTraining):
    """
    Training pipeline for a multi-model setup.

    This class manages the initialization and training configuration of two
    separate models (Model A and Model B) that are trained jointly within the
    same experiment. It is responsible for:
    - Instantiating both models and their optimizers.
    - Initializing independent checkpoint paths for each model.
    - Building the interpolant and its associated loss functions.
    - Restoring training state from available checkpoints.

    The resulting configuration is compatible with the shared training loop
    implemented in the BaseTraining class.
    """

    def build(self):
        model_A, model_B = cast(
            tuple[torch.nn.Module, torch.nn.Module], helpers.config_model(self.model_config, self.interpolant_config, self.device))

        print_param_counts(model_A, model_B)

        assert self.training_data.batch_size is not None, "Batch size cannot be None"
        summary(model_A, (self.training_data.batch_size, self.model_config.in_channels,
                self.model_config.input_size, self.model_config.input_size))

        optimizer_A = helpers.config_optimizer(self.training_config, model_A)
        optimizer_B = helpers.config_optimizer(self.training_config, model_B)

        if self.training_config.cosine_scheduler_with_warmup:
            scheduler_A = helpers.config_scheduler(
                self.training_config, optimizer_A)
            scheduler_B = helpers.config_scheduler(
                self.training_config, optimizer_B)
        else:
            scheduler_A = None
            scheduler_B = None

        if self.training_config.ema:
            ema_A = helpers.config_ema(self.training_config, model_A)
            ema_B = helpers.config_ema(self.training_config, model_B)
        else:
            ema_A = None
            ema_B = None

        self.training_config.ckpt_path = os.path.join(
            self.training_config.ckpt_path, self.model_config.model_name)

        if self.training_config.resume == False and os.path.exists(self.training_config.ckpt_path):
            shutil.rmtree(self.training_config.ckpt_path)

        ckpt_A = ckpt_paths(
            self.training_config.ckpt_path,
            self.model_config.model_name + 'model_A',
        )

        ckpt_B = ckpt_paths(
            self.training_config.ckpt_path,
            self.model_config.model_name + 'model_B',
        )

        initial_distr = helpers.config_initial_distribution(
            self.interpolant_config, self.model_config)

        interpolant, loss_A, loss_B = helpers.config_interpolant(
            self.interpolant_config, self.model_config)
        assert loss_B is not None, "Loss_B cannot be None"

        initial_epoch = initial_step = 0
        initial_best = None

        _, initial_epoch, initial_step = load_checkpoint(
            ckpt_A.latest, model_A, optimizer_A, scheduler_A, ema_A)

        _, _, _ = load_checkpoint(
            ckpt_B.latest, model_B, optimizer_B, scheduler_B, ema_B)

        model_list = [
            ModelEntry(
                name="model_A",
                model=model_A,
                loss=loss_A,
                opt=optimizer_A,
                scheduler=scheduler_A,
                ema=ema_A,
                ckpt_latest=ckpt_A.latest,
                ckpt_best=ckpt_A.best,
            ),
            ModelEntry(
                name="model_B",
                model=model_B,
                loss=loss_B,
                opt=optimizer_B,
                scheduler=scheduler_B,
                ema=ema_A,
                ckpt_latest=ckpt_B.latest,
                ckpt_best=ckpt_B.best,
            )
        ]

        return (
            model_list,
            interpolant,
            initial_distr,
            initial_epoch,
            initial_step,
            initial_best,
        )
