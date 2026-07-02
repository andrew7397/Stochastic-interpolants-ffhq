"""
Base abstractions for training pipelines.

This module defines the BaseTraining class, which provides the common
infrastructure for setting up datasets, preprocessing pipelines, metrics,
logging, and device configuration. Concrete training implementations must
extend this class and implement the build method.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import asdict

import wandb

from satflow.common.config import Config
from satflow.common.interflow.prior import GMM, SimpleNormal
from satflow.common.interflow.stochastic_interpolant import Interpolant
from satflow.common.model_entry import ModelEntry
from satflow.common.utils import setup_device, setup_wandb
from satflow.latent_to_image.fitting import fit
from satflow.latent_to_image.training.data import (setup_data,
                                                   setup_preprocess,
                                                   setup_subdatasets_metrics,
                                                   setup_visual_samples)
from satflow.latent_to_image.training.metrics import setup_metrics


class BaseTraining(ABC):
    """
    Abstract base class defining the common training pipeline setup.

    This class centralizes all the shared initialization logic required by
    concrete training implementations, including:
    - Device selection (CPU / CUDA).
    - Dataset loading and preprocessing pipelines.
    - Metrics initialization.
    - Visualization samples setup.
    - Experiment tracking configuration (Weights & Biases).

    Subclasses must implement the `build` method to provide the concrete
    model(s), interpolant, and checkpoint state required to run the training
    loop.
    """

    def __init__(self, config: Config):
        """
        Initialize the base training environment.

        This method prepares all common components needed by any training
        strategy, such as data loaders, preprocessing pipelines, metrics,
        visualization samples, and experiment logging configuration.

        Args:
            config (Config):
                Global experiment configuration dictionary.
        """

        self.config = config
        self.device = setup_device()

        self.data_config = config.data
        self.model_config = config.model
        self.training_config = config.training
        self.interpolant_config = config.interpolant

        self.training_data, self.validation_data = setup_data(self.data_config)
        self.val_subset_metrics = setup_subdatasets_metrics(
            self.training_config, self.validation_data)

        self.preprocess_pipeline, self.inverse_preprocess_pipeline = setup_preprocess(
            self.data_config)
        self.metrics = setup_metrics(self.device)

        self.visual_samples = setup_visual_samples(
            self.training_data,
            self.validation_data
        )

        self.wandb_config = setup_wandb(
            self.training_data,
            self.training_config,
            self.model_config
        )

    @abstractmethod
    def build(self) -> tuple[list[ModelEntry], Interpolant, Union[SimpleNormal, GMM], int, int, Optional[float]]:
        """
        Build and return all components required to start or resume training.

        Concrete implementations must instantiate and configure:
        - The model list (models, optimizers, loss functions, checkpoint paths).
        - The interpolant.
        - The initial data distribution.
        - The training state restored from checkpoints.

        Returns:
            tuple:
                - model_list (ModelParams):
                    List of model descriptors used by the training loop.
                - interpolant (Interpolant):
                    Configured interpolant object.
                - initial_distribution (SimpleNormal | GMM):
                    Initial data distribution used by the interpolant.
                - initial_epoch (int):
                    Epoch index restored from checkpoint, or 0 if none exists.
                - initial_step (int):
                    Training step restored from checkpoint, or 0 if none exists.
                - initial_best (float | None):
                    Best validation loss restored from checkpoint, or None.
        """

        pass

    def run(self):
        """
        Execute the training loop.

        This method:
        - Calls `build()` to initialize or restore all training components.
        - Initializes the Weights & Biases run.
        - Invokes the main training loop (`fit`) with all required parameters.

        The training process can automatically resume from existing checkpoints
        and previously logged experiments when available.
        """

        (
            model_list,
            interpolant,
            initial_distr,
            initial_epoch,
            initial_step,
            initial_best,
        ) = self.build()

        with wandb.init(
            project=self.training_config.wandb_project_name,
            config=asdict(self.wandb_config),
            id=self.model_config.model_name,
            resume='allow',
            settings=wandb.Settings(init_timeout=180),
            dir=self.training_config.wandb_log_dir,
        ) as wandb_log_run:

            fit(
                model_list,
                interpolant,
                self.interpolant_config,
                initial_distr,
                self.training_data,
                self.validation_data,
                self.val_subset_metrics,
                self.training_config.clip_grad_norm,
                self.preprocess_pipeline,
                self.inverse_preprocess_pipeline,
                self.training_config.max_epochs,
                self.metrics,
                self.training_config.visual_freq_steps,
                self.training_config.validation_freq_steps,
                wandb_log_run,
                self.device,
                self.visual_samples,
                initial_epoch,
                initial_step,
                initial_best,
                RGB_case=True if self.model_config.in_channels == 3 else False,
            )
