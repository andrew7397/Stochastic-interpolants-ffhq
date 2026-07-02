import json
from dataclasses import asdict
from typing import Any, Callable

import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import math

from satflow.common.utils import is_deterministic_learning
from satflow.latent_to_image.models.DiT import DiT_models
# from data.sen2_image_dataset import Sen2ImageDataset
from satflow.data.datasets.ffhq import FFHQDataset

from . import preprocessing
from .config import (Config, DataConfig, DiTArgs, InterpolantConfig,
                     LossConfig, ModelConfig, TrainingConfig)
from .interflow import prior, stochastic_interpolant


def load_config_from_JSON(config_path: str):
    """
    Load a configuration from a JSON file and instantiate the corresponding
    configuration objects.

    This function reads a JSON configuration file from the given path and
    parses its contents into structured configuration objects used by the
    application (e.g., training, model, interpolant, and data settings).

    The JSON file is expected to contain the following top-level keys:
    - "training"
    - "model"
    - "interpolant"
    - "data"

    Each key must map to a dictionary whose entries are compatible with the
    corresponding configuration class constructor.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    Config
        A `Config` instance populated with `TrainingConfig`, `ModelConfig`,
        `InterpolantConfig`, and `DataConfig` objects.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    json.JSONDecodeError
        If the file is not a valid JSON document.
    KeyError
        If one of the required top-level keys is missing from the JSON file.
    TypeError
        If the JSON entries are not compatible with the configuration class
        constructors.
    """
    with open(config_path, "r") as json_file:
        cfg = json.load(json_file)

    return Config(
        training=TrainingConfig(**cfg["training"]),
        model=ModelConfig(**cfg["model"]),
        interpolant=InterpolantConfig(**cfg["interpolant"]),
        data=DataConfig(**cfg["data"]),
    )


def config_dataloader(data_config: DataConfig, split: str):
    """
    Configure a PyTorch DataLoader for the specified dataset split.

    Creates a DataLoader with appropriate batch size, shuffling, and worker settings
    based on the dataset split (train, validation, or test). Training data is shuffled
    while validation and test data maintain their original order.

    Args:
        data_config: Configuration dictionary containing data loading settings.
            Expected keys:
            - data_root (str): Root directory containing the image data
            - train_config_file (str): Path to training set configuration JSON
            - val_config_file (str): Path to validation set configuration JSON
            - test_config_file (str): Path to test set configuration JSON
            - train_batch_size (int): Batch size for training data
            - val_batch_size (int): Batch size for validation and test data
            - num_workers (int): Number of worker processes for data loading
                (0 uses the main process)
        split: Dataset split to load. Must be one of: 'train', 'val', or 'test'

    Returns:
        DataLoader: Configured PyTorch DataLoader for the specified split with:
            - Appropriate batch size for the split
            - Shuffling enabled for training, disabled for validation/test
            - drop_last=True to ensure consistent batch sizes
            - Specified number of worker processes

    Raises:
        ValueError: If split is not one of 'train', 'val', or 'test'

    Notes:
        - Training split uses train_batch_size and shuffle=True
        - Validation and test splits use val_batch_size and shuffle=False
        - All splits use drop_last=True to avoid incomplete batches
    """

    split_config = {
        'train': (data_config.train_config_file, data_config.train_batch_size, True),
        'val': (data_config.val_config_file, data_config.val_batch_size, False),
        'test': (data_config.test_config_file, data_config.val_batch_size, False)
    }

    if split not in split_config:
        raise ValueError(
            f'Unknown dataset split {split}. Use train, val or test.')

    config_file, batch_size, do_shuffle = split_config[split]

    config_file, batch_size, do_shuffle = split_config[split]

    if data_config.dataset_name == 'ffhq':
        # For FFHQ, we use data_root and ignore config files for now
        dataset = FFHQDataset(data_config.data_root, input_size=128, split=split) # TODO: Pass input_size from config properly if needed
    else:
        from data.sen2_image_dataset import Sen2ImageDataset
        dataset = Sen2ImageDataset(data_config.data_root, config_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=data_config.num_workers,
        drop_last=True
    )


def config_preprocess_transform(data_config: DataConfig):
    """
    Configure preprocessing and inverse preprocessing transforms from data configuration.

    Creates a pipeline of transformations to apply to input data before training/inference
    and the corresponding inverse transformations to recover original data from processed output.
    Supports optional clipping and rescaling (standardization or normalization).

    Args:
        data_config: Configuration dictionary containing data preprocessing settings.
            Expected keys:
            - clip (bool): Whether to apply value clipping
            - clip_val (float): Maximum value for clipping (only used if clip is True)
            - rescale_mode (Optional[str]): Type of rescaling to apply.
                Options: 'standardize', 'normalize', or None
            - standardize_mode (Optional[str]): Specifies which mean/std pair to use
                (required when rescale_mode is 'standardize')
            - normalize_min (Optional[float]): Minimum value for normalization range
                (required when rescale_mode is 'normalize')
            - normalize_max (Optional[float]): Maximum value for normalization range
                (required when rescale_mode is 'normalize')

    Returns:
        tuple[Compose, Compose]: A tuple containing:
            - Compose: Forward preprocessing transform pipeline
            - Compose: Inverse preprocessing transform pipeline

        If no transforms are configured, both Compose objects will act as identity transforms.

    Raises:
        ValueError: If rescale_mode is 'standardize' but standardize_mode is None
        ValueError: If rescale_mode contains an unsupported value

    Notes:
        - Transforms are applied in order: clipping (if enabled), then rescaling (if enabled)
        - Inverse transforms are applied in reverse order to recover the original data
        - When rescale_mode is None, no rescaling is applied
    """
    preprocess_transform = []
    inverse_preprocess_transform = []

    if data_config.clip:
        preprocess_transform.append(
            preprocessing.ClipTransform(data_config.clip_val))

    rescale_mode = data_config.rescale_mode

    if rescale_mode == 'standardize':
        standardize_mode = data_config.standardize_mode
        if standardize_mode is None:
            raise ValueError(
                "standardize_mode cannot be None when rescale_mode is 'standardize'")

        mean = preprocessing.standardization_means[standardize_mode]
        std = preprocessing.standardization_stds[standardize_mode]

        preprocess_transform.append(
            preprocessing.StandardizationTransform(mean, std))
        inverse_preprocess_transform.append(
            preprocessing.DestandardizationTransform(mean, std))

    elif rescale_mode == 'normalize':
        preprocess_transform.append(
            preprocessing.NormalizationTransform(
                data_config.normalize_min,
                data_config.normalize_max
            ))
        inverse_preprocess_transform.append(
            preprocessing.DenormalizationTransform(
                data_config.normalize_min,
                data_config.normalize_max
            ))

    elif rescale_mode is not None:
        raise ValueError(
            f"Unsupported preprocessing rescale mode {rescale_mode}")

    return Compose(preprocess_transform), Compose(inverse_preprocess_transform)


def config_model(model_config: ModelConfig, interpolant_config: InterpolantConfig, device: torch.device):
    """
    Configure DiT model(s) from model configuration dictionary and move to device.

    Creates either a single DiT model or a pair of DiT models depending on the
    single_model configuration flag. The model architecture is determined by the
    DiT_type specification. Models are automatically moved to the specified device.

    Args:
        model_config: Configuration dictionary containing model architecture settings.
            Expected keys:
            - DiT_type (str): Type of DiT architecture (e.g., 'DiT-XL/2', 'DiT-XL/4')
            - input_size (int): Spatial dimension of input images
            - input_bands (int): Number of input channels/bands
            - patch_size (int): Size of image patches
            - hidden_size (int): Dimension of hidden layers
            - depth (int): Number of transformer blocks
            - num_heads (int): Number of attention heads
            - mlp_ratio (float): MLP hidden dimension scaling factor
            - single_model (bool): If True, return single model; if False, return two models
        device: PyTorch device to place the model(s) on.

    Returns:
        Union[DiT, tuple[DiT, DiT]]: The configured model(s) on the specified device.
            - DiT: Single model if single_model is True
            - tuple[DiT, DiT]: Two separate models if single_model is False
                (typically used for learning both velocity field and score/denoiser)

    Notes:
        - When single_model is False, both returned models share the same architecture
            but have independent parameters
        - All models are moved to the specified device before being returned
        - For multi-model setup, both models are placed on the same device
    """

    is_deterministic = is_deterministic_learning(
        interpolant_config.type_of_learning)
    DiT_args = DiTArgs(model_config.input_size,
                       model_config.in_channels, model_config.single_model, is_deterministic)

    model_factory = DiT_models[model_config.DiT_type]
    args = asdict(DiT_args)

    if model_config.single_model:
        return model_factory(**args).to(device)

    return model_factory(**args).to(device), model_factory(**args).to(device)


def config_initial_distribution(interpolant_config: InterpolantConfig, model_config: ModelConfig, device: torch.device):
    """
    Configure the initial distribution for the interpolant model.

    Creates either a Gaussian or Gaussian Mixture Model (GMM) base distribution
    based on the configuration parameters. The distribution dimensionality is
    determined by the model's input specifications.

    Args:
        interpolant_config: Configuration dictionary containing interpolant settings.
            Expected keys:
            - gaussian_base_distr (bool): If True, use simple Gaussian distribution;
                if False, use Gaussian Mixture Model
            - mixture_numbers (int): Number of components for GMM (only used when
                gaussian_base_distr is False)
        model_config: Configuration dictionary containing model architecture settings.
            Expected keys:
            - input_bands (int): Number of input channels/bands
            - input_size (int): Spatial dimension of input (assumes square images)

    Returns:
        Union[SimpleNormal, GMM]: The configured initial distribution.
            - SimpleNormal: Standard normal distribution N(0, I) if gaussian_base_distr is True
            - GMM: Gaussian Mixture Model with random means and uniform weights if 
                gaussian_base_distr is False

    Notes:
        - For Gaussian base: Uses zero mean and unit variance
        - For GMM base: Components have uniform weights (automatically set by GMM class),
            random means scaled by 2.0, and random variance matrices scaled by 1/4
    """
    ndim = (model_config.in_channels,
            model_config.input_size, model_config.input_size)

    if interpolant_config.gaussian_base_distr:
        return prior.SimpleNormal(torch.zeros(ndim, device=device), torch.ones(ndim, device=device))

    # Non-Gaussian case: GMM
    mixture_numbers = interpolant_config.mixture_numbers

    # Mean vectors of the gaussian mixture components
    mus = 2.0 * torch.randn(mixture_numbers, ndim[1], device=device)

    # Variance matrices (diagonal only - extracted from full covariance)
    vars_diag = torch.zeros(mixture_numbers, ndim[1], device=device)

    for ii in range(mixture_numbers):
        C = torch.randn(ndim[1], ndim[1], device=device)
        cov_matrix = (C.T @ C + torch.eye(ndim[1], device=device)) / 4.0
        vars_diag[ii] = torch.diagonal(cov_matrix)

    return prior.GMM(loc=mus, var=vars_diag, device=device)


def config_interpolant(interpolant_config: InterpolantConfig, model_config: ModelConfig):
    """
    Configure stochastic interpolant and associated loss functions.

    Creates an interpolant object with specified path and gamma settings, then
    configures the appropriate loss function(s) based on the learning type.
    Different learning types require different combinations of loss functions
    for training the model(s).

    Args:
        interpolant_config: Configuration dictionary containing interpolant settings.
            Expected keys:
            - gaussian_base_distr (bool): If True, use Gaussian base distribution
                and apply one-sided loss functions; if False, use standard loss functions
            - path (str): Interpolation path type (e.g., 'linear', 'trig', 
                'encoding-decoding', 'one-sided-linear', 'one-sided-trig', 'mirror')
            - gamma_type (Optional[str]): Type of gamma scheduling (e.g., 'brownian')
                Only used when gaussian_base_distr is False
            - type_of_learning (str): Specifies what to learn. Options:
                - 'velocity': Learn velocity field only
                - 'vector-score': Learn both vector field and score
                - 'velocity-score': Learn both velocity field and score
                - 'vector-denoiser': Learn both vector field and denoiser
                - 'velocity-denoiser': Learn both velocity field and denoiser
    Returns:
        tuple[Interpolant, Callable] | tuple[Interpolant, Callable, Callable]:
            Returns a tuple containing the interpolant and loss function(s):
            - For 'velocity': (interpolant, loss_fn, None)
            - For dual learning types: (interpolant, first_loss_fn, second_loss_fn)

    Raises:
        NotImplementedError: If type_of_learning is not one of the supported options

    Notes:
        - When gaussian_base_distr is True, gamma_type is set to None and loss functions
            use "one-sided-" prefix for the loss type
        - Loss types: 'b' (velocity), 'v' (vector), 's' (score), 'eta' (denoiser)
        - All loss functions are created with method="shared" for the interpolant
    """
    gaussian_base_distr = interpolant_config.gaussian_base_distr
    type_of_learning = interpolant_config.type_of_learning
    single_model = model_config.single_model

    interpolant = stochastic_interpolant.Interpolant(
        path=interpolant_config.path,
        gamma_type=interpolant_config.gamma_type,
    )

    def make_loss_fn(loss_type: str) -> Callable[[stochastic_interpolant.Velocity, Any, Any, Any, stochastic_interpolant.Interpolant], Any]:
        """Helper to create loss function with correct type prefix."""
        prefixed_type = f"one-sided-{loss_type}" if gaussian_base_distr else loss_type
        return stochastic_interpolant.make_loss("shared", interpolant, prefixed_type)

    if not single_model or type_of_learning == "velocity":
        loss_configs: LossConfig = {
            "velocity": ("b", None),
            "vector-score": ("v", "s"),
            "velocity-score": ("b", "s"),
            "vector-denoiser": ("v", "eta"),
            "velocity-denoiser": ("b", "eta")
        }

    else:
        loss_configs: LossConfig = {
            "vector-score": ("v-s", None),
            "velocity-score": ("b-s", None),
            "vector-denoiser": ("v-eta", None),
            "velocity-denoiser": ("b-eta", None)
        }

    if type_of_learning not in loss_configs:
        raise NotImplementedError(
            f"Unknown type_of_learning: {type_of_learning}")

    loss_type_A, loss_type_B = loss_configs[type_of_learning]

    if loss_type_B is None:
        return interpolant, make_loss_fn(loss_type_A), None

    return interpolant, make_loss_fn(loss_type_A), make_loss_fn(loss_type_B)


def config_optimizer(training_config: TrainingConfig, model: torch.nn.Module):
    """
    Configure AdamW optimizer with optional selective weight decay exclusion.

    Creates an AdamW optimizer with specified learning rate and weight decay settings.
    Supports excluding specific parameters from weight decay based on parameter name
    patterns (e.g., bias terms, normalization layers). When exclusions are specified,
    creates separate parameter groups with different weight decay values.

    Args:
        training_config: Configuration dictionary containing training hyperparameters.
            Expected keys:
            - learning_rate (float): Learning rate for the optimizer
            - weight_decay (float): L2 regularization weight decay coefficient
            - wd_exclude_params (Optional[list[str]]): List of parameter name patterns
                to exclude from weight decay. Parameters whose names contain any of these
                substrings will have weight_decay=0.0 applied
        model: PyTorch model whose parameters will be optimized

    Returns:
        torch.optim.AdamW: Configured AdamW optimizer with:
            - Two parameter groups if weight decay exclusions are specified:
                - Group 1: Parameters with full weight decay
                - Group 2: Excluded parameters with weight_decay=0.0
            - Single parameter group otherwise

    Notes:
        - Common exclusion patterns include 'bias', 'norm', 'bn' for batch norm,
            'ln' for layer norm
        - If weight_decay is 0.0 or wd_exclude_params is None, all parameters
            use the same settings
        - Parameter exclusion is based on substring matching of parameter names
    """

    learning_rate = training_config.learning_rate
    weight_decay = training_config.weight_decay
    wd_exclude_params = training_config.wd_exclude_params

    if weight_decay == 0.0 or not wd_exclude_params:
        params = model.parameters()
    else:
        def should_exclude(param_name: str):
            return any(exclusion in param_name for exclusion in wd_exclude_params)

        wd_params = []
        no_wd_params = []

        for param_name, param in model.named_parameters():
            if should_exclude(param_name):
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0.0}
        ]

    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def config_scheduler(training_config: TrainingConfig, optimizer: torch.optim):
    """
    Configure a learning rate scheduler with warmup and cosine decay.

    This function creates a `LambdaLR` scheduler that:
    - Linearly increases the learning rate during a warmup phase.
    - Applies a cosine decay schedule after warmup.
    - Ensures the learning rate does not fall below a minimum ratio
      of the initial learning rate.

    The learning rate schedule is defined as:

    - Warmup phase (step < num_warmup_steps):
        lr = step / num_warmup_steps

    - Cosine decay phase:
        lr = cosine_decay * (1 - min_lr_ratio) + min_lr_ratio

    where `cosine_decay = 0.5 * (1 + cos(pi * progress))`.

    Args:
        training_config (TrainingConfig):
            Configuration object containing:
            - num_warmup_steps (int): Number of warmup steps.
            - num_training_steps (int): Total number of training steps.
            - min_lr_ratio (float): Minimum learning rate as a fraction
              of the initial learning rate.
        optimizer (torch.optim.Optimizer):
            The optimizer whose learning rate will be scheduled.

    Returns:
        torch.optim.lr_scheduler.LambdaLR:
            A learning rate scheduler implementing warmup followed by
            cosine decay.
    """
    num_warmup_steps = training_config.num_warmup_steps
    num_training_steps = training_config.num_training_steps
    min_lr_ratio = training_config.min_lr_ratio

    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return cosine_decay * (1 - min_lr_ratio) + min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def config_ema(training_config: TrainingConfig, model: torch.nn.Module):
    """
    Configure an Exponential Moving Average (EMA) model.

    This function wraps a model with `torch.optim.swa_utils.AveragedModel`
    to maintain an exponential moving average of its parameters.
    The EMA is updated at each step according to the rule:

        ema = beta_1 * ema + beta_2 * new_parameters

    This technique is commonly used to stabilize training and improve
    evaluation performance, especially in generative models and
    diffusion-based architectures.

    Args:
        training_config (TrainingConfig):
            Configuration object containing:
            - beta_1 (float): Weight applied to the existing EMA parameters.
            - beta_2 (float): Weight applied to the current model parameters.
        model (torch.nn.Module):
            The model to track using exponential moving average.

    Returns:
        torch.optim.swa_utils.AveragedModel:
            A model wrapper that maintains EMA weights.
    """
    beta_1 = training_config.beta_1
    beta_2 = training_config.beta_2

    return AveragedModel(
        model,
        avg_fn=lambda avg, new, n: beta_1 * avg + beta_2 * new
    )