import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .checkpoints import Checkpoints
from .config import ModelConfig, TrainingConfig
from .wandb_config_params import WandbConfigParams


def setup_device():
    """
    Select the computation device to be used for training.

    Returns a `torch.device` object pointing to 'cuda' if a compatible GPU is
    available, otherwise falls back to 'mps' if on Mac M1/M2, otherwise 'cpu'.

    Returns:
        torch.device: The selected computation device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using MPS device")
        return device
    else:
        print("⚠️ Warning: CUDA not available, falling back to CPU")
        return torch.device('cpu')


def ckpt_paths(ckpt_path: str, name: str):
    """
    Generate checkpoint file paths for a given model name.

    Args:
        ckpt_path (str):
            Base directory where checkpoints are stored.
        name (str):
            Model identifier used to build checkpoint file names.

    Returns:
        dict:
            Dictionary containing:
            - 'latest': Path to the latest checkpoint file.
            - 'best': Path to the best checkpoint file.
    """


    os.makedirs(ckpt_path, exist_ok=True)

    return Checkpoints(
        latest=os.path.join(ckpt_path, f"{name}_latest.tar"),
        best=os.path.join(ckpt_path, f"{name}_best.tar"),
    )


def setup_wandb(training_data: DataLoader, training_config: TrainingConfig, model_config: ModelConfig):
    """
    Build the configuration dictionary for a Weights & Biases run.

    The configuration aggregates dataset, training, and model parameters to
    enable experiment tracking and reproducibility.

    Args:
        training_data (DataLoader):
            Training data loader used to extract batch size information.
        training_config (TrainingConfig):
            Training hyperparameters and logging configuration.
        model_config (ModelConfig):
            Model configuration parameters.

    Returns:
        dict:
            Dictionary of parameters passed to the Weights & Biases run.
    """

    return WandbConfigParams(
        training_data.batch_size,
        training_config.learning_rate,
        training_config.weight_decay,
        'SEN12MS-CR-TS',
        model_config.DiT_type,
        model_config.model_name
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], ema: Optional[torch.optim.swa_utils.AveragedModel]):
    """
    Load a training checkpoint from disk and restore model and optimizer states.

    If the checkpoint file does not exist, the function returns default values.

    Args:
        path (str):
            Path to the checkpoint file.
        model (torch.nn.Module):
            Model whose parameters will be restored.
        optimizer (torch.optim.Optimizer | None):
            Optimizer whose state will be restored, if provided.

    Returns:
        tuple:
            - checkpoint (dict | None):
                Loaded checkpoint dictionary, or None if the file does not exist.
            - epoch (int):
                Restored epoch index, or 0 if not available.
            - step (int):
                Restored training step, or 0 if not available.
    """

    if not os.path.exists(path):
        print(f"⚠️ Checkpoint not found at: {path}")
        return None, 0, 0

    print(f"📂 Loading checkpoint: {path}")
    # Always load to the current device to avoid CUDA/CPU mismatches
    device = next(model.parameters()).device
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    
    if ema is not None:
        ema.load_state_dict(ckpt["ema"])

    print(f"✅ Successfully loaded checkpoint (Epoch {ckpt.get('epoch', 0)}, Step {ckpt.get('step', 0)})")
    return ckpt, ckpt.get('epoch', 0), ckpt.get('step', 0)


def is_deterministic_learning(type_of_learning: str) -> bool:
    """
    Determine whether a given learning type corresponds to a deterministic training setup.

    In this context, certain types of learning (e.g., "vector-score", "velocity-score",
    "vector-denoiser", "velocity-denoiser") are considered stochastic or dual-objective.
    All other types are treated as deterministic.

    Args:
        type_of_learning (str): 
            A string identifier representing the type of learning being used.

    Returns:
        bool: 
            True if the learning type is deterministic, False if it is stochastic.

    Examples:
        >>> is_deterministic_learning("velocity-score")
        False
        >>> is_deterministic_learning("velocity")
        True
    """
    return type_of_learning not in {
        "vector-score",
        "velocity-score",
        "vector-denoiser",
        "velocity-denoiser",
    }


def multispectral_to_rgb(
    imgs: torch.Tensor,
    device: torch.device,
    rgb_band_indices=[3, 2, 1],
    clip_min=None,
    clip_max=None,
    eps=1e-6,
    uint8=False,
    RGB_case=False
):
    """
    Convert a batch of multispectral images to RGB format using PyTorch.

    This function selects the specified spectral bands corresponding to
    the RGB channels, optionally applies value clipping, and normalizes
    each image independently to the [0, 1] range using global min–max
    normalization across channels and spatial dimensions. Optionally,
    the output can be converted to 8-bit unsigned integer format.

    The operation is fully vectorized and runs efficiently on both CPU
    and GPU.

    Parameters
    ----------
    imgs : torch.Tensor
        Input tensor containing multispectral images with shape
        (N, C, H, W), where N is the batch size, C the number of spectral
        bands, and H, W the spatial dimensions.
    rgb_band_indices : tuple of int, optional
        Indices of the spectral bands to be used as RGB channels,
        in the order (R, G, B). Default is (3, 2, 1).
    clip_min : float or None, optional
        Minimum value for optional clipping. If None, no lower clipping
        is applied. Default is None.
    clip_max : float or None, optional
        Maximum value for optional clipping. If None, no upper clipping
        is applied. Default is None.
    eps : float, optional
        Small constant added to the denominator during normalization
        to avoid division by zero. Default is 1e-6.
    uint8 : bool, optional
        If True, the output RGB images are scaled to the [0, 255] range
        and returned as ``torch.uint8``. If False, the output is returned
        as ``torch.float32`` in the [0, 1] range. Default is False.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 3, H, W) containing RGB images. The tensor
        is of type ``torch.float32`` with values in [0, 1] if ``uint8``
        is False, or of type ``torch.uint8`` with values in [0, 255] if
        ``uint8`` is True.

    Notes
    -----
    - Min–max normalization is performed independently for each image
      over all RGB channels and spatial dimensions.
    - When ``uint8=True``, scaling and clamping are applied after
      normalization.
    - For quantitative metrics such as FID, the same preprocessing
      pipeline must be applied consistently to both real and generated
      images.
    """

    rgb = imgs[:, rgb_band_indices, :, :] if RGB_case == False else imgs

    # Optional clipping
    if clip_min is not None or clip_max is not None:
        rgb = torch.clamp(rgb, min=clip_min, max=clip_max)

    rgb_zero_one = normalize_from_zero_to_one(rgb, uint8, eps).to(device)
    rgb_minus_one_one = normalize_from_minus_one_to_one(rgb_zero_one, uint8, eps).to(device)


    return rgb_zero_one, rgb_minus_one_one

def normalize_from_zero_to_one(rgb, uint8=False, eps=1e-6):

    # Compute min/max per image (over channels + spatial dims)
    rgb_min = rgb.amin(dim=(1, 2, 3), keepdim=True)
    rgb_max = rgb.amax(dim=(1, 2, 3), keepdim=True)

    # Normalize to [0, 1]
    rgb_zero_one = (rgb - rgb_min) / (rgb_max - rgb_min +
                             eps) if not uint8 else (rgb * 255).clamp(0, 255).to(dtype=torch.uint8)

    return rgb_zero_one

def normalize_from_minus_one_to_one(rgb_zero_one, uint8=False, eps=1e-6):

    rgb_minus_one_one = rgb_zero_one * 2.0 - 1.0

    return rgb_minus_one_one

def print_param_counts(model_A: torch.nn.Module, model_B: Optional[torch.nn.Module]=None):
    """
    Print the number of total, trainable, and frozen parameters for one or two PyTorch models.

    This function computes and displays:
    - the total number of parameters,
    - the number of trainable parameters (requires_grad=True),
    - the number of frozen parameters (requires_grad=False).

    Parameters
    ----------
    model_A : torch.nn.Module
        The primary model whose parameters will be analyzed and printed.

    model_B : torch.nn.Module, optional
        An optional secondary model to compare with `model_A`.
        If provided, its parameter counts will also be printed.
        Default is None.

    Returns
    -------
    None
        This function does not return any value; it prints the results to stdout.
    """

    def _print_single(model: torch.nn.Module, name="Model"):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\n{name}")
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params: {frozen:,}")

    _print_single(model_A, "Model A")

    if model_B is not None:
        _print_single(model_B, "Model B")