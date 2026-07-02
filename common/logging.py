import sys
from numbers import Number
from typing import Any, Optional, Union

import torch
import wandb
from torchvision.utils import make_grid

from data.data_utilities.data_visualization_utilities import multispectral_to_RGB

# more reader-friendly metric names for torchmetrics


def shorten_metric_name(metric_name):
    shortening_dict = {
        'MeanSquaredError': 'mse',
        'StructuralSimilarityIndexMeasure': 'ssim',
        'SpectralAngleMapper': 'sam',
        'PeakSignalNoiseRatio': 'psnr'}

    return shortening_dict.get(metric_name, metric_name)


def log_progress(
    phase: str,
    prefix: str,
    loss: Union[Number, Any],
    losses: dict[str, float],
    step: int,
    norms: Optional[dict[str, list[float]]] = None,
    lrs: Optional[dict[str, float]] = None,  # New parameter
    total_steps: Optional[int] = None
):
    # ANSI color codes
    class Colors:
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"  # New color for LR
        RESET = "\033[0m"

    def format_val(val: float) -> str:
        """Return colored string: green if positive, red if negative."""
        color = Colors.GREEN if val >= 0 else Colors.RED
        return f"{color}{val:8.5f}{Colors.RESET}"

    log_parts = []

    # Main loss
    log_parts.append(f"{phase} -> {prefix}_loss: {format_val(loss)}")

    # Individual losses
    for loss_name, loss_val in losses.items():
        log_parts.append(f"{prefix}_{loss_name}: {format_val(loss_val)}")

    # Norms
    if norms:
        for norm_name, norm_val in norms.items():
            total, mean = norm_val
            log_parts.append(
                f"{prefix}_{norm_name}_norm: Σ {format_val(total)}, μ {format_val(mean)}"
            )

    # Learning Rates (New Section)
    if lrs:
        for lr_name, lr_val in lrs.items():
            # Using scientific notation (e.g., 1e-4) is usually better for LRs
            lr_str = f"{Colors.CYAN}{lr_val:.2e}{Colors.RESET}"
            log_parts.append(f"{prefix}_{lr_name}_lr: {lr_str}")

    # Step info
    step_str = f"[{step:>5d}"
    if total_steps:
        step_str += f"/{total_steps}]"
    else:
        step_str += "]"
    log_parts.append(step_str)

    # Combine all parts and print in-place
    sys.stdout.write("; ".join(log_parts) + "\r")
    sys.stdout.flush()


def log_loss(phase: str, prefix: str, loss: float):
    log_str = f'{phase} -> {prefix}_loss: {loss:>8.5f}; '

    print(log_str + '\n')


def log_wandb(
    wandb_run: wandb.Run,
    prefix: str,
    loss: Union[Number, Any],
    losses: dict[str, float],
    step: int,
    norms: Optional[dict[str, list[float]]] = None,
    lrs: Optional[dict[str, float]] = None
):
    '''
    wandb_run.log({prefix + '_loss': loss}, step=step)

    for loss_name, loss_val in losses.items():
        wandb_run.log(
            {prefix + '_' + loss_name + "_loss": loss_val}, step=step)

    norms_items = norms.items() if norms is not None else {}
    for norm_name, norm_val in norms_items:
        wandb_run.log(
            {prefix + '_' + norm_name + "_total_norm": norm_val[0]}, step=step)

        wandb_run.log(
            {prefix + '_' + norm_name + "_mean_norm": norm_val[1]}, step=step)
    '''

    # Initialize the payload with the main loss
    payload = {f"{prefix}_loss": loss}

    # Add individual losses
    for loss_name, loss_val in losses.items():
        payload[f"{prefix}_{loss_name}_loss"] = loss_val

    # Add norms
    if norms:
        for norm_name, (total, mean) in norms.items():
            payload[f"{prefix}_{norm_name}_total_norm"] = total
            payload[f"{prefix}_{norm_name}_mean_norm"] = mean

    # Add learning rates
    if lrs:
        for lr_name, lr_val in lrs.items():
            payload[f"{prefix}_{lr_name}_lr"] = lr_val

    # Single atomic log call for this step
    wandb_run.log(payload, step=step)


'''
def log_metrics_wandb(wandb_run: wandb.Run, metrics, step: int):

    for metric_name, metric_val in metrics.items():
        wandb_run.log(
            {metric_name: metric_val.item()}, step=step)
'''


def log_metrics_wandb(wandb_run: wandb.Run, metrics: dict[str, Any], step: int):

    log_dict: dict[str, Union[Any, int, float]] = {}

    for metric_name, metric_val in metrics.items():

        # torch tensor
        if hasattr(metric_val, "detach"):
            metric_val = metric_val.detach()

        if hasattr(metric_val, "cpu"):
            metric_val = metric_val.cpu()

        if hasattr(metric_val, "item"):
            metric_val = metric_val.item()

        # numpy scalar
        if hasattr(metric_val, "tolist") and not isinstance(metric_val, (int, float)):
            metric_val = metric_val.tolist()

        log_dict[metric_name] = metric_val

    wandb_run.log(log_dict, step=step)


def log_wandb_images(wandb_run: wandb.Run, list_image_samples: list[torch.Tensor], key: str, RGB_case: bool, step: Optional[int] = None):

    processed_list = []

    for img in list_image_samples:
        # For RGB images, use standard channel order (0,1,2)
        # For multispectral, use satellite band indices (3,2,1)
        rgb_indices = (0, 1, 2) if RGB_case else (3, 2, 1)
        processed_list.append(multispectral_to_RGB(
            img, rgbBandIndices=rgb_indices, clip_min=0.0, clip_max=1e4, RGB_case=RGB_case))

    grid_images = make_grid(processed_list, 2)

    image = wandb.Image(grid_images)
    wandb_run.log({key: image}, step=step)
