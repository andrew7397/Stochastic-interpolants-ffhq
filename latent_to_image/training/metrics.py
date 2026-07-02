
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Union
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def setup_metrics(device: torch.device) -> dict[str, Union[Metric, MetricCollection]]:
    """
    Initialize the evaluation metrics used during training.

    This function sets up both the FID metric and a collection of other
    perceptual metrics, allocating them on the specified device.

    Args:
        device (str or torch.device):
            The device on which the metrics should be allocated ('cpu' or 'cuda').

    Returns:
        dict[str, Union[Metric, MetricCollection]]:
            A dictionary containing:
            - 'fid': the Frechet Inception Distance metric.
            - 'collection': a MetricCollection of additional perceptual metrics
              (e.g., LPIPS), all moved to the specified device.
    """
    try:
        return {
            'fid': FrechetInceptionDistance(feature=2048, normalize=True).to(device),
            'collection': MetricCollection([
                LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
            ])
        }
    except Exception as e:
        print(f"Warning: Failed to initialize metrics due to: {e}")
        print("Continuing without metrics...")
        return {}

