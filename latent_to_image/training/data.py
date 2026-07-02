from typing import Any, Literal

import torch
from satflow.common.config import DataConfig, TrainingConfig
from torch.utils.data import DataLoader, Subset

import satflow.common.config_helpers as helpers


def setup_data(data_config: DataConfig):
    """
    Create training and validation data loaders from the given data configuration.

    Args:
        data_config (DataConfig):
            Configuration describing the dataset and data loader parameters.

    Returns:
        tuple:
            - training_data (DataLoader):
                DataLoader for the training split.
            - validation_data (DataLoader):
                DataLoader for the validation split.
    """

    training_data = helpers.config_dataloader(data_config, 'train')
    validation_data = helpers.config_dataloader(data_config, 'val')

    return training_data, validation_data

def setup_subdatasets_metrics(training_config: TrainingConfig, validation_data: DataLoader):
    """
    TODO
    Create training and validation data loaders from the given data configuration.

    Args:
        data_config (DataConfig):
            Configuration describing the dataset and data loader parameters.

    Returns:
        tuple:
            - training_data (DataLoader):
                DataLoader for the training split.
            - validation_data (DataLoader):
                DataLoader for the validation split.
    """

    n_samples_metrics = training_config.n_samples_metrics
    batch_size_metrics = training_config.batch_size_metrics

    subset_indices_val = torch.randint(
        0, len(validation_data), (n_samples_metrics,))
    sub_val_set = Subset(validation_data.dataset, subset_indices_val)
    sub_val_loader = DataLoader(
        sub_val_set, batch_size=batch_size_metrics, shuffle=False, drop_last=True)

    return sub_val_loader


def setup_preprocess(data_config: DataConfig):
    """
    Build the preprocessing and inverse preprocessing pipelines.

    Args:
        data_config (DataConfig):
            Configuration describing the dataset preprocessing steps.

    Returns:
        tuple:
            - preprocess_pipeline:
                Transform pipeline applied before feeding data to the model.
            - inverse_preprocess_pipeline:
                Transform pipeline used to revert preprocessing for visualization
                or metric computation.
    """

    return helpers.config_preprocess_transform(data_config)


def setup_visual_samples(training_data: DataLoader, validation_data: DataLoader) -> dict[Literal['train', 'val'], list[torch.Tensor]]:
    """
    Select and prepare a small set of samples for visualization during training.

    The function extracts fixed samples from both the training and validation
    datasets and moves them to the specified device. These samples can be used
    for logging qualitative results.

    Args:
        training_data (DataLoader):
            DataLoader for the training dataset.
        validation_data (DataLoader):
            DataLoader for the validation dataset.

    Returns:
        dict:
            Mapping from sample identifiers to tensor samples on the selected
            device.
    """

    train_samples = []
    val_samples = []
    
    for idx in range(6):
        train_item = training_data.dataset[idx]
        val_item = validation_data.dataset[idx]
        
        # Handle datasets that return dicts (e.g., FFHQ) vs raw tensors
        if isinstance(train_item, dict):
            train_samples.append(train_item['input_images'])
        else:
            train_samples.append(train_item)
            
        if isinstance(val_item, dict):
            val_samples.append(val_item['input_images'])
        else:
            val_samples.append(val_item)
    
    return {
        'train': train_samples,
        'val': val_samples,
    }
