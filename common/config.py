from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DataConfig:
    data_root: str
    """ Path to the directory where data is actually stored """

    dataset_name: str
    """ Name of the dataset to use (e.g., 'sen2', 'ffhq') """

    train_config_file: str
    """ Path to json file indicating which images are part of the training set """

    val_config_file: str
    """ Path to json file indicating which images are part of the validation set """

    test_config_file: str
    """ Path to json file indicating which images are part of the test set """

    train_batch_size: int
    """ Batch size to use during training on the training set """

    val_batch_size: int
    """ Batch size to use during evaluation on the validation and test sets """

    num_workers: int
    """ Number of processes to use for data loading. Set 0 to use the main process """

    clip: bool
    """ Whether to clip input data or not """

    clip_val: Optional[float]
    """ If clip==True, this value is used to clip input data in the range [0, clip_val] """

    rescale_mode: Optional[str]
    """ Use either 'standardize' or 'normalize' to rescale input data. If None, no rescale is applied """

    standardize_mode: Optional[str]
    """ If rescale_mode=='standardize', specifies which mean/std pair to use """

    normalize_min: Optional[float]
    """ If rescale_mode=='normalize', specifies the minimum value for 0–1 normalization """

    normalize_max: Optional[float]
    """ If rescale_mode=='normalize', specifies the maximum value for 0–1 normalization """


@dataclass
class ModelConfig:
    model_name: str
    """ Model name, used to identify runs """

    DiT_type: str
    """ DiT model type (e.g., 'DiT-XL/2', 'DiT-XL/4', 'DiT-XL/8') """

    input_size: int
    """ Height of input images """

    in_channels: int
    """ Number of bands in input images (e.g., 13) """

    single_model: bool
    """ Whether to jointly learn velocity/vector field and score/denoiser using a single model """


@dataclass
class InterpolantConfig:
    gaussian_base_distr: bool
    """ Whether to use a Gaussian base distribution """

    mixture_numbers: int
    """ Number of mixture components (e.g., 3) """

    path: str
    """ Type of interpolation path:
        'linear', 'trig', 'encoding-decoding',
        'one-sided-linear', 'one-sided-trig', 'mirror'
    """

    gamma_type: Optional[str]
    """ Gamma type, e.g., None or 'brownian' """

    type_of_learning: str
    """ Type of learning:
        'velocity', 'vector-score', 'velocity-score',
        'vector-denoiser', 'velocity-denoiser'
    """

    n_visual_sampling_steps: int
    """ Number of sampling steps (e.g., 1000) """

    n_metric_sampling_steps: int
    """ Number of sampling steps (e.g., 1000) """

    n_gen_samples: int
    """ Number of samples to generate during evaluation (e.g., 10000) """

    eps: float
    """ Diffusion term (e.g., 6.0) """

    lower_ts: float
    """ Lower endpoint timestamp (e.g., 0.0001) """

    upper_ts: float
    """ Upper endpoint timestamp (e.g., 0.9999) """

    sampler_method: str
    """ Type of sampler:
        deterministic (fixed-step): ['euler', 'midpoint', 'rk4']
        adaptive-step: ['dopri5', 'dopri8', 'adaptive_heun']
        stochastic: ['euler-maruyama', 'heun']
    """

    atol: float
    """ Absolute tolerance of the sampler (e.g., 0.0001) """

    rtol: float
    """ Relative tolerance of the sampler (e.g., 0.0001) """


@dataclass
class TrainingConfig:
    max_epochs: int
    """ Maximum number of training epochs. """

    visual_freq_steps: int
    """ Number of steps (processed batches) between visual runs. """

    validation_freq_steps: int
    """ Number of steps (processed batches) between validation runs. """

    learning_rate: float
    """ Learning rate for the AdamW optimizer. """

    weight_decay: float
    """ Weight decay for the AdamW optimizer. """

    wd_exclude_params: list[str]
    """ List of parameter names to exclude from weight decay. """

    clip_grad_norm: bool
    """ Whether to clip the gradient L2 norm to 1.0. """

    cosine_scheduler_with_warmup: bool
    """ Whether to use a cosine learning rate scheduler with warmup. """

    num_warmup_steps: int
    """ Number of steps for the linear warmup phase in the scheduler. """

    num_training_steps: int
    """ Total number of training steps for the scheduler. """

    min_lr_ratio: float
    """ Minimum learning rate ratio relative to initial LR in cosine decay. """

    ema: bool
    """ Whether to use Exponential Moving Average (EMA) for model parameters. """

    beta_1: float
    """ EMA coefficient for the existing average weights. """

    beta_2: float
    """ EMA coefficient for the current model weights. """

    ckpt_path: str
    """ Path where latest and best checkpoints are stored. """

    wandb_log_dir: str
    """ Path where Weights & Biases logs are stored. """

    wandb_project_name: str
    """ Name of the Weights & Biases project. """

    n_samples_metrics: int
    """ Number of samples to consider for computing metrics (FID, LPIPS). """

    batch_size_metrics: int
    """ Batch size to use during metrics computation. """

    resume: bool
    """ Whether to resume training from the latest checkpoint. """

    metrics_ref_min: Optional[float] = None
    """ Minimum reference value for metrics such as PSNR and SSIM. """

    metrics_ref_max: Optional[float] = None
    """ Maximum reference value for metrics such as PSNR and SSIM. """


@dataclass
class Config:
    data: DataConfig
    """ Data-related configuration """

    model: ModelConfig
    """ Model-related configuration """

    interpolant: InterpolantConfig
    """ Interpolant-related configuration """

    training: TrainingConfig
    """ Training-related configuration """


@dataclass
class DiTArgs:
    input_size: int
    """ Height of input images """

    in_channels: int
    """ Number of input bands """

    single_model: bool
    """ Whether the DiT jointly learns all targets """

    is_deterministic: bool
    """ If the type of learning is deterministic """


LossConfig = dict[
    str,
    tuple[
        Literal[
            'b', 'v', 'b-s', 'v-s', 'b-eta', 'v-eta'
        ],
        Literal['s', 'eta', None]
    ]
]
