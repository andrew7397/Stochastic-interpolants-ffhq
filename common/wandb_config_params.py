from dataclasses import dataclass
from typing import Optional


@dataclass
class WandbConfigParams:
    batch_size: Optional[int]
    learning_rate: float
    weight_decay: float
    dataset: str
    model_type: str
    model_name: str
