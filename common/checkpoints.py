from dataclasses import dataclass


@dataclass
class Checkpoints:
    latest: str
    best: str
