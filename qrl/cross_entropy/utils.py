from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: List[EpisodeStep]