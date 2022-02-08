from typing import List
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Molecule:
    smiles: str
    rank: int = 0
    nearest_direction: int = 0 
    orthogonal_distance: float = 0.0
    fitnesses: List[float] = field(default_factory=lambda:[0.0, 0.0])
    fingerprint: np.ndarray= None
