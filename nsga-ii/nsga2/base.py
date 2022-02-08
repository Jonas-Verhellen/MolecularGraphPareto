import numpy as np
from typing import List
from dataclasses import dataclass, field

@dataclass
class Molecule:
    smiles: str
    rank: int = 0
    crowding_distance: float = 0
    fitnesses: List[float] = field(default_factory=lambda:[0.0, 0.0])
    fingerprint: np.ndarray= None
