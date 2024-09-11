from .gaussian3d import Gaussian3D
from .integrals3d import overlap
from typing import Dict, List
import numpy as np


def get_basis_function_norm_factor(pos: np.ndarray, ang: np.ndarray,
                            coefficients: np.ndarray,
                            exponents: np.ndarray) -> float:
    n = len(coefficients)
    orbital_overlap = np.zeros([n, n])
    for i in range(len(coefficients)):
        ei = exponents[i]
        gi = Gaussian3D(1.0, ei, pos, ang)
        for j in range(i, len(coefficients)):
            ej = exponents[j]
            gj = Gaussian3D(1.0, ej, pos, ang)
            orbital_overlap[i, j] = overlap(gi, gj)
            if j > i:
                orbital_overlap[j, i] \
                    = orbital_overlap[i, j]
    return np.sqrt(coefficients @ orbital_overlap @ coefficients)


class BasisFunction:
    primitives: List[Gaussian3D]
    def __init__(self, 
                 position: np.ndarray, angular: np.ndarray,
                 primitives_dict: Dict[str, List[float]]):
        coefficients = primitives_dict['coefficients']
        exponents = primitives_dict['exponents']
        norm_factor = get_basis_function_norm_factor(
            position, ang, coefficients, exponents)
        for c, e in zip(coefficients, exponents):
            self.primitives.append(Gaussian3D(c, e, position, ang))

    def __len__(self) -> int:
        return len(primatives)

    def __index__(self, index: int) -> Gaussian3D:
        return primitives[index]
