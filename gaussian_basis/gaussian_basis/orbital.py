from .basis_function import BasisFunction
from typing import List, Dict, Tuple, Union
import numpy as np


class Orbital:
    position: np.ndarray
    angular: np.ndarray
    coefficients: np.ndarray
    basis_functions: List[BasisFunction]
    def __init__(self,
                 position: np.ndarray,
                 angular: np.ndarray,
                 orbital_dict: 
                 List[Dict[str, Union[float, Dict[str, List[float]]]]]):
        coeff = []
        self.position = position
        self.angular = angular
        for e in basis_functions:
            c = e['coefficient']
            basis_function_dict = e['primitives']
            coeff.append(c)
            self.basis_functions.append(BasisFunction(position, angular, 
                                                      basis_function_dict))
        self.coefficients = np.array(coeff)

    def __len__(self) -> int:
        return len(basis_functions)

    def __index__(self, index: int) -> Tuple[float, BasisFunction]:
        return coefficients[index], basis_functions[index]
