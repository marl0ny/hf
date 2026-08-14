from closed_shell_system import ClosedShellSystem, \
    angular_number_from_orbital_name, get_principle_from_orbital_name, \
    multiplicity_from_orbital_name
from typing import Dict, List, Union
import numpy as np
from scipy.sparse.linalg import eigsh


class ClosedShellSystemWithPostHF(ClosedShellSystem):

    virtual_orbitals: Dict[str, np.ndarray]
    virtual_energies: Dict[str, np.ndarray]

    def __init__(self, number_of_points: int, extent: float,
                 nuclear_charge: float, number_of_electrons: int,
                 orbital_letters: Union[None, List[str]] = None
                 ):
        self.virtual_orbitals = {}
        self.virtual_energies = {}
        ClosedShellSystem.__init__(
            self, number_of_points, extent,
            nuclear_charge, number_of_electrons, orbital_letters)

    def _single_iter_set_orbitals(self, repulsion: np.ndarray, 
                                      prev_orbitals: Dict[str, np.ndarray],
                                      iter_count: int, n_iterations: int):
        orbital_names = set([self.get_outermost_letter_name(o_name[1])
                                for o_name in self.orbital_names()])
        # if iter_count == n_iterations - 1:
        #     orbital_names = {str(int(e[0]) + 2) + e[1] for e in orbital_names}
            # if '3s' in orbital_names and '2p' not in orbital_names:
            #     orbital_names.add('2p')
        # print(orbital_names)
        for orbital_name in orbital_names:
            exchange = self.get_exchange(orbital_name,
                                            prev_orbitals)
            an = angular_number_from_orbital_name(orbital_name)
            V = self.V + np.diagflat(self.Z/np.abs(self.R - (self.R[-1] + self.DR)))
            H = self.T1 + self.M @ ((an * (an + 1) / 2) *
                                    self.T2 + V
                                    + repulsion - exchange)
            principle_n = get_principle_from_orbital_name(orbital_name)
            count = principle_n
            if 's' in orbital_name:
                count = principle_n
            elif 'p' in orbital_name:
                count = principle_n - 1
            eigval, eigvect = eigsh(H, k=count, M=self.M_SPARSE,
                                    which='LM', sigma=0.0)
            for n in range(count):
                orbital_name2 = f'{1 + n + an}{orbital_name[1]}'
                # print(orbital_name2, n)
                if self.verbose:
                    print(orbital_name2, 
                            ('' if orbital_name2 
                            in self.orbitals.keys() else '(virtual) ') + ': ',
                            27.211386245 * (eigval[n] + self.GLOBAL_SHIFT),
                            'eV')
                if orbital_name2 in self.orbitals.keys():
                    self.orbital_energies[orbital_name2].append(
                        27.211386245 * (eigval[n] + self.GLOBAL_SHIFT))
                    self.orbitals[orbital_name2] \
                        = self.normalize(eigvect.T[n])
                else:
                    self.virtual_energies[orbital_name2] = \
                        27.211386245 * (eigval[n] + self.GLOBAL_SHIFT)
                    self.virtual_orbitals[orbital_name2] = \
                        self.normalize(eigvect.T[n])
    
    def single_iter(self, iter_count: int, n_iterations: int):
        if self.verbose:
            print('Iteration Count: ', iter_count)
        repulsion = 2.0 * sum([
            multiplicity_from_orbital_name(name)
            * self.get_repulsion(self.orbitals[name])
            for name in self.orbitals.keys()
        ])
        orbitals_copy = {name: self.orbitals[name].copy()
                            for name in self.orbitals.keys()}
        self._single_iter_set_orbitals(repulsion,
                                        orbitals_copy, iter_count,
                                        n_iterations
                                        )
        # if iter_count >= 9:
        #     self._average_current_with_previous_if_conv_not_reached(
        #         orbitals_copy
        #     )

    def solve(self, n_iterations: int, verbose: bool = False):
        self.verbose = verbose
        for iter_count in range(n_iterations):
            self.single_iter(iter_count, n_iterations)