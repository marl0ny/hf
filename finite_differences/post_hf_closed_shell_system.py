from closed_shell_system import ClosedShellSystem, \
    angular_number_from_orbital_name, get_principle_from_orbital_name, \
    multiplicity_from_orbital_name
from typing import Dict, List, Union
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh

class ClosedShellSystemWithPostHF(ClosedShellSystem):

    virtual_orbitals: Dict[str, np.ndarray]
    virtual_energies: Dict[str, np.ndarray]

    def __init__(self, number_of_points: int, extent: float,
                 nuclear_charge: float, number_of_electrons: int,
                 orbital_letters: Union[None, List[str]] = None,
                 **kw: dict):
        self.virtual_orbitals = {}
        self.virtual_energies = {}
        ClosedShellSystem.__init__(
            self, number_of_points, extent,
            nuclear_charge, number_of_electrons, orbital_letters,
            **kw)

    def _single_iter_set_orbitals(self, repulsion: np.ndarray, 
                                      prev_orbitals: Dict[str, np.ndarray],
                                      iter_count: int, n_iterations: int):
        orbital_names = set([self.get_outermost_letter_name(o_name[1])
                                for o_name in self.orbital_names()])
        for orbital_name in orbital_names:
            exchange = self.get_exchange(orbital_name,
                                            prev_orbitals)
            an = angular_number_from_orbital_name(orbital_name)
            V = np.copy(self.V)
            if (self._apply_right_bound_reg and 
                iter_count <= self._right_bound_reg_final):
                V += np.diagflat(self.Z/np.abs(self.R - (self.R[-1] + self.DR)))

            H = self.T1 + self.M @ ((an * (an + 1) / 2) *
                                    self.T2 + V
                                    + repulsion - exchange)
            principle_n = get_principle_from_orbital_name(
                self.get_outermost_letter_name(orbital_name[1])
            )
            count = principle_n
            if 's' in orbital_name:
                count = principle_n
            elif 'p' in orbital_name:
                count = principle_n - 1
            elif 'd' in orbital_name:
                count = principle_n - 2
            elif 'f' in orbital_name:
                count = principle_n - 3
            # print(orbital_name, count)
            if self._use_np:
                eigval, eigvect = eigh(H @ self.INV_M_SPARSE.toarray())
            else:
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

    def solve(self, n_iterations: int, verbose: bool = False):
        self.verbose = verbose
        for iter_count in range(n_iterations):
            self.single_iter(iter_count, n_iterations)

    def compute_repulsion_exchange(self,
            oi: np.ndarray, oj: np.ndarray,
            ok: np.ndarray, ol: np.ndarray):
        pass

    def _compute_mp2_term(self, 
                          occ1: str, occ2: str, 
                          virt1: str, virt2: str):
        occ_e1 = self.orbital_energies[occ1]
        occ_e2 = self.orbital_energies[occ2]
        virt_e1 = self.virtual_energies[virt1]
        virt_e2 = self.virtual_energies[virt2]
        occ12_virt12 = self.get_repulsion_exchange(
            self.orbitals[occ1], self.orbitals[occ2],
            self.virtual_orbitals[virt1], self.virtual_orbitals[virt2])
        virt12_occ12 = self.get_repulsion_exchange(
            self.virtual_orbitals[virt1], self.virtual_orbitals[virt2],
            self.orbitals[occ1], self.orbitals[occ2])
        virt12_occ21 = self.get_repulsion_exchange(
            self.virtual_orbitals[virt1], self.virtual_orbitals[virt2],
            self.orbitals[occ2], self.orbitals[occ1])
        term = (
            2.0*occ12_virt12*virt12_occ12 
                / (occ_e1 + occ_e2 - virt_e1 - virt_e2)
            - occ12_virt12*virt12_occ21
                / (occ_e1 + occ_e2 - virt_e1 - virt_e2))
        return  2.0*term if virt1 == virt2 else term

    def compute_mp2_energy(self):
        sum = 0.0
        occupied_names = list(self.orbitals.keys())
        virtual_names = list(self.virtual_orbitals.keys())
        for occ1 in occupied_names:
            for occ2 in occupied_names:
                for virt1_ind, virt1 in enumerate(virtual_names):
                    for virt2_ind in range(virt1_ind, len(virtual_names)):
                        virt2 = virtual_names[virt2_ind]
                        sum += self._compute_mp2_term(
                            occ1, occ2, virt1, virt2
                        )
        return sum