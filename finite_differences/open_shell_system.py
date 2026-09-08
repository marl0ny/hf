from spherically_symmetric_system import *
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from typing import Dict, List


def spin_from_orbital_name(orbital_name):
    if orbital_name[-1] == '+':
        return 1
    elif orbital_name[-1] == '-':
        return -1
    else:
        return 0


def get_outermost_letter_name(orbital_names, letter):
    if letter not in ['s', 'p', 'd', 'f', 'g']:
        raise NotImplementedError
    orbital_values = [int(e[0]) for e in orbital_names if letter in e]
    if orbital_values is []:
        return
    return f'{max(orbital_values)}{letter}'


class UnrestrictedSystem(SphericallySymmetricSystemBase):
    """
    Compute the Hartree-Fock energies and orbitals for spherically symmetric
    unrestricted open shell systems through finite differences
    """
    number_of_electrons: int
    _outermost_count: int
    _outermost_orbital_name: str
    _all_orbital_names: list
    _exc_exp: dict[str, np.ndarray]
    _apply_right_bound_reg: bool
    _right_bound_reg_final: int
    _use_np: bool

    def __init__(self, number_of_points: int, extent: float,
                 nuclear_charge: float, number_of_electrons: int,
                 orbital_letters: Union[None, List[str]] = None,
                 **kw: dict):

        # delta = 0.1 * (64 / number_of_points) ** 2
        delta = (100 / number_of_points) ** 2
        # delta = 5.0 * (64 / number_of_points) ** 2
        if 'delta' in kw.keys():
            delta = kw['delta']
        SphericallySymmetricSystemBase.__init__(self, 
                                                number_of_points, extent,
                                                nuclear_charge, delta,
                                                lambda r, nuc: -nuc/r)
        allowed_number_of_electrons = [i for i in range(40)]
        hydrogen_like_orbitals = self.construct_hydrogen_like_orbitals()
        if not any([number_of_electrons == n
                    for n in allowed_number_of_electrons]):
            raise NotImplementedError
        self.number_of_electrons = number_of_electrons
        all_orbital_names = [
            '1s+', '1s-',
            '2s+', '2s-', '2p+', '2p+', '2p+', '2p-', '2p-', '2p-',
            '3s+', '3s-', '3p+', '3p+', '3p+', '3p-', '3p-', '3p-',
            '4s+', '4s-', 
            '3d+', '3d+', '3d+', '3d+', '3d+',
            '3d-', '3d-', '3d-', '3d-', '3d-',
            '4p+', '4p+', '4p+', '4p-', '4p-', '4p-']
        if not (orbital_letters is None):
            all_orbital_names = orbital_letters
        self._all_orbital_names = all_orbital_names
        self._outermost_orbital_name = all_orbital_names[:number_of_electrons][-1]
        self._outermost_count = len([
            o for o in all_orbital_names[:number_of_electrons]
            if o == self._outermost_orbital_name])
        orbital_designations = set(all_orbital_names[:number_of_electrons])

        if 'init_functions' in kw:
            import json
            with open(kw['init_functions'], 'r') as f:
                contents = ''.join([line for line in f])
            init_orbitals = json.loads(contents)
            for o_name in orbital_designations:
                if o_name[0:2] in init_orbitals \
                    or o_name[0:2] in init_orbitals:
                    o = o_name[0:2]
                    self.init_orbitals[o_name] \
                        = init_orbitals[o]
                    self.orbitals[o_name] \
                        = init_orbitals[o]
                else:
                    self.init_orbitals[o_name] \
                        = hydrogen_like_orbitals[o_name[:2]].copy()
                    self.orbitals[o_name] \
                        = hydrogen_like_orbitals[o_name[:2]]
                self.orbital_energies[o_name] = []
        else:
            for o_name in orbital_designations:
                self.init_orbitals[o_name] \
                    = hydrogen_like_orbitals[o_name[:2]].copy()
                self.orbitals[o_name] \
                    = hydrogen_like_orbitals[o_name[:2]]
                self.orbital_energies[o_name] = []
        self._exc_exp = {}
        r_max = self.R_GREATER_THAN
        r_min = self.R_LESS_THAN
        self._exc_exp['s(s)'] = 1.0 / r_max
        self._exc_exp['s(p)'] = r_min / r_max ** 2
        self._exc_exp['s(d)'] = r_min ** 2 / r_max ** 3
        self._exc_exp['p(s)'] = r_min / r_max ** 2 / 3.0
        self._exc_exp['p(p)'] = 1.0 / r_max + 0.4 * r_min ** 2 / r_max ** 3
        self._exc_exp['p(d)'] = ((3.0 / 7.0) * (r_min**3 / r_max ** 4)
                                    + (2.0 / 3.0) * (r_min / r_max ** 2))
        self._exc_exp['d(s)'] = (1.0 / 5.0) * (r_min**2 / r_max**3)
        self._exc_exp['d(p)'] = ((9.0 / 35.0) * (r_min**3 / r_max**4)
                                    + (2.0 / 5.0) * (r_min / r_max**2))
        self._exc_exp['d(d)'] = ((1.0 / r_max)
                                    + (2.0 / 7.0) * (r_min**4 / r_max**5)
                                    + (2.0 / 7.0) * (r_min**2 / r_max**3))
        self._apply_right_bound_reg = False
        self._right_bound_reg_final = 0
        self._use_np = True

    def get_spin_up_orbital_names(self):
        return set([o_name for o_name in self.orbitals.keys()
                    if '+' in o_name])

    def get_spin_down_orbital_names(self):
            return set([o_name for o_name in self.orbitals.keys() 
                        if '-' in o_name])

    def get_sorted_orbital_names(self, orbital_names: List[str]) -> List[str]:
        sorted_orbital_names = []
        for o in self._all_orbital_names:
            if o in orbital_names and (not o in sorted_orbital_names):
                sorted_orbital_names.append(o)
        return sorted_orbital_names

    def toggle_right_boundary_potential_regulator(
            self, remove_regulator_at: int = 8):
        self._apply_right_bound_reg = not self._apply_right_bound_reg
        self._right_bound_reg_final = remove_regulator_at

    def get_repulsion(self, orbital):
        # orbital2 = np.zeros([N+1])
        # orbital2[1::] = np.conj(orbital)*orbital
        # integrand1 =
        return np.diagflat(cumulative_trapezoid(self.DR
                                                * np.exp(self.S
                                                         * self.DELTA)
                                                * orbital ** 2,
                                                initial=0.0) / self.R
                           + cumulative_trapezoid((self.DR
                                                   * np.exp(self.S
                                                            * self.DELTA)
                                                   * orbital ** 2
                                                   / self.R)[::-1],
                                                  initial=0.0)[::-1])

    def multiplicity_from_orbital_name(self, orbital_name) -> float:
        if orbital_name == self._outermost_orbital_name:
            return self._outermost_count
        return multiplicity_from_orbital_name(orbital_name)

    def _exc_mul_scale_factor(self, other_orbital_name: str) -> float:
            if (self._outermost_orbital_name == other_orbital_name):
                return self.multiplicity_from_orbital_name(other_orbital_name) \
                    / multiplicity_from_orbital_name(other_orbital_name)
            return 1.0

    def get_exchange(self, orbital_name: str,
                     orbitals: Dict[str, np.ndarray]) -> np.ndarray:
        measure = self.DR * np.exp(self.S * self.DELTA)
        exchange = np.zeros([self.N, self.N])
        for other_orbital_name in orbitals.keys():
            s1 = spin_from_orbital_name(orbital_name)
            s2 = spin_from_orbital_name(other_orbital_name)
            mul_factor = self._exc_mul_scale_factor(other_orbital_name)
            if (s1 == s2):
                outer_prod = np.outer(
                    orbitals[other_orbital_name],
                    orbitals[other_orbital_name]
                )
                ang = orbital_name[1]
                other_ang = other_orbital_name[1]
                expansion = self._exc_exp[f'{ang}({other_ang})']
                exchange += (mul_factor 
                             * expansion * outer_prod) @ np.diag(measure)
        return exchange

    def _single_iter_set_orbitals(self, repulsion: np.ndarray,
                                  prev_orbitals: Dict[str, np.ndarray],
                                  iter_count: int):
        spin_up_orbital_names = self.get_spin_up_orbital_names()
        spin_down_orbital_names = self.get_spin_down_orbital_names()
        # print(spin_up_orbital_names)
        # print(spin_down_orbital_names)
        # print(spin_up_orbital_names, '\n', spin_down_orbital_names)
        V = np.copy(self.V)
        if (self._apply_right_bound_reg and 
            iter_count <= self._right_bound_reg_final):
            V += np.diagflat(self.Z/np.abs(self.R - (self.R[-1] + self.DR)))
        for spin_letter, orbital_names in zip(['+', '-'], 
                                              [spin_up_orbital_names,
                                               spin_down_orbital_names]):
            o_names_set = set([
                get_outermost_letter_name(orbital_names, o_name[1])
                for o_name in orbital_names])
            sorted_o_names = self.get_sorted_orbital_names(
                [o + spin_letter for o in o_names_set])
            # if self.verbose:
            #     print('Computing energies for:', *sorted_o_names)
            for o_name_ in sorted_o_names:
                orbital_name = o_name_
                # if self.verbose:
                #     print(orbital_name)
                exchange = self.get_exchange(orbital_name, prev_orbitals)
                an = angular_number_from_orbital_name(orbital_name)
                H = self.T1 + self.M @ ((an * (an + 1) / 2) *
                                        self.T2 + V
                                        + repulsion - exchange)
                principle_n = get_principle_from_orbital_name(orbital_name)
                count = principle_n
                if 'p' in orbital_name:
                    count = principle_n - 1
                if 'd' in orbital_name:
                    count = principle_n - 2
                if self._use_np:
                    eigval, eigvect = eigh(H @ self.INV_M_SPARSE.toarray())
                else:
                    eigval, eigvect = eigsh(H, k=count, M=self.M_SPARSE,
                                            which='LM', sigma=0.0)
                for n in range(count):
                    orbital_name2 = f'{1 + n + an}{orbital_name[1:]}'
                    self.orbital_energies[orbital_name2].append(
                        27.211386245 * (eigval[n] + self.GLOBAL_SHIFT))
                    if self.verbose:
                        print(orbital_name2, ': ',
                              self.orbital_energies[orbital_name2][-1],
                              'eV')
                        self.orbitals[orbital_name2] \
                            = self.normalize(eigvect.T[n])

    def single_iter(self, iter_count: int):
        if self.verbose:
            print('Iteration Count: ', iter_count)
        repulsion = np.zeros([self.N, self.N])
        orbitals_keys = self.orbitals.keys()
        for o_name in self.get_sorted_orbital_names(list(orbitals_keys)):
            mul = self.multiplicity_from_orbital_name(o_name)
            print(f'{o_name} multiplicity: ', mul)
            repulsion += mul * self.get_repulsion(self.orbitals[o_name])
        orbitals_copy = {name: self.orbitals[name].copy()
                         for name in self.orbitals.keys()}
        self._single_iter_set_orbitals(repulsion, orbitals_copy, iter_count)

    def solve(self, n_iterations, verbose=False):
        self.verbose = verbose
        for iter_count in range(n_iterations):
            self.single_iter(iter_count)

    def get_kinetic_energy(self):
        orbitals = self.orbitals
        kinetic_energy = 0.0
        for orbital_name in orbitals.keys():
            orbital = orbitals[orbital_name]
            orbital2 = np.conj(orbital) * orbital
            an = angular_number_from_orbital_name(orbital_name)
            n = self.multiplicity_from_orbital_name(orbital_name)
            orbital_from0 = np.zeros([self.N + 1])
            orbital_from0[1::] = orbital
            k1_orbital = np.zeros([self.N + 1])
            k1_orbital[1::] = self.INV_M @ self.T1 @ orbital
            k1_integrand = orbital_from0 * k1_orbital
            k1_int = simpson(self.DR_0 * np.exp(self.S_0 * self.DELTA)
                             * k1_integrand, even='first')
            k2_integrand = np.zeros([self.N + 1])
            k2_integrand[1::] = ((an * (an + 1) / 2) * self.T2) @ orbital2
            k2_int = simpson(self.DR_0 * np.exp(self.S_0 * self.DELTA)
                             * k2_integrand, even='first')
            # print(orbital_name, n * (k1_int + k2_int))
            kinetic_energy += n * (k1_int + k2_int)
        return kinetic_energy

    def get_potential_energy(self):
        orbitals = self.orbitals
        potential_energy = 0.0
        for orbital_name in orbitals.keys():
            orbital = orbitals[orbital_name]
            n = self.multiplicity_from_orbital_name(orbital_name)
            orbital2 = np.conj(orbital) * orbital
            integrand = np.zeros([self.N + 1])
            # In the limit when r approaches zero, the value of the
            # integrand should also be zero.
            integrand[1::] = -self.Z * orbital2 / self.R
            integral = simpson(self.DR_0 * np.exp(self.S_0 * self.DELTA)
                               * integrand, even='first')
            # print(orbital_name, n * integral)
            potential_energy += n * integral
        return potential_energy

    def get_repulsion_energy(self):
        orbitals = self.orbitals
        repulsion_energy = 0.0
        orbital_names = self.get_sorted_orbital_names(list(orbitals.keys()))
        for orbital_name_i in orbital_names:
            for j in range(len(orbital_names)):
                orbital_name_j = orbital_names[j]
                angular_mul_i = \
                    self.multiplicity_from_orbital_name(orbital_name_i)
                angular_mul_j = \
                    self.multiplicity_from_orbital_name(orbital_name_j)
                print(f'{orbital_name_i}, {orbital_name_j}: ',
                      angular_mul_i, angular_mul_j)
                orbital_i = orbitals[orbital_name_i]
                orbital_j = orbitals[orbital_name_j]
                orbital_from0_i = np.zeros([self.N + 1])
                orbital_from0_i[1::] = orbitals[orbital_name_i]
                repulsion_matrix = self.get_repulsion(orbital_j)
                repulsion_int1 = np.zeros([self.N + 1])
                repulsion_int1[1::] = repulsion_matrix @ orbital_i
                repulsion_energy += (angular_mul_i * angular_mul_j
                                     ) * simpson(self.DR_0 *
                                                 np.exp(self.S_0
                                                        * self.DELTA) *
                                                 np.conj(orbital_from0_i) *
                                                 repulsion_int1)
        return repulsion_energy

    def get_exchange_energy(self):
        orbitals = self.orbitals
        exchange_energy = 0.0
        orbital_names = list(orbitals.keys())
        for orbital_name_i in orbital_names:
            for j in range(len(orbital_names)):
                orbital_name_j = orbital_names[j]
                angular_mul_i = \
                    self.multiplicity_from_orbital_name(orbital_name_i)
                exchange_matrix = self.get_exchange(orbital_name_i,
                                                    {orbital_name_j:
                                                     orbitals[orbital_name_j]})
                orbital_from0_i = np.zeros([self.N + 1])
                orbital_from0_i[1::] = orbitals[orbital_name_i]
                int1 = np.zeros([self.N + 1])
                int1[1::] = exchange_matrix @ orbitals[orbital_name_i]
                exchange_energy += (angular_mul_i
                                    * simpson(self.DR_0 *
                                              np.exp(self.S_0 * self.DELTA) *
                                              orbital_from0_i * int1))
        return exchange_energy

    def get_total_energy(self):
        return (self.get_kinetic_energy()
                + self.get_potential_energy()
                + self.get_repulsion_energy() / 2.0
                - self.get_exchange_energy() / 2.0)
