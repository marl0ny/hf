from spherically_symmetric_system import *
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from typing import Dict


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

    def __init__(self, number_of_points: int, extent: float,
                 nuclear_charge: float, number_of_electrons: int):

        delta = 0.1 * (64 / number_of_points) ** 2
        # delta = 5.0 * (64 / number_of_points) ** 2
        SphericallySymmetricSystemBase.__init__(self, 
                                                number_of_points, extent,
                                                nuclear_charge, delta,
                                                lambda r, nuc: -nuc/r)
        allowed_number_of_electrons = [1, 2, 3, 4, 7, 10, 
                                       11, 12, 15, 18, 19, 20]
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
            '4d+', '4d+', '4d+', '4d+', '4d+',
            '4d-', '4d-', '4d-', '4d-', '4d-',
            '4p+', '4p+', '4p+', '4p-', '4p-', '4p-']
        orbital_designations = set(all_orbital_names[:number_of_electrons])
        for o_name in orbital_designations:
            self.init_orbitals[o_name] \
                = hydrogen_like_orbitals[o_name[:2]].copy()
            self.orbitals[o_name] \
                = hydrogen_like_orbitals[o_name[:2]]
            self.orbital_energies[o_name] = []

    def get_spin_up_orbital_names(self):
        return set([o_name for o_name in self.orbitals.keys()
                    if '+' in o_name])

    def get_spin_down_orbital_names(self):
        return set([o_name for o_name in self.orbitals.keys() 
                    if '-' in o_name])

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

    def get_exchange(self, orbital_name: str,
                     orbitals: Dict[str, np.ndarray]) -> np.ndarray:
        exchange = np.zeros([self.N, self.N])
        for other_orbital_name in orbitals.keys():
            s1 = spin_from_orbital_name(orbital_name)
            s2 = spin_from_orbital_name(other_orbital_name)
            outer_prod = np.outer(self.DR * np.exp(self.S * self.DELTA)
                                  * orbitals[other_orbital_name],
                                  orbitals[other_orbital_name])
            if 's' in orbital_name:
                if 's' in other_orbital_name and s1 == s2:
                    exchange += outer_prod / self.R_GREATER_THAN
                elif 'p' in other_orbital_name and s1 == s2:
                    exchange += outer_prod * self.R_LESS_THAN / \
                                self.R_GREATER_THAN ** 2
            elif 'p' in orbital_name:
                if 's' in other_orbital_name and s1 == s2:
                    exchange += outer_prod * self.R_LESS_THAN / \
                                self.R_GREATER_THAN ** 2 / 3.0
                if 'p' in other_orbital_name and s1 == s2:
                    exchange += (outer_prod / self.R_GREATER_THAN
                                 + 0.4 * outer_prod
                                 * self.R_LESS_THAN ** 2 /
                                 self.R_GREATER_THAN ** 3
                                 )
        return exchange

    def _single_iter_set_orbitals(self, repulsion: np.ndarray,
                                  prev_orbitals: Dict[str, np.ndarray]):
        spin_up_orbital_names = self.get_spin_up_orbital_names()
        spin_down_orbital_names = self.get_spin_down_orbital_names()
        # print(spin_up_orbital_names)
        # print(spin_down_orbital_names)
        # print(spin_up_orbital_names, '\n', spin_down_orbital_names)
        for spin_letter, orbital_names in zip(['+', '-'], 
                                              [spin_up_orbital_names,
                                               spin_down_orbital_names]):
            for o_name_ in set([get_outermost_letter_name(orbital_names,
                                                          o_name[1])
                                for o_name in orbital_names]):
                orbital_name = o_name_ + spin_letter
                # if self.verbose:
                #     print(orbital_name)
                exchange = self.get_exchange(orbital_name, prev_orbitals)
                an = angular_number_from_orbital_name(orbital_name)
                H = self.T1 + self.M @ ((an * (an + 1) / 2) *
                                        self.T2 + self.V
                                        + repulsion - exchange)
                principle_n = get_principle_from_orbital_name(orbital_name)
                count = principle_n
                if 'p' in orbital_name:
                    count = principle_n - 1
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

    def single_iter(self, iter_count):
        if self.verbose:
            print('Iteration Count: ', iter_count)
        repulsion = np.zeros([self.N, self.N])
        orbital_count = self.number_of_electrons
        for o_name in set([o[:2] for o in self.orbitals.keys()]):
            apparent_mult = 2*multiplicity_from_orbital_name(o_name)
            mult = orbital_count if \
                (orbital_count - apparent_mult) < 0 else apparent_mult
            o_name2 = (f"{o_name}{'+'}"
                       if f"{o_name}{'+'}" in self.orbitals
                       else f"{o_name}{'-'}")
            repulsion += mult * self.get_repulsion(self.orbitals[o_name2])
            orbital_count -= mult
        orbitals_copy = {name: self.orbitals[name].copy()
                         for name in self.orbitals.keys()}
        self._single_iter_set_orbitals(repulsion, orbitals_copy)

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
            n = multiplicity_from_orbital_name(orbital_name)
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
            n = multiplicity_from_orbital_name(orbital_name)
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
        orbital_names = list(orbitals.keys())
        for orbital_name_i in orbital_names:
            for j in range(len(orbital_names)):
                orbital_name_j = orbital_names[j]
                spin_mul = 2
                angular_mul_i = multiplicity_from_orbital_name(orbital_name_i)
                angular_mul_j = multiplicity_from_orbital_name(orbital_name_j)
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
                angular_mul_i = multiplicity_from_orbital_name(orbital_name_i)
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
