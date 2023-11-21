from spherically_symmetric_system import *
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from typing import Dict, List, Union


class ClosedShellSystem(SphericallySymmetricSystemBase):
    """
    Used for numerically computing the Hartree-Fock energies and orbitals
    of spherically symmetric closed shell systems through finite differences.
    """

    def __init__(self, number_of_points: int, extent: float,
                 nuclear_charge: float, number_of_electrons: int,
                 orbital_letters: Union[None, List[str]] = None
                 ):

        delta = (64 / number_of_points) ** 2
        SphericallySymmetricSystemBase.__init__(self, 
                                                number_of_points, extent,
                                                nuclear_charge, delta,
                                                lambda r, nuc: -nuc/r)

        orbitals = self.construct_hydrogen_like_orbitals()
        if orbital_letters is None:
            allowed_number_of_electrons = [2, 4, 10, 12, 18, 20]
            if not any([number_of_electrons == n
                        for n in allowed_number_of_electrons]):
                raise NotImplementedError
            orbital_letters = ['1s',
                               '2s', '2p', '2p', '2p',
                               '3s', '3p', '3p', '3p',
                               '4s']
        orbital_designations = set(orbital_letters[:number_of_electrons//2])
        for o_name in orbital_designations:
            self.init_orbitals[o_name] = orbitals[o_name].copy()
            self.orbitals[o_name] = orbitals[o_name]
            self.orbital_energies[o_name] = []

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
            outer_prod = np.outer(self.DR * np.exp(self.S * self.DELTA)
                                  * orbitals[other_orbital_name],
                                  orbitals[other_orbital_name])
            if 's' in orbital_name:
                if 's' in other_orbital_name:
                    exchange += outer_prod / self.R_GREATER_THAN
                elif 'p' in other_orbital_name:
                    exchange += outer_prod * self.R_LESS_THAN / \
                                self.R_GREATER_THAN ** 2
            elif 'p' in orbital_name:
                if 's' in other_orbital_name:
                    exchange += outer_prod * self.R_LESS_THAN / \
                                self.R_GREATER_THAN ** 2 / 3.0
                if 'p' in other_orbital_name:
                    exchange += (outer_prod / self.R_GREATER_THAN
                                 + 0.4 * outer_prod
                                 * self.R_LESS_THAN ** 2 /
                                 self.R_GREATER_THAN ** 3
                                 )
        return exchange

    def _single_iter_set_orbitals_no_ortho(self, repulsion: np.ndarray, 
                                           prev_orbitals: 
                                           Dict[str, np.ndarray]):
        for orbital_name in self.orbital_names():
            exchange = self.get_exchange(orbital_name,
                                         prev_orbitals)
            an = angular_number_from_orbital_name(orbital_name)
            principle_n = get_principle_from_orbital_name(orbital_name)
            count = principle_n
            if 'p' in orbital_name:
                count -= 1
            H = self.T1 + self.M @ ((an * (an + 1) / 2) *
                                    self.T2 + self.V
                                    + repulsion - exchange)
            eigval, eigvect = eigsh(H, k=count, M=self.M_SPARSE,
                                    which='LM', sigma=0.0)
            self.orbital_energies[orbital_name].append(
                    27.211386245 * (eigval[count-1] + self.GLOBAL_SHIFT))
            if self.verbose:
                print(orbital_name, ': ',
                      self.orbital_energies[orbital_name][-1],
                      'eV')
            self.orbitals[orbital_name] \
                    = self.normalize(eigvect.T[count-1])

    def _single_iter_set_orbitals(self, repulsion: np.ndarray, 
                                  prev_orbitals: Dict[str, np.ndarray]):
        for orbital_name in set([self.get_outermost_letter_name(o_name[1])
                                 for o_name in self.orbital_names()]):
            exchange = self.get_exchange(orbital_name,
                                         prev_orbitals)
            an = angular_number_from_orbital_name(orbital_name)
            H = self.T1 + self.M @ ((an * (an + 1) / 2) *
                                    self.T2 + self.V
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
        repulsion = 2.0 * sum([
            multiplicity_from_orbital_name(name)
            * self.get_repulsion(self.orbitals[name])
            for name in self.orbitals.keys()
        ])
        orbitals_copy = {name: self.orbitals[name].copy()
                         for name in self.orbitals.keys()}
        self._single_iter_set_orbitals(repulsion,
                                       orbitals_copy)

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
            # print(orbital_name, 2.0 * n * (k1_int + k2_int))
            kinetic_energy += 2.0 * n * (k1_int + k2_int)
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
            # print(orbital_name, 2.0 * n * integral)
            potential_energy += 2.0 * n * integral
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
                repulsion_energy += (spin_mul ** 2
                                     * angular_mul_i * angular_mul_j
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
                exchange_energy += ((2.0
                                    * angular_mul_i)
                                    * simpson(self.DR_0 *
                                              np.exp(self.S_0 * self.DELTA) *
                                              orbital_from0_i * int1))
        return exchange_energy

    def get_total_energy(self):
        return (self.get_kinetic_energy()
                + self.get_potential_energy()
                + self.get_repulsion_energy() / 2.0
                - self.get_exchange_energy() / 2.0)


if __name__ == '__main__':
    # system = ClosedShellSystem(number_of_points=1400, extent=15.0,
    #                            nuclear_charge=20, number_of_electrons=20)
    nuclear_charge_ = 10
    number_of_electrons_ = 10
    system = ClosedShellSystem(number_of_points=1000, extent=5.0,
                               nuclear_charge=nuclear_charge_,
                               number_of_electrons=number_of_electrons_)
    system.solve(n_iterations=12)
    print(system.get_kinetic_energy())
    print(system.get_potential_energy())
    print(system.get_total_energy())

    import json
    import matplotlib.pyplot as plt
    orbitals_dict = {}
    if nuclear_charge_ == 10 and number_of_electrons_ == 10:
        for o in system.orbitals.keys():
            orbitals_dict[o] = {'r': list(system.R),
                                'values': list(system.orbitals[o]
                                               * np.exp(0.5*system.S
                                                        * system.DELTA
                                                        )
                                               )
                                }
            plt.plot(orbitals_dict[o]['r'],
                     orbitals_dict[o]['values'])
    plt.show()
    plt.close()
    with open(f'../data/{nuclear_charge_}p{number_of_electrons_}e.json',
              'w') as f:
        json.dump(orbitals_dict, f)

