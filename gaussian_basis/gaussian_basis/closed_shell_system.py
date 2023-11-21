from typing import List, Tuple
import numpy as np
from scipy.linalg import eigh
from . import extension
from .matrices import get_overlap_matrix, get_kinetic_matrix
from .matrices import get_nuclear_potential_matrix
from .matrices import get_two_electron_integrals_tensor


class ClosedShellSystem:

    orbitals: np.ndarray
    orbitals_count: np.ndarray
    overlap: np.ndarray
    kinetic: np.ndarray
    nuclear: np.ndarray
    h: np.ndarray
    two_electron_integrals: np.ndarray
    energies: np.ndarray
    nuclear_configuration: List[List[Tuple[np.ndarray, int]]]

    def __init__(self, **kw):
        primitives = kw['primitives']
        self.orbitals = kw['orbitals']
        nuclear_config = kw['nuclear_config']
        self.nuclear_configuration = nuclear_config
        self.orbitals_count = self.orbitals.shape[0]
        self.energies = np.zeros([self.orbitals_count])
        if 'use_ext' in kw and kw['use_ext']:
            self.init_using_ext(primitives)
        else:
            self.overlap = get_overlap_matrix(primitives)
            self.kinetic = get_kinetic_matrix(primitives)
            self.nuclear = get_nuclear_potential_matrix(primitives,
                                                        nuclear_config)
            self.h = self.kinetic + self.nuclear
            self.two_electron_integrals = \
                get_two_electron_integrals_tensor(primitives)

    def init_using_ext(self, primitives):
        size = 6
        gaussian_arr = np.zeros([size*len(primitives)], order='C',
                                dtype=np.double)
        # gaussian_arr_long = gaussian_arr.view(dtype=np.int_)
        gaussian_arr_short = gaussian_arr.view(dtype=np.short)
        for i, g in enumerate(primitives):
            # Write the orbital exponent 
            # and amplitude for the Gaussian3D object
            gaussian_arr[size*i] = g.orbital_exponent()
            gaussian_arr[size*i + 1] = g.amplitude()
            gaussian_arr_short[4*(size*i + 2)] = int(g.angular()[0])
            gaussian_arr_short[4*(size*i + 2) + 1] = int(g.angular()[1])
            gaussian_arr_short[4*(size*i + 2) + 2] = int(g.angular()[2])
            gaussian_arr_short[4*(size*i + 2) + 3] = 0
            gaussian_arr[size*i + 3: size*i + 6] = g.position()
            # Now write the members of each of the individual Gaussian1Ds
            # offset = size*i + 2
            # g1d_size = 3
            # # x
            # gaussian_arr[offset] = g.position()[0]
            # gaussian_arr_long[offset + 1] = int(g.angular()[0])
            # gaussian_arr[offset + 2] = g.orbital_exponent()
            # # y
            # gaussian_arr[offset + g1d_size] =g.position()[1]
            # gaussian_arr_long[offset 
            #                     + g1d_size + 1] = int(g.angular()[1])
            # gaussian_arr[offset + g1d_size + 2] = g.orbital_exponent()
            # # z
            # gaussian_arr[offset + 2*g1d_size] = g.position()[2]
            # gaussian_arr_long[offset
            #                     + 2*g1d_size + 1] = int(g.angular()[2])
            # gaussian_arr[offset + 2*g1d_size + 2] = g.orbital_exponent()
        # print(gaussian_arr)
        # print(gaussian_arr_long)
        nuclear_config = self.nuclear_configuration
        nuc_loc = np.zeros([3*len(nuclear_config)],
                           dtype=np.double, order='C')
        charges = np.zeros([len(nuclear_config)],
                           dtype=np.intc, order='C')
        for i, e in enumerate(self.nuclear_configuration):
            nuc_loc[3*i: 3*i + 3] = e[0]
            charges[i] = e[1]
        print(nuc_loc)
        print(charges)
        self.overlap = np.zeros(2*[len(primitives)], 
                                dtype=np.double, order='C')
        self.kinetic = np.zeros(2*[len(primitives)], 
                                dtype=np.double, order='C')
        self.nuclear = np.zeros(2*[len(primitives)],
                                dtype=np.double, order='C')
        self.two_electron_integrals = np.zeros(
            4*[len(primitives)], dtype=np.double, order='C')
        extension.compute_overlap(self.overlap, gaussian_arr)
        extension.compute_kinetic(self.kinetic, gaussian_arr)
        extension.compute_nuclear(self.nuclear, 
                                  nuc_loc, charges,
                                  gaussian_arr)
        extension.compute_two_electron_integrals(
            self.two_electron_integrals, gaussian_arr)
        self.h = self.kinetic + self.nuclear
        # import matplotlib.pyplot as plt
        # plt.imshow(self.overlap)
        # plt.show(); plt.close()
        # plt.imshow(self.kinetic)
        # plt.show(); plt.close()
        # plt.imshow(self.nuclear)
        # plt.show(); plt.close()

    def solve(self, iter_count):
        orbitals = self.orbitals
        energies = self.energies
        two_electron_integrals = self.two_electron_integrals
        h = self.h
        for i in range(iter_count):
            repulsion = 2.0*np.einsum('ijkl,mk,ml->ij',
                                      two_electron_integrals,
                                      orbitals, orbitals)
            exchange = np.einsum('ijkl,mj,ml->ik',
                                 two_electron_integrals,
                                 orbitals, orbitals)
            fock = h + repulsion - exchange
            energies, eigenvectors = eigh(
                fock, b=self.overlap,
                subset_by_index=[0, self.orbitals_count-1])
            orbitals = eigenvectors.T
        self.orbitals = orbitals
        self.energies = energies
        return energies, orbitals
    
    def get_kinetic_energy(self):
        return 2.0*np.einsum('ij,ni,nj->', self.kinetic, 
                             * 2*[self.orbitals])

    def get_nuclear_potential_energy(self):
        return 2.0*np.einsum('ij,ni,nj->', self.nuclear,
                             * 2*[self.orbitals])

    def get_repulsion_exchange_energy(self):
        repulsion_e = np.einsum('ijkl,ni,nj,mk,ml->',
                                self.two_electron_integrals,
                                * 4*[self.orbitals],
                                optimize='greedy')
        exchange_e = np.einsum('ijkl,ni,mj,nk,ml->',
                               self.two_electron_integrals,
                               * 4*[self.orbitals],
                               optimize='greedy')
        return 2.0*repulsion_e - exchange_e

    def get_nuclear_configuration_energy(self):
        val = 0.0
        if len(self.nuclear_configuration) > 1:
            for i, c_i in enumerate(self.nuclear_configuration):
                for j in range(i+1, len(self.nuclear_configuration)):
                    c_j = self.nuclear_configuration[j]
                    r_i, q_i = c_i
                    r_j, q_j = c_j
                    r = np.linalg.norm(r_i - r_j)
                    val += q_i*q_j/r
            return val
        else:
            return 0.0

    def get_total_energy(self):
        return (self.get_kinetic_energy()
                + self.get_nuclear_potential_energy()
                + self.get_repulsion_exchange_energy()
                + self.get_nuclear_configuration_energy())
