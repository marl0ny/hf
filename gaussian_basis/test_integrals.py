from gaussian_basis import *
from gaussian_basis import ClosedShellSystem
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.integrals3d import *
import numpy as np
import unittest
import json


def get_angular_number(orbital_name: str) -> int:
    letter = orbital_name[1]
    if letter == 's':
        return 0
    if letter == 'p':
        return 1
    if letter == 'd':
        return 2
    if letter == 'f':
        return 3


def get_multiplicity_from_orbital_name(orbital_name):
    letters_numbers = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
    for letter in letters_numbers.keys():
        if letter in orbital_name:
            return letters_numbers[letter]
    raise NotImplementedError


def get_orbital_norm_factor(pos: np.ndarray, ang: np.ndarray,
                            coefficients: np.ndarray,
                            exponents: np.ndarray):
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


def get_h_like_system_energy_from_gaussians(nuc_charge: int,
                                            letter: str,
                                            orbitals_dict_all: dict) -> float:
    orbitals_dict = orbitals_dict_all[letter]
    nuc_position = [[np.array([0.2, -1.0, 1.0]), nuc_charge]]
    basis_funcs = []
    wf_list = []
    pos = np.array([0.2, -1.0, 1.0])
    ang = np.array([0, 0, 0], dtype=int)
    if letter == '2s':
        ang = np.array([0, 0, 0], dtype=int)
    if letter == '2p' or letter == '3p':
        ang = np.array([0, 0, 1], dtype=int)
    if letter == '3d':
        ang = np.array([0, 1, 1], dtype=int)
    size = 0
    for c, e in zip(orbitals_dict['coefficients'],
                    orbitals_dict['exponents']):
        basis_funcs.append(Gaussian3D(1.0, e, pos=pos, ang=ang))
        wf_list.append(c)
        size += 1
    wavefunc = np.array(wf_list)
    overlap_m = np.zeros([size, size])
    kinetic_m = np.zeros([size, size])
    potential_m = np.zeros([size, size])
    for i in range(kinetic_m.shape[0]):
        for j in range(i, kinetic_m.shape[1]):
            gi, gj = basis_funcs[i], basis_funcs[j]
            overlap_m[i, j] = overlap(gi, gj)
            kinetic_m[i, j] = kinetic(gi, gj)
            potential_m[i, j] = nuclear(gi, gj, nuc_position)
            if j > i:
                overlap_m[j, i] = overlap_m[i, j]
                kinetic_m[j, i] = kinetic_m[i, j]
                potential_m[j, i] = potential_m[i, j]
    hamiltonian = kinetic_m + potential_m
    return (wavefunc @ hamiltonian @ wavefunc) / \
            (wavefunc @ overlap_m @ wavefunc)


def neon_hf_energies(ne_orbitals_dict):
    nuc_pos = np.array([0.0, 0.0, 0.0])
    data = OrbitalPrimitivesBuilder(position=nuc_pos,
                                    orbitals_dict=ne_orbitals_dict)
    # number_of_basis_funcs = 0
    # number_of_orbitals = 0
    # for o in data.keys():
    #     for k in range(get_multiplicity_from_orbital_name(o)):
    #         number_of_basis_funcs += len(data[o]['coefficients'])
    #         number_of_orbitals += 1
    # orbitals = np.zeros([number_of_orbitals, number_of_basis_funcs])
    #
    # basis_funcs = []
    # orbital_count = 0
    # basis_func_count = 0
    # for o in data.keys():
    #     for k in range(get_multiplicity_from_orbital_name(o)):
    #         ang = np.zeros([3])
    #         ang[k] = get_angular_number(o)
    #         coefficients = data[o]['coefficients']
    #         exponents = data[o]['exponents']
    #         norm_factor = get_orbital_norm_factor(
    #             orbitals_pos, ang, coefficients, exponents)
    #         for c, e in zip(coefficients, exponents):
    #             basis_funcs.append(Gaussian3D(1.0, e,
    #                                           orbitals_pos, ang))
    #             orbitals[orbital_count,
    #                      basis_func_count] = c/norm_factor
    #             basis_func_count += 1
    #         orbital_count += 1

    system = ClosedShellSystem(primitives=data.primitives(),
                               orbitals=data.orbitals(),
                               nuclear_config=[[nuc_pos, 10.0]],
                               use_ext=True)
    system.solve(5)
    kinetic_e = system.get_kinetic_energy()
    potential_e = system.get_nuclear_potential_energy()
    rep_ex_e = system.get_repulsion_exchange_energy()
    return {'e': kinetic_e + potential_e + rep_ex_e,
            'ke': kinetic_e, 'pe': potential_e}


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        string = ''
        with open('../data/1p1e-8gaussians.json', 'r') as f:
            for line in f:
                string += line
            self.h_orbitals_dict = json.loads(string)
        string = ''
        with open('../data/2p1e-8gaussians.json', 'r') as f:
            for line in f:
                string += line
            self.he_cation_orbitals_dict = json.loads(string)
        string = ''

        # 377s for 8 gaussians
        # got -3478.151599441746 for hf energy
        # repulsion exchange eV
        # 1800.1917704764494 -328.5441270092453

        with open('../data/10p10e-4gaussians.json', 'r') as f:
            for line in f:
                string += line
            ne_orbitals_dict = json.loads(string)
        self.ne_energies_dict = neon_hf_energies(ne_orbitals_dict)

    def test_hydrogen_1s_energy(self):
        energy = -0.5
        letter = '1s'
        energy_gb = get_h_like_system_energy_from_gaussians(
            1, letter, self.h_orbitals_dict)
        self.assertAlmostEqual(energy_gb, energy, delta=1e-4)

    def test_hydrogen_2s_energy(self):
        energy = -0.125
        letter = '2s'
        energy_gb = get_h_like_system_energy_from_gaussians(
            1, letter, self.h_orbitals_dict)
        self.assertAlmostEqual(energy_gb, energy, delta=1e-4)

    def test_hydrogen_2p_energy(self):
        energy = -0.125
        letter = '2p'
        energy_gb = get_h_like_system_energy_from_gaussians(
            1, letter, self.h_orbitals_dict)
        self.assertAlmostEqual(energy_gb, energy, delta=1e-4)

    # def test_hydrogen_3s_energy(self):
    #     energy = -0.5/3.0**2
    #     letter = '3s'
    #     energy_gb = get_h_like_system_energy_from_gaussians(
    #         1, letter, self.h_orbitals_dict)
    #     self.assertAlmostEqual(energy_gb, energy, delta=1e-4)
    #
    # def test_hydrogen_3p_energy(self):
    #     energy = -0.5/3.0**2
    #     letter = '3p'
    #     energy_gb = get_h_like_system_energy_from_gaussians(
    #         1, letter, self.h_orbitals_dict)
    #     self.assertAlmostEqual(energy_gb, energy, delta=1e-4)
    #
    # def test_hydrogen_3d_energy(self):
    #     energy = -0.5/3.0**2
    #     letter = '3d'
    #     energy_gb = get_h_like_system_energy_from_gaussians(
    #         1, letter, self.h_orbitals_dict)
    #     self.assertAlmostEqual(energy_gb, energy, delta=1e-4)

    def test_he_cation_1s_energy(self):
        energy = -4.0*0.5
        letter = '1s'
        energy_gb = get_h_like_system_energy_from_gaussians(
            2, letter, self.he_cation_orbitals_dict)
        self.assertAlmostEqual(energy_gb, energy, delta=1e-3)

    def test_he_cation_2s_energy(self):
        energy = -0.5*4.0/2.0**2
        letter = '2s'
        energy_gb = get_h_like_system_energy_from_gaussians(
            2, letter, self.he_cation_orbitals_dict)
        self.assertAlmostEqual(energy_gb, energy, delta=1e-4)

    def test_he_cation_2p_energy(self):
        energy = -0.5*4.0/2.0**2
        letter = '2p'
        energy_gb = get_h_like_system_energy_from_gaussians(
            2, letter, self.he_cation_orbitals_dict)
        self.assertAlmostEqual(energy_gb, energy, delta=1e-4)

    def test_ne_hf_energy(self):
        e_dict = self.ne_energies_dict
        self.assertAlmostEqual(27.211386245 * e_dict['e'], -3497,
                               delta=3497*0.001,
                               # msg=f"{27.211386245*e_dict['e']} eV"
                               )

    def test_ne_hf_kinetic_energy(self):
        e_dict = self.ne_energies_dict
        self.assertAlmostEqual(27.211386245*e_dict['ke'], 3496,
                               delta=3496*0.001,
                               # msg=f"{27.211386245*e_dict['ke']} eV"
                               )

    def test_ne_hf_potential_energy(self):
        e_dict = self.ne_energies_dict
        self.assertAlmostEqual(27.211386245*e_dict['pe'], -8466,
                               delta=8466*0.001,
                               # msg=f"{27.211386245*e_dict['pe']} eV"
                               )



unittest.main()

