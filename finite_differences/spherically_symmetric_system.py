import numpy as np
import scipy.sparse as sparse
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from typing import Union, Callable, Dict


def angular_number_from_orbital_name(orbital_name: str) -> int:
    letters_numbers = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    for letter in letters_numbers.keys():
        if letter in orbital_name:
            return letters_numbers[letter]
    raise NotImplementedError


def multiplicity_from_orbital_name(orbital_name):
    letters_numbers = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
    for letter in letters_numbers.keys():
        if letter in orbital_name:
            return letters_numbers[letter]
    raise NotImplementedError


def get_principle_from_orbital_name(orbital_name):
    return int(orbital_name[0])


def letter_from_orbital_name(orbital_name):
    return orbital_name[1]


class SphericallySymmetricSystemBase:
    """
    Used for numerically computing the Hartree-Fock energies and orbitals
    of spherically symmetric systems through finite differences.

    N: number of points used in the discretization of the orbitals
    L: physical extent of the simulation, starting from the origin
    S_0:
    DS:
    DELTA:
    RP:
    R_0: Array of spatial positions, including the origin at 0
    R: Array of spatial positions that skips the origin at 0
    DR_0: Array of distances between each point in R, including the origin
    DR: Array of distances between each point in R,
    where the origin at 0 is skipped.
    Z: Nuclear charge
    V: Numpy array for the potential
    DIAG_M:
    M_SPARSE:
    INV_M_SPARSE:
    M:
    INV_M:
    R_GREATER_THAN:
    R_LESS_THAN:
    init_orbitals: dictionary containing the initial trial orbitals,
    where the keys are the orbital names.
    orbitals: dictionary of the computed orbitals,
    where the keys are the orbital names.
    orbital_energies: dictionary of the orbital energies, where the
    keys are the orbital names.
    GLOBAL_SHIFT: a global shift applied to the potential so that its
    minimum value is zero.
    T1: discretized Laplacian part of the kinetic energy operator
    T2: Angular dependance part of the kinetic energy operator
    """
    N: int
    L: float
    S_0: np.ndarray
    DS: float
    DELTA: float
    RP: float
    R_0: np.ndarray
    R: np.ndarray
    DR: np.ndarray
    DR_0: np.ndarray
    Z: float
    V: np.ndarray
    DIAG_M: np.ndarray
    M_SPARSE: sparse.dia_matrix
    INV_M_SPARSE: sparse.dia_matrix
    M: np.ndarray
    INV_M: np.ndarray
    R_GREATER_THAN: np.ndarray
    R_LESS_THAN: np.ndarray
    init_orbitals: Dict[str, np.ndarray]
    orbitals: Dict[str, np.ndarray]
    orbital_energies: Dict[str, list]
    GLOBAL_SHIFT: float
    T1: np.ndarray
    T2: np.ndarray
    verbose: bool

    def __init__(self, number_of_points: int, extent: float,
                 nuclear_charge: float, delta: float,
                 potential_f: Union[Callable, np.ndarray]):

        self.N = number_of_points
        self.L = extent

        self.S_0 = np.linspace(0.0, self.L, self.N + 1, dtype=float)
        self.DS = self.S_0[1] - self.S_0[0]
        self.S = self.S_0[1::]
        self.DELTA = delta
        self.RP = self.L / (np.exp(self.DELTA * self.S_0[-1]) - 1.0)
        self.R_0 = self.RP * (np.exp(self.DELTA * self.S_0) - 1.0)
        self.R = self.RP * (np.exp(self.DELTA * self.S) - 1.0)
        self.DR = (self.DELTA * self.RP *
                   np.exp(self.S * self.DELTA) * self.DS)
        self.DR_0 = (self.DELTA * self.RP *
                     np.exp(self.S_0 * self.DELTA) * self.DS)

        self.R_GREATER_THAN = np.vectorize(lambda r1, r2: r1
                                           if r1 > r2 else r2
                                           )(*np.meshgrid(self.R, self.R))
        self.R_LESS_THAN = np.vectorize(lambda r1, r2: r1
                                        if r1 < r2 else r2
                                        )(*np.meshgrid(self.R, self.R))

        self.DIAG_M = np.exp(2.0 * self.S * self.DELTA
                             ) * self.RP ** 2 * self.DELTA ** 2
        self.M_SPARSE = sparse.spdiags(self.DIAG_M, np.array([0]),
                                       self.N, self.N)
        self.INV_M_SPARSE = sparse.spdiags(1.0 / self.DIAG_M,
                                           np.array([0]), self.N, self.N)
        self.M, self.INV_M = \
            self.M_SPARSE.toarray(), self.INV_M_SPARSE.toarray()
        div2 = self.get_spatial_derivative(2)

        self.init_orbitals = {}
        self.orbitals = {}
        self.orbital_energies = {}

        self.T1 = -0.5 * (div2
                          - np.diagflat(self.DELTA ** 2 / 4.0))  # Kinetic
        self.T2 = np.diagflat(1 / self.R ** 2)

        self.Z = nuclear_charge
        potential = np.ndarray([self.N])
        if type(potential_f) == np.ndarray:
            potential = potential_f
        elif 'function' in str(type(potential_f)):
            potential = potential_f(self.R, self.Z)
        self.GLOBAL_SHIFT = np.amin(potential)
        self.V = np.diagflat(potential - self.GLOBAL_SHIFT)

        self.verbose = False

    def normalize(self, psi: np.ndarray):
        psi_ = np.zeros([1+self.N])
        psi_[1::] = psi
        return psi/np.sqrt(simpson(self.DR_0*np.exp(self.S_0 * self.DELTA)
                                   * psi_*np.conj(psi_)))

    def construct_hydrogen_like_orbitals(self):
        orbitals = dict()
        orbitals['1s'] = self.normalize(self.R * np.exp(-self.Z * self.R)
                                        / np.exp(0.5 * self.S
                                                 * self.DELTA))
        orbitals['2s'] = self.normalize(self.R * (2 - self.Z * self.R)
                                        * np.exp(-self.Z * self.R / 2.0)
                                        / np.exp(0.5 * self.S
                                                 * self.DELTA))
        orbitals['2p'] = self.normalize(self.R ** 2
                                        * np.exp(-self.Z * self.R / 2.0)
                                        / np.exp(0.5 * self.S * self.DELTA))
        orbitals['3s'] = self.normalize(self.R * np.exp(-self.Z *
                                                        self.R / 3.0)
                                        * (27.0 - 18.0 * self.Z * self.R
                                        + 2.0 * self.Z ** 2 * self.R ** 2)
                                        / np.exp(0.5 * self.S * self.DELTA))
        orbitals['3p'] = self.normalize(self.R ** 2 *
                                        np.exp(-self.Z * self.R / 3.0)
                                        * (6.0 - self.Z * self.R)
                                        / np.exp(0.5 * self.S * self.DELTA))
        orbitals['4s'] = self.normalize(self.R * np.exp(-self.Z
                                                        * self.R / 3.0)
                                        * (27.0 - 18.0 * self.Z * self.R
                                            + 2.0 * self.Z ** 2 * self.R ** 2)
                                        / np.exp(0.5 * self.S * self.DELTA))
        return orbitals

    def get_spatial_derivative(self, number_of, discretization_type='sin'):
        if discretization_type == 'sin':
            arr_1d = np.arange(1, self.N + 1)
            dst = np.vectorize(lambda k, n:
                               np.sin(np.pi * k * n / (self.N + 1)))(
                *np.meshgrid(arr_1d, arr_1d))
            e2 = -np.diagflat(np.pi ** 2 * arr_1d ** 2 /
                              (self.L + 2 * self.DR) ** 2)
            if number_of == 2:
                return 2.0 * (dst.T @ e2 @ dst) / self.N
            else:
                raise NotImplementedError
        elif discretization_type == '2nd order':
            zeros, ones = np.zeros([self.N], dtype=float), \
                          np.ones([self.N], dtype=float)
            derivative_diags = np.array([-ones, zeros, ones]) / (2 * self.DS)
            laplacian_diags \
                = np.array([ones, -2.0 * ones, ones]) / self.DS ** 2
            if number_of == 1:
                return sparse.spdiags(derivative_diags,
                                      np.array([-1, 0, 1]),
                                      self.N, self.N).toarray()
            elif number_of == 2:
                return sparse.spdiags(laplacian_diags,
                                      np.array([-1, 0, 1]),
                                      self.N, self.N).toarray()
            else:
                raise NotImplementedError

    def orbital_names(self):
        return self.orbitals.keys()

    def get_outermost_letter_name(self, letter):
        if letter not in ['s', 'p', 'd', 'f', 'g']:
            raise NotImplementedError
        orbital_names = self.orbitals.keys()
        orbital_values = [int(e[0]) for e in orbital_names if letter in e]
        if orbital_values is []:
            return
        return f'{max(orbital_values)}{letter}'

    def get_innermost_letter_name(self, letter):
        if letter not in ['s', 'p', 'd', 'f', 'g']:
            raise NotImplementedError
        orbital_names = self.orbitals.keys()
        orbital_values = [int(e[0]) for e in orbital_names if letter in e]
        if orbital_values is []:
            return
        return f'{min(orbital_values)}{letter}'

    def get_s_orbitals(self):
        return {k: self.orbitals[k]
                for k in self.orbitals.keys() if 's' in k}

    def get_p_orbitals(self):
        return {k: self.orbitals[k]
                for k in self.orbitals.keys() if 'p' in k}

    def get_d_orbitals(self):
        return {k: self.orbitals[k]
                for k in self.orbitals.keys() if 'd' in k}

    def get_initial_orbital(self, orbital_name) -> np.ndarray:
        orbital = self.init_orbitals[orbital_name]
        return orbital * np.exp(0.5 * self.S * self.DELTA)

    def get_orbital(self, orbital_name) -> np.ndarray:
        orbital = self.orbitals[orbital_name]
        return orbital * np.exp(0.5 * self.S * self.DELTA)
