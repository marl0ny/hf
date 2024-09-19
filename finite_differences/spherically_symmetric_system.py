import numpy as np
import scipy.sparse as sparse
from scipy.integrate import simpson
from typing import Union, Callable, Dict


def angular_number_from_orbital_name(orbital_name: str) -> int:
    """Get the orbital angular number given the name of the orbital.
    """
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


def get_principle_from_orbital_name(orbital_name: str) -> int:
    """Extract the principal quantum number from the name of the orbital.
    """
    return int(orbital_name[0])


def letter_from_orbital_name(orbital_name: str) -> int:
    """Extract the orbital angular letter from the name of the orbital.
    """
    return orbital_name[1]


class SphericallySymmetricSystemBase:
    """Base class for managing the Hartree-Fock energies and orbitals
    of spherically symmetric systems through finite differences. This base 
    class's responsibilities include storing the initial trial and final
    orbital solutions and energies, the mesh discretization,
    which in general is non-uniform, those matrix operators such as the 
    kinetic or overlap which are dependant on the underlying mesh
    discretization used, and the potential, which is assumed to be Coulombic.
    The actual Hartree-Fock computations are handled in derived classes.

    To help improve accuracy, especially at the vicinity of the origin where
    there is a singularity due to the Coulomb potential, use a non-uniform 
    finite difference discretization in which a higher density of points are
    placed closer to the origin. This is introduced in [1], where if R denotes
    the array of grid points used in the discretization, then
        R[i] = RP*(exp(i*DELTA) - 1), i = 0..N-1,
    where N is the total number of points used, and RP and DELTA are constants
    which determine the points distribution. From here the Schrodinger equation 
    and other relations can be recast in terms of i by applying
    the chain rule.

        1. Jos Thijssen, Computational Physics, Second Edition
           Exercise 5.1, pg 116-117. 

    @param N: number of points used in the discretization of the orbitals
    @param L: physical extent of the simulation, starting from the origin
    @param S_0: Array of evenly spaced points, which is used to construct
    the non-uniform, non-equally spaced array of spatial positions R. At its
    zeroth index is the origin at 0.
    @param DS: Distance between each point in S and S0.
    @param S: S = S_0[1::]
    @param DELTA: Factor that controls grid non-uniformity
    @param RP: Another factor for controlling grid non-uniformity
    @param R_0: Array of spatial positions, including the origin at 0
    @param R: Array of spatial positions that skips the origin at 0. Used in
    computations where there is a singularity at the origin.
    @param DR_0: Array of distances between each point in R, including the origin
    @param DR: Array of distances between each point in R,
    where the origin at 0 is skipped.
    @param Z: Nuclear charge
    @param V: Numpy array for the potential
    @param DIAG_M: Diagonal of the overlap matrix.
    @param M_SPARSE: Sparse overlap matrix.
    @param INV_M_SPARSE: Inverse of the sparse overlap matrix.
    @param M: Numpy array of the overlap matrix.
    @param INV_M: Inverse of the numpy array of the sparse overlap matrix.
    @param R_GREATER_THAN: 2D Numpy array, where for an index i, j and
    spatial positions R, R_GREATER_THAN[i, j] = R[i] if R[i] > R[j] else R[j].
    @param R_LESS_THAN: 2D Numpy array, where for an index i, j and spatial
    positions R, R_GREATER_THAN[i, j] = R[i] if R[i] < R[j] else R[j].
    @param init_orbitals: dictionary containing the initial trial orbitals,
    where the keys are the orbital names.
    @param orbitals: dictionary of the computed orbitals,
    where the keys are the orbital names.
    @param orbital_energies: dictionary of the orbital energies, where the
    keys are the orbital names.
    @param GLOBAL_SHIFT: a global shift applied to the potential so that its
    minimum value is zero.
    @param T1: discretized Laplacian part of the kinetic energy operator
    @param T2: Angular dependance part of the kinetic energy operator
    """
    N: int
    L: float
    S_0: np.ndarray
    DS: float
    S: np.ndarray
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
                 potential_f: Union[Callable[[np.ndarray, int], np.ndarray],
                                    np.ndarray]):
        """Constructor.

        @param number_of_points: number of points to use.
        @param extent: extent of the simulation, in Hartree atomic units.
        Note that the simulation always starts from the origin.
        @param nuclear_charge: the charge at the origin.
        @param delta: controls the spacing between each point.
        @param potential_f: The potential. If this is a function, this must
        take as argument the spatial positions array and nuclear charge.
        """

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

    def normalize(self, psi: np.ndarray) -> np.ndarray:
        """Normalize the input wave function.

        @param psi: input wave function.
        """
        psi_ = np.zeros([1+self.N])
        psi_[1::] = psi
        return psi/np.sqrt(simpson(self.DR_0*np.exp(self.S_0 * self.DELTA)
                                   * psi_*np.conj(psi_)))

    def construct_hydrogen_like_orbitals(self):
        """Hydrogen-like orbitals. These come from here:

        http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/hydwf.html
        """
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

    def get_spatial_derivative(self, number_of: int, 
                               discretization_type: str = 'sin'
                               ) -> Union[np.ndarray, sparse.dia_matrix]:
        """Get the matrix that performs the spatial derivative. Currently
        only the first and second derivative are supported.

        @param number_of: order of differentiation.

        @param discretization_type: Either 'sin' or '2nd_order'. 
        
        If discretization_type is 'sin', the derivative matrix is composed of
        first that linear transform that takes an input position vector to
        the sin function basis, the diagonal derivative matrix in this 
        basis, and finally the linear transform that goes back to the original
        basis. Note that this corresponds to boundary conditions where things
        vanish at the endpoints. The matrix returned is a Numpy array.

        If on the other hand discretization_type is '2nd_order', return a 
        sparse matrix of the finite difference 2nd order approximation
        for the given spatial derivative.
        """
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
        """Get the names of the orbitals.
        """
        return self.orbitals.keys()

    def get_outermost_letter_name(self, letter: str) -> Union[None, str]:
        """Given an orbital angular momentum letter, get the highest principal 
        quantum number which contains an orbital that corresponds to that 
        letter. 

        @param letter: orbital angular momentum letter
        """
        if letter not in ['s', 'p', 'd', 'f', 'g']:
            raise NotImplementedError
        orbital_names = self.orbitals.keys()
        orbital_values = [int(e[0]) for e in orbital_names if letter in e]
        if orbital_values is []:
            return
        return f'{max(orbital_values)}{letter}'

    def get_innermost_letter_name(self, letter):
        """Given an orbital angular momentum letter, get the lowest principal 
        quantum number which contains an orbital that corresponds to that 
        letter. 

        @param letter: orbital angular momentum letter
        """
        if letter not in ['s', 'p', 'd', 'f', 'g']:
            raise NotImplementedError
        orbital_names = self.orbitals.keys()
        orbital_values = [int(e[0]) for e in orbital_names if letter in e]
        if orbital_values is []:
            return
        return f'{min(orbital_values)}{letter}'

    def get_s_orbitals(self) -> Dict[str, np.ndarray]:
        """Get a dictionary of all s orbitals. The keys are the name of
        the orbitals, and the values are the arrays.
        """
        return {k: self.orbitals[k]
                for k in self.orbitals.keys() if 's' in k}

    def get_p_orbitals(self) -> Dict[str, np.ndarray]:
        """Get a dictionary of all p orbitals. The keys are the name of
        the orbitals, and the values are the arrays.
        """
        return {k: self.orbitals[k]
                for k in self.orbitals.keys() if 'p' in k}

    def get_d_orbitals(self) -> Dict[str, np.ndarray]:
        """Get a dictionary of all d orbitals. The keys are the name of
        the orbitals, and the values are the arrays.
        """
        return {k: self.orbitals[k]
                for k in self.orbitals.keys() if 'd' in k}

    def get_initial_orbital(self, orbital_name: str) -> np.ndarray:
        """Get the initial trial orbitals. The keys are the name of
        the orbitals, and the values are the arrays.
        """
        orbital = self.init_orbitals[orbital_name]
        return orbital * np.exp(0.5 * self.S * self.DELTA)

    def get_orbital(self, orbital_name: str) -> np.ndarray:
        """Get the computed orbital, given its name.
        """
        orbital = self.orbitals[orbital_name]
        return orbital * np.exp(0.5 * self.S * self.DELTA)
