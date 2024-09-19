from typing import Union, List, Dict
import numpy as np
import json
from .gaussian3d import Gaussian3D
from .integrals3d import overlap
from .matrices import get_overlap_matrix
from .molecular_geometry import MolecularGeometry
from .closed_shell_system import ClosedShellSystemFromPrimitives


def get_multiplicity_from_orbital_name(orbital_name: str) -> int:
    letters_numbers = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
    for letter in letters_numbers.keys():
        if letter in orbital_name:
            return letters_numbers[letter]
    raise NotImplementedError


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


def get_orbital_norm_factor(pos: np.ndarray, ang: np.ndarray,
                            coefficients: np.ndarray,
                            exponents: np.ndarray) -> float:
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

def get_basis_function_norm_factor(pos: np.ndarray, ang: np.ndarray,
                                   coefficients: np.ndarray,
                                   exponents: np.ndarray) -> float:
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


def get_orbitals_dict_from_file(filename: str) -> dict:
    string = ''
    with open(filename, 'r') as f:
        for line in f:
            string += line
        d = json.loads(string)
        return d

class OrbitalPrimitivesBuilder:

    orbital_coefficients: np.ndarray
    primitives: Union[List[Gaussian3D], List[List[Gaussian3D]]]

    def __init__(self, **kw):
        if 'position' in kw and 'orbitals_dict' in kw:
            position = kw['position']
            orbitals_dict = kw['orbitals_dict']
            self.constructor_position_orbitals_dict(
                position, orbitals_dict
            )
        elif 'orbitals' in kw and 'basis_functions' in kw:
            self.primitives = []
            for basis_functions in kw['basis_functions']:
                for g in basis_functions:
                    self.primitives.append(g)
            total_basis_func_count = len(self.primitives)
            total_orbital_count = sum([o.shape[0] for o in kw['orbitals']])
            self.orbital_coefficients = np.zeros([
                total_orbital_count, total_basis_func_count])
            orbital_count = 0
            basis_func_count = 0
            for o in kw['orbitals']:
                self.orbital_coefficients[
                    orbital_count: (orbital_count + o.shape[0]),
                    basis_func_count: (basis_func_count + o.shape[1])] = o
                orbital_count += o.shape[0]
                basis_func_count += o.shape[1]
        else:
            self.orbital_coefficients = np.array([])
            self.primitives = []

    def constructor_position_orbitals_dict(self, position, orbitals_dict):
        self.constructor_position_primitive_orbitals_dict(
            position, orbitals_dict
        )

    def constructor_position_basis_func_orbitals_dict(
            self, position, orbitals_dict):
        total_basis_func_count = 0
        total_orbitals_count = 0
        for o_name in orbitals_dict.keys():
            for k in range(min(3,
                               get_multiplicity_from_orbital_name(o_name))):
                total_basis_func_count += len(orbitals_dict[o_name])
                total_orbitals_count += 1
        self.orbital_coefficients = np.zeros([total_orbitals_count,
                                              total_basis_func_count])
        for o_name in orbitals_dict.keys():
            orbital_data = orbitals_dict[o_name]
            for k in range(
                        min(3, get_multiplicity_from_orbital_name(o_name))):
                ang = np.zeros([3])
                ang[k] = get_angular_number(o_name)
                for basis_func in orbital_data:
                    coefficients = basis_func['coefficients']
                    exponents = basis_func['exponents']
                    norm_factor = get_basis_function_norm_factor(
                        position, ang, coefficients, exponents
                    )
                    basis_func_primitives = []
                    for c, e in zip(coefficients, exponents):
                        basis_func_primitives.append(
                            Gaussian3D(c, e, position, ang))
                    self.primitives.append()
                    self.orbital_coefficients[
                        orbital_count, basis_func_count
                    ] = 1.0/norm_factor
                    basis_func_count += 1
            orbital_count += 1

    def constructor_position_primitive_orbitals_dict(
            self, position, orbitals_dict):
        total_basis_func_count = 0
        total_orbitals_count = 0
        for o_name in orbitals_dict.keys():
            for k in range(min(3,
                               get_multiplicity_from_orbital_name(o_name))):
                total_basis_func_count += len(
                    orbitals_dict[o_name]['coefficients'])
                total_orbitals_count += 1
        self.orbital_coefficients = np.zeros([total_orbitals_count,
                                              total_basis_func_count])

        orbital_count = 0
        basis_func_count = 0
        self.primitives = []
        for o_name in orbitals_dict.keys():
            orb_data = orbitals_dict[o_name]
            for k in range(min(3,
                               get_multiplicity_from_orbital_name(o_name))):
                ang = np.zeros([3])
                ang[k] = get_angular_number(o_name)
                coefficients = orb_data['coefficients']
                exponents = orb_data['exponents']
                norm_factor = get_orbital_norm_factor(
                    position, ang, coefficients, exponents
                )
                for c, e in zip(coefficients, exponents):
                    self.primitives.append(
                        Gaussian3D(1.0, e, position, ang))
                    self.orbital_coefficients[
                        orbital_count, basis_func_count
                    ] = c/norm_factor
                    basis_func_count += 1
                orbital_count += 1

    def get_primitives(self) -> List[Gaussian3D]:
        return self.primitives

    def orbitals(self) -> np.ndarray:
        return self.orbital_coefficients

    def __add__(self,
                other: Union['OrbitalPrimitivesBuilder', int]
                ) -> 'OrbitalPrimitivesBuilder':
        if isinstance(other, int):
            return OrbitalPrimitivesBuilder(
                orbitals=[self.orbitals()],
                basis_functions=[self.get_primitives()])
        orbitals = [self.orbitals(), other.orbitals()]
        basis_functions = [self.get_primitives(), other.get_primitives()]
        return OrbitalPrimitivesBuilder(orbitals=orbitals,
                                        basis_functions=basis_functions)

    def __radd__(self,
                 other: Union['OrbitalPrimitivesBuilder', int]
                 ) -> 'OrbitalPrimitivesBuilder':
        if isinstance(other, int):
            return OrbitalPrimitivesBuilder(
                orbitals=[self.orbitals()],
                basis_functions=[self.get_primitives()])
        orbitals = [self.orbitals(), other.orbitals()]
        basis_functions = [self.get_primitives(), other.get_primitives()]
        return OrbitalPrimitivesBuilder(orbitals=orbitals,
                                        basis_functions=basis_functions)

    def get_orbital_exponents(self) -> np.ndarray:
        exponents = [g.orbital_exponent() for g in self.primitives]
        exponents.sort()
        return np.array(exponents)

    def set_number_of_orbitals(self, n: int):
        orbitals = self.orbitals()
        if n < orbitals.shape[0]:
            # overlap_mat = get_overlap_matrix(self.get_primitives())
            # orbital = np.zeros(orbitals.shape[1])
            # for i in range(n, orbitals.shape[0]):
            #     orbitals[n-i] += orbitals[i]
            # orbital /= np.sqrt(orbital @ overlap_mat @ orbital)
            # for i in range(n):
            #     orbitals[i] += orbital
            #     orbitals[i] /= np.sqrt(orbitals[i] 
            #                            @ overlap_mat @ orbitals[i])
            # for i in range(n, orbitals.shape[0]):
            #     for j in range(n):
            #         orbitals[j] += orbitals[i]
                    # orbitals[j] /= np.sqrt(orbitals[j]
                    #                        @ overlap_mat
                    #                        @ orbitals[j])
            self.orbital_coefficients = orbitals[:n]

            # arr_1d0 = np.arange(1, n + 1)
            # arr_1d1 = np.arange(1, self.orbital_coefficients.shape[1] + 1)
            # dst = np.vectorize(lambda k, n:
            #                    np.sin(np.pi*k*n/(n + 1)))(
            #     *np.meshgrid(arr_1d0, arr_1d1, indexing='ij'))
            # print(dst.shape)
            # self.orbital_coefficients = dst
        else:
            new_orbitals = np.zeros([n, orbitals.shape[1]])
            new_orbitals[:orbitals.shape[1]] = orbitals
            self.orbital_coefficients = new_orbitals


def get_build_from_primitives(geom: MolecularGeometry,
                              orbitals_dicts:
                              Dict[str, Dict[str, List[float]]],
                              n_orbitals: int
                              ) -> OrbitalPrimitivesBuilder:
    builds = []
    for k in geom.config:
        print(k)
        orbitals_dict = orbitals_dicts[k]
        builds.append([OrbitalPrimitivesBuilder(position=geom[k][i],
                                                orbitals_dict=orbitals_dict
                                                )
                       for i in range(len(geom[k]))])
    data = sum([sum(e) for e in builds])
    data.set_number_of_orbitals(n_orbitals)
    return data


def make_system_from_geometry_and_build(
        geom: MolecularGeometry,
        build: OrbitalPrimitivesBuilder) -> ClosedShellSystemFromPrimitives:
    return ClosedShellSystemFromPrimitives(primitives=build.get_primitives(),
                             orbitals=build.orbitals(),
                             nuclear_config=geom.get_nuclear_configuration(),
                             use_ext=True)


def make_system_from_primitives(geom: MolecularGeometry,
                                orbitals_dicts:
                                Dict[str, Dict[str, List[float]]],
                                n_orbitals: int
                                ) -> ClosedShellSystemFromPrimitives:
    data = get_build_from_primitives(geom, orbitals_dicts, n_orbitals)
    return ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                             orbitals=data.orbitals(),
                             nuclear_config=geom.get_nuclear_configuration(),
                             use_ext=True)
