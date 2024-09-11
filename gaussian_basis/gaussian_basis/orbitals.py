from typing import List, Dict, Tuple, Union
from copy import deepcopy
# from .orbital import Orbital
import numpy as np
from . import extension
from .molecular_geometry import MolecularGeometry
from .gaussian3d import Gaussian3D


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


class AtomicOrbitals:
    """Spherically symmetric orbitals centred around a specified point.
    To be clear this class does not represent a single orbital, but encapsulates a 
    composition of multiple spherically symmetric orbitals such as those found
    for an atom.
    """
    position: np.ndarray
    _dict_repr: Dict[str, List[Dict[str, Union[float, Dict[str, List[float]]]]]]

    def __init__(
            self, 
            position: np.ndarray,
            orbitals_dict: 
            Dict[str, List[Dict[str, Union[float, Dict[str, List[float]]]]]]
            ):
        """
        Constructor for the AtomicOrbitals class.
        
        position: position of which the spherically symmetric orbitals are
        centered.
        orbitals_dict: data that describes the orbitals. 
        It has the following form:

        {'1s': [{'coefficient': 1.0, 
                 'primitives': {'coefficients': [1.0, 1.0, 1.0], 
                                 'exponents': [1.0, 1.0, 1.0]}},
                {'coefficient': 1.0, 
                 'primitives': {'coefficients': [1.0, 0.5, 0.25],
                                'exponents': [1.0, 2.0, 3.0]}}
               ],
         ...
        }
        """
        self.position = position.copy()
        self._dict_repr = deepcopy(orbitals_dict)

    def get_dictionary_of_parameters(
        self
        ) -> Dict[str, List[Dict[str, Union[float, Dict[str, List[float]]]]]]:
        """
        Return a dictionary of parameters. This should have the form:

        {'1s': [{'coefficient': 1.0, 
                 'primitives': {'coefficients': [1.0, 1.0, 1.0], 
                                 'exponents': [1.0, 1.0, 1.0]}},
                {'coefficient': 1.0, 
                 'primitives': {'coefficients': [1.0, 0.5, 0.25],
                                'exponents': [1.0, 2.0, 3.0]}}
               ]
        }
        """
        return self._dict_repr

    def dictionary_of_parameters(
        self
        ) -> Dict[str, List[Dict[str, Union[float, Dict[str, List[float]]]]]]:
        """Wrapper for get_dictionary_of_parameters
        """
        return self.get_dictionary_of_parameters()

    def parameters(
        self
        ) -> Dict[str, List[Dict[str, Union[float, Dict[str, List[float]]]]]]:
        """Wrapper for get_dictionary_of_parameters
        """
        return self.dictionary_of_parameters()

    def get_position(self):
        return self.position
    
    def number_of_orbitals(self):
        orbital_count = 0
        for k in self.dictionary_of_parameters():
            orbital_count += get_multiplicity_from_orbital_name(k)
        return orbital_count


def number_of_primitives_and_basis_functions(
    orbitals_list: List[AtomicOrbitals]) -> Tuple[int]:
    """ Given a list of AtomicOrbitals instances,
    return a tuple containing the total number of primitive 
    Gaussians, and the total number of basis functions.
    """
    basis_func_count = 0
    primitives_count = 0
    for orbital in orbitals_list:
        orbitals_dict = orbital.get_dictionary_of_parameters()
        for o_name in orbitals_dict:
            for k in range(min(3,
                            get_multiplicity_from_orbital_name(o_name))):
                basis_func_list = orbitals_dict[o_name]
                for basis_func in basis_func_list:
                    basis_func_count += 1
                    primitives_count += \
                        len(basis_func['primitives']['coefficients'])
    return primitives_count, basis_func_count


def encode_orbitals_dict_as_array(
    orbitals_list: List[AtomicOrbitals]) -> np.ndarray:
    primitives_count, basis_func_count\
         = number_of_primitives_and_basis_functions(orbitals_list)
    sizeof_double = 8
    sizeof_long = 8
    sizeof_short = 2
    sizeof_basis_func = 16
    sizeof_gaussian3d = 48
    total_size = (sizeof_long + basis_func_count*sizeof_basis_func 
                  + primitives_count*sizeof_gaussian3d)
    arr_bytes = np.zeros([total_size], order='C', dtype=np.uint8)
    arr_short = arr_bytes.view(dtype=np.int16)
    arr_double = arr_bytes.view(dtype=np.double)
    arr_long = arr_bytes.view(dtype=np.int64)
    arr_long[0] = basis_func_count
    basis_func_index = 0
    primitives_index = 0
    for orbital in orbitals_list:
        position = orbital.get_position()
        orbitals_dict = orbital.get_dictionary_of_parameters()
        for o_name in orbitals_dict:
            for k in range(min(3,
                           get_multiplicity_from_orbital_name(o_name))):
                ang = np.zeros([3])
                ang[k] = get_angular_number(o_name)
                print(o_name, ang)
                for basis_func in orbitals_dict[o_name]:
                    # coefficient = basis_func['coefficient']
                    primitives = basis_func['primitives']
                    offset = 1
                    index = ((sizeof_basis_func // sizeof_long)
                             *basis_func_index)
                    arr_long[offset
                             + index] = len(primitives['coefficients'])
                    for c, e in zip(primitives['coefficients'],
                                    primitives['exponents']):
                        offset = 1 + (sizeof_basis_func // 
                                    sizeof_long) * basis_func_count
                        index = (sizeof_gaussian3d 
                                // sizeof_double)*primitives_index
                        arr_double[offset + index] = e
                        arr_double[offset + index + 1] = c
                        arr_double[offset + index + 3] = position[0]
                        arr_double[offset + index + 4] = position[1]
                        arr_double[offset + index + 5] = position[2]
                        offset1 = (sizeof_long // sizeof_short
                                + basis_func_count 
                                * (sizeof_basis_func // sizeof_short))
                        index = (sizeof_gaussian3d * primitives_index
                                 ) // sizeof_short
                        offset2 = (2*sizeof_double) // sizeof_short
                        arr_short[offset1 + index + offset2] = int(ang[0])
                        arr_short[offset1 + index + offset2 + 1] = int(ang[1])
                        arr_short[offset1 + index + offset2 + 2] = int(ang[2])
                        # print(arr_short[offset1 + index + offset2],
                        #       arr_short[offset1 + index + offset2 + 1],
                        #       arr_short[offset1 + index + offset2 + 2])
                        primitives_index += 1
                    basis_func_index += 1
    extension.set_pointer_addresses(arr_bytes)
    return arr_bytes


class Orbitals:

    atomics: List[AtomicOrbitals]
    # _packed_to_arr=False
    # _arr: np.ndarray
    
    def __init__(
        self, 
        o:
        Union[
            'Orbitals', 
            List[AtomicOrbitals],
            Dict[str, Dict[str, Union[float, Dict[str, List[float]]]]],
            None
            ],
        position=None
        ):
        # List of instances of AtomicOrbitals, where each instance corresponds
        # to the orbitals of a single atom.
        self.atomics = []
        if isinstance(o, Orbitals):
            for e in o.atomics:
                self.atomics.append(
                    AtomicOrbitals(e.position, e.parameters()))
        elif isinstance(o, list):
            for e in o:
                if isinstance(e, AtomicOrbitals):
                    self.atomics.append(
                        AtomicOrbitals(e.position, e.parameters()))
        elif isinstance(o, dict) and position is not None:
            if len(position) == 3:
                x = np.array([position[0], position[1], position[2]],
                             dtype=np.float64)
                self.atomics.append(AtomicOrbitals(x, o))

    def number_of_orbitals(self) -> int:
        return np.sum([
            atomic_orbitals.number_of_orbitals() 
            for atomic_orbitals in self.atomics])
    
    def get_primitives(self) -> List[List[Gaussian3D]]:
        """Return a list of list of each of the primitive Gaussians,
        where each sublist gives the primitives for a single basis function.
        """
        primitive_functions = []
        for atomic_orbitals in self.atomics:
            position = atomic_orbitals.get_position()
            orbitals_dict = atomic_orbitals.get_dictionary_of_parameters()
            for orbital_name in orbitals_dict:
                multiplicity = \
                    get_multiplicity_from_orbital_name(orbital_name)
                for k in range(min(3, multiplicity)):
                    angular = np.zeros([3])
                    angular[k] = get_angular_number(orbital_name)
                    basis_func_primitives = []
                    for basis_func in orbitals_dict[orbital_name]:
                        coefficients = \
                            basis_func['primitives']['coefficients']
                        exponents = basis_func['primitives']['exponents']
                        basis_func_primitives = [
                            Gaussian3D(c, e, position, angular)
                            for c, e in zip(coefficients, exponents)
                        ]
                        primitive_functions.append(basis_func_primitives)
        return primitive_functions



    def __add__(self, 
                other: Union['Orbitals', int]) -> 'Orbitals':
        # self._packed_to_arr = False
        if isinstance(other, int):
            return Orbitals(
                [AtomicOrbitals(e.position, e.parameters()) 
                 for e in self.atomics])
        else:
            new_o = Orbitals(None)
            for o1 in self.atomics:
                new_o.atomics.append(
                    AtomicOrbitals(o1.position, o1.parameters()))
            for o2 in other.atomics:
                new_o.atomics.append(
                    AtomicOrbitals(o2.position, o2.parameters()))
            return new_o

    # def pack_to_array(self):
    #     if not self._packed_to_arr:
    #         self._arr = encode_orbitals_dict_as_array(self.atomics)


def get_orbitals_from_geometry(
    geom: MolecularGeometry, 
    orbitals_dicts: 
    Dict[str,
         Dict[str, List[Dict[str, Union[float, Dict[str, List[float]]]]]]]
) -> Orbitals:
    orbitals = Orbitals([])
    for k in geom.config:
        orbitals_dict = orbitals_dicts[k]
        for position in geom[k]:
            orbitals.atomics.append(AtomicOrbitals(position, orbitals_dict))
    return orbitals

        
