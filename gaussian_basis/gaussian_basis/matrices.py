from .integrals3d import *
from .gaussian3d import Gaussian3D
from typing import List
import numpy as np


def get_overlap_matrix(g: List[Gaussian3D]) -> np.ndarray:
    n = len(g)
    overlap_matrix = np.zeros([n, n])
    for i in range(len(g)):
        for j in range(i, len(g)):
            overlap_matrix[i, j] = overlap(g[i], g[j])
            if j > i:
                overlap_matrix[j, i] = overlap_matrix[i, j]
    return overlap_matrix


def get_kinetic_matrix(g: List[Gaussian3D]) -> np.ndarray:
    n = len(g)
    kinetic_matrix = np.zeros([n, n])
    for i in range(len(g)):
        for j in range(i, len(g)):
            kinetic_matrix[i, j] = kinetic(g[i], g[j])
            if j > i:
                kinetic_matrix[j, i] = kinetic_matrix[i, j]
    return kinetic_matrix


def get_nuclear_potential_matrix(g: List[Gaussian3D],
                                 nuc_loc_charge_lst: 
                                 List[List[Union[np.ndarray, int]]]
                                 ) -> np.ndarray:
    n = len(g)
    nuclear_matrix = np.zeros([n, n])
    for i in range(len(g)):
        for j in range(i, len(g)):
            nuclear_matrix[i, j] = nuclear(g[i], g[j], nuc_loc_charge_lst)
            if j > i:
                nuclear_matrix[j, i] = nuclear_matrix[i, j]
    return nuclear_matrix


def two_electron_integrals_inner(a, b, 
                                 g: List[Gaussian3D], arr: np.ndarray):
    primitives_count = len(g)
    for c in range(primitives_count):
        for d in range(c, primitives_count):
            arr[a, b, c, d] = repulsion(g[a], g[b], g[c], g[d])
            if d > c:
                arr[a, b, d, c] = arr[a, b, c, d]


def get_two_electron_integrals_tensor(g: List[Gaussian3D]) -> np.ndarray:
    primitives_count = len(g)
    arr = np.zeros(4*[primitives_count])
    for a in range(primitives_count):
        for b in range(a, primitives_count):
            two_electron_integrals_inner(a, b, g=g, arr=arr)
            if b > a:
                arr[b, a] = arr[a, b]
    return arr

# def get_two_electron_integrals_tensor(g: List[Gaussian3D]) -> np.ndarray:
#     primitives_count = len(g)
#     arr = np.zeros(4*[primitives_count])
#     for a in range(primitives_count):
#         g_a = g[a]
#         if g_a.amplitude() != 0.0:
#             for b in range(primitives_count):
#                 g_b = g[b]
#                 if g_b.amplitude() != 0.0:
#                     for c in range(primitives_count):
#                         g_c = g[c]
#                         if g_c.amplitude() != 0.0:
#                             for d in range(primitives_count):
#                                 g_d = g[d]
#                                 arr[a, b, c, d] = repulsion(g_a, g_b,
#                                                             g_c, g_d)
#     return arr


def get_repulsion_exchange_matrix(orbitals: np.ndarray, 
                                  g: List[Gaussian3D]) -> np.ndarray:
    """ This function assumes everything is closed shell.
    """
    orbitals_count = len(orbitals[0])
    primitives_count = len(g)
    matrix = np.zeros([primitives_count, 
                       primitives_count])
    for n in range(primitives_count):
        for m in range(primitives_count):
            for i in range(orbitals_count):
                for a in range(primitives_count):
                    o_a = orbitals[i, a]*g[a]
                    for b in range(primitives_count):
                        o_b = orbitals[i, b]*g[b]
                        matrix[n, m] \
                            += (2.0*repulsion(g[n], g[m], o_a, o_b)
                                - repulsion(g[n], o_a, o_b, g[m]))
    return matrix
