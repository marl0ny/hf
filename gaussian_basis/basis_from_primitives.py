import numpy as np
from gaussian_basis import *
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.integrals3d import *
from gaussian_basis import get_orbitals_dict_from_file


DESCRIPTION = \
"""This script takes in a json file of fit parameters for a
summation of Gaussian functions representing an atomic orbital. 
This input json file uses the following naming convention:

<n_protons>p<n_electrons>e_<n_gaussians>gaussians.json,

where <n_protons> is the number protons in the atom,
<n_electrons> is the number of electrons, and <n_gaussians>
is the number of Gaussian basis functions used. 

It must use the following structure:

{"<orbital_name>": {"coefficients": [<coefficient_values>],
                    "exponents": [<exponent_values>]}
 ...
},

where orbital_name must either be 1s, 2s, 2p, 3s, etc.
The coefficients key gives a list of the amplitudes for each of the
Gaussian functions, while the exponents key gives their corresponding
exponent values.

The script returns a json file with the following naming convention:

<n_protons>p<n_electrons>e_<orbital_name><grouping_1>...<grouping_n>...json,

where n_protons denotes the number of protons, and n_electrons denotes the
number of electrons. To describe how Gaussians are grouped into basis 
functions, the file name of the json output contains the following pattern:

_<orbital_name><grouping_1>...<grouping_n>,

where orbital_name is either 1s, 2s, 2p, 3s, etc.
<grouping_1> is an integer value that denotes the number of Gaussians
for the first basis function. This is followed by <grouping_2>, which
counts the number of Gaussians for the second basis function. Next comes
<grouping_3>, and so on to <grouping_n>, where n is the total number of
basis functions. 

As an example, for the json file name

10p10e_1s6_2s6_2p33.json,

there are 10 protons and 10 electrons. The 1s orbital contains a single
basis function with 6 Gaussian functions. Similarly the 2s orbital also
uses a single basis function with 6 Gaussians. The 2p orbital on the 
other hand is split into 2 basis functions, each with 3 Gaussian functions.

For another example, for the file

10p10e_1s33_2s42_2p_111111.json,

the 1s orbital has 2 basis functions with 3 Gaussians, the 2s orbital
uses 2 basis functions where the first one contains 4 Gaussians and the
second 2, and the 2p orbital has 6 basis functions, each with 
a single Gaussian.

Example Usage:

python3 basis_from_primitives.py 10p10e-6gaussians.json -1s 6 -2s 6 -3s 4,1,1

The script takes as input a 10 proton, 10 electron Hartree-Fock 
atomic orbital system where each orbital is fitted with 6 Gaussians 
as described by the input json file.
It then generates a json file where for the 1s orbital there is a single
basis function with 6 Gaussians, for the 2s orbital there is likewise only
one basis function with 6 Gaussians, and for the 3s orbital there are three
basis functions, with 4, 1, and 1 Gaussian functions respectively.
"""


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


def get_norm_factor(pos: np.ndarray, ang: np.ndarray,
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


def gaussian(x: np.ndarray, amp: float, orb_exp: float) -> np.ndarray:
    return amp*np.exp(-orb_exp*x**2)


if __name__ == '__main__':

    import sys
    import json
    import re
    import argparse

    parser = argparse.ArgumentParser(
        description=DESCRIPTION)
    parser.add_argument('filename')
    help_str = 'For the {} orbital, specify the number of Gaussian '
    help_str += 'primitives alloted to each basis function.'
    help_str += 'The input must be formatted as '
    help_str += '<c1>,<c2>,...<cn>, where c1 specifies the number '
    help_str += 'of Gaussians for the first basis function, '
    help_str += 'c2 for the second, and so on to cn for the '
    help_str += 'last (nth) basis function. '
    help_str += 'As an example, the input '
    help_str += '3,1,2 '
    help_str += 'allots 3 Gaussian function for the first basis function'
    help_str += ' 1 Gaussian for the second basis function, '
    help_str += 'and 2 Gaussians for the third.'
    parser.add_argument('-1s', required=False, help=help_str.format('1s'))
    parser.add_argument('-2s', required=False, help=help_str.format('2s'))
    parser.add_argument('-2p', required=False, help=help_str.format('2p'))
    args_dict = vars(parser.parse_args())
    filename = args_dict['filename']
    groupings_dict = {}
    e_count = int(re.search(r'[0-9]+e', filename).group(0)[:-1])
    p_count = int(re.search(r'[0-9]+p', filename).group(0)[:-1])
    g_count = int(re.search(r'[0-9]+gaussians', 
                            filename).group(0).strip('gaussians'))
    for name in ['1s', '2s', '2p']:
        if args_dict[name] is not None:
            groupings_dict[name] = [int(e) for e in 
                                    str(args_dict[name]).split(',')]
        else:
            groupings_dict[name] = [g_count]
    for name in groupings_dict:
        if sum(groupings_dict[name]) != g_count:
            print(f'Invalid number of primitives for {name}.')
            sys.exit()

    if filename is None:
        sys.exit()

    with open(filename, 'r') as f:
        primitives_dict = json.load(f)

    new_dict = {}
    for name in primitives_dict:
        orbital_data0 = primitives_dict[name]
        coefficients = orbital_data0['coefficients']
        exponents = orbital_data0['exponents']
        sort_key = np.argsort(exponents)[::-1]
        coefficients = np.array([coefficients[sort_key[i]]
                                 for i in range(sort_key.shape[0])])
        exponents = np.array([exponents[sort_key[i]]
                              for i in range(sort_key.shape[0])])
        ang = np.array([get_angular_number(name), 0.0, 0.0])
        pos = np.zeros([3])
        orbital_norm = get_norm_factor(pos, ang, coefficients, exponents)
        new_dict[name] = []
        groupings = groupings_dict[name]
        c = 0
        for n in groupings:
            basis_func_c = coefficients[c:c+n]
            basis_func_e = exponents[c:c+n]
            basis_func_norm = get_norm_factor(pos, ang,
                                              basis_func_c, basis_func_e)
            new_dict[name].append({'coefficient': basis_func_norm/orbital_norm,
                                   'primitives': {'coefficients': 
                                                  list(basis_func_c/basis_func_norm),
                                                  'exponents': 
                                                  list(basis_func_e)
                                                  }
                                  })
            c += n
    # print(new_dict)
    
    new_filename = f'{p_count}p{e_count}e'
    for name in groupings_dict:
        new_filename += '_' + name
        tmp = [str(e) + 'g' if e > 9 else str(e) 
               for e in groupings_dict[name]]
        for e in tmp:
            new_filename += e
    new_filename = '../data/' + new_filename + '.json'
    with open(new_filename, 'w') as f:
        json.dump(new_dict, f, indent=4)

    # import sys; sys.exit()
    for name in new_dict:
        coefficients = []
        exponents = []
        for basis_func_dict in new_dict[name]:
            coefficients.extend([basis_func_dict['coefficient']
                                *e for e in 
                                basis_func_dict['primitives']['coefficients']])
            exponents.extend(basis_func_dict['primitives']['exponents'])
        ang = np.array([get_angular_number(name), 0.0, 0.0])
        pos = np.zeros([3])
        orbital_norm = get_norm_factor(pos, ang, 
                                       np.array(coefficients),
                                       np.array(exponents))
        print(name, orbital_norm)

    import matplotlib.pyplot as plt
    with open('../data/10p10e_fd.json', 'r') as f:
        data = json.load(f)
    show_r_scaled_plots = True
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, o in enumerate(data.keys()):
        r_ = np.array(data[o]['r'])
        dr = r_[1] - r_[0]
        values = np.array(data[o]['values'])
        values = values/np.sqrt(dr*np.dot(values, np.conj(values)))
        gauss_sum = np.zeros([len(r_)])
        for basis_func_dict in new_dict[o]:
            co = basis_func_dict['coefficient']
            for c, e in zip(basis_func_dict['primitives']['coefficients'],
                            basis_func_dict['primitives']['exponents']):
                gauss_sum += r_**get_angular_number(o)*gaussian(r_, co*c, e)
        # for c, e in zip(gauss_data[o]['coefficients'],
        #                 gauss_data[o]['exponents']):
        #     gauss_sum += r_**get_angular_number(o)*gaussian(r_, c, e)
        if show_r_scaled_plots:
            plt.plot(r_, values*np.amax(np.abs(r_*gauss_sum))/
                                        np.max(np.abs(values)),
                     label=f'Original: {o}', color=cols[i])
            plt.plot(r_, r_*gauss_sum, label=f'Gaussian fit: {o}',
                     color=cols[i], linestyle='--')
        else:
            # plt.plot(r_, values/r_, label=f'Original: {o}')
            plt.plot(r_, gauss_sum, label=f'Gaussian fit: {o}')
    plt.legend()
    plt.show()
    plt.close()
