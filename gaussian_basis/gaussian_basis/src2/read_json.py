import json


def get_orbitals_data_string(spec):
    with open(f'./data/{spec}.json') as f:
        lines = [line for line in f]
        contents = ''.join(lines)
        data = json.loads(contents)

    orbitals_data_string = ''
    for o in data:
        orbitals_data_string += '{ "' + o.strip(' ') + '", {' + '\n'
        for b in data[o]:
            coefficient_string = str(b['coefficient'])
            primitives = b['primitives']
            print(primitives)
            coefficients_string = \
                '{' + str(primitives['coefficients']).strip(']').strip('[')\
                + '}'
            exponents_string = \
                '{' + str(primitives['exponents']).strip(']').strip('[')\
                + '}'
            primitives_string = \
                '{' + '\n        ' + coefficients_string + ',\n' + '        ' + exponents_string + '}'
            basis_func_string = '    {' + coefficient_string + ', ' + primitives_string + '\n    },'
            orbitals_data_string += basis_func_string + '\n'
        orbitals_data_string += '}},\n'

    lines = []
    for line in orbitals_data_string.split('\n'):
        line2 = '    ' + line
        lines.append(line2)

    orbitals_data_string = '{{\n' + '\n'.join(lines) + '\n}};'
    return orbitals_data_string


ATOMIC_DATA_STR = """#include "orbitals_description.hpp"

using namespace orbital_description_data;

namespace atomic_data_descriptions {

"""

with open('atomic_data.hpp', 'w') as f:
    f.write(ATOMIC_DATA_STR)
    specs = ['10p10e_1s5_2s311_2p311',
             '1p1e_1s21_2s21_2p21']
    for spec in specs:
        f.write(f'static const OrbitalsData ORB_{spec.upper()} = ')
        orbitals_data_string = get_orbitals_data_string(spec)
        for line in orbitals_data_string:
            f.write(line)
        f.write('\n\n')
    f.write('}\n')
        