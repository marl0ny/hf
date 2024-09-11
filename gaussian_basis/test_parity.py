import unittest
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import OrbitalPrimitivesBuilder, get_orbitals_dict_from_file



def get_energies():
    ne_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
    nuc_pos = np.array([0.0, 0.0, 0.0])
    data = OrbitalPrimitivesBuilder(position=nuc_pos,
                                    orbitals_dict=ne_dict)
    system_python_only = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                                           orbitals=data.orbitals(),
                                           nuclear_config=[[nuc_pos, 10.0]])
    system_ext = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                                   orbitals=data.orbitals(),
                                   nuclear_config=[[nuc_pos, 10.0]],
                                   use_ext=True)
    system_ext.solve(10)
    system_python_only.solve(10)
    return {'ext': {'e': system_ext.get_total_energy(),
                     'ke': system_ext.get_kinetic_energy(),
                     'pe': system_ext.get_nuclear_potential_energy(),
                     },
            'python_only': {
                'e': system_python_only.get_total_energy(),
                'ke': system_python_only.get_kinetic_energy(),
                'pe': system_python_only.get_nuclear_potential_energy(),
                 },
           }


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.energies = get_energies()

    def test_compare_total_energy(self):
        self.assertAlmostEqual(self.energies['python_only']['e'],
                               self.energies['ext']['e'],
                               delta=1e-20)

    def test_compare_kinetic_energy(self):
        self.assertAlmostEqual(self.energies['python_only']['ke'],
                               self.energies['ext']['ke'],
                               delta=1e-20) 

    def test_compare_potential_energy(self):
        self.assertAlmostEqual(self.energies['python_only']['pe'],
                               self.energies['ext']['pe'],
                               delta=1e-20) 


if __name__ == '__main__':
    unittest.main()