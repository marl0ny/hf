"""Plot the closed-shell Hartree-Fock potential energy curve of the
Hydrogen molecule.
"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import MolecularGeometry
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem


t1 = perf_counter_ns()


hydrogen_dict = get_orbitals_dict_from_file(
    '../data/1p1e_1s4_2s4_2p4.json')

energies = []
positions = []
n_points = 100
for i in range(n_points):
    r2_x = 0.75 + (i/n_points)
    positions.append(r2_x)
    geom = MolecularGeometry()
    geom.add_atom('H', np.array([0.0, 0.0, 0.0]))
    geom.add_atom('H', np.array([r2_x, 0.0, 0.0]))
    orbitals = get_orbitals_from_geometry(
        geom, {'H': hydrogen_dict}
    )
    system = ClosedShellSystem(1, orbitals, geom.get_nuclear_configuration())
    system.solve(10)
    print(system.energies)
    energies.append(system.get_total_energy())

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

plt.title('Hartree-Fock Closed Shell Energy for $H_2$')
plt.plot(positions, energies)
plt.xlabel('r (a.u.)')
plt.ylabel('E (a.u.)')
plt.show()
plt.close()

