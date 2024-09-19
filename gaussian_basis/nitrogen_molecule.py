"""Plot the closed-shell Hartree-Fock potential energy curve of the
Nitrogen molecule.
"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

unrestricted_nitrogen_dict \
    = get_orbitals_dict_from_file('../data/7p7e-6gaussians.json')
nitrogen_dict = {
    '1s': unrestricted_nitrogen_dict['1s+'],
    '2s': unrestricted_nitrogen_dict['2s+'],
    '2p': unrestricted_nitrogen_dict['2p+'],
}
# nitrogen_dict \
#     = get_orbitals_dict_from_file('../data/8p8e-6gaussians.json')

energies = []
positions = []
n_points = 10
for i in range(n_points):
    r2_x = 1.2 + (i / n_points)
    positions.append(r2_x)
    nitrogen_r1 = np.array([0.0, 0.0, 0.0])
    nitrogen_r2 = np.array([r2_x, 0.0, 0.0])
    nitrogen_r1_dat = OrbitalPrimitivesBuilder(
        position=nitrogen_r1, orbitals_dict=nitrogen_dict)
    nitrogen_r2_dat = OrbitalPrimitivesBuilder(
        position=nitrogen_r2, orbitals_dict=nitrogen_dict)
    dat = nitrogen_r1_dat + nitrogen_r2_dat
    dat.set_number_of_orbitals(7)
    system = ClosedShellSystemFromPrimitives(primitives=dat.get_primitives(),
                               orbitals=dat.orbitals(),
                               nuclear_config=[[nitrogen_r1, 7.0],
                                               [nitrogen_r2, 7.0]],
                               use_ext=True)
    system.solve(10)
    energies.append(system.get_total_energy())

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

print(positions[np.argmin(energies)])
print(np.amin(energies))

plt.title('Hartree-Fock Closed Shell Energy for $N_2$')
plt.plot(positions, energies)
plt.xlabel('r (a.u.)')
plt.ylabel('E (a.u.)')
plt.show()
plt.close()
