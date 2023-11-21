from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-8gaussians.json')
                 ['1s']}

energies = []
positions = []
n_points = 100
for i in range(n_points):
    r2_x = 0.25 + 4.0*(i/n_points)
    positions.append(r2_x)
    h_r1 = np.array([0.0, 0.0, 0.0])
    h_r2 = np.array([r2_x, 0.0, 0.0])
    h_r1_dat = OrbitalPrimitivesBuilder(position=h_r1,
                                        orbitals_dict=hydrogen_dict)
    h_r2_dat = OrbitalPrimitivesBuilder(position=h_r2,
                                        orbitals_dict=hydrogen_dict)
    dat = h_r1_dat + h_r2_dat
    system = ClosedShellSystem(primitives=dat.primitives(),
                               orbitals=dat.orbitals()[:-1],
                               nuclear_config=[[h_r1, 1.0],
                                               [h_r2, 1.0]])
    system.solve(10)
    energies.append(system.get_total_energy())

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

plt.title('Hartree-Fock Closed Shell Energy for $H_2$')
plt.plot(positions, energies)
plt.xlabel('r (a.u.)')
plt.ylabel('E (a.u.)')
plt.show()
plt.close()

