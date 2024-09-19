"""Plot the closed-shell Hartree-Fock potential energy curve of the
Hydrogen Fluoride.
"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

energies = []
positions = []
n_points = 30

for i in range(n_points):
    fluorine_position = np.array([0.0, 0.0, 0.0])
    hydrogen_position = np.array([1.6 + i*0.4/n_points,
                                  0.0, 0.0])
    positions.append(hydrogen_position[0])
    hydrogen_dict = {'1s':
                     get_orbitals_dict_from_file(
                     '../data/1p1e-4gaussians.json')
                     ['1s']
                    }
    fluorine_dict = get_orbitals_dict_from_file(
        '../data/9p10e-5gaussians.json')

    data_h = OrbitalPrimitivesBuilder(position=hydrogen_position,
                                    orbitals_dict=hydrogen_dict)
    data_f = OrbitalPrimitivesBuilder(position=fluorine_position,
                                    orbitals_dict=fluorine_dict)
    data = data_f + data_h

    system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                            orbitals=data.orbitals()[:-1],
                            nuclear_config=[[hydrogen_position, 1.0],
                                            [fluorine_position, 9.0]],
                            use_ext=True)

    system.solve(15)
    print(system.energies)
    print(total_energy := system.get_total_energy())
    energies.append(total_energy)

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

plt.title('Hartree-Fock Closed Shell Energy for HF')
plt.plot(positions, energies)
plt.xlabel('r (a.u.)')
plt.ylabel('E (a.u.)')
plt.show()
plt.close()
