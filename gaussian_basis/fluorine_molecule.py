from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
import matplotlib.pyplot as plt

t1 = perf_counter_ns()

fluorine_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')

energies = []
positions = []
n_points = 10
for i in range(n_points):
    fluorine_pos1 = np.array([0.0, 0.0, 0.0])
    fluorine_pos2 = np.array([2.0 + i/n_points, 0.0, 0.0])
    positions.append(fluorine_pos2[0])
    dat_f_pos1 = OrbitalPrimitivesBuilder(position=fluorine_pos1,
                                        orbitals_dict=fluorine_dict)
    dat_f_pos2 = OrbitalPrimitivesBuilder(position=fluorine_pos2,
                                        orbitals_dict=fluorine_dict)
    data = dat_f_pos1 + dat_f_pos2
    data.set_number_of_orbitals(9)

    system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                            orbitals=data.orbitals(),
                            nuclear_config=[[fluorine_pos1, 9.0],
                                            [fluorine_pos2, 9.0]],
                            use_ext=True)

    system.solve(10)
    print(system.energies)
    print(total_energy := system.get_total_energy())
    energies.append(total_energy)

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

plt.title('Hartree-Fock Closed Shell Energy for $F_2$')
plt.plot(positions, energies)
plt.xlabel('r (a.u.)')
plt.ylabel('E (a.u.)')
plt.show()
plt.close()