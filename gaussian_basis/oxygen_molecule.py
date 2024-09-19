"""Plot the closed-shell Hartree-Fock potential energy curve of the
Oxygen molecule.
"""
from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
import matplotlib.pyplot as plt


t1 = perf_counter_ns()

# oxygen_dict = get_orbitals_dict_from_file(
#     '../data/8p8e_1s6_2s111111_2p111111.json'
# )
oxygen_dict = get_orbitals_dict_from_file(
    '../data/8p8e-5gaussians.json')

energies = []
positions = []
for i in range(10):
    oxygen_pos1 = np.array([0.0, 0.0, 0.0])
    oxygen_pos2 = np.array([2.1 + i*0.03, 0.0, 0.0])
    positions.append(oxygen_pos2[0])

    dat_oxygen_pos1 = OrbitalPrimitivesBuilder(position=oxygen_pos1,
                                            orbitals_dict=oxygen_dict)
    dat_oxygen_pos2 = OrbitalPrimitivesBuilder(position=oxygen_pos2,
                                            orbitals_dict=oxygen_dict)
    data = dat_oxygen_pos1 + dat_oxygen_pos2
    data.set_number_of_orbitals(8)

    system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                            orbitals=data.orbitals(),
                            nuclear_config=[[oxygen_pos1, 8.0],
                                            [oxygen_pos2, 8.0]],
                            use_ext=True)
    system.solve(20)
    print(system.energies)
    print(total_energy := system.get_total_energy())
    energies.append(total_energy)


t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

print(positions[np.argmin(energies)])
print(np.min(energies))
plt.plot(np.array(positions), np.array(energies))
plt.title('Closed Shell Hartree-Fock Energy Curve for ' + r'$O_2$')
plt.ylabel('Energy (a.u.)')
plt.xlabel('Bond length (a.u.)')
plt.show()
plt.close()