from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder

t1 = perf_counter_ns()

oxygen_pos1 = np.array([2.2, 0.0, 0.0])
carbon_pos = np.array([0.0, 0.0, 0.0])
oxygen_pos2 = np.array([-2.2, 0.0, 0.0])

oxygen_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
carbon_dict = get_orbitals_dict_from_file('../data/7p8e-5gaussians.json')
dat_oxygen_pos1 = OrbitalPrimitivesBuilder(position=oxygen_pos1,
                                          orbitals_dict=oxygen_dict)
dat_oxygen_pos2 = OrbitalPrimitivesBuilder(position=oxygen_pos2,
                                           orbitals_dict=oxygen_dict)
dat_carbon = OrbitalPrimitivesBuilder(position=carbon_pos,
                                      orbitals_dict=carbon_dict)
data = dat_oxygen_pos1 + dat_oxygen_pos2 + dat_carbon
data.set_number_of_orbitals(11)


system = ClosedShellSystem(primitives=data.primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=[[oxygen_pos1, 8.0],
                                           [carbon_pos, 6.0],
                                           [oxygen_pos2, 8.0]],
                           use_ext=True)
system.solve(20)
print(system.energies)
print(system.get_total_energy())

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')