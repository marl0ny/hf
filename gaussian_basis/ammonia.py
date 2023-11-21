from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder

t1 = perf_counter_ns()

n_position = np.array([0.0, 0.0, 0.0])
r = 1.9
sqrt_3 = np.sqrt(3.0)
s_val = np.sqrt((2.0/3.0)*(1.0 - np.cos(2.0*np.pi*0.32)))
c_val = np.sqrt(1.0 - s_val**2)
# print(s_val, c_val)
h_positions_list = [
    r*np.array([s_val, 0.0, c_val]),
    r*np.array([-0.5*s_val, sqrt_3*s_val/2.0, c_val]),
    r*np.array([-0.5*s_val, -sqrt_3*s_val/2.0, c_val]),
]
h_positions = np.array(h_positions_list)

unrestricted_nitrogen_dict \
    = get_orbitals_dict_from_file('../data/7p7e-6gaussians.json')
nitrogen_dict = {
    '1s': unrestricted_nitrogen_dict['1s+'],
    '2s': unrestricted_nitrogen_dict['2s+'],
    '2p': unrestricted_nitrogen_dict['2p+'],
}
hydrogen_dict = {
    '1s':
    get_orbitals_dict_from_file('../data/1p2e-4gaussians.json')['1s']}

dat_n = OrbitalPrimitivesBuilder(position=n_position,
                                 orbitals_dict=nitrogen_dict)
dat_h_list = [OrbitalPrimitivesBuilder(position=h_positions[i],
                                       orbitals_dict=hydrogen_dict)
              for i in range(h_positions.shape[0])]
data = dat_n + dat_h_list[0] + dat_h_list[1] + dat_h_list[2]
data.set_number_of_orbitals(5)

system = ClosedShellSystem(primitives=data.primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=[[n_position, 7],
                                           [h_positions[0], 1],
                                           [h_positions[1], 1],
                                           [h_positions[2], 1],
                                           ],
                           use_ext=True)
system.solve(10)
print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())


t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

