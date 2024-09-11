from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder

t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e-5gaussians.json')
oxygen_dict = get_orbitals_dict_from_file(
        '../data/10p10e-5gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }

carbon_pos = np.array([0.0, 0.0, 0.0])
oxygen_pos = np.array([2.7, 0.0, 0.0])
r_oh = 1.8
r_ch = 2.07
a = np.pi*(1.0 - (108.870/180))
b1 = np.pi/2.0
b2 = b1 + 2.0*np.pi/3.0
b3 = b1 + 4.0*np.pi/3.0
c, s = np.cos(2.0*np.pi*(109.91/180)), np.sin(2.0*np.pi*(109.91/180))
h_positions = np.array([
    oxygen_pos + r_oh*np.array([np.cos(a), 0.0, np.sin(a)]),
    r_ch*np.array([c, s*np.cos(b1), s*np.sin(b1)]),
    r_ch*np.array([c, s*np.cos(b2), s*np.sin(b2)]),
    r_ch*np.array([c, s*np.cos(b3), s*np.sin(b3)])])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(h_positions.T[0], h_positions.T[1], h_positions.T[2])
# ax.set_xlabel('x')
# ax.set_xlabel('y')
# ax.set_xlabel('z')
# ax.scatter(carbon_pos[0], carbon_pos[1], carbon_pos[2])
# ax.scatter(oxygen_pos[0], oxygen_pos[1], oxygen_pos[2])
# plt.show()

dat_c = OrbitalPrimitivesBuilder(position=carbon_pos,
                                 orbitals_dict=carbon_dict)
dat_o = OrbitalPrimitivesBuilder(position=oxygen_pos,
                                 orbitals_dict=oxygen_dict)
dat_h_list = [OrbitalPrimitivesBuilder(
              position=h_positions[i],
              orbitals_dict=hydrogen_dict) for i in range(len(h_positions))]
data = dat_c + dat_o + dat_h_list[0] + dat_h_list[1] \
       + dat_h_list[2] + dat_h_list[3]
data.set_number_of_orbitals(9)

system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=[[carbon_pos, 6.0],
                                           [oxygen_pos, 8.0],
                                           [h_positions[0], 1.0],
                                           [h_positions[1], 1.0],
                                           [h_positions[2], 1.0],
                                           [h_positions[3], 1.0]],
                           use_ext=True)
system.solve(10)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')
