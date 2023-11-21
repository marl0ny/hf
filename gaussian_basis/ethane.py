from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }

cc_r = 2.9
ch_r = 2.06
a = 110.905/180*np.pi
c_positions = np.array([[-cc_r/2.0, 0.0, 0.0],
                        [cc_r/2.0, 0.0, 0.0]])
phi = 2.0*np.pi/3.0
c, s = np.cos(np.pi - a), np.sin(np.pi - a)
h_positions_list = [
    c_positions[1] + ch_r*np.array([c, s*np.cos(phi*0.0), s*np.sin(phi*0.0)]),
    c_positions[1] + ch_r*np.array([c, s*np.cos(phi*1.0), s*np.sin(phi*1.0)]),
    c_positions[1] + ch_r*np.array([c, s*np.cos(phi*2.0), s*np.sin(phi*2.0)]),
]
for i in range(3):
    e = h_positions_list[i]
    c, s = np.cos(np.pi), np.sin(np.pi)
    h_positions_list.append(np.array([-e[0],
                                      c*e[1] - s*e[2],
                                      s*e[1] + c*e[2]]))
h_positions = np.array(h_positions_list)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(h_positions.T[0], h_positions.T[1], h_positions.T[2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.scatter(c_positions.T[0], c_positions.T[1], c_positions.T[2])
# plt.show()

dat_h_list = [OrbitalPrimitivesBuilder(
              position=h_positions[i],
              orbitals_dict=hydrogen_dict) for i in range(len(h_positions))]
dat_c_list = [OrbitalPrimitivesBuilder(
              position=c_positions[i],
              orbitals_dict=carbon_dict) for i in range(len(c_positions))]
data = sum(dat_c_list) + sum(dat_h_list)
data.set_number_of_orbitals(9)

nuclear_config = [[r, 6.0] for r in c_positions] +\
                 [[r, 1.0] for r in h_positions]
system = ClosedShellSystem(primitives=data.primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=nuclear_config,
                           use_ext=True)
system.solve(10)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')