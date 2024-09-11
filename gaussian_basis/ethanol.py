from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import make_ch3, make_ch2, make_oh


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e-5gaussians.json')
oxygen_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }

ch3 = make_ch3(2.04, 2.04, 2.04, np.pi*110.5/180.0,
               np.array([-1.0, 0.0, 0.0]),
               np.array([0.0, 0.0, 1.0])
               ) + np.array([2.86, 0.0, 0.0])
a1 = np.pi*(107.2/180.0)
ch2 = make_ch2(2.05, 2.05, np.pi*108.0/180.0,
               np.array([np.cos(a1), 0.0, np.sin(a1)]),
               np.array([0.0, 1.0, 0.0]))
axis = np.array([np.cos(a1), 0.0, -np.sin(a1)])
c_o = 2.7
oh = make_oh(1.8, axis) + c_o*axis
geom = ch3 + ch2 + oh

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2])
ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2])
ax.scatter(geom['O'].T[0], geom['O'].T[1], geom['O'].T[2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

dat_h_list = [OrbitalPrimitivesBuilder(position=geom['H'][i],
                                       orbitals_dict=hydrogen_dict)
              for i in range(len(geom['H']))]
dat_o_list = [OrbitalPrimitivesBuilder(position=geom['O'][i],
                                       orbitals_dict=oxygen_dict)
              for i in range(len(geom['O']))]
dat_c_list = [OrbitalPrimitivesBuilder(position=geom['C'][i],
                                       orbitals_dict=carbon_dict)
              for i in range(len(geom['C']))]
data = sum(dat_o_list) + sum(dat_h_list) + sum(dat_c_list)
# data.set_number_of_orbitals(13)

nuclear_config = [[r, 6.0] for r in geom['C']] +\
                 [[r, 1.0] for r in geom['H']] +\
                 [[r, 8.0] for r in geom['O']]
system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=nuclear_config,
                           use_ext=True)
system.solve(10)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')
