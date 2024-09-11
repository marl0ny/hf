from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import make_ch3, make_ch2


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e-6gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }

ch3_0 = make_ch3(2.07, 2.07, 2.07, np.pi*(111.800/180.0),
                 np.array([-1.0, 0.0, 0.0]), 
                 np.array([0.0, 0.0, -1.0])) + np.array([2.88, 0.0, 0.0])
a = np.pi*112.4/180.0
b = (2.0*np.pi - a)/2.0
ch2_r = np.array([np.cos(b), 0.0, -np.sin(b)])
ch2 = make_ch2(2.07, 2.07, np.pi*(106.100/180.0),
               ch2_r, np.array([0.0, 1.0, 0.0]))
ch3_r = np.array([np.cos(a), 0.0, np.sin(a)])
ch3_1 = make_ch3(2.07, 2.07, 2.07, np.pi*(111.800/180.0), 
                 -ch3_r,
                 np.array([-np.sin(a), 0.0, np.cos(a)])) + 2.88*ch3_r
geom = ch3_0 + ch2 + ch3_1

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2])
ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

dat_h_list = [OrbitalPrimitivesBuilder(position=geom['H'][i],
                                       orbitals_dict=hydrogen_dict)
              for i in range(len(geom['H']))]
dat_c_list = [OrbitalPrimitivesBuilder(position=geom['C'][i],
                                       orbitals_dict=carbon_dict)
              for i in range(len(geom['C']))]
data = sum(dat_c_list) + sum(dat_h_list)
data.set_number_of_orbitals(13)

nuclear_config = [[r, 6.0] for r in geom['C']] +\
                 [[r, 1.0] for r in geom['H']]
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
