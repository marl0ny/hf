from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import make_system_from_primitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import make_co, make_ch3


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e-4gaussians.json')
# carbon_dict_ = get_orbitals_dict_from_file('../data/7p7e-4gaussians.json')
# carbon_dict = {
#     '1s': carbon_dict_['1s+'],
#     '2s': carbon_dict_['2s+'],
#     '2p': carbon_dict_['2p+'],
# }
oxygen_dict = get_orbitals_dict_from_file(
    '../data/8p8e-6gaussians.json')
# oxygen_dict = get_orbitals_dict_from_file(
#     '../data/10p10e-6gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }
co = make_co(2.3)
a = 0.674*np.pi
ch3_1 = make_ch3(2.05, 2.05, 2.05, 0.61*np.pi,
                 np.array([-np.sin(a), 0.0, -np.cos(a)]),
                 np.array([-np.cos(a), 0.0, np.sin(a)])
                 ) + 2.85*np.array([np.sin(a), 0.0, np.cos(a)])
ch3_2 = make_ch3(2.05, 2.05, 2.05, 0.61*np.pi,
                 np.array([np.sin(a), 0.0, -np.cos(a)]),
                 np.array([np.cos(a), 0.0, np.sin(a)])
                 ) + 2.85*np.array([-np.sin(a), 0.0, np.cos(a)])
geom = co + ch3_1 + ch3_2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# for e in geom.get_nuclear_configuration():
#     print(e)
#     ax.scatter([e[0][0]], [e[0][1]], [e[0][2]])
ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2])
ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2])
ax.scatter(geom['O'].T[0], geom['O'].T[1], geom['O'].T[2])
ax.scatter([0.0, 0.0], [-4.0, 4.0], [0.0, 0.0], alpha=0.0)
ax.scatter([0.0, 0.0], [0.0, 0.0], [-4.0, 4.0], alpha=0.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

system = make_system_from_primitives(
    geom, {'H': hydrogen_dict, 'C': carbon_dict, 'O': oxygen_dict}, 16)

t2 = perf_counter_ns()

system.solve(10)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())

t3 = perf_counter_ns()

print(f'Construction time: {(t2 - t1)/1000000000.0}s')
print(f'Total time taken: {(t3 - t1)/1000000000.0}s')
