from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import make_system_from_primitives
from gaussian_basis import get_build_from_primitives
from gaussian_basis import make_system_from_geometry_and_build
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import MolecularGeometry

from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem


t1 = perf_counter_ns()

# carbon_dict = get_orbitals_dict_from_file('../data/7p8e-3gaussians.json')
# carbon_dict_ = get_orbitals_dict_from_file('../data/7p7e-4gaussians.json')
# carbon_dict = {
#     '1s': carbon_dict_['1s+'],
#     '2s': carbon_dict_['2s+'],
#     '2p': carbon_dict_['2p+'],
# }
# hydrogen_dict = {'1s':
#                  get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
#                  ['1s']}
carbon_dict = get_orbitals_dict_from_file('../data/7p8e_1s5_2s32_2p32.json')
hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s4.json')


geom = MolecularGeometry()
angle = np.pi/3.0
for i in range(6):
    geom.add_atom('C', 2.6*np.array([np.cos(i*angle), np.sin(i*angle), 0.0]))
    geom.add_atom('H', 4.6*np.array([np.cos(i*angle), np.sin(i*angle), 0.0]))

# for i, e in enumerate(geom.get_nuclear_configuration()):
#     print(e)
#     plt.scatter([e[0][0]], [e[0][1]])
plt.scatter(geom['H'].T[0], geom['H'].T[1])
plt.scatter(geom['C'].T[0], geom['C'].T[1])
plt.show()
plt.close()

# data = get_build_from_primitives(
#     geom, {'H': hydrogen_dict, 'C': carbon_dict}, 21)
# system = make_system_from_geometry_and_build(geom, data)

orbitals = get_orbitals_from_geometry(geom, {'H': hydrogen_dict, 'C': carbon_dict})
system = ClosedShellSystem(21, orbitals, geom.get_nuclear_configuration())

t2 = perf_counter_ns()

system.solve(200)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t3 = perf_counter_ns()

s = np.linspace(-5.5, 5.5, 220)
x, y = np.meshgrid(s, s)
u = np.zeros(x.shape)
primitive_func_list = orbitals.get_primitives()
for k, orbital in enumerate(system.orbitals):

    o = sum([orbital[i]*sum([p([x, y, 0.0]) for p in primitive_func_list[i]])
             for i in range(len(orbital))])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_title(r'$\phi_'
                 + f'{k}(x, y, z=0)$'
                   f'\nE = {system.energies[k]} a.u.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\phi_' + f'{k}(x, y)$')
    ax.plot_surface(x, y, o)
    plt.show()
    plt.close()
    u += o**2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$|\psi(x, y, z=0)|$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel(r'$|\psi(x, y)|$')
ax.plot_surface(x, y, np.sqrt(u))
plt.show()
plt.close()

print(f'Construction time: {(t2 - t1)/1000000000.0}s')
print(f'Total time taken: {(t3 - t1)/1000000000.0}s')
