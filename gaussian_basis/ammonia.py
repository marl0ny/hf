from time import perf_counter_ns
import numpy as np
from gaussian_basis import make_system_from_primitives
from gaussian_basis import get_build_from_primitives
from gaussian_basis import make_system_from_geometry_and_build
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import MolecularGeometry
import matplotlib.pyplot as plt
from gaussian_basis import Orbitals, get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem

import pprint


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
    '1s': get_orbitals_dict_from_file('../data/1p1e-4gaussians.json')['1s'],
    '2s': get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')['2s'],
}
hydrogen_dict = {
    '1s': get_orbitals_dict_from_file('../data/10p10e-6gaussians.json')['1s']
}
nitrogen_dict = get_orbitals_dict_from_file('../data/10p10e-6gaussians.json')
# hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e-4gaussians.json')

geom = MolecularGeometry()
for i in range(3):
    geom.add_atom('H', h_positions[i])
geom.add_atom('N', n_position)

hydrogen_dict2 = get_orbitals_dict_from_file(
     # '../data/1p1e_1s8.json'
     # '../data/10p10e_1s6_2s6_2p6.json'
     '../data/1p1e_1s4.json'
     # '../data/10p10e_1s6_2s42_2p42.json'
     )
# del hydrogen_dict2['2p']
# del hydrogen_dict2['2s']
nitrogen_dict2 = get_orbitals_dict_from_file(
    # '../data/10p10e_1s6_2s6_2p6.json'
    # '../data/10p10e_1s8_2s41111_2p41111.json'
    # '../data/10p10e_1s6_2s42_2p42.json'
    '../data/7p7e_1s6_2s3111_2p3111.json'
    )
pprint.pprint(hydrogen_dict2)
orbitals = get_orbitals_from_geometry(geom, {'H': hydrogen_dict2, 'N': nitrogen_dict2})
for atomic_orbitals in orbitals.atomics:
    print('Position: ', atomic_orbitals.position)
    print('Number of orbitals', atomic_orbitals.number_of_orbitals())
    pprint.pprint(atomic_orbitals.get_dictionary_of_parameters())
system = ClosedShellSystem(5, orbitals, geom.get_nuclear_configuration())


# data = get_build_from_primitives(
#     geom, {'H': hydrogen_dict, 'N': nitrogen_dict}, 5)
# system = make_system_from_geometry_and_build(geom, data)

plt.imshow(system.orbitals)
plt.title("Orbital matrix (before iteration)")
plt.xlabel('basis function index')
plt.ylabel('orbital index')
plt.show()
plt.close()

plt.imshow(system.overlap)
plt.title("Overlap matrix")
plt.show()
plt.close()

system.solve(10)

plt.imshow(system.orbitals)
plt.title("Orbital matrix (after iteration)")
plt.xlabel('basis function index')
plt.ylabel('orbital index')
plt.show()
plt.close()

print('Orbital energies: ', system.energies)
print('Nuclear configuration energy: ',
     system.get_nuclear_configuration_energy())
print('Kinetic energy: ',
      system.get_kinetic_energy())
print('Nuclear energy: ',
      system.get_nuclear_potential_energy())
print('Electron repulsion-exchange energy: ',
      system.get_repulsion_exchange_energy())
print('Total energy: ', system.get_total_energy())


t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')


import plotly.graph_objects as go
s = np.linspace(-3.2, 3.2, 100)
ds = s[1] - s[0]
x, y, z = np.meshgrid(s, s, s)
u = np.zeros(x.shape)
primitive_func_list = orbitals.get_primitives()
for k, orbital in enumerate(system.orbitals):
    o = sum([orbital[i]*sum([p([x, y, z]) for p in primitive_func_list[i]])
             for i in range(len(orbital))])
    o /= np.sqrt(np.sum(np.conj(o)*o)*ds**3)
    u += o**2

print(np.sum(u*ds**3))
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=2.0*u.flatten(), isomin=0.1, isomax=2.0,
    opacity=0.1, surface_count=20,  
))
fig.show()

# import plotly.graph_objects as go
# s = np.linspace(-3.2, 3.2, 100)
# ds = s[1] - s[0]
# x, y, z = np.meshgrid(s, s, s)
# u = np.zeros(x.shape)
# for k, orbital in enumerate(system.orbitals):
#     o = sum([orbital[i]*data.get_primitives()[i]([x, y, z])
#              for i in range(len(orbital))])
#     o /= np.sqrt(np.sum(np.conj(o)*o)*ds**3)
#     u += o**2

# print(np.sum(u*ds**3))
# fig = go.Figure(data=go.Volume(
#     x=x.flatten(), y=y.flatten(), z=z.flatten(),
#     value=2.0*u.flatten(), isomin=0.1, isomax=2.0,
#     opacity=0.1, surface_count=20,  
# ))
# fig.show()
