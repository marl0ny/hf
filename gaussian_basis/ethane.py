"""Compute the lowest Hartree-Fock energy of Carbon Dioxide,
and plot its probability density. This uses the experimental geometry
of the molecule, which is found in a NIST database:

    Experimental data for C2H6 (Ethane).
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives, ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis.molecular_geometry import MolecularGeometry


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e_1s32_2s32_2p32.json')
hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s4_2s1111_2p1111.json')

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

import matplotlib.pyplot as plt
fig = plt.figure()
plt.title('Ethane Nuclei Locations')
ax = fig.add_subplot(projection='3d')
ax.scatter(h_positions.T[0], h_positions.T[1], h_positions.T[2],
           label='H positions')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(c_positions.T[0], c_positions.T[1], c_positions.T[2],
           label='C positions')
ax.scatter([0.0, 0.0], [0.0, 0.0], [2.0, -2.0], alpha=0.0)
ax.legend()
plt.show()

geom = MolecularGeometry()
for c_position in c_positions:
    geom.add_atom('C', c_position)
for h_position in h_positions:
    geom.add_atom('H', h_position)

orbitals = get_orbitals_from_geometry(
    geom, {'H': hydrogen_dict, 'C': carbon_dict})
system = ClosedShellSystem(9, orbitals, geom.get_nuclear_configuration())
system.solve(20)

# Old ground state energy: -78.2155


print("Orbital energies: ", system.energies)
print("Nuclear potential energy: ", system.get_nuclear_configuration_energy())
print("Total energy: ", system.get_total_energy())
t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

import plotly.graph_objects as go
s = np.linspace(-3.0, 3.0, 100)
ds = s[1] - s[0]
x, y, z = np.meshgrid(s, s, s)
u = np.zeros(x.shape)
primitive_func_list = orbitals.get_primitives()
for k, orbital in enumerate(system.orbitals):
    o = sum([orbital[i]*sum([p([x, y, z]) for p in primitive_func_list[i]])
             for i in range(len(orbital))])
    o /= np.sqrt(np.sum(np.conj(o)*o)*ds**3)
    u += o**2

# print('Summation or orbital norms: ', np.sum(u*ds**3))
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=2.0*u.flatten(), isomin=0.1, isomax=1.0,
    opacity=0.1, surface_count=20,  
))
fig.show()