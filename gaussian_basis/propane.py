"""Compute the lowest Hartree-Fock energy of Propane,
and plot its probability density. This uses the experimental geometry
of the molecule, which is found in a NIST database:

    Experimental data for C3H8 (Propane).
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import make_ch3, make_ch2
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem
from gaussian_basis.molecular_geometry import MolecularGeometry

# Using primitives:
# -117.2162

t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e_1s3111_2s3111_2p111111.json')
hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s4.json')
# hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s4_2s1111_2p1111.json')

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
plt.title('Propane Nuclei Locations')
ax = fig.add_subplot(projection='3d')
ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2], label='H')
ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2], label='C')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()


orbitals = get_orbitals_from_geometry(geom,
                                      {
                                          'H': hydrogen_dict,
                                          'C': carbon_dict
                                      })
system = ClosedShellSystem(13, orbitals, geom.get_nuclear_configuration())
system.solve(10)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

import plotly.graph_objects as go
s_x = np.linspace(-4.5, 4.5, 128)
s_y = np.linspace(-2.5, 2.5, 128)
s_z = np.linspace(-3.0, 5.0, 128)
ds3 = (s_x[1] - s_x[0])*(s_y[1] - s_y[0])*(s_z[1] - s_z[0])
x, y, z = np.meshgrid(s_x, s_y, s_z)
u = np.zeros(x.shape)
primitive_func_list = orbitals.get_primitives()
for k, orbital in enumerate(system.orbitals):
    o = sum([orbital[i]*sum([p([x, y, z]) for p in primitive_func_list[i]])
             for i in range(len(orbital))])
    o /= np.sqrt(np.sum(np.conj(o)*o)*ds3)
    u += o**2
print(np.sum(u*ds3))
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=2.0*u.flatten(), isomin=0.1, isomax=2.0,
    opacity=0.1, surface_count=20,  
))
fig.show()
