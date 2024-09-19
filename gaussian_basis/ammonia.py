"""Compute the lowest Hartree-Fock energy of Ammonia,
and plot its probability density. This uses the experimental geometry
of the molecule, which is found in a NIST database:

    Experimental data for NH3 (Ammonia)
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import MolecularGeometry
import matplotlib.pyplot as plt
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem


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

geom = MolecularGeometry()
for i in range(3):
    geom.add_atom('H', h_positions[i])
geom.add_atom('N', n_position)

hydrogen_dict = get_orbitals_dict_from_file(
     '../data/1p1e_1s4.json'
     )

nitrogen_dict = get_orbitals_dict_from_file(
    '../data/7p7e_1s6_2s3111_2p3111.json'
)
orbitals = get_orbitals_from_geometry(geom, {'H': hydrogen_dict, 'N': nitrogen_dict})
system = ClosedShellSystem(5, orbitals, geom.get_nuclear_configuration())

system.solve(10)

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

# print("Sum of orbitals squared: ", np.sum(u*ds**3))
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=2.0*u.flatten(), isomin=0.1, isomax=2.0,
    opacity=0.1, surface_count=20,  
))
fig.show()

