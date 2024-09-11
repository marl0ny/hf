from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import MolecularGeometry
from gaussian_basis import get_orbitals_from_geometry

t1 = perf_counter_ns()

oxygen_pos1 = np.array([2.2, 0.0, 0.0])
carbon_pos = np.array([0.0, 0.0, 0.0])
oxygen_pos2 = np.array([-2.2, 0.0, 0.0])

geom = MolecularGeometry()
geom.add_atom('O', oxygen_pos1)
geom.add_atom('C', carbon_pos)
geom.add_atom('O', oxygen_pos2)
oxygen_dict = get_orbitals_dict_from_file('../data/10p10e_1s2111_2s2111_2p2111.json')
carbon_dict = get_orbitals_dict_from_file('../data/7p8e_1s3111_2s3111_2p111111.json')

orbitals = get_orbitals_from_geometry(geom, {'O': oxygen_dict, 'C': carbon_dict})
system = ClosedShellSystem(11, orbitals, geom.get_nuclear_configuration())

system.solve(20)
print(system.energies)
print(system.get_total_energy())

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

import plotly.graph_objects as go
s_x = np.linspace(-3.2, 3.2, 128)
s_yz = np.linspace(-2.8, 2.8, 100)
ds = s_yz[1] - s_yz[0]
x, y, z = np.meshgrid(s_x, s_yz, s_yz)
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
