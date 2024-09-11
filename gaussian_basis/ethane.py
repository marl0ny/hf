from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives, ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis import get_orbitals_from_geometry


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e-4gaussians.json')
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

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(h_positions.T[0], h_positions.T[1], h_positions.T[2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(c_positions.T[0], c_positions.T[1], c_positions.T[2])
ax.scatter([0.0, 0.0], [0.0, 0.0], [2.0, -2.0], alpha=0.0)
plt.show()

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
system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=nuclear_config,
                           use_ext=True)
system.solve(20)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

import plotly.graph_objects as go
s = np.linspace(-3.0, 3.0, 100)
ds = s[1] - s[0]
x, y, z = np.meshgrid(s, s, s)
u = np.zeros(x.shape)
for k, orbital in enumerate(system.orbitals):
    o = sum([orbital[i]*data.get_primitives()[i]([x, y, z])
             for i in range(len(orbital))])
    o /= np.sqrt(np.sum(np.conj(o)*o)*ds**3)
    u += o**2

print(np.sum(u*ds**3))
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=2.0*u.flatten(), isomin=0.1, isomax=1.0,
    opacity=0.1, surface_count=20,  
))
fig.show()