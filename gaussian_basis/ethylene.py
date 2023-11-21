from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

cc_r = 2.5
c_positions = np.array([[cc_r/2.0, 0.0, 0.0],
                        [-cc_r/2.0, 0.0, 0.0]])
cc_v = np.array([cc_r, 0.0, 0.0])
ch_r = 2.02
a = np.pi/3.0
h_positions_list = [
    ch_r*np.array([np.cos(a), np.sin(a), 0.0]) + cc_v/2.0,
    ch_r*np.array([np.cos(a), -np.sin(a), 0.0]) + cc_v/2.0,
    ch_r*np.array([np.cos(np.pi + a), np.sin(np.pi + a), 0.0]) - cc_v/2.0,
    ch_r*np.array([np.cos(np.pi - a), np.sin(np.pi - a), 0.0]) - cc_v/2.0
]
h_positions = np.array(h_positions_list)

# plt.scatter(h_positions.T[0], h_positions.T[1])
# plt.scatter(c_positions.T[0], c_positions.T[1])
# plt.show()
# plt.close()

carbon_dict = get_orbitals_dict_from_file('../data/7p8e-5gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p2e-3gaussians.json')
                 ['1s']
                 }

dat_c_list = [OrbitalPrimitivesBuilder(position=c_positions[i],
                                       orbitals_dict=carbon_dict)
              for i in range(c_positions.shape[0])]
dat_h_list = [OrbitalPrimitivesBuilder(position=h_positions[i],
                                       orbitals_dict=hydrogen_dict)
              for i in range(h_positions.shape[0])]
data = dat_c_list[0] + dat_c_list[1] \
       + dat_h_list[0] + dat_h_list[1] + dat_h_list[2] + dat_h_list[3]
data.set_number_of_orbitals(8)

t2 = perf_counter_ns()
system = ClosedShellSystem(primitives=data.primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=[[c_positions[0], 6.0],
                                           [c_positions[1], 6.0],
                                           [h_positions[0], 1.0],
                                           [h_positions[1], 1.0],
                                           [h_positions[2], 1.0],
                                           [h_positions[3], 1.0]],
                           use_ext=True
                           )
t3 = perf_counter_ns()
system.solve(20)
print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t4 = perf_counter_ns()

s = np.linspace(-5.0, 5.0, 100)
x, y = np.meshgrid(s, s)
u = np.zeros(x.shape)
for k, orbital in enumerate(system.orbitals):
    o = sum([orbital[i]*data.primitives()[i]([x, y, 0.0])
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

print(f'Construction time: {(t3 - t2)/1000000000.0}s')
print(f'Iteration time: {(t4 - t3)/1000000000.0}s')
print(f'Total time taken: {(t4 - t1)/1000000000.0}s')
