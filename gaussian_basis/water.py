from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

hydrogen_dict = {'1s': get_orbitals_dict_from_file(
                 '../data/1p1e-3gaussians.json')['1s']}
oxygen_dict = get_orbitals_dict_from_file(
    '../data/10p10e-5gaussians.json')
oxygen_pos = np.array([0.0, 0.0, 0.0])

N_ANGLES = 11
N_DISTANCES = 10
angles = [np.pi/3.0 + k/N_ANGLES*np.pi for k in range(N_ANGLES)]
distances = [1.2 + k/N_DISTANCES for k in range(N_DISTANCES)]
angle_distances_i, angle_distances_j\
 = np.meshgrid(angles, distances, indexing='ij')
energies = np.zeros(angle_distances_i.shape)

for i, angle in enumerate(angles):
    for j, d in enumerate(distances):
        # angle = np.pi*0.58
        # angle = np.pi
        # angle = np.pi*0.5
        h_pos1 = d*np.array([np.cos(np.pi/4.0 + angle/2.0),
                            np.sin(np.pi/4.0 + angle/2.0), 0.0])
        h_pos2 = d*np.array([np.cos(np.pi/4.0 - angle/2.0),
                            np.sin(np.pi/4.0 - angle/2.0), 0.0])
        positions = np.array([oxygen_pos, h_pos1, h_pos2]).T
        # plt.scatter(positions[0], positions[1])
        # plt.show()
        # plt.close()

        dat_h_pos1 = OrbitalPrimitivesBuilder(position=h_pos1,
                                            orbitals_dict=hydrogen_dict)
        dat_h_pos2 = OrbitalPrimitivesBuilder(position=h_pos2,
                                            orbitals_dict=hydrogen_dict)
        dat_oxygen = OrbitalPrimitivesBuilder(position=oxygen_pos,
                                            orbitals_dict=oxygen_dict)
        data = dat_oxygen + dat_h_pos1 + dat_h_pos2
        # data.set_number_of_orbitals(5)

        system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                                   orbitals=data.orbitals()[:5],
                                   nuclear_config=[[h_pos1, 1.0],
                                                   [h_pos2, 1.0],
                                                   [oxygen_pos, 8.0]],
                                   use_ext=True)

        system.solve(15)
        print(system.energies)
        print(total_energy := system.get_total_energy())
        energies[i, j] = total_energy

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')


energies_flat = np.reshape(energies,
                           (energies.shape[0]*energies.shape[1]))
indices = (np.argmin(energies_flat) // energies.shape[1],
           np.argmin(energies_flat) % energies.shape[1])
print('Energy: ', np.amin(energies_flat))
print(f'H-O-H angle: {angles[indices[0]]}')
print(f'H-O distance: {distances[indices[1]]}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$H_2O$' + ' Potential Energy Curve')
ax.set_xlabel('H-O-H angle (radians)')
ax.set_ylabel('O-H bond distance (a.u.)')
ax.set_zlabel('Energy (a.u.)')
ax.plot_surface(angle_distances_i,
                angle_distances_j,
                energies)
plt.show()
plt.close()

# s = np.linspace(-4.0, 4.0, 100)
# x, y = np.meshgrid(s, s)
# u = np.zeros(x.shape)
# for k, orbital in enumerate(system.orbitals):
#     o = sum([orbital[i]*data.get_primitives()[i]([x, y, 0.0])
#              for i in range(len(orbital))])
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     ax.set_title(r'$\phi_'
#                  + f'{k}(x, y, z=0)$'
#                    f'\nE = {system.energies[k]} a.u.')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel(r'$\phi_' + f'{k}(x, y)$')
#     ax.plot_surface(x, y, o)
#     plt.show()
#     plt.close()
#     u += o**2
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.set_title(r'$|\psi(x, y, z=0)|$')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel(r'$|\psi(x, y)|$')
# ax.plot_surface(x, y, np.sqrt(u))
# plt.show()
# plt.close()

