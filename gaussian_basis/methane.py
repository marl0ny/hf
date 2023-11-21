from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder

t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }
carbon_pos = np.array([0.0, 0.0, 0.0])

N_ANGLES = 5
N_DISTANCES = 10
angles = [np.pi/4.0 + k/N_ANGLES*np.pi/2.0 for k in range(N_ANGLES)]
distances = [1.75 + k/N_DISTANCES for k in range(N_DISTANCES)]
angle_distances_i, angle_distances_j\
 = np.meshgrid(angles, distances, indexing='ij')
energies = np.zeros(angle_distances_i.shape)

for i, theta in enumerate(angles):
    for j, r in enumerate(distances):
        phi = 2.0*np.pi/3.0
        h_positions = [r*np.array([0.0, 0.0, 1.0]),
                       r*np.array([np.cos(0.0*phi)*np.sin(theta),
                                   np.sin(0.0*phi)*np.sin(theta),
                                   np.cos(theta)]),
                       r*np.array([np.cos(phi)*np.sin(theta),
                                   np.sin(phi)*np.sin(theta),
                                   np.cos(theta)]),
                       r*np.array([np.cos(2.0*phi)*np.sin(theta),
                                   np.sin(2.0*phi)*np.sin(theta),
                                   np.cos(theta)])]
        print(theta, np.arccos(np.dot(h_positions[1], h_positions[2])/r**2))

        dat_carbon = OrbitalPrimitivesBuilder(position=carbon_pos,
                                              orbitals_dict=carbon_dict)
        dat_h_list = [OrbitalPrimitivesBuilder(position=h_positions[i],
                                               orbitals_dict=hydrogen_dict)
                      for i in range(len(h_positions))]
        data = dat_carbon + (dat_h_list[0] + dat_h_list[1]
                             + dat_h_list[2] + dat_h_list[3])
        data.set_number_of_orbitals(5)

        system = ClosedShellSystem(primitives=data.primitives(),
                                   orbitals=data.orbitals(),
                                   nuclear_config=[[carbon_pos, 6.0],
                                                   [h_positions[0], 1.0],
                                                   [h_positions[1], 1.0],
                                                   [h_positions[2], 1.0],
                                                   [h_positions[3], 1.0]],
                                   use_ext=True)
        system.solve(10)
        print(system.energies)
        print(system.get_nuclear_configuration_energy())
        print(total_energy := system.get_total_energy())
        energies[i, j] = total_energy

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

energies_flat = np.reshape(energies,
                           (energies.shape[0]*energies.shape[1]))
indices = (np.argmin(energies_flat) // energies.shape[1],
           np.argmin(energies_flat) % energies.shape[1])
print('Energy: ', np.amin(energies_flat))
print(f'angle: {angles[indices[0]]}')
print(f'C-H bond distance: {distances[indices[1]]}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$CH_4$' + ' Potential Energy')
ax.set_xlabel('angle (radians)')
ax.set_ylabel('C-H bond distance (a.u.)')
ax.set_zlabel('Energy (a.u.)')
ax.plot_surface(angle_distances_i,
                angle_distances_j,
                energies)
plt.show()
plt.close()
