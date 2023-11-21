from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

hydrogen_dict = {'1s': get_orbitals_dict_from_file(
                 '../data/1p1e-3gaussians.json')['1s']}
carbon_dict = get_orbitals_dict_from_file(
    '../data/7p8e-4gaussians.json')

N_CC = 10
N_CH = 10
cc_distances = [1.0 + 2.0*k/N_CC for k in range(N_CC)]
ch_distances = [1.0 + 2.0*k/N_CH for k in range(N_CH)]
cc_ch_i, cc_ch_j = np.meshgrid(cc_distances, ch_distances, indexing='ij')
energies = np.zeros(cc_ch_i.shape)

for i, cc_distance in enumerate(cc_distances):
    for j, ch_distance in enumerate(ch_distances):
        carbon_pos1 = np.array([0.0, 0.0, 0.0])
        carbon_pos2 = np.array([cc_distance, 0.0, 0.0])
        hydrogen_pos1 = np.array([carbon_pos1[0] - ch_distance, 0.0, 0.0])
        hydrogen_pos2 = np.array([carbon_pos2[0] + ch_distance, 0.0, 0.0])
        positions = np.array([carbon_pos1, carbon_pos2,
                              hydrogen_pos1, hydrogen_pos2]).T
        # plt.scatter(positions[0], positions[1])
        # plt.show()
        # plt.close()
        dat_c_pos1 = OrbitalPrimitivesBuilder(position=carbon_pos1,
                                              orbitals_dict=carbon_dict)
        dat_c_pos2 = OrbitalPrimitivesBuilder(position=carbon_pos2,
                                              orbitals_dict=carbon_dict)
        dat_h_pos1 = OrbitalPrimitivesBuilder(position=hydrogen_pos1,
                                              orbitals_dict=hydrogen_dict)
        dat_h_pos2 = OrbitalPrimitivesBuilder(position=hydrogen_pos2,
                                              orbitals_dict=hydrogen_dict)
        data = dat_c_pos1 + dat_c_pos2 + dat_h_pos1 + dat_h_pos2
        data.set_number_of_orbitals(7)
        system = ClosedShellSystem(primitives=data.primitives(),
                                   orbitals=data.orbitals(),
                                   nuclear_config=[[hydrogen_pos1, 1.0],
                                                   [hydrogen_pos2, 1.0],
                                                   [carbon_pos1, 6.0],
                                                   [carbon_pos2, 6.0]],
                                   use_ext=True)
        system.solve(10)
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
print(f'C-C bond distance (a.u.): {cc_distances[indices[0]]}')
print(f'C-H bond distance (a.u.): {ch_distances[indices[1]]}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$C_2H_2$' + ' Potential Energy Surface')
ax.set_xlabel('C-C bond distance (a.u.)')
ax.set_ylabel('C-H bond distance (a.u.)')
ax.set_zlabel('Energy (a.u.)')
ax.plot_surface(cc_ch_i, cc_ch_j, energies)
plt.show()
plt.close()
