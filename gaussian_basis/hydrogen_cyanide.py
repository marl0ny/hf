from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


t1 = perf_counter_ns()

# This takes quite a while to finish
# Got 1.8 a.u. for C-N, 2.0 a.u. for C-H, and -92.5 a.u. for total energy.
hydrogen_dict = {'1s': get_orbitals_dict_from_file(
                 '../data/1p1e-3gaussians.json')['1s']}
carbon_dict = get_orbitals_dict_from_file(
    '../data/10p10e-5gaussians.json')
# unrestricted_nitrogen_dict \
#     = get_orbitals_dict_from_file('../data/7p7e-6gaussians.json')
# nitrogen_dict = {
#     '1s': unrestricted_nitrogen_dict['1s+'],
#     '2s': unrestricted_nitrogen_dict['2s+'],
#     '2p': unrestricted_nitrogen_dict['2p+'],
# }
nitrogen_dict = get_orbitals_dict_from_file(
    '../data/7p8e-4gaussians.json')
nitrogen_dict['2s'] \
    = get_orbitals_dict_from_file('../data/7p7e-6gaussians.json')['2s+']

N_CN = 5
N_CH = 5
cn_distances = [1.9 + 0.3*k/N_CN for k in range(N_CN)]
ch_distances = [1.9 + 0.3*k/N_CH for k in range(N_CH)]
cn_ch_i, cn_ch_j = np.meshgrid(cn_distances, ch_distances, indexing='ij')
energies = np.zeros(cn_ch_i.shape)

for i, cn_distance in enumerate(cn_distances):
    for j, ch_distance in enumerate(ch_distances):
        nitrogen_pos = np.array([-cn_distance, 0.0, 0.0])
        carbon_pos = np.array([0.0, 0.0, 0.0])
        hydrogen_pos = np.array([ch_distance, 0.0, 0.0])
        dat_nitrogen = OrbitalPrimitivesBuilder(position=nitrogen_pos,
                                                orbitals_dict=nitrogen_dict)
        dat_carbon = OrbitalPrimitivesBuilder(position=carbon_pos,
                                              orbitals_dict=carbon_dict)
        dat_hydrogen = OrbitalPrimitivesBuilder(position=hydrogen_pos,
                                                orbitals_dict=hydrogen_dict)
        data = dat_nitrogen + dat_carbon + dat_hydrogen
        data.set_number_of_orbitals(7)
        system = ClosedShellSystem(primitives=data.primitives(),
                                   orbitals=data.orbitals(),
                                   nuclear_config=[[nitrogen_pos, 7.0],
                                                   [carbon_pos, 6.0],
                                                   [hydrogen_pos, 1.0]],
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
print(f'C-N bond distance (a.u.): {cn_distances[indices[0]]}')
print(f'C-H bond distance (a.u.): {ch_distances[indices[1]]}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$HCN$' + ' Potential Energy Surface')
ax.set_xlabel('C-N bond distance (a.u.)')
ax.set_ylabel('C-H bond distance (a.u.)')
ax.set_zlabel('Energy (a.u.)')
ax.plot_surface(cn_ch_i, cn_ch_j, energies)
plt.show()
plt.close()
