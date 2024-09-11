from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import make_ch2

# -113.5
carbon_dict = get_orbitals_dict_from_file('../data/7p8e-6gaussians.json')
oxygen_dict = get_orbitals_dict_from_file('../data/10p10e-6gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }

t1 = perf_counter_ns()

ch_distances = [1.75 + 0.25*i/3.0 for i in range(3)]
co_distances = [1.75 + 0.25*i/3.0 for i in range(3)]
hch_angles = [0.5*np.pi + i/3.0*(0.2*np.pi) for i in range(3)]
min_energy = 0.0
co_dist_at_min_energy = 100.0
ch_dist_at_min_energy = 100.0
hch_angle_at_min_energy = 2.0*np.pi
for i, ch_distance in enumerate(ch_distances):
    for j, co_distance in enumerate(co_distances):
        for k, hch_angle in enumerate(hch_angles):
            geom = make_ch2(ch_distance, ch_distance,
                            hch_angle,
                            np.array([0.0, 0.0, -1.0]),
                            np.array([1.0, 0.0, 0.0]))
            geom.add_atom('O', np.array([0.0, 0.0, co_distance]))

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2])
            # ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2])
            # ax.scatter(geom['O'].T[0], geom['O'].T[1], geom['O'].T[2])
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            # plt.show()

            dat_h_list = [OrbitalPrimitivesBuilder(position=geom['H'][i],
                                                   orbitals_dict=hydrogen_dict)
                          for i in range(len(geom['H']))]
            dat_c_list = [OrbitalPrimitivesBuilder(position=geom['C'][i],
                                                   orbitals_dict=carbon_dict)
                          for i in range(len(geom['C']))]
            dat_o_list = [OrbitalPrimitivesBuilder(position=geom['O'][i],
                                                   orbitals_dict=oxygen_dict)
                          for i in range(len(geom['O']))]

            data = sum(dat_c_list) + sum(dat_h_list) + sum(dat_o_list)
            data.set_number_of_orbitals(8)
            nuclear_config = [[r, 6.0] for r in geom['C']] +\
                             [[r, 8.0] for r in geom['O']] +\
                             [[r, 1.0] for r in geom['H']]
            system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
                                       orbitals=data.orbitals(),
                                       nuclear_config=nuclear_config,
                                       use_ext=True)
            system.solve(10)
            print(system.get_nuclear_configuration_energy())
            print(total_energy := system.get_total_energy())
            if min_energy > total_energy:
                min_energy = total_energy
                ch_dist_at_min_energy = ch_distance
                co_dist_at_min_energy = co_distance
                hch_angle_at_min_energy = hch_angle

print(min_energy)
print('C-H bond distance: ', ch_dist_at_min_energy, ' a.u.')
print('C-O bond distance: ', co_dist_at_min_energy, ' a.u.')
print('H-C-H angle:', hch_angle_at_min_energy/np.pi, 'pi radians')

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')
