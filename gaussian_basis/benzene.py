from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis.molecular_geometry import MolecularGeometry


t1 = perf_counter_ns()

# carbon_dict = get_orbitals_dict_from_file('../data/7p8e-4gaussians.json')
carbon_dict_ = get_orbitals_dict_from_file('../data/7p7e-4gaussians.json')
carbon_dict = {
    '1s': carbon_dict_['1s+'],
    '2s': carbon_dict_['2s+'],
    '2p': carbon_dict_['2p+'],
}
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }

geom = MolecularGeometry()
angle = np.pi/3.0
for i in range(6):
    geom.add_atom('C', 2.6*np.array([np.cos(i*angle), np.sin(i*angle), 0.0]))
    geom.add_atom('H', 4.6*np.array([np.cos(i*angle), np.sin(i*angle), 0.0]))

plt.scatter(geom['H'].T[0], geom['H'].T[1])
plt.scatter(geom['C'].T[0], geom['C'].T[1])
plt.show()
plt.close()

dat_h_list = [OrbitalPrimitivesBuilder(position=geom['H'][i],
                                       orbitals_dict=hydrogen_dict)
              for i in range(len(geom['H']))]
dat_c_list = [OrbitalPrimitivesBuilder(position=geom['C'][i],
                                       orbitals_dict=carbon_dict)
              for i in range(len(geom['C']))]
data = sum(dat_c_list) + sum(dat_h_list)
data.set_number_of_orbitals(21)

nuclear_config = [[r, 6.0] for r in geom['C']] +\
                 [[r, 1.0] for r in geom['H']]
system = ClosedShellSystem(primitives=data.primitives(),
                           orbitals=data.orbitals(),
                           nuclear_config=nuclear_config,
                           use_ext=True)
system.solve(25)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

