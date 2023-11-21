from time import perf_counter_ns
import numpy as np
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
import matplotlib.pyplot as plt


t1 = perf_counter_ns()

oxygen_dict = get_orbitals_dict_from_file(
        '../data/8p8e-5gaussians.json')

energies = []
positions = []
for i in range(10):
    oxygen_pos1 = np.array([0.0, 0.0, 0.0])
    oxygen_pos2 = np.array([2.1 + i*0.03, 0.0, 0.0])
    positions.append(oxygen_pos2[0])

    dat_oxygen_pos1 = OrbitalPrimitivesBuilder(position=oxygen_pos1,
                                            orbitals_dict=oxygen_dict)
    dat_oxygen_pos2 = OrbitalPrimitivesBuilder(position=oxygen_pos2,
                                            orbitals_dict=oxygen_dict)
    data = dat_oxygen_pos1 + dat_oxygen_pos2
    data.set_number_of_orbitals(8)

    system = ClosedShellSystem(primitives=data.primitives(),
                            orbitals=data.orbitals(),
                            nuclear_config=[[oxygen_pos1, 8.0],
                                            [oxygen_pos2, 8.0]],
                            use_ext=True)
    system.solve(20)
    print(system.energies)
    print(total_energy := system.get_total_energy())
    energies.append(total_energy)

    # s = np.linspace(-5.0, 5.0, 100)
    # x, y = np.meshgrid(s, s)
    # u = np.zeros(x.shape)
    # for k, orbital in enumerate(system.orbitals):
    #     o = sum([orbital[i] * data.primitives()[i]([x, y, 0.0])
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
    #     u += o ** 2
    #
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_title(r'$|\psi(x, y, z=0)|$')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel(r'$|\psi(x, y)|$')
    # ax.plot_surface(x, y, np.sqrt(u))
    # plt.show()
    # plt.close()

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

print(positions[np.argmin(energies)])
print(np.min(energies))
plt.plot(np.array(positions), np.array(energies))
plt.title('Closed Shell Hartree-Fock Energy Curve for ' + r'$O_2$')
plt.ylabel('Energy (a.u.)')
plt.xlabel('Bond length (a.u.)')
plt.show()
plt.close()