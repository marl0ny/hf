import numpy as np
from gaussian_basis import ClosedShellSystem
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder


def get_geometry(oh_length: float) -> 'dict[str, np.ndarray]':
    return {'O': np.array([0.0, 0.0, 0.0]),
            'H': np.array([oh_length, 0.0, 0.0])}


if __name__ == '__main__':
    from time import perf_counter_ns
    import matplotlib.pyplot as plt

    t1 = perf_counter_ns()
    hydrogen_dict = {'1s': get_orbitals_dict_from_file(
        '../data/1p1e-4gaussians.json')['1s']}
    oxygen_dict = get_orbitals_dict_from_file(
        '../data/10p10e-6gaussians.json')

    energies = []
    positions = []
    n_points = 30

    for i in range(n_points):
        geom = get_geometry(1.2 + i/n_points)
        positions.append(1.2 + i/n_points)
        data_h = OrbitalPrimitivesBuilder(position=geom['H'],
                                          orbitals_dict=hydrogen_dict)
        data_o = OrbitalPrimitivesBuilder(position=geom['O'],
                                          orbitals_dict=oxygen_dict)
        data = data_o + data_h
        data.set_number_of_orbitals(5)
        system = ClosedShellSystem(primitives=data.primitives(),
                                   orbitals=data.orbitals(),
                                   nuclear_config=[[geom['H'], 1.0],
                                                   [geom['O'], 8.0]],
                                   use_ext=True)
        system.solve(20)
        print(system.energies)
        print(total_energy := system.get_total_energy())
        energies.append(total_energy)

    t2 = perf_counter_ns()
    print(f'Time taken: {(t2 - t1)/1000000000.0}s')

    # #1.92 a.u., -75.2 a.u
    plt.title('Hartree-Fock Closed Shell Energy for HF')
    plt.plot(positions, energies)
    plt.xlabel('r (a.u.)')
    plt.ylabel('E (a.u.)')
    plt.show()
    plt.close()
