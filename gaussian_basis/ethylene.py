"""Compute the lowest Hartree-Fock energy of Ethylene.
This uses the experimental geometry of the molecule, 
which is found in a NIST database:

    Experimental data for C2H4 (Ethylene).
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import get_build_from_primitives
from gaussian_basis import make_system_from_geometry_and_build
from gaussian_basis.molecular_geometry import MolecularGeometry
from gaussian_basis import get_orbitals_dict_from_file


t1 = perf_counter_ns()

carbon_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
hydrogen_dict = {'1s':
                 get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
                 ['1s']
                 }
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

geom = MolecularGeometry()
for i in range(len(h_positions)):
    geom.add_atom('H', h_positions[i])
for j in range(len(c_positions)):
    geom.add_atom('C', c_positions[j])


data = get_build_from_primitives(
    geom, {'C': hydrogen_dict, 'H': carbon_dict}, 8)
system = make_system_from_geometry_and_build(geom, data)

t2 = perf_counter_ns()
system.solve(20)
t3 = perf_counter_ns()

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t4 = perf_counter_ns()

s = np.linspace(-5.0, 5.0, 100)
x, y = np.meshgrid(s, s)
u = np.zeros(x.shape)
for k, orbital in enumerate(system.orbitals):
    o = sum([orbital[i]*data.get_primitives()[i]([x, y, 0.0])
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
