"""Compute the lowest Hartree-Fock energy of Benzene,
then plot each orbital and the total probability density. 
This uses the experimental geometry of the molecule, 
which is found in a NIST database:

    Experimental data for C6H6 (Benzene).
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import MolecularGeometry

from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem


t1 = perf_counter_ns()

# carbon_dict = get_orbitals_dict_from_file('../data/7p8e_1s33_2s33_2p33.json')
carbon_dict = get_orbitals_dict_from_file('../data/7p8e_1s32_2s32_2p32.json')
hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s22.json')
# hydrogen_dict = get_orbitals_dict_from_file(
#     '../data/1p1e_1s4.json')


geom = MolecularGeometry()
angle = np.pi/3.0
for i in range(6):
    geom.add_atom('C', 2.6*np.array([np.cos(i*angle), np.sin(i*angle), 0.0]))
    geom.add_atom('H', 4.6*np.array([np.cos(i*angle), np.sin(i*angle), 0.0]))

plt.title('Benzene Nuclei Locations')
plt.scatter(geom['H'].T[0], geom['H'].T[1], label='H')
plt.scatter(geom['C'].T[0], geom['C'].T[1], label='C')
plt.legend()
plt.show()
plt.close()

orbitals = get_orbitals_from_geometry(geom, {'H': hydrogen_dict, 'C': carbon_dict})
system = ClosedShellSystem(21, orbitals, geom.get_nuclear_configuration())

t2 = perf_counter_ns()

system.solve(200)

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())
t3 = perf_counter_ns()

s = np.linspace(-5.5, 5.5, 220)
x, y = np.meshgrid(s, s)
u = np.zeros(x.shape)
primitive_func_list = orbitals.get_primitives()
for k, orbital in enumerate(system.orbitals):

    o = sum([orbital[i]*sum([p([x, y, 0.0]) for p in primitive_func_list[i]])
             for i in range(len(orbital))])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_title(r'$\phi_'
                 + r"{" + str(k) + r"}(x, y, z=0)$"
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

print(f'Construction time: {(t2 - t1)/1000000000.0}s')
print(f'Total time taken: {(t3 - t1)/1000000000.0}s')


def plotly_render():
    import plotly.graph_objects as go
    orbitals = get_orbitals_from_geometry(
        geom, {'H': hydrogen_dict, 'C': carbon_dict})
    s_xy = np.linspace(-5.5, 5.5, 128)
    s_z = np.linspace(-1.75, 1.75, 130)
    ds3 = (s_z[1] - s_z[0])*(s_xy[1] - s_xy[0])**2
    x, y, z = np.meshgrid(s_xy, s_xy, s_z)
    u = np.zeros(x.shape)
    primitive_func_list = orbitals.get_primitives()
    for k, orbital in enumerate(system.orbitals):
        o = sum([orbital[i]*sum([p([x, y, z]) 
                                 for p in primitive_func_list[i]])
                for i in range(len(orbital))])
        o /= np.sqrt(np.sum(np.conj(o)*o)*ds3)
        u += o**2    
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=2.0*u.flatten(), isomin=0.1, isomax=2.0,
        opacity=0.1, surface_count=20,  
    ))
    fig.show()


plotly_render()