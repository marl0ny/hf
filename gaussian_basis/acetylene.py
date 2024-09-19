"""Compute the Hartree-Fock ground state energy of Acetylene.
Iterate over a predefined list of C-C and C-H distances,
then choose the configuration that gives the lowest
ground state energy. A 3D plot of the probability density for 
this geometry is then produced.
"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem
from gaussian_basis.molecular_geometry import MolecularGeometry


t1 = perf_counter_ns()

hydrogen_dict = get_orbitals_dict_from_file(
    '../data/1p1e_1s21_2s21_2p21.json')
carbon_dict \
    = get_orbitals_dict_from_file(
        '../data/10p10e_1s111111_2s111111_2p111111.json')

N_CC = 10
N_CH = 10
cc_distances = [1.0 + 2.0*k/N_CC for k in range(N_CC)]
ch_distances = [1.0 + 2.0*k/N_CH for k in range(N_CH)]
cc_ch_i, cc_ch_j = np.meshgrid(cc_distances, ch_distances, indexing='ij')
energies = np.zeros(cc_ch_i.shape)

def get_acetylene_geometry(
        cc_distance: float, ch_distance: float) -> MolecularGeometry:
    carbon_pos1 = np.array([0.0, 0.0, 0.0])
    carbon_pos2 = np.array([cc_distance, 0.0, 0.0])
    hydrogen_pos1 = np.array([carbon_pos1[0] - ch_distance, 0.0, 0.0])
    hydrogen_pos2 = np.array([carbon_pos2[0] + ch_distance, 0.0, 0.0])
    geom = MolecularGeometry()
    geom.add_atom('C', carbon_pos1)
    geom.add_atom('C', carbon_pos2)
    geom.add_atom('H', hydrogen_pos1)
    geom.add_atom('H', hydrogen_pos2)
    return geom
    
# When using the one primitives per basis function implementation:
# Time taken: 313.71501s
# Energy:  -76.07563665587556
# C-C bond distance (a.u.): 2.2
# C-H bond distance (a.u.): 1.8

for i, cc_distance in enumerate(cc_distances):
    for j, ch_distance in enumerate(ch_distances):
        geom = get_acetylene_geometry(cc_distance, ch_distance)
        orbitals = get_orbitals_from_geometry(
            geom, {'H': hydrogen_dict, 'C': carbon_dict}
        )
        system = ClosedShellSystem(
            7, orbitals, geom.get_nuclear_configuration()
        )
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

def plotly_render():
    import plotly.graph_objects as go
    geom = get_acetylene_geometry(
        cc_distances[indices[0]], ch_distances[indices[1]])
    orbitals = get_orbitals_from_geometry(
        geom, {'H': hydrogen_dict, 'C': carbon_dict}
    )
    system = ClosedShellSystem(
        7, orbitals, geom.get_nuclear_configuration()
    )
    system.solve(10)
    s_x = np.linspace(-3.0, 5.0, 128)
    s_yz = np.linspace(-2.0, 2.0, 128)
    ds3 = (s_x[1] - s_x[0])*(s_yz[1] - s_yz[0])**2
    x, y, z = np.meshgrid(s_x, s_yz, s_yz)
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