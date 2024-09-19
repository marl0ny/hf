"""Compute the Hartree-Fock ground state energy of water.
Iterate over a predefined list of H-O-H angular separations and
O-H distances, then choose the configuration that gives the lowest
ground state energy. A 3D plot of the probability density for 
this geometry is then produced. The potential energy surface for
the given configurations is plotted as well.
"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem
from gaussian_basis.molecular_geometry import MolecularGeometry


t1 = perf_counter_ns()

oxygen_dict = get_orbitals_dict_from_file(
    '../data/10p10e_1s2111_2s2111_2p2111.json')
hydrogen_dict = get_orbitals_dict_from_file(
    '../data/1p1e_1s21_2s21_2p21.json')

oxygen_pos = np.array([0.0, 0.0, 0.0])

N_ANGLES = 11
N_DISTANCES = 10
angles = [np.pi/3.0 + k/N_ANGLES*np.pi for k in range(N_ANGLES)]
distances = [1.2 + k/N_DISTANCES for k in range(N_DISTANCES)]
angle_distances_i, angle_distances_j\
 = np.meshgrid(angles, distances, indexing='ij')
energies = np.zeros(angle_distances_i.shape)

def get_h2o_geometry(angle: float, d: float) -> MolecularGeometry:
    """Get molecular geometry of an H2O molecule.

    @param angle: H-O-H angular separation in radians
    @param d: Distance of O-H in Hartrees 
    (the O-H bonds have equal distances)

    """
    h_pos1 = d*np.array([np.cos(np.pi/4.0 + angle/2.0),
                        np.sin(np.pi/4.0 + angle/2.0), 0.0])
    h_pos2 = d*np.array([np.cos(np.pi/4.0 - angle/2.0),
                        np.sin(np.pi/4.0 - angle/2.0), 0.0])
    geom = MolecularGeometry()
    geom.add_atom('H', h_pos1)
    geom.add_atom('H', h_pos2)
    geom.add_atom('O', oxygen_pos)
    return geom

for i, angle in enumerate(angles):
    for j, d in enumerate(distances):
        geom = get_h2o_geometry(angle, d)
        orbitals = get_orbitals_from_geometry(
            geom, {'H': hydrogen_dict, 'O': oxygen_dict}
        )
        system = ClosedShellSystem(
            5, orbitals, geom.get_nuclear_configuration())
        system.solve(15)

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
print(f'H-O-H angle: {angles[indices[0]]}')
print(f'H-O distance: {distances[indices[1]]}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$H_2O$' + ' Potential Energy Curve')
ax.set_xlabel('H-O-H angle (radians)')
ax.set_ylabel('O-H bond distance (a.u.)')
ax.set_zlabel('Energy (a.u.)')
ax.plot_surface(angle_distances_i,
                angle_distances_j,
                energies)
plt.show()
plt.close()


def plotly_render():
    import plotly.graph_objects as go
    geom = get_h2o_geometry(angles[indices[0]], distances[indices[1]])
    orbitals = get_orbitals_from_geometry(
        geom, {'H': hydrogen_dict, 'O': oxygen_dict})
    system = ClosedShellSystem(5, orbitals, geom.get_nuclear_configuration())
    system.solve(15)
    s_xy = np.linspace(-3.5, 3.5, 128)
    s_z = np.linspace(-1.5, 1.5, 128)
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
