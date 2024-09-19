"""Compute the Hartree-Fock ground state energy of Methane.
Iterate over a predefined list of H-C-H angular separations and
C-H distances, then choose the configuration that gives the lowest
ground state energy. A 3D plot of the probability density for 
this geometry is then produced. The potential energy surface for
the given configurations is plotted as well.
"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import ClosedShellSystemFromPrimitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis import OrbitalPrimitivesBuilder
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem
from gaussian_basis.molecular_geometry import MolecularGeometry

t1 = perf_counter_ns()

# Using primitives
# Time taken: 132.16239725s
# Energy:  -40.00116703206716
# angle: 2.0420352248333655
# C-H bond distance: 2.05

# carbon_dict = get_orbitals_dict_from_file(
#   '../data/10p10e_1s5_2s2111_2p2111.json')
# carbon_dict = get_orbitals_dict_from_file(
#   '../data/7p8e_1s3111_2s3111_2p111111.json')
carbon_dict = get_orbitals_dict_from_file(
    '../data/10p10e_1s111111_2s111111_2p111111.json')
hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s4.json')

# carbon_dict = get_orbitals_dict_from_file('../data/10p10e-5gaussians.json')
# hydrogen_dict = {'1s':
#                  get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')
#                  ['1s'],
#                  # '2s':
#                  # get_orbitals_dict_from_file('../data/1p1e-3gaussians.json')['2s'],
#                  }

carbon_pos = np.array([0.0, 0.0, 0.0])

N_ANGLES = 5
N_DISTANCES = 10
angles = [np.pi/4.0 + k/N_ANGLES*np.pi/2.0 for k in range(N_ANGLES)]
distances = [1.75 + k/N_DISTANCES for k in range(N_DISTANCES)]
angle_distances_i, angle_distances_j\
 = np.meshgrid(angles, distances, indexing='ij')
energies = np.zeros(angle_distances_i.shape)

for i, theta in enumerate(angles):
    for j, r in enumerate(distances):
        phi = 2.0*np.pi/3.0
        h_positions = [r*np.array([0.0, 0.0, 1.0]),
                       r*np.array([np.cos(0.0*phi)*np.sin(theta),
                                   np.sin(0.0*phi)*np.sin(theta),
                                   np.cos(theta)]),
                       r*np.array([np.cos(phi)*np.sin(theta),
                                   np.sin(phi)*np.sin(theta),
                                   np.cos(theta)]),
                       r*np.array([np.cos(2.0*phi)*np.sin(theta),
                                   np.sin(2.0*phi)*np.sin(theta),
                                   np.cos(theta)])]
        print(theta, np.arccos(np.dot(h_positions[1], h_positions[2])/r**2))

        geom = MolecularGeometry()
        geom.add_atom('C', carbon_pos)
        for h_position in h_positions:
            geom.add_atom('H', h_position)
        orbitals = get_orbitals_from_geometry(
            geom, {'H': hydrogen_dict, 'C': carbon_dict})
        system = ClosedShellSystem(
            5, orbitals, geom.get_nuclear_configuration()
        )
        system.solve(10)
        
        # dat_carbon = OrbitalPrimitivesBuilder(position=carbon_pos,
        #                                       orbitals_dict=carbon_dict)
        # dat_h_list = [OrbitalPrimitivesBuilder(position=h_positions[i],
        #                                        orbitals_dict=hydrogen_dict)
        #               for i in range(len(h_positions))]
        # data = dat_carbon + (dat_h_list[0] + dat_h_list[1]
        #                      + dat_h_list[2] + dat_h_list[3])
        # data.set_number_of_orbitals(5)

        # system = ClosedShellSystemFromPrimitives(primitives=data.get_primitives(),
        #                            orbitals=data.orbitals(),
        #                            nuclear_config=[[carbon_pos, 6.0],
        #                                            [h_positions[0], 1.0],
        #                                            [h_positions[1], 1.0],
        #                                            [h_positions[2], 1.0],
        #                                            [h_positions[3], 1.0]],
        #                            use_ext=True)
        # system.solve(10)

        print(system.energies)
        print(system.get_nuclear_configuration_energy())
        print(total_energy := system.get_total_energy())
        energies[i, j] = total_energy

t2 = perf_counter_ns()
print(f'Time taken: {(t2 - t1)/1000000000.0}s')

energies_flat = np.reshape(energies,
                           (energies.shape[0]*energies.shape[1]))
indices = (np.argmin(energies_flat) // energies.shape[1],
           np.argmin(energies_flat) % energies.shape[1])
print('Energy: ', np.amin(energies_flat))
print(f'angle: {angles[indices[0]]}')
print(f'C-H bond distance: {distances[indices[1]]}')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_title(r'$CH_4$' + ' Potential Energy')
ax.set_xlabel('angle (radians)')
ax.set_ylabel('C-H bond distance (a.u.)')
ax.set_zlabel('Energy (a.u.)')
ax.plot_surface(angle_distances_i,
                angle_distances_j,
                energies)
plt.show()
plt.close()

def plotly_render():
    import plotly.graph_objects as go
    geom = MolecularGeometry()
    geom.add_atom('C', carbon_pos)
    for h_position in h_positions:
        geom.add_atom('H', h_position)
    orbitals = get_orbitals_from_geometry(
        geom, {'H': hydrogen_dict, 'C': carbon_dict})
    system = ClosedShellSystem(
        5, orbitals, geom.get_nuclear_configuration()
    )
    system.solve(10)
    s_x = np.linspace(-2.2, 3.0, 128)
    s_y = np.linspace(-3.0, 3.0, 128)
    s_z = np.linspace(-2.1, 3.1, 128)
    ds3 = (s_z[1] - s_z[0])*(s_y[1] - s_y[0])*(s_x[1] - s_x[0])
    x, y, z = np.meshgrid(s_x, s_y, s_z)
    u = np.zeros(x.shape)
    primitive_func_list = orbitals.get_primitives()
    for orbital in system.orbitals:
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