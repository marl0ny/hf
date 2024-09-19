"""Compute the lowest Hartree-Fock energy of Acetone,
and plot its probability density. This uses the experimental geometry
of the molecule, which is found in a NIST database:

    Experimental data for CH3COCH3 (Acetone).
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import make_co, make_ch3
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem

# When using the one primitives per basis function implementation:
# 119.72313498564577
# -189.61626040101655
# Construction time: 21.461366166s
# Total time taken: 32.359685416s

t1 = perf_counter_ns()

carbon_dict \
    = get_orbitals_dict_from_file(
        '../data/7p8e_1s321_2s3111_2p111111.json')
oxygen_dict = get_orbitals_dict_from_file(
    '../data/8p8e_1s6_2s3111_2p111111.json')
hydrogen_dict = get_orbitals_dict_from_file('../data/1p1e_1s4.json')
co = make_co(2.3)
a = 0.674*np.pi
ch3_1 = make_ch3(2.05, 2.05, 2.05, 0.61*np.pi,
                 np.array([-np.sin(a), 0.0, -np.cos(a)]),
                 np.array([-np.cos(a), 0.0, np.sin(a)])
                 ) + 2.85*np.array([np.sin(a), 0.0, np.cos(a)])
ch3_2 = make_ch3(2.05, 2.05, 2.05, 0.61*np.pi,
                 np.array([np.sin(a), 0.0, -np.cos(a)]),
                 np.array([np.cos(a), 0.0, np.sin(a)])
                 ) + 2.85*np.array([-np.sin(a), 0.0, np.cos(a)])
geom = co + ch3_1 + ch3_2

fig = plt.figure()
plt.title("Position of Acetone Nuclei")
ax = fig.add_subplot(projection='3d')
# for e in geom.get_nuclear_configuration():
#     print(e)
#     ax.scatter([e[0][0]], [e[0][1]], [e[0][2]])
ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2], label='H')
ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2], label='C')
ax.scatter(geom['O'].T[0], geom['O'].T[1], geom['O'].T[2], label='O')
ax.scatter([0.0, 0.0], [-4.0, 4.0], [0.0, 0.0], alpha=0.0)
ax.scatter([0.0, 0.0], [0.0, 0.0], [-4.0, 4.0], alpha=0.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()

orbitals = get_orbitals_from_geometry(
    geom, {'H': hydrogen_dict, 'C': carbon_dict, 'O': oxygen_dict}
)
system = ClosedShellSystem(
    16, orbitals, geom.get_nuclear_configuration()
)
system.solve(50)

t2 = perf_counter_ns()

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())

t3 = perf_counter_ns()

print(f'Construction time: {(t2 - t1)/1000000000.0}s')
print(f'Total time taken: {(t3 - t1)/1000000000.0}s')


def plotly_render():
    import plotly.graph_objects as go
    s = np.linspace(-4.5, 4.5, 128)
    ds3 = (s[1] - s[0])**3
    x, y, z = np.meshgrid(s, s, s)
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
