"""Compute the lowest Hartree-Fock energy of Formic acid.
This uses the experimental geometry of the molecule, 
which is found in a NIST database:

    Experimental data for HCOOH (Formic acid)
    Computational Chemistry Comparison and Benchmark DataBase,
    https://cccbdb.nist.gov/exp2x.asp.

"""
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_basis import make_system_from_primitives
from gaussian_basis import get_orbitals_dict_from_file
from gaussian_basis.molecular_geometry import make_oh, make_co
from gaussian_basis import get_orbitals_from_geometry
from gaussian_basis import ClosedShellSystem

t1 = perf_counter_ns()

# Using one primitive per basis function:
# 70.05032635921498
# -146.5767552606344
# Construction time: 190.226541875s
# Total time taken: 196.067997792s

carbon_dict = get_orbitals_dict_from_file(
    # '../data/10p10e_1s111111_2s111111_2p111111.json'
    '../data/7p8e_1s111111_2s111111_2p111111.json'
    )
oxygen_dict = get_orbitals_dict_from_file(
    '../data/10p10e_1s111111_2s111111_2p111111.json'
)
hydrogen_dict = get_orbitals_dict_from_file(
    '../data/1p1e_1s4_2s4_2p4.json'
)

a = 2.0*np.pi/3.0
coh = make_co(2.27, np.array([np.cos(np.pi + a), 0.0, np.sin(np.pi + a)]))
coh.add_atom('H', 2.07*np.array([np.cos(np.pi - a*0.9),
                                 0.0, np.sin(np.pi - a*0.9)]))
oh = make_oh(1.84, np.array([np.cos(0.6*np.pi), 0.0, np.sin(0.6*np.pi)])
             ) + 2.55*np.array([np.cos(np.pi), 0.0, np.sin(np.pi)])
geom = coh + oh

fig = plt.figure()
plt.title('Formic Acid Nuclei Locations')
ax = fig.add_subplot(projection='3d')
for e in geom.get_nuclear_configuration():
    charge = e[1]
    ax.scatter(
        [e[0][0]], [e[0][1]], [e[0][2]],
        label=f"{'C' if charge == 6 else ('H' if charge == 1 else 'O')}")
# ax.scatter(geom['H'].T[0], geom['H'].T[1], geom['H'].T[2])
# ax.scatter(geom['C'].T[0], geom['C'].T[1], geom['C'].T[2])
# ax.scatter(geom['O'].T[0], geom['O'].T[1], geom['O'].T[2])
ax.scatter([-4.0, 4.0], [0.0, 0.0], [0.0, 0.0], alpha=0.0)
ax.scatter([0.0, 0.0], [-4.0, 4.0], [0.0, 0.0], alpha=0.0)
ax.scatter([0.0, 0.0], [0.0, 0.0], [-4.0, 4.0], alpha=0.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()

orbitals = get_orbitals_from_geometry(
    geom, {'H': hydrogen_dict, 'O': oxygen_dict, 'C': carbon_dict}
)
system = ClosedShellSystem(12, orbitals, geom.get_nuclear_configuration())
system.solve(20)

t2 = perf_counter_ns()

print(system.energies)
print(system.get_nuclear_configuration_energy())
print(system.get_total_energy())

t3 = perf_counter_ns()

print(f'Construction time: {(t2 - t1)/1000000000.0}s')
print(f'Total time taken: {(t3 - t1)/1000000000.0}s')
