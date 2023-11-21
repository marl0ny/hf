from open_shell_system import *
import matplotlib.pyplot as plt
import numpy as np
import json

atoms = {
    # 'H': {'N': 1024, 'extent': 17.0,
    #       'nuclear charge': 1, 'electron count': 1,
    #       'iterations': 20,
    #      },
    # 'Li': {'N': 1024, 'extent': 14.0,
    #        'nuclear charge': 3, 'electron count': 3,
    #        'iterations': 10,
    #       },
    'N': {'N': 1000, 'extent': 7.5,
          'nuclear charge': 7, 'electron count': 7,
          'iterations': 10,
          },
    # 'Na': {'N': 1000, 'extent': 12.0,
    #        'nuclear charge': 11, 'electron count': 11,
    #        'iterations': 12
    #       },
    # 'P': {'N': 1400, 'extent': 8.0,
    #       'nuclear charge': 15, 'electron count': 15,
    #       'iterations': 12,
    #       },
    # 'K': {'N': 1500, 'extent': 14.0,
    #       'nuclear charge': 19, 'electron count': 19,
    #       'iterations': 12,
    #       # -16121.354305388597
    #       # -5.281611735804714
    #      },
}


for name in atoms.keys():
    atom = atoms[name]
    system = UnrestrictedSystem(atom['N'], atom['extent'],
                                atom['nuclear charge'],
                                atom['electron count'])
    system.solve(n_iterations=atom['iterations'], verbose=True)
    plt.title(r'Hartree-Fock Orbital Energies for ${'
              + name + '}$')
    plt.xlabel('Iteration Count')
    plt.ylabel('Energy (eV)')
    for k in system.orbital_names():
        plt.plot(system.orbital_energies[k])
    file_name = ''.join([c for c in name if c not in ['{', '}', '^']])
    plt.savefig(f'{file_name}_energies.png')
    plt.show()
    plt.close()
    plt.title(r'Hartree-Fock Orbitals for ${'
              + name + '}$ (Radial Profile)')
    plt.xlabel('Radius (a.u.)')
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(system.orbitals.keys()) > len(cols):
        for i in range(len(system.orbitals.keys()) - len(cols)):
            cols.append('gray') 
    orbitals_dict = {}
    for k, orbital_name in enumerate(system.orbital_names()):
        init_orbital = np.abs(system.get_initial_orbital(orbital_name))
        orbital = np.abs(system.get_orbital(orbital_name))
        plt.plot(system.R, init_orbital,
                 label=r'Initial $|r\phi_{' + orbital_name + r'}(r)|$',
                 color=cols[k],
                 linestyle='--')
        plt.plot(system.R, orbital,
                 label=r'Final $|r\phi_{' + orbital_name + r'}(r)|$',
                 color=cols[k])
        orbitals_dict[orbital_name] = \
            {'r': list(system.R),
             'values': list(system.get_orbital(orbital_name))
             }
    plt.legend()
    plt.savefig(f'{file_name}_orbitals.png')
    plt.show()
    plt.close()
    print(27.211386245 * system.get_total_energy())
    with open(f"../data/{atom['nuclear charge']}p"
              + f"{atom['electron count']}e_fd.json", "w") as f:
        json.dump(orbitals_dict, f)

