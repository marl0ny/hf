from closed_shell_system import *
import matplotlib.pyplot as plt
import numpy as np
import json


atoms = {
    # 'H^{-}': {'N': 1024, 'extent': 17.0,
    #           'nuclear charge': 1, 'electron count': 2,
    #           'iterations': 20,
    #           },
    # 'He': {'N': 1024, 'extent': 7.0,
    #        'nuclear charge': 2, 'electron count': 2,
    #        'iterations': 20,
    #        },
    # 'Li^{+}': {'N': 1024, 'extent': 5.0,
    #            'nuclear charge': 3, 'electron count': 2,
    #            'iterations': 10,
    #            },
    # 'Be': {'N': 1024, 'extent': 10.0,
    #        'nuclear charge': 4, 'electron count': 4,
    #        'iterations': 10,
    #        },
    # 'C^{2+}': {'N': 1000, 'extent': 7.0,
    #            'nuclear charge': 6, 'electron count': 4,
    #            'iterations': 12,
    #            },
    # 'C^{4-}': {'N': 1000, 'extent': 10.0,
    #            'nuclear charge': 6, 'electron count': 10,
    #            'iterations': 12,
    #            },
    # 'N^{3-}': {'N': 1000, 'extent': 10.0,
    #            'nuclear charge': 7, 'electron count': 10,
    #            'iterations': 12,
    #            },
    # 'O^{2-}': {'N': 1000, 'extent': 5.5,
    #            'nuclear charge': 8, 'electron count': 10,
    #            'iterations': 51,
    #            },
    # 'F^{-}': {'N': 1000, 'extent': 5.7,
    #           'nuclear charge': 9, 'electron count': 10,
    #           'iterations': 51,
    #           },
    'N^{-}': {'N': 1000, 'extent': 6.5,
               'nuclear charge': 7, 'electron count': 8,
               'iterations': 30,
           },
    # 'O': {'N': 1000, 'extent': 7.0,
    #        'nuclear charge': 8, 'electron count': 8,
    #        'iterations': 12,
    #        },
    # 'Ne': {'N': 1000, 'extent': 5.0,
    #        'nuclear charge': 10, 'electron count': 10,
    #        'iterations': 12,
    #        },
    # 'Na^{+}': {'N': 1000, 'extent': 5.0,
    #            'nuclear charge': 11, 'electron count': 10,
    #            'iterations': 12,
    #            },
    # 'Mg': {'N': 1400, 'extent': 12.0,
    #        'nuclear charge': 12, 'electron count': 12,
    #        'iterations': 10,
    #        },
    # 'Cl^{-}': {'N': 1400, 'extent': 10.0,
    #        'nuclear charge': 17, 'electron count': 18,
    #        'iterations': 15,
    #        },
    # 'Ar': {'N': 1400, 'extent': 7.0,
    #        'nuclear charge': 18, 'electron count': 18,
    #        'iterations': 10,
    #        },
    # 'Ca': {'N': 1400, 'extent': 12.0,
    #        'nuclear charge': 20, 'electron count': 20,
    #        'iterations': 12,
    #        },

}

for name in atoms.keys():
    atom = atoms[name]
    system = ClosedShellSystem(atom['N'], atom['extent'],
                               atom['nuclear charge'],
                               atom['electron count'],
                               orbital_letters=['1s', '2p', '2p', '2p']
                               )
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
