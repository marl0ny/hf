from open_shell_system import *
import matplotlib.pyplot as plt
import numpy as np
import json
from time import perf_counter

atoms = {
    'H': {'N': 2400, 'extent': 15.0,
          'nuclear charge': 1, 'electron count': 1,
          'iterations': 20,
          'delta': 0.035**2
         },
    # 'Li': {'N': 1024, 'extent': 14.0,
    #        'nuclear charge': 3, 'electron count': 3,
    #        'iterations': 10,
    #       },
    # 'B': {'N': 1450, 'extent': 14.5,
    #       'nuclear charge': 5, 'electron count': 5,
    #       'iterations': 10},
    # 'C': {'N': 1828, 'extent': 9.0,
    #       'nuclear charge': 6, 'electron count': 6,
    #       'iterations': 10},
    # 'N': {'N': 1828, 'extent': 7.0,
    #       'nuclear charge': 7, 'electron count': 7,
    #       'iterations': 10,
    #       # delta = (100 / number_of_points) ** 2
    #       # -54.40516560484707
    #       # -1480.4399749966826 eV
    #       },
    # 'O': {'N': 1000, 'extent': 7.0,
    #        'nuclear charge': 8, 'electron count': 8,
    #        'iterations': 10,
    #        },
    # 'F': {'N': 1000, 'extent': 7.5,
    #       'nuclear charge': 9, 'electron count': 9,
    #       'iterations': 12,
    #       },
    # 'Na': {'N': 1000, 'extent': 12.0,
    #        'nuclear charge': 11, 'electron count': 11,
    #        'iterations': 12
    #       },
    # 'Al': {'N': 1828, 'extent': 13.0,
    #        'nuclear charge': 13, 'electron count': 13,
    #        'iterations': 12,
    #        },
    # 'Si': {'N': 1400, 'extent': 13.5,
    #        'nuclear charge': 14, 'electron count': 14,
    #        'iterations': 12,
    #        },
    # 'P': {'N': 1828, 'extent': 8.0,
    #       'nuclear charge': 15, 'electron count': 15,
    #       'iterations': 12,
    #       'delta': 0.04**2
    #        #   -340.277058448542
    #        #   -9259.410467755717 eV
    #       },
    # 'S': {'N': 1400, 'extent': 7.5,
    #        'nuclear charge': 16, 'electron count': 16,
    #        'iterations': 12,
    #         },
    # 'Cl': {'N': 1400, 'extent': 7.0,
    #        'nuclear charge': 17, 'electron count': 17,
    #        'iterations': 12,
    #        },
    # 'K': {'N': 1828, 'extent': 11.0,
    #       'nuclear charge': 19, 'electron count': 19,
    #       'iterations': 12,
    #       'delta': 0.045**2
    #       # -596.2795776541122
    #       # -16225.59389755152 eV
    #      },
    # 'Sc': {'N': 2400, 'extent': 9.0,
    #        'nuclear charge': 21, 'electron count': 21,
    #        'iterations': 21,
    #        'delta': 0.105**2
    #        # -758.0012205286561
    #        # -20626.263985986683 eV
    #        },
    # 'V': {'N': 2400, 'extent': 7.5,
    #       'nuclear charge': 23, 'electron count': 23,
    #       'iterations': 21,
    #       'delta': 0.105**2
    #       # -940.8915102478887
    #       # -25602.962299996674 eV
    #       },
    # 'Mn': {'N': 2400, 'extent': 8.0,
    #        'nuclear charge': 25, 'electron count': 25,
    #        'iterations': 21,
    #        'delta': 0.105**2
    #        # -1147.4367763200848
    #        # -31223.3453121635 eV
    #        },
    # 'Co': {'N': 2400, 'extent': 7.0,
    #         'nuclear charge': 27, 'electron count': 27,
    #         'iterations': 27,
    #         'delta': 0.115**2,
    #         # Previous:
    #         # -1378.0361593189602
    #         # -37498.27419080459 eV
    #         'init_functions': '../data/28p28e_fd.json', 
    #         },
    # 'Cu': {'N': 2400, 'extent': 7.6,
    #        'nuclear charge': 29, 'electron count': 29,
    #        'iterations': 27,
    #        'delta': 0.105**2,
    #        # -1634.6117996565097
    #        # -44480.05304108784 eV
    #        'init_functions': '../data/28p28e_fd.json', 
    #        },
    # 'Ga': {'N': 2400, 'extent': 7.6,
    #        'nuclear charge': 31, 'electron count': 31,
    #        'iterations': 27,
    #        'delta': 0.105**2,
    #        'init_functions': '../data/32p32e_fd.json', 
    #         # -1918.0307057823225
    #         # -52192.27436481273 eV
    #        },
    # 'As': {'N': 2400, 'extent': 7.5,
    #        'nuclear charge': 33, 'electron count': 33,
    #        'iterations': 27,
    #        'delta': 0.075**2,
    #        # -2227.8368705693747
    #        # -60622.52957591533 eV
    #        'init_functions': '../data/34p34e_fd.json', 
    #        },
    # 'Br': {'N': 2400, 'extent': 7.5,
    #        'nuclear charge': 35, 'electron count': 35,
    #        'iterations': 27,
    #        'delta': 0.05**2,
    #        # -2564.0735015218957
    #        # -69771.9944104819 eV
    #        'init_functions': '../data/36p36e_fd.json', 
    #        },
}


for name in atoms.keys():
    t1 = perf_counter()
    atom = atoms[name]
    kw = {}
    if 'delta' in atom:
        kw['delta'] = atom['delta']
    if 'init_functions' in atoms:
        kw['init_functions'] = atom['init_functions']
    system = UnrestrictedSystem(atom['N'], atom['extent'],
                                atom['nuclear charge'],
                                atom['electron count'],
                                **kw)
    if atom['electron count'] >= 25:
        if name in ['Co', 'Ni', 'Cu']:
            system.toggle_right_boundary_potential_regulator(
                                    remove_regulator_at=16)
        elif name in ['Ga', 'Co']:
            system.toggle_right_boundary_potential_regulator(
                                    remove_regulator_at=20)
        else:
            system.toggle_right_boundary_potential_regulator(
                remove_regulator_at=8)
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
    total_energy = system.get_total_energy()
    print(total_energy)
    print(27.211386245 * total_energy, 'eV')
    with open(f"../data/{atom['nuclear charge']}p"
              + f"{atom['electron count']}e_fd.json", "w") as f:
        json.dump(orbitals_dict, f)
    t2 = perf_counter()
    print('Time taken:', t2 - t1, "s")
