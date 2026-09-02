from closed_shell_system import *
from post_hf_closed_shell_system import *
import matplotlib.pyplot as plt
import numpy as np
import json
from time import perf_counter


atoms = {
    # 'He': {'N': 1024, 'extent': 7.0,
    #        'nuclear charge': 2, 'electron count': 2,
    #        'iterations': 20,
    #        # Fully spherically symmetric
    #        # -2.8624957111495077
    #        },
    # 'Be': {'N': 1400, 'extent': 10.75,
    #        'nuclear charge': 4, 'electron count': 4,
    #        'iterations': 20,
    #        # Fully spherically symmetric
    #        # -14.573995502919988
    #        },
    # 'C': {'N': 512, 'extent': 11.0,
    #        'nuclear charge': 6, 'electron count': 6,
    #        'iterations': 15,
    #       },
    # 'O': {'N': 1400, 'extent': 7.0,
    #        'nuclear charge': 8, 'electron count': 8,
    #        'iterations': 15,
    #        },
    # 'Ne': {'N': 1828, 'extent': 5.25,
    #        'nuclear charge': 10, 'electron count': 10,
    #        'iterations': 15,
    #        # Fully spherically symmetric
    #        # -128.55391465196834
    #        },
    # 'Mg': {'N': 2600, 'extent': 9.25,
    #        'nuclear charge': 12, 'electron count': 12,
    #        'iterations': 12,
    #        # Fully spherically symmetric
    #        # -199.51443003479284
    #        'delta': 0.025**2
    #        },
    # 'Si': {'N': 1024, 'extent': 13.5,
    #        'nuclear charge': 14, 'electron count': 14,
    #        'iterations': 16,
    #        },
    # 'S': {'N': 1400, 'extent': 7.5,
    #       'nuclear charge': 16, 'electron count': 16,
    #       'iterations': 15,
    #       },
    # 'Ar': {'N': 2048, 'extent': 6.0,
    #        'nuclear charge': 18, 'electron count': 18,
    #        'iterations': 12,
    #        # Fully spherically symmetric
    #        # -526.4429151268422
    #        'delta': 0.04**2
    #        },
    # 'Ca': {'N': 1400, 'extent': 12.0,
    #        'nuclear charge': 20, 'electron count': 20,
    #        # Fully spherically symmetric
    #        'iterations': 12,
    #        },
    # 'Ti': {'N': 1828, 'extent': 7.25,
    #        'nuclear charge': 22, 'electron count': 22,
    #        'iterations': 21,
    #        # -846.0505051754807
    #        'delta': 0.105**2
    #        },
    # 'Cr': {'N': 2400, 'extent': 7.75,
    #        'nuclear charge': 24, 'electron count': 24,
    #        'iterations': 21,
    #        'delta': 0.105**2
    #        # -1040.3253330968612
    #        },
    # 'Fe': {'N': 2400, 'extent': 7.75,
    #        'nuclear charge': 26, 'electron count': 26,
    #        'iterations': 21,
    #        'delta': 0.105**2,
    #        # -1258.6203513093235
    #        },
    # 'Ni': {'N': 2400, 'extent': 7.70,
    #            'nuclear charge': 28, 'electron count': 28,
    #            'delta': 0.085**2,
    #            'iterations': 27
    #            # -1502.474769407017
    #            # -40884.42127370165 eV
    #            },
    # 'Zn': {'N': 2400, 'extent': 7.5,
    #        'nuclear charge': 30, 'electron count': 30,
    #        'iterations': 26,
    #        'delta': 0.105**2,
    #        # Fully spherically symmetric
    #        },
    'Ge': {'N': 2400, 'extent': 7.5,
            'nuclear charge': 32, 'electron count': 32,
            'iterations': 26,
            'delta': 0.105**2,
            # previous
            # -2068.752943474861
            # -56293.635390375086 eV
            },
    # 'Se': {'N': 2400, 'extent': 8.0,
    #        'nuclear charge': 34, 'electron count': 34,
    #        'iterations': 21,
    #        'delta': 0.04**2
    #        # -2391.220427776233
    #        },
    # 'Kr': {'N': 1828, 'extent': 7.5,
    #         'nuclear charge': 36, 'electron count': 36,
    #         'iterations': 17,
    #         'delta': 0.05**2
    #        # Fully spherically symmetric
    #         # -2736.627738653648
    #         },
    # 'Sr': {'N': 1828, 'extent': 14.5,
    #        'nuclear charge': 38, 'electron count': 38,
    #        'iterations': 15,
    #        },
    # 'Zr': {'N': 1400, 'extent': 12.5,
    #        'nuclear charge': 40, 'electron count': 40,
    #        'iterations': 15,
    #        # -3451.65759132418
    #        },
    # 'Mo': {'N': 1400, 'extent': 11.0,
    #        'nuclear charge': 42, 'electron count': 42,
    #        'iterations': 15,
    #        # Approx. -3880
    #        },
    # 'Ru': {'N': 1400, 'extent': 11.5,
    #        'nuclear charge': 44, 'electron count': 44,
    #        'iterations': 15,
    #        # -4331.900226454583
    #        },

    ## TODO: these do not work properly yet! #################################
    # 'Pd': {'N': 1024, 'extent': 7.5,
    #        'nuclear charge': 46, 'electron count': 46,
    #        'iterations': 15,
    #        'orbital_letters':
    #            ['1s',
    #             '2s', '2p', '2p', '2p',
    #             '3s', '3p', '3p', '3p',
    #             '4s', '3d', '3d', '3d', '3d', '3d',
    #             '4p', '4p', '4p',
    #             '4d', '4d', '4d', '4d', '4d'],
    #         'delta': 0.15**2
    #        },
    # 'Cd': {'N': 1400, 'extent': 9.0,
    #        'nuclear charge': 48, 'electron count': 48,
    #        'iterations': 15,
    #        },
    ##########################################################################

    # 'Sn': {'N': 1400, 'extent': 10.0,
    #        # -5881.166746421344 -- This is way off!!!
    #        'nuclear charge': 50, 'electron count': 50,
    #        'iterations': 15,
    #        },
    # 'Te': {'N': 1828, 'extent': 9.0,
    #        'nuclear charge': 52, 'electron count': 52,
    #        # -6521.673794329942
    #        'iterations': 15,
    #        },
    # 'Xe': {'N': 1828, 'extent': 8.5,
    #        'nuclear charge': 54, 'electron count': 54,
    #        'iterations': 15,
    #        'delta': 0.035 ** 2,
    #        # Fully spherically symmetric
    #        },
    # 'Ba': {'N': 1024, 'extent': 16.0,
    #        'nuclear charge': 56, 'electron count': 56,
    #        'iterations': 12,
    #        # Fully spherically symmetric
    #        },

}

for name in atoms.keys():
    t1 = perf_counter()
    atom = atoms[name]
    # system = ClosedShellSystemWithPostHF(atom['N'], atom['extent'],
    #                            atom['nuclear charge'],
    #                            atom['electron count'],
    #                            # orbital_letters=['1s', '2p', '2p', '2p']
    #                            )
    kw = {}
    if 'orbital_letters' in atom.keys():
        kw['orbital_letters'] = atom['orbital_letters']
    if 'delta' in atom.keys():
        kw['delta'] = atom['delta']
    system = ClosedShellSystem(atom['N'], atom['extent'],
                               atom['nuclear charge'],
                               atom['electron count'],
                               **kw
                               # orbital_letters=['1s', '2p', '2p', '2p']
                               )
    if atom['electron count'] > 20 or name in ['C', 'Si', 'S']:
        # TODO: Not necessary for Titanium.
        print(name)
        if name in ['Ni', 'Ge']:
            system.toggle_right_boundary_potential_regulator(
                        remove_regulator_at=16)
        elif name in ['Zn']:
            system.toggle_right_boundary_potential_regulator(
                    remove_regulator_at=20)
        else:
            system.toggle_right_boundary_potential_regulator(
                remove_regulator_at=8)
    # plt.imshow(system.R_GREATER_THAN)
    # plt.show()
    # plt.close()
    # plt.imshow(system.R_LESS_THAN)
    # plt.show()
    # plt.close()
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
                 color=cols[k] if k < len(cols) else 'red',
                 linestyle='--')
        plt.plot(system.R, orbital,
                 label=r'Final $|r\phi_{' + orbital_name + r'}(r)|$',
                 color=cols[k] if k < len(cols) else 'red')
        orbitals_dict[orbital_name] = \
            {'r': list(system.R),
             'values': list(system.get_orbital(orbital_name))
             }
    # for orbital_name in system.virtual_orbitals:
    #     virtual_orbital = np.abs(system.virtual_orbitals[orbital_name])
    #     plt.plot(system.R, virtual_orbital,
    #              label=r'Virtual $|r\phi_{' + orbital_name + r'}(r)|$',
    #              color='gray', linestyle='--', alpha=0.25)
    plt.legend()
    plt.savefig(f'{file_name}_orbitals.png')
    plt.show()
    plt.close()
    print(system.get_total_energy())
    print(27.211386245 * system.get_total_energy(), 'eV')
    with open(f"../data/{atom['nuclear charge']}p"
              + f"{atom['electron count']}e_fd.json", "w") as f:
        json.dump(orbitals_dict, f)
    t2 = perf_counter()
    print("Time taken:", t2 - t1, "s")
