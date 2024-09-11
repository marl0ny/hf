from typing import Dict, List, Union
import numpy as np


def copy_position_list(positions: List[np.ndarray]) -> List[np.ndarray]:
    return [r.copy() for r in positions]


def copy_config(config: Dict[str, List[np.ndarray]]
                ) -> Dict[str, List[np.ndarray]]:
    return {k: copy_position_list(config[k]) for k in config}


def rotate(r, angle, rotation_axis, offset):
    def mult(a, b):
        return np.array([a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                         a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                         a[0]*b[2] + a[2]*b[0] + a[3]*b[1] - a[1]*b[3],
                         a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1]])
    c, s = np.cos(angle/2.0), np.sin(-angle/2.0)
    norm = np.linalg.norm(rotation_axis)
    rot = np.array([c,
                    s*rotation_axis[0]/norm, 
                    s*rotation_axis[1]/norm, 
                    s*rotation_axis[2]/norm])
    rot_inv = np.array([rot[0], -rot[1], -rot[2], -rot[3]])
    q0 = np.array([1.0, 
                   r[0] - offset[0], r[1] - offset[1], r[2] - offset[2]])
    qf = mult(rot_inv, mult(q0, rot))
    return np.array([qf[1] + offset[0],
                     qf[2] + offset[1],
                     qf[3] + offset[2]])


def cross(a, b):
    # print(a.shape, b.shape)
    return np.array([a[1]*b[2] - a[2]*b[1],
                     -a[0]*b[2] + a[2]*b[0],
                     a[0]*b[1] - a[1]*b[0]])


class MolecularGeometry:

    config: Dict[str, List[np.ndarray]]

    def __init__(self, **kw):
        """MolecularGeometry constructor.

        If an argument is supplied, it must be the key-value pair

        config={<element>: <positions>, ...},

        where <element> is the atomic number of an atom and
        <positions> is a list of Numpy arrays of length 3
        where each numpy array gives the position for where
        this atom can be found.

        As an example,
        
        config={'H': [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
                'O': [np.array([0.0, 0.0, 0.0])]}

        places hydrogen atoms at (1, 0, 0) and (0, 1, 0) and an oxygen
        atom at the origin.
        """
        self.config = {}
        if 'config' in kw:
            self.config = kw['config']
    
    def add_atom(self, atomic_symbol: 'str', position: np.ndarray):
        if atomic_symbol not in self.config:
            self.config[atomic_symbol] = []
        self.config[atomic_symbol].append(position)

    def __getitem__(self, atomic_symbol: str) -> np.ndarray:
        return np.array(self.config[atomic_symbol])

    def __add__(
            self,
            other: Union[np.ndarray, 'MolecularGeometry', int]
            ) -> 'MolecularGeometry':
        if isinstance(other, np.ndarray):
            config = copy_config(self.config)
            # print(config)
            for k in config:
                # print(config[k])
                for i in range(len(config[k])):
                    config[k][i] += other
            return MolecularGeometry(config=config)
        elif type(self) == type(other):
            config = copy_config(self.config)
            other_config = copy_config(other.config)
            for k in other_config:
                if k not in config:
                    config[k] = []
                config[k].extend(other_config[k])
            return MolecularGeometry(config=config)
        MolecularGeometry(config=copy_config(self.config))

    def __radd__(self, other: Union[np.ndarray, 'MolecularGeometry', int]
                 ) -> 'MolecularGeometry':
        return self.__add__(other)

    def __sub__(self, other: np.ndarray) -> 'MolecularGeometry':
        return self.__add__(-other)

    def __mul__(self, other: Union[float, np.ndarray]
                ) -> 'MolecularGeometry':
        config = copy_config(self.config)
        for k in config:
            for i in range(config[k]):
                config['k'][i] *= other
        return MolecularGeometry(config=config)
    
    def rotate(self, angle: float, rotation_axis: np.ndarray,
               offset: np.ndarray = np.zeros([3])
               ):
        for k in self.config:
            for i, r in enumerate(self.config[k]):
                self.config[k][i] = rotate(r, angle, rotation_axis, offset)

    def get_nuclear_configuration(self) -> List[List[Union[np.ndarray, int]]]:
        """ Get the configuration of the nuclear charges.

        It returns a list of the form

        [<position>: <charge>, ...],

        where <position> is a length 3 numpy array that indicates
        the position of the charge and <charge> is its charge strength.

        """
        nuc_config = []
        for k in self.config:
            if k == 'H':
                charge = 1
            elif k == 'C':
                charge = 6
            elif k == 'N':
                charge = 7
            elif k == 'O':
                charge = 8
            elif k == 'F':
                charge = 9
            else:
                raise NotImplementedError
            for i in range(len(self.config[k])):
                nuc_config.append([self.config[k][i].copy(), charge])
        return nuc_config


def make_oh(oh_length: float,
            axis=np.array([0.0, 0.0, 1.0])) -> 'MolecularGeometry':
    a = axis/np.linalg.norm(axis)
    geom = MolecularGeometry(config={
        'O': [np.array([0.0, 0.0, 0.0])],
        'H': [oh_length*a]
    })
    return geom


def make_co(co_length: float,
            axis=np.array([0.0, 0.0, 1.0])) -> 'MolecularGeometry':
    a = axis/np.linalg.norm(axis)
    geom = MolecularGeometry(config={
        'C': [np.array([0.0, 0.0, 0.0])],
        'O': [co_length*a]
    })
    return geom


def make_ch2(ch1: float, ch2: float,
             hch_angle: float,
             axis1: np.ndarray, axis2: np.ndarray
             ) -> 'MolecularGeometry':
    a1 = axis1/np.linalg.norm(axis1)
    a2 = axis2/np.linalg.norm(axis2)
    return MolecularGeometry(config={
        'C': [np.array([0.0, 0.0, 0.0])],
        'H': [ch1*(a1*np.cos(hch_angle/2.0) + a2*np.sin(hch_angle/2.0)), 
              ch2*(a1*np.cos(hch_angle/2.0) - a2*np.sin(hch_angle/2.0))],
    })


def make_ch3(ch1: float, ch2: float, ch3: float,
             ach_angle: float,
             axis1: np.ndarray, axis2: np.ndarray
             ) -> 'MolecularGeometry':
    a = axis1/np.linalg.norm(axis1)
    x = axis2/np.linalg.norm(axis2)
    y = cross(a, x)
    p1 = x
    phi = 2.0*np.pi/3.0
    p2 = x*np.cos(phi) + y*np.sin(phi)
    p3 = x*np.cos(2.0*phi) + y*np.sin(2.0*phi)
    return MolecularGeometry(config={
        'C': [np.array([0.0, 0.0, 0.0])],
        'H': [ch1*(a*np.cos(ach_angle) + p1*np.sin(ach_angle)),
              ch2*(a*np.cos(ach_angle) + p2*np.sin(ach_angle)),
              ch3*(a*np.cos(ach_angle) + p3*np.sin(ach_angle))],
    })


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # config = {'H': [np.array([1.0, 0.0, 0.0]),
    #                 np.array([-1.0, 0.0, 0.0])],
    #           'C': [np.array([0.0, 0.0, 0.0])]}
    # config2 = copy_config(config)
    # config2['H'].append(np.array([0.0, 1.0, 0.0]))
    # config2['C'][0][2] = -0.4
    # geom = MolecularGeometry(config=config2)
    # print(config2)
    # geom.rotate(np.pi/4.0, np.array([0.0, 0.0, 1.0]))
    # print(geom.config)

    geom1 = make_ch2(1.5, 1.5, 3.0 * np.pi / 2.0,
                     np.array([1.0, 0.0, 0.0]),
                     np.array([0.0, 0.0, 1.0]))
    plt.scatter(geom1['C'].T[0], geom1['C'].T[2])
    plt.scatter(geom1['H'].T[0], geom1['H'].T[2])
    plt.show()
    plt.close()

    geom2 = make_ch3(2.0, 1.5, 1.5, np.pi/4.0,
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(geom2['H'].T[0],
               geom2['H'].T[1],
               geom2['H'].T[2])
    ax.scatter(geom2['C'].T[0],
               geom2['C'].T[1],
               geom2['C'].T[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.close()
