from .gaussian1d import Gaussian1D
import numpy as np


class Gaussian3D:
    pos: np.ndarray
    ang: np.ndarray
    orb_exp: float
    amp: float
    gaussian0: Gaussian1D
    gaussian1: Gaussian1D
    gaussian2: Gaussian1D

    def __init__(self, amp: float, orb_exp: float,
                 pos: np.ndarray, ang: np.ndarray):
        if pos.shape != (3,) or ang.shape != (3,):
            raise Exception
        self.amp = amp
        self.pos = pos
        self.ang = ang
        self.orb_exp = orb_exp
        self.gaussian0 = Gaussian1D(orb_exp, pos[0], int(ang[0]))
        self.gaussian1 = Gaussian1D(orb_exp, pos[1], int(ang[1]))
        self.gaussian2 = Gaussian1D(orb_exp, pos[2], int(ang[2]))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.amp*self.gaussian0(r[0]) * \
               self.gaussian1(r[1])*self.gaussian2(r[2])

    def __mul__(self, other: float) -> 'Gaussian3D':
        return Gaussian3D(other*self.amp,
                          self.orb_exp, self.pos, self.ang)

    def __rmul__(self, other: float) -> 'Gaussian3D':
        return Gaussian3D(other*self.amp,
                          self.orb_exp, self.pos, self.ang)

    def get_gaussian1d(self, index: int) -> Gaussian1D:
        if index == 0:
            return self.gaussian0
        elif index == 1:
            return self.gaussian1
        elif index == 2:
            return self.gaussian2
        else:
            raise IndexError

    def position(self):
        return self.pos

    def orbital_exponent(self):
        return self.orb_exp

    def angular(self):
        return self.ang

    def amplitude(self):
        return self.amp


def product_center(g: Gaussian3D, h: Gaussian3D) -> np.ndarray:
    return (g.orbital_exponent()*g.position()
            + h.orbital_exponent()*h.position()) / \
           (g.orbital_exponent() + h.orbital_exponent())
