import numpy as np


class Gaussian1D:
    r0: float
    ang: int
    orb_exp: float

    def __init__(self, orb_exp: float, r0: float, ang: int):
        self.orb_exp = orb_exp
        self.r0 = r0
        self.ang = ang

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ang = self.ang
        r0 = self.r0
        orb_exp = self.orb_exp
        return (x - r0)**ang*np.exp(-orb_exp*(x - r0)**2)

    def __add__(self, n: int) -> 'Gaussian1D':
        position = self.position()
        orb_exp = self.orbital_exponent()
        angular = self.angular()
        return Gaussian1D(orb_exp=orb_exp, r0=position,
                          ang=angular + n)

    def __sub__(self, n: int) -> 'Gaussian1D':
        position = self.position()
        orb_exp = self.orbital_exponent()
        angular = self.angular()
        return Gaussian1D(orb_exp=orb_exp, r0=position,
                          ang=angular - n)

    def position(self) -> float:
        return self.r0

    def orbital_exponent(self) -> float:
        return self.orb_exp

    def angular(self) -> int:
        return self.ang
