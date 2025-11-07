import numpy as np

class Dynamics:
    """
    m * d2p = u + m*g
    Intégration semi-implicite (stable et simple).
    """
    def __init__(self, mass, dt, g=np.array([0.0, 0.0, -9.81])):
        self.m = float(mass)
        self.dt = float(dt)
        self.g = np.array(g, dtype=float)
        self.p = np.zeros(3)  # position
        self.v = np.zeros(3)  # vitesse

    def reset(self, p=(0, 0, 0.2), v=(0, 0, 0)):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)

    def step(self, force_world: np.ndarray):
        """
        force_world: np.array shape (3,) en N, dans le repère monde.
        Retourne (p, v) après un pas.
        """
        a = self.g + force_world / self.m
        self.v = self.v + self.dt * a
        self.p = self.p + self.dt * self.v
        return self.p.copy(), self.v.copy()


