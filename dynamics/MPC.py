import numpy as np

class MPC:
    """
    MPC minimal horizon=1 (équivalent PD/LQR) :
    a_des = Kp*(p_ref - p) + Kd*(v_ref - v)
    Puis u = m*(a_des - g) pour compenser la gravité.
    """
    def __init__(self, dt: float, kp: float = 4.0, kd: float = 2.5, a_limit: float = 8.0):
        self.dt = float(dt)
        self.kp = float(kp)
        self.kd = float(kd)
        self.a_limit = float(a_limit)

    def control(self, p: np.ndarray, v: np.ndarray, p_ref: np.ndarray, v_ref: np.ndarray | None = None):
        if v_ref is None:
            v_ref = np.zeros(3)
        a = self.kp * (p_ref - p) + self.kd * (v_ref - v)
        # saturation sur la norme d'accélération
        n = np.linalg.norm(a)
        if n > self.a_limit:
            a = a * (self.a_limit / (n + 1e-9))
        return a