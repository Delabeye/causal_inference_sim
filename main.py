import numpy as np
from dynamics.Dynamics import Dynamics
from dynamics.MPC import MPC

if __name__ == "__main__":
    # --- Démo single agent ---
    dt = 0.02
    sys =Dynamics(mass = 1.5, dt = dt)
    sys.reset(p=(0, 0, 0.2), v=(0, 0, 0))

    mpc = MPC(dt=dt, kp=4.0, kd=2.5, a_limit=8.0)

    p_ref = np.array([5.0, 2.0, 1.0])   # objectif simple
    v_ref = np.zeros(3)

    T = 20.0
    steps = int(T / dt)
    traj = np.zeros((steps, 6))

    for k in range(steps):
        a_des = mpc.control(sys.p, sys.v, p_ref, v_ref)      # accélération désirée
        u = sys.m * (a_des - sys.g)                          # force monde (compensation gravité)
        p, v = sys.step(u)
        traj[k, :3] = p
        traj[k, 3:] = v

    print("Position finale:", traj[-1, :3])
