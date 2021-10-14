import numpy as np


def get_traj_name(traj):
    coeffs = traj(1) - traj(0)
    constants = traj(0)
    powers = np.ones(len(traj(0)))

    str = ""
    for coeff, constant, power in zip(coeffs, constants, powers):
        str += "({}*t+{})**{}".format(coeff, constant, power)

    return str
