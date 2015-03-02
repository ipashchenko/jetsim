import math
import numpy as np


def velsum(v, u):
    """
    Relativistic sum of two 3-velocities ``u`` and ``vf``.
        u, vf - 3-velocities [c]
    """

    gamma_v = 1. / math.sqrt(1. - np.linalg.norm(v))

    return (1. / (1. + v.dot(u))) * (v + (1. / gamma_v) * u +
                                     (gamma_v / (1. + gamma_v)) * v.dot(u) * v)


def luminosity_distance(z, H_0=73.0, omega_M=0.3, omega_V=0.7, format="cm"):
    """
    Given redshift z, Hubble constant H_0 [km/s/Mpc] and
    density parameters omega_M and omega_V, returns luminosity
    distance.
    """

    from scipy.integrate import quad

    if format == "cm":
        return 9.26 * 10.0 ** 27.0 * (H_0 / 100.0) ** (-1.) * (1. + z) * quad(lambda x: (omega_M * (1. + x ** 3) + omega_V) ** (-0.5), 0, z)[0]
    elif format == "Mpc":
        return 3000.0 * (H_0 / 100.0) ** (-1.) * (1. + z) * quad(lambda x: (omega_M * (1. + x ** 3) + omega_V) ** (-0.5), 0, z)[0]
    else:
        raise Exception('Format=\"cm\" or \"Mpc\"')


def transfer_stokes()
