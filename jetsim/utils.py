import math
import numpy as np


def velsum(v, u):
    """
    Relativistic sum of two 3-velocities ``u`` and ``v``.
        u, v - 3-velocities [c]
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


def transfer_stokes(stokes1, v1, v2, n1, n2, bf2):
    """
    Transfer stokes vector from frame that has velocity v1 in observer frame to
    frame that has velocity v2 in observer frame. Index 2 means value in second
    (final) rest frame. Index 1 means value in first (initial) rest frame.

    :param stokes1:
        Stokes vector in RF that has velocity v1 relative to observer frame.
    :param v1:
        Velocity of first frame relative to observer frame.
    :param v2:
        Velocity of second frame relative to observer frame.
    :param n1:
        Direction of propagation in first RF.
    :param n2:
        Direction of propagation in second RF.
    :param bf2:
        Direction of B-field in second RF.
    :return:
        Stokes vector in second rest frame.
    """
    # Find Doppler factor of v2 relative to v1
    v2r1 = velsum(v2, -v1)
    G2r1 = 1. / math.sqrt(1. - v2r1.dot(v2r1))
    D2r1 = 1. / (G2r1 * (1. - n2.dot(v2r1)))
    I1, Q1, U1, V1 = stokes1
    LP1 = math.sqrt(Q1 ** 2. + U1 ** 2.)
    chi1 = math.atan2(U1, Q1)
    # Polarization angle in first RF
    e1 = np.array([n1[2] * math.sin(chi1),
                   math.cos(chi1),
                   -n1[0] * math.sin(chi1)])
    # Polarization angle in second RF
    e2 = G2r1 * (e1 - (G2r1 / (G2r1 + 1)) * e1.dot(v2r1) * v2r1 +
                 np.cross(v2r1, np.cross(n1, e1)))
    I2 = I1 / D2r1 ** 3.
    V2 = V1 / D2r1 ** 3.
    LP2 = LP1 / D2r1 ** 3.
    chi2 = math.acos(((bf2 - bf2.dot(n2) * n2) / np.linalg.norm(bf2 - bf2.dot(n2) * n2)) * e2 / np.linalg.norm(e2))
    Q2 = LP2 * math.cos(2. * chi2)
    U2 = LP2 * math.sin(2. * chi2)
    return np.array([I2, Q2, U2, V2])
