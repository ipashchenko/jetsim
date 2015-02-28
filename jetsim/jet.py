import math
import numpy as np

eps = 10**(-5)

class Jet(object):
    """
    Class that represents jet's physical properties: distribution of magnetic
    field, particle density, velocity flow.
    """
    def __init__(self):
        pass

    def integrate(self, ray, back, front):
        """
        Integrate along ray from back to front.
        """
        pass


def bf_fi(x, y, z, z_0=1., bf_fi_0=10**(-1)):
    """
    Fi-component of B-field as function of z-cylinder coordinate ``z`` and value
    of ``z`` where ``bf_fi`` equals specified value ``bf_fi_0``.
    :param z:
    :param z_0:
    :param bf_fi_0:
    :return:
    """
    return bf_fi_0 * z_0 / z


def bf_z(x, y, z, z_0=1, bf_z_0=10**(-1)):
    """
    Z-component of B-field as function of z-cylinder coordinate ``z`` and value
    of ``z`` where ``bf_z`` equals specified value ``bf_z_0``.
    :param z:
    :param z_0:
    :param bf_z_0:
    :return:
    """
    return bf_z_0 * (z_0 / z) ** 2.


def bf(x, y, z):
    """
    Vector of B-field at N points (x, y, z) in rectangular coordinates.
    """
    return np.array([-bf_z(x, y, z) * y / np.sqrt(x * x + y * y),
                     bf_fi(x, y, z) * x / np.sqrt(x * x + y * y),
                     bf_z(x, y, z)])


def n_z(x, y, z, z_0=1., n_z_0=1.):
    return n_z_0 * (z_0 / z) ** 2.


def beta(x, y, z, z_0=1, theta_0=0., gamma0=10.):
    """
    Velocity field of jet.
    :param x, y, z:
        Rectangular coordinates. (3, N,) N-number of points
    :param gamma0:
        Lorentz-factor of jet at ``z_0`` & ``theta_0``
    :return:
        (3, N,) array of vectors of velocity of jet at given N point ``xyz`` in
        rectangular coordinates.
    """
    # r-component of velocity in spherical coordinates
    value = math.sqrt(1. - 1. / gamma0 ** 2.)
    result =  np.array([value * x / np.sqrt(x * x + y * y + z * z),
                        value * y / np.sqrt(x * x + y * y + z * z),
                        value * z / np.sqrt(x * x + y * y + z * z)])
    return result

