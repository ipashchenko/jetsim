import math
import numpy as np


class BField(object):
    def bf_ord(self, x, y, z):
        """
        Vector of ordered B-field component at N points (x, y, z) in rectangular
        coordinates.
        """
        raise NotImplementedError

    def bf_rnd(self, x, y, z, fraction=None):
        """
        Vector of random B-field component at N points (x, y, z) in rectangular
        coordinates.
        """
        raise NotImplementedError

    def bf(self, x, y, z, fraction=0.3):
        return self.bf_ord(x, y, z) + self.bf_rnd(x, y, z, fraction=fraction)


# TODO: Add initialization based on initial pitch angle
class BFHelical(object):
    """
    Class that represents helical B-field.
    """
    def __init__(self, z0=1., bf_fi_0=10**(-1), bf_z_0=10**(-1),
                 pitch_angle_0=None):
        self.z0 = z0
        self.bf_fi_0 = bf_fi_0
        self.bf_z_0 = bf_z_0
        self.pitch_angle_0 = pitch_angle_0

    def bf_fi(self, x, y, z):
        """
        Fi-component of B-field as function of z-cylinder coordinate ``z`` and value
        of ``z`` where ``bf_fi`` equals specified value ``bf_fi_0``.
        :param z:
        :return:
        """
        return self.bf_fi_0 * self.z_0 / z

    def bf_z(self, x, y, z):
        """
        Z-component of B-field as function of z-cylinder coordinate ``z`` and value
        of ``z`` where ``bf_z`` equals specified value ``bf_z_0``.
        :param z:
        :return:
        """
        return self.bf_z_0 * (self.z_0 / z) ** 2.

    def bf_ord(self, x, y, z):
        """
        Vector of B-field at N points (x, y, z) in rectangular coordinates.
        """
        return np.array([-self.bf_z(x, y, z) * y / np.sqrt(x * x + y * y),
                         self.bf_fi(x, y, z) * x / np.sqrt(x * x + y * y),
                         self.bf_z(x, y, z)])

    def bf_rnd(self, x, y, z, fraction=None):
        bf_ord = self.bf_ord(x, y, z)
        return np.linalg.norm(bf_ord) * fraction * (1. / math.sqrt(3)) *\
               np.array([1. / math.sqrt(3),
                         1. / math.sqrt(3),
                         1. / math.sqrt(3)])
