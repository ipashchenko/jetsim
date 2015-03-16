import math
import numpy as np


# TODO: ``BFRandom`` can be initialized with some value at ``z_0`` or
# ``BFHelical`` can be given some ``fraction_rnd``. What to use?
# TODO: Create ``BFOrdered`` & ``BFRandom`` classes?
class BField(object):
    """
    Basic class that represents magnetic field.

    :param z_0: (optional)
        Distance at which referenced values are given [pc]. (default:
        ``1.``)

    :param fraction_rnd: (optional)
        Fraction of energy in random component of the magnetic field.
        (default: ``0.``)

    """
    def __init__(self, z_0=1., fraction_rnd=0.):
        self.z_0 = z_0
        self.fraction_rnd = fraction_rnd

    def bf_ord(self, x, y, z):
        """
        Vector of ordered B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        raise NotImplementedError

    def bf_rnd(self, x, y, z):
        """
        Vector of random B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        raise NotImplementedError

    def bf(self, x, y, z):
        """
        Vector of full magnetic field at N points (x, y, z) in rectangular
        coordinates.

        """
        return self.bf_ord(x, y, z) + self.bf_rnd(x, y, z)


# That class describes different types of randomness for B-fields.
class BFRandom(object):
    """
    Class that represents random (tangled) magnetic field.

    :param z_0: (optional)
        Distance at which referenced values are given [pc]. (default:
        ``1.``)

    """
    def __init__(self, z_0=1.):
        pass


# TODO: Add initialization based on initial pitch angle
class BFHelical(BField):
    """
    Class that represents helical B-field.

        :param z_0:
            Distance at which referenced values are given [pc].

        :param bf_fi_0: (optional)
            Fi-component of the magnetic field at ``z_0``.

        :param bf_z_0: (optional)
            Z-component of the magnetic field at ``z_0``.

        :param pitch_angle_0: (optional)
            Pitch angle of the helical magnetic field at ``z_0``.

    """
    def __init__(self, z_0=1., bf_fi_0=10.**(-1.), bf_z_0=10.**(-1.),
                 pitch_angle_0=None, fraction_rnd=0.5):
        super(BFHelical, self).__init__(z_0=z_0, fraction_rnd=fraction_rnd)
        self.bf_fi_0 = bf_fi_0
        self.bf_z_0 = bf_z_0
        self.pitch_angle_0 = pitch_angle_0

    def bf_fi(self, x, y, z):
        """
        Fi-component of B-field as function of z-cylinder coordinate ``z``.

        """
        return self.bf_fi_0 * self.z_0 / z

    def bf_z(self, x, y, z):
        """
        Z-component of B-field as function of z-cylinder coordinate ``z``.

        """
        return self.bf_z_0 * (self.z_0 / z) ** 2.

    def bf_ord(self, x, y, z):
        """
        Vector of ordered B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        return np.array([-self.bf_z(x, y, z) * y / np.sqrt(x * x + y * y),
                         self.bf_fi(x, y, z) * x / np.sqrt(x * x + y * y),
                         self.bf_z(x, y, z)])

    def bf_rnd(self, x, y, z):
        """
        Vector of random B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        bf_ord = self.bf_ord(x, y, z)
        return np.linalg.norm(bf_ord) * self.fraction_rnd * (1. / math.sqrt(3))\
               * np.array([1. / math.sqrt(3),
                           1. / math.sqrt(3),
                           1. / math.sqrt(3)])
