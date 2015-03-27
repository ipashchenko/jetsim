import numpy as np
from utils import generate_ndim_random_directions


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

    def bf_ord_vec(self, ps):
        """
        Vectors of ordered B-field component at N points in rectangular
        coordinates.

        """
        raise NotImplementedError

    def bf_rnd(self, x, y, z):
        """
        Vector of random B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        raise NotImplementedError

    def bf_rnd_vec(self, ps):
        """
        Vectors of random B-field component at N points in rectangular
        coordinates.

        """
        raise NotImplementedError

    def bf(self, x, y, z):
        """
        Vector of full magnetic field at N points (x, y, z) in rectangular
        coordinates.

        """
        return self.bf_ord(x, y, z) + self.bf_rnd(x, y, z)

    def bf_vec(self, ps):
        """
        Vectors of full magnetic field at N points in rectangular
        coordinates.

        """
        return self.bf_ord_vec(ps) + self.bf_rnd_vec(ps)


# That class describes different types of randomness for B-fields.
class BFRandom(object):
    """
    Class that represents random (tangled) magnetic field.

    :param z_0: (optional)
        Distance at which referenced values are given [pc]. (default:
        ``1.``)
    :param bf_0: (optional)
        Value of B-field at reference distance [G]. (default: ``0.1``)

    """
    def __init__(self, z_0=1., bf_0=0.1):
        self.z_0 = z_0
        self.bf_0 = bf_0

    def generate_n_random_directions(self, n):
        """
        Generate n random directions.

        :param n:
            Number of directions to generate.
        :return:
            List of ``n`` random directions.

        """
        return generate_ndim_random_directions(3, k=n)

    def bf(self, x, y, z):
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
    def __init__(self, z_0=1., bf_fi_0=1., bf_z_0=1.,
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

    def bf_fi_vec(self, ps):
        """
        Fi-components of B-field as function of z-cylinder coordinates ``z``.

        """
        return self.bf_fi_0 * self.z_0 / ps[:, 2]

    def bf_z(self, x, y, z):
        """
        Z-component of B-field as function of z-cylinder coordinate ``z``.

        """
        return self.bf_z_0 * (self.z_0 / z) ** 2.

    def bf_z_vec(self, ps):
        """
        Z-components of B-field as function of z-cylinder coordinate ``z``.

        """
        return self.bf_z_0 * (self.z_0 / ps[:, 2]) ** 2.

    def bf_ord(self, x, y, z):
        """
        Vector of ordered B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        return np.array([-self.bf_z(x, y, z) * y / np.sqrt(x * x + y * y),
                         self.bf_fi(x, y, z) * x / np.sqrt(x * x + y * y),
                         self.bf_z(x, y, z)])

    def bf_ord_vec(self, ps):
        """
        Vectors of ordered B-field components at N points in rectangular
        coordinates.

        """

        xs = ps[:, 0]
        ys = ps[:, 1]
        zs = ps[:, 2]
        bx = -self.bf_z_vec(ps) * ys / np.sqrt(xs * xs + ys * ys)
        by = self.bf_fi_vec(ps) * xs / np.sqrt(xs * xs + ys * ys)
        bz = self.bf_z_vec(ps)
        return np.asarray((bx, by, bz,)).T

    def bf_rnd(self, x, y, z):
        """
        Vector of random B-field component at N points (x, y, z) in rectangular
        coordinates.

        """
        bf_ord = self.bf_ord(x, y, z)
        n_rnd = generate_ndim_random_directions()[0]
        return np.linalg.norm(bf_ord) * self.fraction_rnd * n_rnd

    def bf_rnd_vec(self, ps):
        """
        Vectors of random B-field component at N points in rectangular
        coordinates.

        """
        bf_ord_vec = self.bf_ord_vec(ps)
        n_rnd = np.asarray(generate_ndim_random_directions(k=len(ps)))
        return np.linalg.norm(bf_ord_vec, axis=1)[:, np.newaxis] *\
               self.fraction_rnd * n_rnd

    def bf(self, x, y, z):
        """
        Vector of full magnetic field at N points (x, y, z) in rectangular
        coordinates.

        """
        return self.bf_ord(x, y, z) + self.bf_rnd(x, y, z)

    def bf_vec(self, ps):
        """
        Vectors of full magnetic field at N points in rectangular
        coordinates.

        """
        return self.bf_ord_vec(ps) + self.bf_rnd_vec(ps)
