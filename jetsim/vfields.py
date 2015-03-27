import math
import numpy as np

class VField(object):
    """
    Basic class that represents velocity field in observer frame.
    """
    def v(self, x, y, z):
        """
        Velocity field of jet in observer rest frame in triangular coordinates.
        :param x, y, z:
            Rectangular coordinates. (3, N,) N-number of points
        """
        raise NotImplementedError

    def v_vec(self, ps):
        """
        Velocity field of jet in observer rest frame in triangular coordinates.
        :param ps:
            Numpy array with shape (N, 3,) of rectangular coordinates. N -
            number of points
        """
        raise NotImplementedError


class FlatVField(VField):
    """
    Velocity field directed along z-direction.
    """
    def __init__(self, gamma0=10.):
        self.gamma0 = gamma0
        # z-component of velocity in triangular coordinates
        self.v_z = math.sqrt(1. - 1. / self.gamma0 ** 2.)

    def v(self, x, y, z):
        return  np.array([0., 0., np.sign(z) * self.v_z])

    def v_vec(self, ps):
        vx = np.zeros(len(ps), dtype=float)
        vy = np.zeros(len(ps), dtype=float)
        vz = np.sign(ps[:, 2]) * self.v_z
        return  np.asarray((vx, vy, vz,)).T


class CentralVField(VField):
    """
    Velocity field directed from central point.
    """
    def __init__(self, gamma0=10.):
        self.gamma0 = gamma0

    def v(self, x, y, z):
        # r-component of velocity in spherical coordinates
        v_r = math.sqrt(1. - 1. / self.gamma0 ** 2.)
        return  np.array([v_r * x / np.sqrt(x * x + y * y + z * z),
                          v_r * y / np.sqrt(x * x + y * y + z * z),
                          v_r * z / np.sqrt(x * x + y * y + z * z)])

    def v_vec(self, ps):
        # r-component of velocity in spherical coordinates
        v_r = math.sqrt(1. - 1. / self.gamma0 ** 2.)
        xs = ps[:, 0]
        ys = ps[:, 1]
        zs = ps[:, 2]

        vx = v_r * xs / np.sqrt(xs * xs + ys * ys + zs * zs)
        vy = v_r * ys / np.sqrt(xs * xs + ys * ys + zs * zs)
        vz = v_r * zs / np.sqrt(xs * xs + ys * ys + zs * zs)

        return np.asarray((vx, vy, vz,)).T
