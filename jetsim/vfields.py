import math
import numpy as np

class VField(object):
    """
    Basic class that represents velocity field in observer frame.
    """
    def v(self, x, y, z):
        """
        Velocity field of jet in observer rest frame.
        :param x, y, z:
            Rectangular coordinates. (3, N,) N-number of points
        """
        raise NotImplementedError


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
