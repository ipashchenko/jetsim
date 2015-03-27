class NField(object):
    """
    Basic class that represents particle density in observer rest frame.
    """
    def n(self, x, y, z):
        raise NotImplementedError

    def n_vec(self, ps):
        raise NotImplementedError


class BKNField(NField):
    """
    Class that describes Blandford-Konigle particle density with r**(-2)
    dependence on z-distance
    """
    def __init__(self, z_0=1., n_0=500.):
        self.z_0 = z_0
        self.n_0 = n_0

    def n(self, x, y, z):
        return self.n_0 * (self.z_0 / z) ** 2.

    def n_vec(self, ps):
        """
        :param ps:
            Numpy array with shape (N, 3,) where N is the number of points.
        :return:
            Numpy array of values in N points.
        """
        return self.n_0 * (self.z_0 / ps[:, 2]) ** 2.
