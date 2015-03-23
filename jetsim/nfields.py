class NField(object):
    """
    Basic class that represents particle density in observer rest frame.
    """
    def n(self, x, y, z):
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
