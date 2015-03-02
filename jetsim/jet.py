import math
import numpy as np
from bfields import BFHelical
from vfields import CentralVField
from nfields import BKNField

eps = 10**(-5)
m_e = 0
e = 0
c = 0

# plasma frequency (for electrons)
# nu_p = math.sqrt(n_e * e ** 2. / (math.pi * m_e))
# larmor frequency (for electrons)
# nu_B = e * B / (2. * math.pi * m_e * c)
# eta_0 (for electrons)
# eta_0 = math.pi * nu_p ** 2. * nu_B * m_e / c
# k_0 (for electrons), nu - frequency
# k_0 = math.pi * nu_p ** 2. * nu_B / (c * nu ** 2.)

# emission coeff. (for electrons)
# eta_I = eta_0*math.sin(theta)*(nu_B*math.sin(theta)/nu)**((s-1.)/2.)*(3.**(s/2.)/(2.*(s+1.)))*Gamma(s/4.+19./12.)*Gamma(s/4.-1./12.)
# absorbtion coeff.
# k_I = k_0*math.sin(theta)*(nu_B*math.sin(theta)/nu)**(s/2.)*(3.**((s+1.)/2.)/4.)*Gamma(s/4.+11./16.)*Gamma(s/4.+1./6.)

class Jet(object):
    """
    Class that represents jet's physical properties: distribution of magnetic
    field, particle density, velocity flow.
    """
    def __init__(self, bfield=BFHelical, vfield=CentralVField, nfield=BKNField,
                 bf_kwargs=None, vf_kwargs=None, nf_kwargs=None):
        if bf_kwargs is not None:
            self.bfield = bfield(bf_kwargs)
        else:
            self.bfield = bfield()
        if vf_kwargs is not None:
            self.vfield = vfield(bf_kwargs)
        else:
            self.vfield = vfield()
        if nf_kwargs is not None:
            self.nfield = nfield(nf_kwargs)
        else:
            self.nfield = nfield()

    def integrate(self, ray):
        """
        Integrate along ray.
        """
        pass

    def D(self, n, x, y, z):
        """
        Returns Doppler factor for point ``(x, y, z)`` and direction ``n`` in
        observer frame.
        :param n:
        :params x, y, z:
        :return:
        """
        v = self.vf(x, y, z)
        G = 1. / np.sqrt(1. - v.dot(v))
        return 1. / (G * (1. - n.dot(v)))

    def G(self, x, y, z):
        v = self.vf(x, y, z)
        return 1. / np.sqrt(1. - v.dot(v))

    def vf(self, x, y, z):
        """
        Velocity field of jet in observer rest frame.
        :param x, y, z:
            Rectangular coordinates. (3, N,) N-number of points
        """
        return self.vfield.v(x, y, z)

    def bf(self, x, y, z):
        """
        Returns vector of B-field of jet in observer frame.
        """
        return self.bfield.bf(x, y, z)

    def n_j(self, n, x, y, z):
        """
        Direction in plasma rest-frame.
        """
        v = self.vf(x, y, z)
        G = 1. / np.sqrt(1. - v.dot(v))
        return (n + G * v * (G * n.dot(v) / (G + 1.) - 1.)) /\
               (G * (1. - n.dot(v)))

    def bf_j(self, x, y, z):
        """
        Direction of B-filed in plasma rest-frame.
        """
        G = self.G(x, y, z)
        B = self.bf(x, y, z)
        v = self.vf(x, y, z)
        return ((1. + G) * B + G ** 2. * B.dot(v) * v) /\
               ((1. + G) * np.sqrt(1. + G ** 2. * (B.dot(v)) ** 2.))




def integrate(ray, t1, t2):
    tmin = min(t1, t2)
    tmax = max(t1, t2)
    p1 = ray.point(tmin)
    p2 = ray.point(tmax)
    if p1[2] > 0 and p2[2] < 0:
        p1 = ray.point(-1000)
    dt = tmax - tmin


