import numpy as np
from geometry import Cone
from bfields import BFHelical
from vfields import FlatVField
from nfields import BKNField
from utils import AlongBorderException, k_I, source_func, m_e, q_e


# All vectors returned by methods are in triangular coordinates
class Jet(object):
    """
    Class that represents jet's physical properties: geometry, distribution of
    magnetic field, particle density, velocity flow.
    """
    def __init__(self, geometry=Cone, bfield=BFHelical, vfield=FlatVField,
                 nfield=BKNField, geo_kwargs=None, bf_kwargs=None,
                 vf_kwargs=None, nf_kwargs=None, m=m_e, q=q_e, s=2.5):
        if geo_kwargs is not None:
            self.geometry = geometry((0., 0., 0.,), (0., 0., 1.,), bf_kwargs)
        else:
            self.geometry = geometry((0., 0., 0.,), (0., 0., 1.,))
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

        # Particle's properties (move to method?)
        self.m = m
        self.q = q
        self.s = s

        # Frequency in observer frame
        self.nu = None

    # TODO: If optical depth is Lorenz-invariant then traverse from front of jet
    # to tau=tau_max in observer frame first to set initial point.
    def transfer_stokes_along_ray(self, ray, stokes=None, n=100, max_tau=None,
                                  max_delta=0.01):
        """
        Transfer stokes vector along given ray.

        :param ray:
            Instance of Ray class. Ray along which to make polaization transfer.
        :param stokes (optioanl):
            Initial (background) stokes vector.
        :param n (optioanl):
            Number of cells to use.
        :param max_tau (optioanl):
            Maximum optical depth to traverse into jet. Ignore all emission with
            tau > max_tau.
        :param max_delta:
            Maximum fractional change of physical quantities (B-field, n, v) in
            neighbor cells. If fractional change is more then max_delta =>
            divide cell in two (recursevly).
        """
        if stokes is None:
            stokes = np.zeros(4, dtype=float)
        else:
            stokes = np.array(stokes)
        try:
            t1, t2 = self.geometry.hit(ray)
            # 1) Make default n cells
            dt = abs(t2 - t1) / n
            # Parameters of edges of cells
            t_edges = [t1 + i * dt for i in xrange(n)]
            # Parametrs of centers of cells
            t_cells = [t1 + (i + 0.5) * dt for i in xrange(n - 1)]
            # Edges of cells
            p_edges = [ray.point(t) for t in t_edges]
            # Centers of cells
            p_cells = [ray.point(t) for t in t_cells]
            # 2) For each cell check that relative ratio of B, n, v in plasma
            # rest frame less then user specified value ``max_delta``.
            pass
            # 3) Split cells in two where it is not so. Thus we have ``n`` +
            # delta cells
            pass
            # 4) Going from front of jet inside and find tau = sum(k_I * dl)
            # If tau < tau_max => OK. If not => use only first N cells where
            # tau < tau_max
            pass
            # Here we got N ``t`` values in ts
            # Now numerically integrate:
            # 1) Going from background into jet
            pass
            # 2) Cycle inside jet
            for i, p in enumerate(p_cells):
                x, y, z = p
                # Calculate physical distance between cell edges
                dp = np.linalg.norm(p_edges[i + 1] - p_edges[i])
                # Calculate optical depth
                dtau = self.k_I(x, y, z, -ray.direction)
                # Calculate source function
                s_func = self.source_func_j(x, y, z, -ray.direction)
                # This adds to stokes vector in current cell rest frame
                dI = (s_func - stokes[0]) * dtau
                stokes[0] = stokes[0] + dI
            # 3) Coming out of jet
            # 4) Boost in observer rest frame

        # If ``hit`` returns ``None`` => no interception of ray with jet.
        except TypeError:
            result = stokes
        except AlongBorderException:
            # Going to max_tau if given or just traversing along border
            pass

        return result

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

    # FIXME: It is not direction but vector?
    def bf_j(self, x, y, z):
        """
        Direction of B-field in plasma rest-frame.
        """
        G = self.G(x, y, z)
        B = self.bf(x, y, z)
        v = self.vf(x, y, z)
        return ((1. + G) * B + G ** 2. * B.dot(v) * v) /\
               ((1. + G) * np.sqrt(1. + G ** 2. * (B.dot(v)) ** 2.))

    def nf_j(self, x, y, z):
        """
        Particle density in plasma rest-frame.
        :params x, y, z:
        :return:
        """
        return self.nfield.n(x, y, z) / self.G(x, y, z)

    def k_I(self, x, y, z, n):
        """
        Absorption coefficient in point (x, y, z) calculated in plasma rest
        frame.
        :params x, y, z:
        :param n:
            Vector of direction in observer rest frame.
        :return:
        """
        n_j = self.nf_j(x, y, z)
        B_j = self.bf_j(x, y, z)
        n_j = self.n_j(n, x, y, z)
        sin_theta = np.cross(n_j, B_j) / np.linalg.norm(B_j)
        return k_I(self.nu, n_j, B_j, sin_theta, s=self.s, q=self.q, m=self.m)

    def source_func_j(self, x, y, z, n):
        """
        Source function in point (x, y, z) calculated in plasma rest frame.
        :params x, y, z:
        :param n:
            Vector of direction in observer rest frame.
        :return:
        """
        n_j = self.nf_j(x, y, z)
        B_j = self.bf_j(x, y, z)
        n_j = self.n_j(n, x, y, z)
        sin_theta = np.cross(n_j, B_j) / np.linalg.norm(B_j)
        return source_func(self.nu, n_j, sin_theta, s=self.s, q=self.q, m=self.m)

    def k_I_j(self, x, y, z):
        """
        Absorbtion coefficient in point (x, y, z) calculated in plasma rest
        frame.
        :params x, y, z:
        :return:
        """
        pass
