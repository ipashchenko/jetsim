import math
import numpy as np
from geometry import Cone
from bfields import BFHelical
from vfields import FlatVField
from nfields import BKNField
from utils import AlongBorderException, k_I, source_func, m_e, q_e,\
    transfer_stokes


# All vectors returned by methods are in triangular coordinates
# FIXME: Does particle density transforms through G?
# TODO: Use some symbols for direction in methods (like self.ubf)
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
            # Default opening angle is pi/36
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

        # Cosmological redshift
        self.z = None
        # Frequency in observer frame [Hz]
        self.nu_obs = None
        self.nu = None

    def set_redshift(self, z):
        """
        Set cosmological redshift and shift observing frequency.

        :param z:
            Cosmological redshift of jet.

        """
        self.z = z
        try:
            self.nu = self.nu_obs * (1. + self.z)
        except ValueError:
            raise Exception("Set observation frequency using"
                            "Jet.set_obs_frequency method!")

    def set_obs_frequency(self, nu):
        """
        Set observing frequency.

        :param nu:
            Observing requency [GHz].

        """
        self.nu_obs = nu

    # TODO: If optical depth is Lorenz-invariant then traverse from front of jet
    # to tau=``tau_max`` in observer frame first to set initial point.
    # TODO: Add option of choosing ``n`` using only ``max_delta`` & ``max_tau``.
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
        # Optical depth
        tau = 0.
        if stokes is None:
            stokes = np.zeros(4, dtype=float)
        else:
            stokes = np.array(stokes)
        try:
            t1, t2 = self.geometry.hit(ray)
            # 1) Make default ``n`` cells
            dt = abs(t2 - t1) / n
            # Parameters of edges of cells
            t_edges = [min(t1, t2) + i * dt for i in xrange(n)]
            # Parametrs of centers of cells
            t_cells = [min(t1, t2) + (i + 0.5) * dt for i in xrange(n - 1)]
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
            # TODO: Do this outside ``Jet`` class! Here only transfer inside
            # jet.
            pass
            # 2) Cycle inside jet
            for i, p in enumerate(p_cells):
                x, y, z = p
                # Calculate physical distance between cell edges [cm]
                dp = 3.085677 * 10 ** 18. * np.linalg.norm(p_edges[i + 1] -
                                                           p_edges[i])
                # Calculate optical depth
                dtau = self.k_I_j(x, y, z, -ray.direction) * dp
                tau = tau + dtau
                # Calculate source function
                s_func = self.source_func_j(x, y, z, -ray.direction)
                # This adds to stokes vector in current cell rest frame
                dI = (s_func - stokes[0]) * dtau
                stokes[0] = stokes[0] + dI
            # 3) Coming out of jet
                pass
            # Stokes I can't be negative
            stokes[0][np.where(stokes[0] < 0)] = 0.
            # 4) Boost to observer rest frame
            stokes = transfer_stokes(stokes, self.vfield.v(x, y, z),
                                     np.zeros(3),
                                     self.n_j(-ray.direction, x, y, z),
                                     self.bf(x, y, z))

            result = stokes

        # If ``hit`` returns ``None`` => no interception of ray with jet.
        except TypeError:
            result = stokes
        except AlongBorderException:
            # Going to max_tau if given or just traversing along border
            pass

        return result, tau

    # TODO: One can use utils.doppler_factor function with ``v1`` = 0.
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

    # TODO: I don't need this method?
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

    def bf_j(self, x, y, z):
        """
        Returns vector of B-field of jet in plasma rest frame.
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

    def nu_j(self, n, x, y, z):
        """
        Frequency that corresponds to ``nu_obs`` in plasma rest frame.
        """
        return self.nu / self.D(n, x, y, z)

    # TODO: Use (7) if Lyutikov et al. 2005
    def bf(self, x, y, z):
        """
        Direction (unit vector) of B-field in observer rest-frame.
        """
        G = self.G(x, y, z)
        B_j = self.bf_j(x, y, z)
        B_j = B_j / np.linalg.norm(B_j)
        v = self.vf(x, y, z)
        return (1. / math.sqrt(1. - np.dot(B_j, v) ** 2.)) * (B_j - (G / (1. + G)) * np.dot(B_j, v) * v)
        #return ((1. + G) * B + G ** 2. * B.dot(v) * v) /\
        #       ((1. + G) * np.sqrt(1. + G ** 2. * (B.dot(v)) ** 2.))

    def nf_j(self, x, y, z):
        """
        Particle density in plasma rest-frame.
        :params x, y, z:
        :return:
        """
        return self.nfield.n(x, y, z) / self.G(x, y, z)

    def k_I_j(self, x, y, z, n):
        """
        Absorption coefficient in point (x, y, z) calculated in plasma rest
        frame.
        :params x, y, z:
        :param n:
            Vector of direction in observer rest frame.
        :return:
        """
        nf_j = self.nf_j(x, y, z)
        B_j = self.bf_j(x, y, z)
        n_j = self.n_j(n, x, y, z)
        nu_j = self.nu_j(n, x, y, z)
        sin_theta = np.linalg.norm(np.cross(n_j, B_j)) / np.linalg.norm(B_j)
        return k_I(nu_j, nf_j, np.linalg.norm(B_j), sin_theta, s=self.s,
                   q=self.q, m=self.m)

    def source_func_j(self, x, y, z, n):
        """
        Source function in point (x, y, z) calculated in plasma rest frame.
        :params x, y, z:
        :param n:
            Vector of direction in observer rest frame.
        :return:
        """
        nf_j = self.nf_j(x, y, z)
        B_j = self.bf_j(x, y, z)
        n_j = self.n_j(n, x, y, z)
        nu_j = self.nu_j(n, x, y, z)
        sin_theta = np.linalg.norm(np.cross(n_j, B_j)) / np.linalg.norm(B_j)
        return source_func(nu_j, nf_j, np.linalg.norm(B_j), sin_theta,
                           s=self.s, q=self.q, m=self.m)

    # TODO: Make it coefficient on observer frame for ``tau`` calculation.
    # TODO: Caution! self.bf is direction!
    def k_I(self, x, y, z, n):
        """
        Absorbtion coefficient in point (x, y, z) calculated in plasma rest
        frame.
        :params x, y, z:
        :return:
        """
        nf = self.nfield.n(x, y, z)
        B = self.bfield.bf(x, y, z)
        sin_theta = np.linalg.norm(np.cross(n, B)) / np.linalg.norm(B)
        return k_I(self.nu, nf, np.linalg.norm(B), sin_theta, s=2.5, q=q_e,
                   m=m_e)


if __name__ == '__main__':
    jet = Jet(nu=5., z=0.5)
    from transfer import Transfer
    transf = Transfer(jet, (10, 10,), math.pi/4)
    origin = np.array([0., 0.5, 3.])
    direction = transf.los_direction
    from geometry import Ray
    ray = Ray(origin, direction)
    n = -np.array(direction)

    t1, t2 = jet.geometry.hit(ray)
    k = 100
    dt = abs(t2 - t1) / k
    t_edges = [t2 + i * dt for i in xrange(k)]
    t_cells = [min(t1, t2) + (i + 0.5) * dt for i in xrange(k - 1)]
    p_edges = [ray.point(t) for t in t_edges]
    p_cells = [ray.point(t) for t in t_cells]
    i = 50
    p = p_cells[i]
    x, y, z = p
    dp = np.linalg.norm(p_edges[i + 1] - p_edges[i])
    B_j = jet.bf_j(x, y, z)
    nf_j = jet.nf_j(x, y, z)
    nu = jet.nu
    from utils import k_0, nu_plasma, nu_b
    print "nu_plasma", nu_plasma(nf_j)
    print "nu_b", nu_b(np.linalg.norm(B_j))
    print "k_0", k_0(nu, nf_j, np.linalg.norm(B_j))

    dtau = jet.k_I_j(x, y, z, n) * dp
    print "dtau", dtau
    s_func = jet.source_func_j(x, y, z, -ray.direction)
    print "source_func", s_func

    out_stokes = jet.transfer_stokes_along_ray(ray)
