import math
import numpy as np


class AlongBorderException(Exception):
    pass

eps = 10**(-5)
# Mass of electron
m_e = 0
# Charge of electron
e = -1.6 * 10 ** (-19)
# Speed of light [cm / s]
c = 3. * 10 ** 8

# Plasma frequency (default - for electrons)
def nu_plasma(n, q=e, m=m_e):
    """
    Returns plasma frequency for particles with charge ``q`` and mass ``m``.
    Default are electrons/positrons.
    :param n:
        Concentration [cm ** (-3)]
    :param q (optional):
        Particle's charge. Default is ``e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Plasma frequency [Hz].
    """
    return math.sqrt(n * q ** 2. / (math.pi * m))


# Larmor frequency (default - for electrons)
def nu_b(B, q=e, m=m_e):
    """
    Returns larmor frequency for particles with charge ``q`` and mass ``m``.
    Default are electrons/positrons.
    :param B:
        Magnetic field [G]
    :param q (optional):
        Particle's charge. Default is ``e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Larmor frequency [Hz].
    """
    return q * B / (2. * math.pi * m * c)

# eta_0 (default - for electrons)
def eta_0(n, B, q=e, m=m_e):
    """
    Coefficient ``eta_0`` in emission coefficient.
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param q (optional):
        Particle's charge. Default is ``e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Coefficient ``eta_0`` used in expression for emission coefficient.
    """
    return math.pi * nu_plasma(n, q=q, m=m) ** 2. * nu_b(B, q=q, m=m) * m / c


# k_0 (default - for electrons)
def k_0(nu, n, B, q=e, m=m_e):
    """
    Coefficient ``k_0`` in absorption coefficient.
    :param nu:
        Frequency of radiation [Hz].
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param q (optional):
        Particle's charge. Default is ``e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Coefficient ``k_0`` used in expression for absorption coefficient.
    """
    return math.pi * nu_plasma(n, q=q, m=m) ** 2. * nu_b(B, q=q, m=m) /\
           (c * nu ** 2.)

# emission coeff. (for electrons)
# eta_I = eta_0*math.sin(theta)*(nu_B*math.sin(theta)/nu)**((s-1.)/2.)*(3.**(s/2.)/(2.*(s+1.)))*Gamma(s/4.+19./12.)*Gamma(s/4.-1./12.)
# absorbtion coeff.
# k_I = k_0*math.sin(theta)*(nu_B*math.sin(theta)/nu)**(s/2.)*(3.**((s+1.)/2.)/4.)*Gamma(s/4.+11./16.)*Gamma(s/4.+1./6.)


def velsum(v, u):
    """
    Relativistic sum of two 3-velocities ``u`` and ``v``.
        u, v - 3-velocities [c]
    """

    gamma_v = 1. / math.sqrt(1. - np.linalg.norm(v))

    return (1. / (1. + v.dot(u))) * (v + (1. / gamma_v) * u +
                                     (gamma_v / (1. + gamma_v)) * v.dot(u) * v)


def luminosity_distance(z, H_0=73.0, omega_M=0.3, omega_V=0.7, format="cm"):
    """
    Given redshift z, Hubble constant H_0 [km/s/Mpc] and
    density parameters omega_M and omega_V, returns luminosity
    distance.
    """

    from scipy.integrate import quad

    if format == "cm":
        return 9.26 * 10.0 ** 27.0 * (H_0 / 100.0) ** (-1.) * (1. + z) *\
               quad(lambda x: (omega_M * (1. + x ** 3) + omega_V) ** (-0.5),
                    0, z)[0]
    elif format == "Mpc":
        return 3000.0 * (H_0 / 100.0) ** (-1.) * (1. + z) *\
               quad(lambda x: (omega_M * (1. + x ** 3) + omega_V) ** (-0.5),
                    0, z)[0]
    else:
        raise Exception('Format=\"cm\" or \"Mpc\"')


def transfer_stokes(stokes1, v1, v2, n1, bf2):
    """
    Transfer stokes vector from frame that has velocity v1 in observer frame to
    frame that has velocity v2 in observer frame. Index 2 means value in second
    (final) rest frame. Index 1 means value in first (initial) rest frame.

    :param stokes1:
        Stokes vector in RF that has velocity v1 relative to observer frame.
    :param v1:
        Velocity of first frame relative to observer frame.
    :param v2:
        Velocity of second frame relative to observer frame.
    :param n1:
        Direction of propagation in first RF.
    :param bf2:
        Direction of B-field in second RF.
    :return:
        Stokes vector in second rest frame.
    """
    # Find Doppler factor of v2 relative to v1
    v2r1 = velsum(v2, -v1)
    G2r1 = 1. / math.sqrt(1. - v2r1.dot(v2r1))
    # Direction of propagation in second RF.
    n2 =  (n1 + G2r1 * v2r1 * (G2r1 * n1.dot(v2r1) / (G2r1 + 1.) - 1.)) / \
          (G2r1 * (1. - n1.dot(v2r1)))
    D2r1 = 1. / (G2r1 * (1. - n2.dot(v2r1)))

    I1, Q1, U1, V1 = stokes1
    LP1 = math.sqrt(Q1 ** 2. + U1 ** 2.)
    chi1 = math.atan2(U1, Q1)
    # Polarization angle in first RF
    e1 = np.array([n1[2] * math.sin(chi1),
                   math.cos(chi1),
                   -n1[0] * math.sin(chi1)])
    # Polarization angle in second RF
    e2 = G2r1 * (e1 - (G2r1 / (G2r1 + 1)) * e1.dot(v2r1) * v2r1 +
                 np.cross(v2r1, np.cross(n1, e1)))
    I2 = I1 / D2r1 ** 3.
    V2 = V1 / D2r1 ** 3.
    LP2 = LP1 / D2r1 ** 3.
    chi2 = math.acos(((bf2 - bf2.dot(n2) * n2) / np.linalg.norm(bf2 - bf2.dot(n2) * n2)) * e2 / np.linalg.norm(e2))
    Q2 = LP2 * math.cos(2. * chi2)
    U2 = LP2 * math.sin(2. * chi2)

    return np.array([I2, Q2, U2, V2])


def transform_to_lab(stokes, v):
    pass
