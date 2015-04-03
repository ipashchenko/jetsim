import math
import numpy as np


mas_to_rad = 4.8481368 * 1E-09
rad_to_mas = 1. / mas_to_rad

# Parsec [cm]
pc = 3.0857 * 10 ** 18
# Mass of electron [g]
m_e = 9.109382 * 10 ** (-28)
# Mass of proton [g]
m_p = 1.672621 * 10 ** (-24)
# Charge of electron [C]
q_e = 1.602176 * 10 ** (-19)
# Charge of proton [C]
q_p = 1.602176 * 10 ** (-19)
# Speed of light [cm / s]
c = 3. * 10 ** 10


class AlongBorderException(Exception):
    pass


# Plasma frequency (default - for electrons)
def nu_plasma(n, q=q_e, m=m_e):
    """
    Returns plasma frequency for particles with charge ``q`` and mass ``m``.
    Default are electrons/positrons.
    :param n:
        Concentration [cm ** (-3)]
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Plasma frequency [Hz].
    """
    return np.sqrt(n * q ** 2. / (math.pi * m))


# Larmor frequency (default - for electrons)
def nu_b(B, q=q_e, m=m_e):
    """
    Returns larmor frequency for particles with charge ``q`` and mass ``m``.
    Default are electrons/positrons.
    :param B:
        Magnetic field [G]
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Larmor frequency [Hz].
    """
    return q * B / (2. * math.pi * m * c)


# TODO: I dont' need it: just use nu_b * sin(n, B)
# Larmor frequency with sin(n, B) (default - for electrons)
def nu_b_tr(n, B, q=q_e, m=m_e):
    """
    Returns larmor frequency for particles with charge ``q`` and mass ``m``.
    Default are electrons/positrons.
    :param n:
        Direction of emission.
    :param B:
        Magnetic field vecotr [G]
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Larmor frequency [Hz].
    """
    return q * abs(np.cross(n, B)) / (2. * np.linalg.norm(B) * math.pi * m * c)


# eta_0 (default - for electrons)
def eta_0(n, B, q=q_e, m=m_e):
    """
    Coefficient ``eta_0`` in emission coefficient.
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Coefficient ``eta_0`` used in expression for emission coefficient.
    """
    return math.pi * nu_plasma(n, q=q, m=m) ** 2. * nu_b(B, q=q, m=m) * m / c


# k_0 (default - for electrons)
def k_0(nu, n, B, q=q_e, m=m_e):
    """
    Coefficient ``k_0`` in absorption coefficient.
    :param nu:
        Frequency of radiation [Hz].
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
        Coefficient ``k_0`` used in expression for absorption coefficient.
    """
    return math.pi * nu_plasma(n, q=q, m=m) ** 2. * nu_b(B, q=q, m=m) /\
        (c * nu ** 2.)


def eta_I(nu, n, B, sin_theta, s=2.5, q=q_e, m=m_e):
    """
    Emission coefficient.
    :param nu:
        Frequency of radiation [Hz].
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param sin_theta:
        Sin of angle between direction of emission and magnetic field.
    :param s (optional):
        Power law index of electron energy distribution. Default is 2.5
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
    """
    return eta_0(n, B, q=q, m=m) * sin_theta *\
           (nu_b(B, q=q, m=m) * sin_theta / nu) ** ((s - 1.) / 2.) *\
           (3. ** (s / 2.) / (2. * (s + 1.))) *\
           math.gamma(s / 4. + 19. / 12.) * math.gamma(s / 4. - 1. / 12.)


def k_I(nu, n, B, sin_theta, s=2.5, q=q_e, m=m_e):
    """
    Absorption coefficient.
    :param nu:
        Frequency of radiation [Hz].
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param sin_theta:
        Sin of angle between direction of emission and magnetic field.
    :param s (optional):
        Power law index of electron energy distribution. Default is 2.5
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
    """
    return k_0(nu, n, B, q=q, m=m) * sin_theta *\
           (nu_b(B, q=q, m=m) * sin_theta / nu) ** (s / 2.) *\
           (3. ** ((s + 1.) / 2.) / 4.) *\
           math.gamma(s / 4. + 11. / 16.) * math.gamma(s / 4. + 1. / 6.)


def source_func(nu, n, B, sin_theta, s=2.5, q=q_e, m=m_e):
    """
    Source function
    :param nu:
        Frequency of radiation [Hz].
    :param n:
        Concentration [cm ** (-3)]
    :param B:
        Magnetic field [G]
    :param sin_theta:
        Sin of angle between direction of emission and magnetic field.
    :param s (optional):
        Power law index of electron energy distribution. Default is 2.5
    :param q (optional):
        Particle's charge. Default is ``q_e``.
    :param m (optional):
        Particle's mass. Default is ``m_e``.
    :return:
    """
    return eta_I(nu, n, B, sin_theta, s=s, q=q, m=m) / k_I(nu, n, B, sin_theta,
                                                           s=s, q=q, m=q)


def velsum(v, u):
    """
    Relativistic sum of two 3-velocities ``u`` and ``v``.
        u, v - 3-velocities [c]
    """

    gamma_v = 1. / math.sqrt(1. - np.linalg.norm(v))

    return (1. / (1. + v.dot(u))) * (v + (1. / gamma_v) * u +
                                     (gamma_v / (1. + gamma_v)) * v.dot(u) * v)


def boost_direction(v1, v2, n1):
    """
    :param v1:
        Velocity of first frame relative to observer frame.
    :param v2:
        Velocity of second frame relative to observer frame.
    :param n1:
        Direction of propagation in first RF that moves with velocity ``v1``.
    :return:
        Direction in RF that moves with velocity ``v2``.
    """
    v2r1 = velsum(v2, -v1)
    G2r1 = 1. / math.sqrt(1. - v2r1.dot(v2r1))
    # Direction of propagation in second RF.
    return (n1 + G2r1 * v2r1 * (G2r1 * n1.dot(v2r1) / (G2r1 + 1.) - 1.)) /\
           (G2r1 * (1. - n1.dot(v2r1)))


def doppler_factor(v1, v2, n1):
    """
    Function that calculates Doppler factor for RF2 that has velocity ``v2``
    relative to RF1 that has velocity ``v1`` and direction in RF1 ``n1``.
    :param v1:
        Velocity of first frame relative to observer frame.
    :param v2:
        Velocity of second frame relative to observer frame.
    :param n1:
        Direction of propagation in first RF.
    :return:
        Value of Doppler factor.
    :note:
        To find Doppler factor for emission boosted by jet moving with velocity
        v_jet relative to observer (observer has velocity v_obs=0) use:

        >>>doopler_factor(0, v_jet, n_obs)

        To find Doppler factor of emission deboosted (in jet RF):
        >>>n_jet = boost_direction(v_jet, 0, n_obs)
        >>>doppler_factor(v_jet, 0, n_jet)
    """
    v2r1 = velsum(v2, -v1)
    G2r1 = 1. / math.sqrt(1. - v2r1.dot(v2r1))
    D2r1 = 1. / (G2r1 * (1. - n1.dot(v2r1)))
    return D2r1


# G = 10.
# v2 = np.array([0, 0, math.sqrt(G**2-1)/G])
# v1 = np.array([0.0, 0, 0])
# n1 = np.array([-sin(1/G), 0, cos(1/G)])
# stokes1 = array([1., 0, 0, 0])
# TODO: add optional arg ``n2`` - direction in final rest frame. Thus make
# ``n1`` also optional.
def transfer_stokes(stokes1, v1, v2, n1, bf2):
    """
    Transfer stokes vector from frame (1) that has velocity v1 in observer frame
    to frame (2) that has velocity v2 in observer frame. Index 2 means value in
    second (final) rest frame. Index 1 means value in first (initial) rest
    frame.

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
    # Find Doppler factor of v2 relative to v1 and direction n1 in first RF.
    v2r1 = velsum(v2, -v1)
    G2r1 = 1. / math.sqrt(1. - v2r1.dot(v2r1))
    # Direction of propagation in second RF.
    # array([-0.9999986 ,  0.        ,  0.00167561])
    n2 = (n1 + G2r1 * v2r1 * (G2r1 * n1.dot(v2r1) / (G2r1 + 1.) - 1.)) / \
         (G2r1 * (1. - n1.dot(v2r1)))
    D2r1 = 1. / (G2r1 * (1. - n1.dot(v2r1)))
    # print "D = ", D2r1

    I1, Q1, U1, V1 = stokes1
    LP1 = math.sqrt(Q1 ** 2. + U1 ** 2.)
    chi1 = math.atan2(U1, Q1)
    # Polarization angle in first RF
    # array([ 0.,  1.,  0.])
    e1 = np.array([n1[2] * math.sin(chi1),
                   math.cos(chi1),
                   -n1[0] * math.sin(chi1)])
    # Polarization angle in second RF
    # array([ 0.        ,  0.09983356,  0.        ])
    e2 = G2r1 * (e1 - (G2r1 / (G2r1 + 1)) * e1.dot(v2r1) * v2r1 +
                 np.cross(v2r1, np.cross(n1, e1)))
    # FIXME: There should be * (compare v1=0 v2~c)
    I2 = I1 / D2r1 ** 3.
    V2 = V1 / D2r1 ** 3.
    LP2 = LP1 / D2r1 ** 3.
    chi2 = math.acos(((bf2 - bf2.dot(n2) * n2) / np.linalg.norm(bf2 - bf2.dot(n2) * n2)).dot(e2 / np.linalg.norm(e2)))
    Q2 = LP2 * math.cos(2. * chi2)
    U2 = LP2 * math.sin(2. * chi2)

    return np.array([I2, Q2, U2, V2])


def comoving_transverse_distance(z, H_0=73.0, omega_M=0.3, omega_V=0.7,
                                 format="pc"):
    """
    Given redshift ``z``, Hubble constant ``H_0`` [km/s/Mpc] and
    density parameters ``omega_M`` and ``omega_V``, returns comoving transverse
    distance (see arXiv:astro-ph/9905116v4 formula 14). Angular diameter
    distance is factor (1 + z) lower and luminosity distance is the same factor
    higher.

    """
    from scipy.integrate import quad
    fmt_dict = {"cm": 9.26 * 10.0 ** 27.0, "pc": 3. * 10 ** 9, "Mpc": 3000.0}

    result = (H_0 / 100.0) ** (-1.) * quad(lambda x: (omega_M * (1. + x ** 3) +
                                                      omega_V) ** (-0.5),
                                           0, z)[0]
    try:
        return fmt_dict[format] * result
    except KeyError:
        raise Exception('Format  \"pc\", \"cm\" or \"Mpc\"')


def pc_to_mas(z):
    """
    Return scale factor that convert from parsecs to milliarcseconds .

    """
    # Angular distance in pc
    d_a = comoving_transverse_distance(z, format='pc') / (1. + z)
    # Angle in radians
    angle = 1. / d_a
    return rad_to_mas * angle


def mas_to_pc(z):
    """
    Return scale factor that convert from milliarcseconds to parsecs.

    """
    # Angular distance in pc
    d_a = comoving_transverse_distance(z, format='pc') / (1. + z)
    return mas_to_rad * d_a


def generate_ndim_random_directions(n=3, k=1):
    """
    Generate ``k`` random unit vectors in n-dimensional space.

    :param n:
        Dimension of space.
    :param k: (optional)
        NUmber of vectors to generate.
    :return:
        List of k n-dim vectors.

    :note:
        http://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
        by Jim Lewis

    """
    result = list()
    vec_count = 0
    while vec_count < k:
        # Generate n uniformly distributed values a[0..n-1] over [-1, 1]
        a = np.random.uniform(low=-1, high=1, size=n)
        r_squared = sum(a ** 2.)
        if 0 < r_squared <= 1:
            # Normalize to length 1
            b = a / math.sqrt(r_squared)
            # Add vector b[0..n-1] to output list
            result.append(b)
            vec_count += 1
        else:
            # Reject this sample
            continue

    return result


def enlarge(arr, indxs, k):
    """
    Enlarge array ``arr`` using mask such way that ``False`` values are deleted
    from ``arr`` and on their places new subarrays are added with values
    linearly interpolating ``True`` values.
    :param arr:
        Numpy 1D array to be enlarged.
    :param indxs:
        Iterable if indexes of elements of ``arr`` to be substituted by ``k``
        elements each.
    :param k:
        Number of elements to substitute those indexed by ``indxs`` in ``arr``.
        Single number or iterable of length ``len(indxs)``.
    :return:
        Enlarged numpy 1D array with values in added elements linearly
        interpolated.

    """
    # If ``k`` is single number then create array of ``k`` with length equal
    # ``len(indxs)``
    try:
        assert len(k) == len(indxs)
        k = np.asarray(k)
    except AssertionError:
        k = k * np.ones(len(indxs), dtype=int)

    # Create empty enlarged array
    # new_arr = np.empty(len(arr) + len(indxs) * (k - 1), dtype=float)
    new_arr = np.empty(len(arr) + sum(k - 1), dtype=float)
    # Find new indexes of elements that won't be substituted in new array
    indxs_old = np.delete(np.indices(arr.shape)[0], indxs)
    # Get this values from original array
    new_arr[i_(indxs_old, indxs, k)] = arr[indxs_old]
    # Interpolate/extrapolate substituted values in enlarged array
    indxs_where_to_interp = np.delete(np.indices(new_arr.shape)[0],
                                      i_(indxs_old, indxs, k))
    new_arr[indxs_where_to_interp] = np.interp(indxs_where_to_interp,
                                               i_(indxs_old, indxs, k),
                                               arr[np.asarray(indxs_old)])
    return new_arr


def i_(indxs_old, indxs, k):
    """
    Returns indexes of elements that were not enlarged in new enlarged array.
    :param indxs_old:
        Indexes (in original 1D array) of elements that won't be substituted in
        new enlarged 1D array.
    :param indxs:
        Indexes of elements in original 1D array that will be substituted in new
        enlarged 1D array.
    :param k:
        One element is substituted by ``k`` elements if k is number or each i-th
        element from ``indxs`` (i=0, len(indxs)) is substituted by ``k[i]``.
    :return:
        Numpy array of indexes that were not substituted (in new 1D array).

    """
    indxs_old = np.asarray(indxs_old)
    indxs = np.asarray(indxs)

    # Number of substituted elements before current element
    temp = np.sum(np.array((indxs_old - indxs[:, np.newaxis]) > 0, dtype=int),
                  axis=0)
    k = np.insert(k, 0, 0)

    return np.cumsum(k)[temp] + indxs_old - temp