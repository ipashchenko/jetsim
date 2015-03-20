import math
import numpy as np
from scipy import signal


def gaussianBeam(size_x, bmaj, bmin, bpa, size_y=None):
    """
    Generate and return a 2D Gaussian function
    of dimensions (size_x,size_y).

    See Briggs PhD (Appendix B) for details.

    :param size_x:
        Size of first dimension [pixels].
    :param bmaj:
        Beam major axis size [pixels].
    :param bmin:
        Beam minor axis size [pixels].
    :param bpa:
        Beam positional angle [deg].
    :param size_y (optional):
        Size of second dimension. Default is ``size_x``.
    :return:
        Numpy.ndarray of size (``size_x``, ``size_y``,).
    """
    size_y = size_y or size_x
    x, y = np.mgrid[-size_x: size_x + 1, -size_y: size_y + 1]
    # Constructing parameters of gaussian from ``bmaj``, ``bmin``, ``bpa``.
    a0 = 1. / (0.5 * bmaj) ** 2.
    c0 = 1. / (0.5 * bmin) ** 2.
    theta = math.pi * (bpa + 90.) / 180.
    a = math.log(2) * (a0 * math.cos(theta) ** 2. +
                       c0 * math.sin(theta) ** 2.)
    b = (-(c0 - a0) * math.sin(2. * theta)) * math.log(2.)
    c = math.log(2) * (a0 * math.sin(theta) ** 2. +
                       c0 * math.cos(theta) ** 2.)

    g = np.exp(-a * x ** 2. - b * x * y - c * y ** 2.)
    # FIXME: It is already normalized?
    # return g/g.sum()
    return g


# TODO: bmaj & bmin must be in pixels!!!
class Beam(object):
    """
    Class that represents central part of point spread function.
    """
    def __init__(self, bmaj=None, bmin=None, bpa=None, size=None):
        # Number of pixels for bmaj & bmin
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        # Size of map to convolve with
        self.size = size
        self.image = gaussianBeam(self.size[0], self.bmaj, self.bmin, self.bpa,
                                  self.size[1])

    # Convolve with any object that has ``image`` attribute
    def convolve(self, image_like):
        return signal.fftconvolve(self.image, image_like, mode='same')
