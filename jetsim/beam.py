from scipy import signal
from data_io import get_fits_image_info
from utils import gaussianBeam


# * Resolution could depend on position in map or frequency. Should i associate
# instance of ``Beam`` with each pixel in ``Image`` instance in such cases?
# * Beam is connected with ``CCModel`` instance naturally by deconvolution
# process? NO! Delta functions can be used in modelling without CLEAN!
def create_clean_beam_from_fits(fname):
    """
    Create instance of ``CleanBeam`` from FITS-file of CLEAN image.
    """
    imsize, pixref, (bmaj, bmin, bpa), pixsize = get_fits_image_info(fname)
    return CleanBeam(bmaj, bmin, bpa, imsize)


class Beam(object):
    """
    Basic class that represents point spread function.
    """
    def __init__(self):
        self.image = None

    # Convolve with any object that has ``image`` attribute
    def convolve(self, image_like):
        return signal.fftconvolve(self.image, image_like.image, mode='same')


class DirtyBeam(Beam):
    """
    Class that represents point spread function.
    """
    def fit_central_part(self):
        """
        Fit central part of dirty beam and return instance of ``CleanBeam``.
        """
        pass


# TODO: bmaj & bmin must be in pixels!!!
class CleanBeam(Beam):
    """
    Class that represents central part of point spread function.
    """
    def __init__(self, bmaj=None, bmin=None, bpa=None, size=None):
        self.bmaj = bmaj
        self.bmin = bmin
        self.bpa = bpa
        self.size = size
        self.image = gaussianBeam(self.size[0], self.bmaj, self.bmin, self.bpa,
                                  self.size[1])
