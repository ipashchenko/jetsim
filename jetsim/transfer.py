import math
import numpy as np
from geometry import Ray
from utils import transfer_stokes


# TODO: I need more fine grid close to BH.
class Transfer(object):
    def __init__(self, jet, imsize, los_angle, pixsize=(1., 1.,)):
        self.jet = jet
        self.los_direction = (math.sin(los_angle), 0, -math.cos(los_angle))
        self.yscale = pixsize[0]
        self.zscale = pixsize[1]
        self.imsize = imsize
        self.image = {'I': np.zeros(imsize), 'Q': np.zeros(imsize),
                      'U': np.zeros(imsize), 'V': np.zeros(imsize)}
        # Arrays of coordinates
        y, z = np.meshgrid(np.arange(imsize[0]), np.arange(imsize[1]))
        y = y - imsize[0] / 2. + 0.5
        z = z - imsize[0] / 2. + 0.5
        # Array (imsize, 2,) of pixels coodinates. array[i, j] - coordinate of
        # pixel (i, j)
        image_coordinates = np.dstack((y, z))
        # Coordinates in jet frame
        jet_coordinates = image_coordinates.copy()
        jet_coordinates[..., 0] = image_coordinates[..., 0] * pixsize[0]
        jet_coordinates[..., 1] = image_coordinates[..., 1] * pixsize[1] /\
            math.sin(self.jet.geometry.angle)
        # Add zero x-coordinate for points in (yz)-jet plane
        jet_coordinates = np.dstack((np.zeros(imsize), jet_coordinates))
        self.image_coordinates = image_coordinates
        self.jet_coordinates = jet_coordinates

    def iter_row(self, row):
        for column in xrange(self.imsize[1]):
            origin = self.jet_coordinates[row, column, ...]
            yield (Ray(origin=origin, direction=self.los_direction),
                   (column, row))

    def __iter__(self):
        for row in xrange(self.imsize[0]):
            yield self.iter_row(row)

    def transfer(self, n=100, max_tau=None, max_delta=0.01):
        image = np.zeros(self.imsize)
        for row in self:
            for ray, pixel in row:
                stokes = self.jet.transfer_stokes_along_ray(ray, n=n,
                                                            max_tau=max_tau,
                                                            max_delta=max_delta)
                for i, stok in enumerate('I', 'Q', 'U', 'V'):
                    self.image[stok][pixel] = stokes[i]
        return image

    def transfer_along_ray(self, ps):
        """
        Transfer stokes parameters along LOS using ``n`` specified points.
        :param ps:
        :return:
        """
        stokes1 = np.zeros(4, dtype=float)
        for i in range(len(ps)):
            dl = np.linalg.norm(ps[i], ps[i + 1])
            k_I = self.jet.k_I_j(ps[i + 1])
            dtau = k_I * dl
            stokes0 = stokes1.copy()
            # Boost stokes0 from RF0 to RF1
            stokes1 = transfer_stokes(stokes0, self.jet(ps[i]),
                                      self.jet(ps[i + 1]), -self.los_direction,
                                      self.jet.bf_j(ps[i + 1]))
            stokes1 = stokes1 + np.array([self.jet.source_func_j(ps[i + 1]) *
                                          dtau - stokes1[0] * dtau, 0., 0., 0.])
