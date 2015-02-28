import math
import numpy as np
from geometry import Cone, Ray


class Transfer(object):
    def __init__(self, imsize, los_angle, pixsize=(1., 1.,)):
        self.geometry = Cone(origin=(0., 0., 0.), direction=(0., 0., 1.),
                             angle=math.pi/6.)
        self.los_direction = (math.sin(los_angle), 0, -math.cos(los_angle))
        self.yscale = pixsize[0]
        self.zscale = pixsize[1]
        self.imsize = imsize
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
                                  math.sin(self.geometry.angle)
        # Add zero x-coordinate for points in (yz)-jet plane
        jet_coordinates = np.dstack((np.zeros(imsize), jet_coordinates))
        self.image_coordinates = image_coordinates
        self.jet_coordinates = jet_coordinates

    def iter_row(self, row):
        for column in xrange(self.imsize[1]):
            origin = self.jet_coordinates[row, column, ...]
            yield (Ray(origin = origin, direction = self.los_direction),
                   (column, row))

    def __iter__(self):
        for row in xrange(self.imsize[0]):
            yield self.iter_row(row)

    def generate_interceptions(self):
        image = np.zeros(self.imsize)
        for row in self:
            for ray, pixel in row:
                try:
                    p1, p2 = self.geometry.hit(ray)
                    dp = np.linalg.norm(p1 - p2)
                    image[pixel] = dp
                    print dp
                except TypeError:
                    dp = 0.
                    image[pixel] = dp
        return image



