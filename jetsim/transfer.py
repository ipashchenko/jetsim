import math
import time
import numpy as np
from multiprocessing import Pool
from geometry import Ray
from jet import Jet


def unwrap_self_transfer(arg, **kwargs):
    return Transfer.transfer_along_rays(*arg, **kwargs)


# TODO: I need more fine grid close to BH.
# TODO: Keep redshift and cosmological stuff here?
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

    # TODO: Use multiprocessing
    def transfer(self, n=100, max_tau=None, max_delta=0.01):
        for row in self:
            for ray, pixel in row:
                # print "processing pixel ", pixel
                stokes = self.jet.transfer_stokes_along_ray(ray, n=n,
                                                            max_tau=max_tau,
                                                            max_delta=max_delta)
                for i, stok in enumerate(['I', 'Q', 'U', 'V']):
                    self.image[stok][pixel] = stokes[i]

    def transfer_along_rays(self, rays, pixels, n=100, max_tau=None,
                            max_delta=0.01):
        for ray, pixel in zip(rays, pixels):
            print "processing pixel ", pixel
            stokes = self.jet.transfer_stokes_along_ray(ray, n=n,
                                                        max_tau=max_tau,
                                                        max_delta=max_delta)
            return stokes
            # for i, stok in enumerate(['I', 'Q', 'U', 'V']):
            #     print "adding to image stokes ", stok
            #     print "value", stokes[i]
            #     self.image[stok][pixel] = stokes[i]

    def transfer_mp(self):
        pool = Pool(processes=4)
        rays_all = [[(r[0], r[1]) for r in row] for row in self]
        rays_list = list()
        pixels_list = list()
        for i in range(len(rays_all)):
            rays, pixels = zip(*rays_all[i])
            rays = list(rays)
            pixels = list(pixels)
            rays_list.append(rays)
            pixels_list.append(pixels)

        result = pool.map(unwrap_self_transfer, zip([self]*len(rays_list),
                                                    rays_list, pixels_list))
        pool.close()
        pool.join()
        return result


if __name__ == '__main__':

    jet = Jet(nu=5., z=0.5)
    transfer = Transfer(jet, (50, 50,), math.pi/6., pixsize=(0.05, 0.05,))
    t1 = time.time()
    transfer.transfer(n=50)
    # result = transfer.transfer_mp()
    t2 = time.time()
    print t2 - t1
