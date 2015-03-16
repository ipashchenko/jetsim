import math
import time
import numpy as np
from multiprocessing import Pool
from geometry import Ray
from jet import Jet
from utils import mas_to_pc


def unwrap_self_transfer(arg, **kwargs):
    return Transfer.transfer_along_rays(*arg, **kwargs)


# TODO: I need more fine grid close to BH.
# TODO: Keep redshift and cosmological stuff here?
class Transfer(object):
    """
    Class that describes transfer of radiation through jet.

    :param jet:
        Instance of ``Jet`` class.
    :param imsize:
        Tuple of image sizes for each of 2 map dimensions (int x int).
    :param los_angle:
        L.O.S. angle.
    :param pixsize:
        Tuple of pixel sizes in mas for each of 2 map dimensions.
    :param z: (optional)
        Cosmological redshift.
    :param nu_obs: (optional)
        Observing frequency [GHz].
    """
    def __init__(self, jet, los_angle, imsize, pixsize, z=0.5, nu_obs=5.):
        self.jet = jet
        self.jet.set_obs_frequency(nu_obs)
        self.jet.set_redshift(z)
        self.los_angle = los_angle
        self.los_direction = (math.sin(los_angle), 0, -math.cos(los_angle))
        self.yscale = pixsize[0] * mas_to_pc(z)
        self.zscale = pixsize[1] * mas_to_pc(z)
        self.pixsize = pixsize
        self.imsize = imsize
        self.z = z
        self.nu_obs = nu_obs
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
        jet_coordinates[..., 0] = image_coordinates[..., 0] * self.yscale
        jet_coordinates[..., 1] = image_coordinates[..., 1] * self.zscale /\
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

    jet = Jet()
    transfer = Transfer(jet, los_angle=0.1, imsize=(100, 100,),
                        pixsize=(0.0005, 0.0005,), z=0.1, nu_obs=5.)
    t1 = time.time()
    transfer.transfer(n=50)
    # result = transfer.transfer_mp()
    t2 = time.time()
    print t2 - t1
