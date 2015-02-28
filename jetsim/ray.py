import math
import numpy as np


class Jet(object):
    """
    Class that represents jet's physical properties: distribution of magnetic
    field, particle density, velocity flow.
    """
    def __init__(self):
        pass

    def integrate(self, ray, back, front):
        """
        Integrate along ray from back to front.
        """
        pass


class Geometry(object):
    """
    Class that represents different possible jet geometries and
    interceptions of rays with jet's boundaries.
    """
    def hit(self, ray):
        raise NotImplementedError


class Cone(Geometry):
    def __init__(self, origin, direction, angle):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.angle = float(angle)

    def hit(self, ray):
        # return (back, front,)
        # 1. Find coefficients A, B, C
        angle = self.angle
        direction = self.direction
        dp = ray.origin - self.origin
        expr1 = ray.direction - np.dot(ray.direction, direction) * direction
        expr2 = dp - np.dot(dp, direction) * direction
        a = math.cos(angle) ** 2 * np.dot(expr1, expr1) -\
            math.sin(angle) ** 2 * np.dot(ray.direction, direction) ** 2
        b = 2 * math.cos(angle) ** 2 * np.dot(expr1, expr2) -\
            2 * math.sin(angle) ** 2 * np.dot(ray.direction, direction) * np.dot(dp, direction)
        c = math.cos(angle) ** 2 * np.dot(expr2, expr2) -\
            math.sin(angle) ** 2 * np.dot(dp, direction) ** 2
        d = b ** 2 - 4. * a * c
        t1 = (-b + math.sqrt(d)) / (2. * a)
        t2 = (-b - math.sqrt(d)) / (2. * a)
        return ray.point(min(t1, t2)), ray.point(max(t1,t2))


class Parabolic(Geometry):
    pass


class Cylinder(Geometry):
    pass


class Ray(object):
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))

    def point(self, t):
        return self.origin + self.direction * t


class ViewPlane(object):
    """
    Class that represents sky image of jet.
    """
    def __init__(self, imsize, pixel_size, direction):
        self.imsize = imsize
        self.pixel_size = pixel_size
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))

    def iter_row(self, row):
        for column in xrange(self.imsize[0]):
            origin = np.zeros(3)
            origin[0] = 0.
            origin[1] = self.pixel_size*(row - self.imsize[1] / 2 + 0.5)
            origin[2] = 100.0
            yield (Ray(origin = origin, direction = self.direction), (column, row))

    def __iter__(self):
        for row in xrange(self.imsize[1]):
            yield self.iter_row(row)


class Tracer(object):
    def __init__(self, transfer):
        self.transfer = transfer

    def trace_ray(self, ray):
        try:
            back, front = self.transfer.geometry.hit(ray)
            stokes = self.transfer.jet.intergate(ray, back, front)
        except ValueError:
            return (0., 0., 0., 0.)


class Transfer(object):
    def __init__(self, imsize, pixsize, geometry=Cone, geoargs=None,
                 geokwargs=None, jetargs=None, jetkwargs=None):
        self.viewplane = ViewPlane(imsize=imsize, pixsize=pixsize)
        self.geometry = Cone(center=(0.,0.,0.,), angle=0.5)
        self.jet = Jet(jetargs, jetkwargs)

    def transfer(self):
        image = np.array(self.viewplane.imsize)
        tracer = Tracer(self)
        for row in self.viewplane:
            for ray, pixel in row:
                image[pixel] = tracer.trace_ray(ray)


