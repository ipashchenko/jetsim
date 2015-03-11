import math
import numpy as np


class AlongBorderException(Exception):
    pass

class Geometry(object):
    """
    Class that represents different possible jet geometries and
    interceptions of rays with jet's boundaries.
    """
    def hit(self, ray):
        raise NotImplementedError


class Cone(Geometry):
    def __init__(self, origin, direction, angle=math.pi / 12.):
        self.origin = np.array(origin)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.angle = float(angle)

    def hit(self, ray):
        # return (back, front,)
        angle = self.angle
        direction = self.direction
        dp = ray.origin - self.origin
        expr1 = ray.direction - np.dot(ray.direction, direction) * direction
        expr2 = dp - np.dot(dp, direction) * direction
        a = math.cos(angle) ** 2 * np.dot(expr1, expr1) - \
            math.sin(angle) ** 2 * np.dot(ray.direction, direction) ** 2
        b = 2 * math.cos(angle) ** 2 * np.dot(expr1, expr2) - \
            2 * math.sin(angle) ** 2 * np.dot(ray.direction, direction) * np.dot(dp, direction)
        c = math.cos(angle) ** 2 * np.dot(expr2, expr2) - \
            math.sin(angle) ** 2 * np.dot(dp, direction) ** 2
        # print "a : ", a
        # print "b : ", b
        # print "c : ", c
        d = b ** 2 - 4. * a * c
        if d < 0:
            print "No interceptions!"
            return None
        if a == 0:
            raise AlongBorderException
        # print "Descriminant : ", d
        t1 = (-b + math.sqrt(d)) / (2. * a)
        t2 = (-b - math.sqrt(d)) / (2. * a)
        # print "Solutions : ", t1, t2
        #return ray.point(min(t1, t2)), ray.point(max(t1,t2))
        return t1, t2


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


def get_interception(yz, angle, yscale=1., zscale=1., cone_origin=(0, 0, 0),
                     cone_direction=(0, 0, 1), cone_angle=math.pi/6):
    """
    Given coordinates ``yz`` in image plane and ``angle`` between LOS and jets
    direction returns points of interception of LOS and jets border.
    :param yz:
    :param angle:
    :return:
    """
    z = yz[1] * zscale / math.sin(angle)
    y = yz[0] * yscale
    x = 0.
    ray = Ray(origin=(x, y, z),
              direction=(math.sin(angle), 0, -math.cos(angle)))
    cone = Cone(origin=cone_origin, direction=cone_direction, angle=cone_angle)
    return cone.hit(ray)


def get_thickness(yz, angle, yscale=1., zscale=1., cone_origin=(0, 0, 0),
                     cone_direction=(0, 0, 1), cone_angle=math.pi/6):
    try:
        result = get_interception(yz=yz, angle=angle, yscale=yscale, zscale=zscale,
                                  cone_origin=cone_origin,
                                  cone_direction=cone_direction,
                                  cone_angle=cone_angle)
        print result
        # If no interception
        if result is None:
            thickness = 0
            print "no interceptions"
        else:
            print "2 interceptions"
            # If there's two interceptions
            # If both interceptions with jet or with counterjet
            if (result[1][2] > 0 and result[0][2] > 0) or (result[1][2] < 0 and result[0][2] < 0):
                print "both interceptions with jet or counterjet"
                delta = result[1] - result[0]
                thickness = np.linalg.norm(delta)
            # If one interception with jet and other with counter jet
            elif (result[1][2] > 0 and result[0][2] < 0) or (result[1][2] < 0 and result[0][2] > 0):
                print "one interc. with jet & other with counterjet"
                thickness = None
            else:
                thickness = 0

    except AlongBorderException:
        thickness = 0

    return thickness

