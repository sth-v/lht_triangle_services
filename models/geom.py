import json
from functools import cached_property, lru_cache

from mmcore.baseitems import Matchable
from mmcore.geom.base import Point

from models.connections import *


class CutPanels:
    def __init__(self, crvs):
        object.__init__(self)
        self._crvs = crvs

    @cached_property
    def a(self):
        return rs.surface.AddPlanarSrf(self._crvs)[0]

    def is_intersect(self, yy):
        jj = []
        for j in range(len(self._crvs)):
            jj.append(not rss.PlanarCurveCollision(yy, self._crvs[j]))
            return not all(jj)

    def intersect(self, yy):
        drf = rs.AddPlanarSrf(rs.coerceguid(yy))
        return rss.IntersectBreps(drf, self.a)

    def union(self, yy):
        drf = rs.AddPlanarSrf(rs.coerceguid(yy))
        return rss.BooleanUnion(drf, self.a)

    @lru_cache(maxsize=1024)
    def solve(self, y):
        return map(self.cut, filter(self.is_intersect, y))

    def __call__(self, triangles):
        self.cutted_triangles = self.solve(triangles)
        return self.cutted_triangles


class PointRhp(Point, Matchable):
    __match_args__ = "x", "y"

    def to_rhino(self) -> rg.Point3d:
        return rg.Point3d(*self.xyz)

    def toJSON(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_rhino(cls, point: rg.Point3d) -> 'PointRhp':
        return PointRhp(x=point.X, y=point.Y, z=point.Z)


class SimpleTriangle(Matchable):
    __match_args__ = "a", "b", "c"

    def to_shapely(self):
        return shapely.Polygon([self.a, self.b, self.c])


def check(x, y, eps=0.1):
    x.Transform(rg.Transform.PlanarProjection(rg.Plane.WorldXY))
    [yy.Transform(rg.Transform.PlanarProjection(rg.Plane.WorldXY)) for yy in y]
    ap = rg.AreaMassProperties.Compute(x)
    dst = list(map(lambda yy: (yy, ap.Centroid.DistanceTo(yy.ClosestPoint(ap.Centroid))), y))
    dst.sort(key=lambda xx: xx[1])
    if dst[0][1] < eps:
        return all(
            [(not rg.Curve.PlanarCurveCollision(x, crv.ToNurbsCurve(), rg.Plane.WorldXY, 0.1)) for crv in
             list(dst[0][0].Curves2D)
             ])
    else:
        return False


def main_func(aa, bb, eps=0.1):
    for aaa in aa:
        yield check(aaa, bb, eps=eps)


# a=list(main_func(x,y, eps=40))
def check2(x, y, eps=0.1):
    x.Transform(rg.Transform.PlanarProjection(rg.Plane.WorldXY))
    [yy.Transform(rg.Transform.PlanarProjection(rg.Plane.WorldXY)) for yy in y]
    ap = rg.AreaMassProperties.Compute(x)
    dst = list(map(lambda yy: (yy, ap.Centroid.DistanceTo(yy.ClosestPoint(ap.Centroid))), y))
    dst.sort(key=lambda xx: xx[1])
    if dst[0][1] < eps:
        return all(
            [(rg.Curve.PlanarCurveCollision(x, crv.ToNurbsCurve(), rg.Plane.WorldXY, 0.1)) for crv in
             list(dst[0][0].Curves2D)
             ])
    else:
        return False


def mask2(aa, bb, eps=300):
    for aaa in aa:
        yield check2(aaa, bb, eps=eps)


import shapely
from shapely import Polygon, prepare

def ordered_compare(a, b):
    dct=dict(eq=[], not_eq=[])
    for k,(i, j) in enumerate(zip(a, b)):
        if i==j:
            dct["eq"].append(k)
        else:
            dct["not_eq"].append(k)
    return dct


def strictly_contains_mask(mask):
    prepare(mask)

    def masked(items):
        return shapely.contains_properly(mask, items).tolist()

    return masked
