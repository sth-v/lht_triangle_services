import json
from enum import IntEnum
from functools import cached_property, lru_cache
from typing import Any

import numpy as np

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

def to_ls(points):
    for pts in points:
        ptts = []
        for pt in pts:
            ptts.append((pt.X, pt.Y, pt.Z))

        yield shapely.LineString(ptts)
def strictly_contains_mask(mask):
    prepare(mask)

    def masked(items):
        return shapely.contains_properly(mask, items).tolist()

    return masked

def to_poly(self):
        for pts in self.curve.points:
            ptts = []
            for pt in pts:
                ptts.append((pt.X, pt.Y, pt.Z))

            yield shapely.Polygon(ptts)

import pydantic
from collections import ChainMap
class Link(object):
    __slots__ = 'prev', 'next', 'key', 'data'



class IntersectConditionStatus(IntEnum):
    outside = 0
    standard = 1
    cutted = 2
    excluded = 3

from collections import ChainMap, namedtuple

ChainMap()
class IntersectConditionServiceSolution( pydantic.BaseModel):
    index: int
    uuid: pydantic.UUID4
    reason: tuple[bool, bool]
    status: IntersectConditionStatus





class IntersectConditionService(Matchable):
    #IntersectConditionService   __match_args__ = "curve", "offset"
    curve: Any
    offset: float


    @property
    def front_side_curve(self):

        return self.curve.to_shapely()
    @property
    def back_side_curve(self):
        return self.front_side_curve.to_shapely().buffer(self.offset)

    def condition(self, pts):

        ptts = []
        for pt in pts:
            ptts.append((pt.X, pt.Y, pt.Z))

        return shapely.intersects(self.front_side_curve, shapely.LineString(ptts)),\
            shapely.intersects(self.back_side_curve,shapely.LineString(ptts))

    def check(self, triangles):
        for i, tr in enumerate(triangles):
            res = self.condition(tr)

            if res == (False, False):
                yield {"index": i, "status": "excluded", "reason": res, "comment": "Панель вне зоны примыкания",
                       "id": 0}
            elif res == (False, True):
                yield {"index": i, "status": "standard", "reason": res,
                       "comment": "Расстояние до стены позволяет оставить панель цельной", "id": 1}
            elif res == (True, True):
                yield {"index": i, "status": "cutted", "reason": res, "comment": "Подрезная паналь", "id": 2}
            else:
                yield {"index": i, "status": "excluded", "reason": res,
                       "comment": "Панель не доходит до внешней границы стены и может быть полностью исключена",
                       "id": 3}


def divide_by_count(crv, count):
    c = crv.ToNurbsCurve()
    for t in np.linspace(c.Domain.T0, c.Domain.T1, count):
        yield Point(c.PointAt(t).X, c.PointAt(t).Y, c.PointAt(t).X)


from fastapi import FastAPI
app=FastAPI()
