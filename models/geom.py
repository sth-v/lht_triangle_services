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
