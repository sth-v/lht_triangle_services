import io
import json
import os
import sys
import time
from enum import Enum
from types import TracebackType
from typing import Iterator, ContextManager, Type

import dotenv
import numpy as np
import scipy.spatial
from scipy.spatial import KDTree
from models.utils import decode_sv
import multiprocessing
import threading
import asyncio

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
from scipy.spatial import distance
import more_itertools
from mmcore.collections.multi_description import ElementSequence
import mmcore.baseitems.descriptors
from cxmdata import CxmData
from mmcore.baseitems import Matchable

from mmcore.geom.base import Polygon

from functools import lru_cache, cached_property

from models.geom import PointRhp, CutPanels

PRIMARY = "runitime:lht:ceiling:L2W:"
TOLERANCE = 20

from models.graphql import GQlEntity, InsertTypeMarkOne, UpdateTypeMarksByPKPartialTemp, TypeMarksReg, \
    TypeMarksByPkPartial, UpdatePanelTriangleByPKPartialTemp, PanelTriangleByPkPartial, InsertPanelTriangle, \
    PanelTriangleReg
from models.connections import *

# import Rhino.Geometry as rgg
logger_buf = io.StringIO()

triangles = CxmData(redis_conn.get(PRIMARY + "triangles")).decompress()
trianglesv2 = json.loads(redis_conn.get(PRIMARY + "pnl_json").decode())
mask = CxmData(redis_conn.get(PRIMARY + "mask")).decompress()
pnl_ns = CxmData(redis_conn.get(PRIMARY + "pnl_namespace")).decompress()
# tps = [[10, 20], [11, 21], [12, 22], [10, 20]]

TOLERANCE = 100
cutter = CutPanels(mask)


class TypeTagDesc(mmcore.baseitems.descriptors.DataDescriptor):

    def mutate(self):
        return GQlEntity(UpdateTypeMarksByPKPartialTemp(self.name))

    def query(self):
        return TypeMarksByPkPartial(self.name)

    def __get__(self, inst, own):
        r = self.query()(x=inst.x, y=inst.y)
        # print("r:  ", r)
        return r[0][self.name]

    def __set__(self, inst, v):
        self.mutate()(**{
            'x': inst.x,
            'y': inst.y,
            self.name: v
        }
                      )


class PanelTriangleDesc(mmcore.baseitems.descriptors.DataDescriptor):

    def mutate(self):
        return GQlEntity(UpdatePanelTriangleByPKPartialTemp(self.name))

    def query(self):
        return PanelTriangleByPkPartial(self.name)

    def __get__(self, inst, own):
        r = self.query()(x=inst.x, y=inst.y)
        # print("r:  ", r)
        return r[0][self.name]

    def __set__(self, inst, v):
        self.mutate()(**{
            'x': inst.x,
            'y': inst.y,
            self.name: v
        }
                      )


class TypeTag(Matchable):
    __match_args__ = "x", "y"
    tag = TypeTagDesc()
    uuid = TypeTagDesc()
    floor = TypeTagDesc()
    z = TypeTagDesc()
    _x = 0
    _y = 0
    _rhino_point = None

    def __init__(self, x, y, z=0.0, tag="A", no_post=False, **kwargs):
        self.x = x
        self.y = y
        if not no_post:
            dct = dict(InsertTypeMarkOne(x=self.x, y=self.y, z=z, tag=tag, **kwargs))

        super().__init__(self.x, self.y)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._rhino_point = self.to_rhino()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = round(v / TOLERANCE, 0) * TOLERANCE

    @classmethod
    def get_sequence(cls) -> ElementSequence:
        return ElementSequence(
            list(TypeTag(dct["x"], dct["y"], no_post=True) for dct in more_itertools.flatten(TypeMarksReg())))

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = round(v / TOLERANCE, 0) * TOLERANCE

    def distance(self, other):
        return distance.euclidean([self.x, self.y], [other.x, other.y])

    def __eq__(self, other):
        return (self.x == round(other.x / TOLERANCE, 0) * TOLERANCE) \
            and (self.y == round(other.y / TOLERANCE, 0) * TOLERANCE)

    @property
    def point(self):
        return PointRhp(self.x, self.y, 0.0)

    def to_rhino(self):
        return rg.Point3d(self.x, self.y, 0.0)

    @property
    def rhino_point(self):
        return self._rhino_point


def logtime(f):
    def wrp(*args, **kwargs):
        s = time.time()
        res = f(*args, **kwargs)
        delta = time.time() - s
        print(f'{f.__name__} time: {delta}')
        return res

    return wrp


class PanelTriangle(Matchable):
    __match_args__ = "centroid", "subtype", "tag", "cutted", "extra", "outside"
    extra = PanelTriangleDesc()
    cutted = PanelTriangleDesc()
    subtype = PanelTriangleDesc()
    tag = PanelTriangleDesc()
    uuid = PanelTriangleDesc()
    centroid = PanelTriangleDesc()
    _tag = "A"
    _x = None
    _y = None

    @logtime
    @property
    def mark(self):
        return self.tag + "-" + self.subtype

    @logtime
    def __init__(self, centroid, no_post=False, **kwargs):
        self.x = centroid["x"]
        self.y = centroid["y"]

        super().__init__(centroid, **kwargs)

        # self.a, self.b, self.c, = points
        if not no_post:
            InsertPanelTriangle(x=centroid.x, y=centroid.y, **kwargs)

    # def to_rhino(self):
    #    self._rh_tri = rg.Triangle3d(self.a.to_rhino(), self.b.to_rhino(), self.c.to_rhino())
    #    return self._rh_tri

    @classmethod
    def get_sequence(cls) -> ElementSequence:
        return ElementSequence(
            list(PanelTriangle(**dict((item, dct[item]) for item in cls.__match_args__), no_post=True) for dct in
                 more_itertools.flatten(PanelTriangleReg())))

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = round(v / TOLERANCE, 0) * TOLERANCE

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = round(v / TOLERANCE, 0) * TOLERANCE


class PaginateGhNamespace(Iterator):
    def __init__(self, data):
        self.headers = data.keys()
        self.data = zip(*data.values())

    def _next_page(self):
        return dict(zip(self.headers, next(self.data)))

    @classmethod
    def from_redis(cls, key):
        return cls(data=CxmData(redis_conn.get(key)).decompress())

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_page()


class RedisCxmDataBridge:
    iterator = PaginateGhNamespace
    name = None

    def __init__(self, key=None, primary=PRIMARY, conn=redis_conn, **kwargs):
        super().__init__()
        self.key = key
        self.primary = primary
        self.conn = conn
        self.__dict__ |= kwargs

    def __set_name__(self, owner, name):
        self.name = name

    def get_data(self):
        if self.name is None:
            key = self.primary + self.key
        else:
            key = self.primary + self.name
        return CxmData(redis_conn.get(key)).decompress()

    def get(self):
        return self.iterator(self.get_data())

    def __get__(self, instance, owner):
        return self.get()


from mmcore.gql.client import query, mutate, GqlString


class Floor(str, Enum):
    L2W = "L2W"
    L2 = "L2"
    L1 = "L1"
    B1 = "B1"


class Triangles(ElementSequence):
    floor: Floor
    query: query
    mutation: mutate
    mutation = mutate(GqlString("lht_ceiling", "panels"), {
        "mark",
        "tag",
        "x",
        "y"}, {
                          "objects": []
                      })

    def __init_subclass__(cls, floor: Floor = None, **kwargs):
        cls.floor = floor
        cls.query = query("lht_ceiling_panels(where: {floor: {_eq: $floor}}})".replace("$floor", floor), {
            "mark"
            "centroid",
            "tag",
            "subtype",
            "floor"
        }, variables=dict(floor=cls.floor))
        cls.mutation = mutate(GqlString("lht_ceiling", "panels"), {
            "mark",
            "tag",
            "x",
            "y"}, {
                              "objects": []
                          })

        super().__init_subclass__(**kwargs)



    def __init__(self, seq):
        super().__init__(seq=seq)
        self.aaa()


    @classmethod
    def from_redis(cls):
        sl = cls(trianglesv2)

        return sl

    def commit(self):
        return self.mutation(variables={"objects": self._seq})

    def __array__(self):
        return np.stack([self["x"], self["y"]], axis=-1)

    @property
    def array(self):
        return self.__array__()

    def solve_kd(self, leafsize=3, **kwargs):
        self._kd = KDTree(data=self.array, leafsize=leafsize, **kwargs)
        return self.kd

    def aaa(self):
        ctr = ElementSequence(self["centroid"])
        self["x"], self["y"] = ctr["x"], ctr["y"]
        self["mark"] = list(map(lambda x: f'{x[0]}-{x[1]}', zip(self["tag"], self["subtype"])))

    @classmethod
    def from_db(cls):
        return cls(cls.query(full_root=True)["data"]["lht_ceiling_type_marks"])

    @property
    def kd(self):
        return self._kd

    def tag_match(self, j):
        dist, ind = self.kd.query([j["x"], j["y"]])
        if dist > 200:

            self.get_from_index(ind)["tag"] = "A"
        else:
            self.get_from_index(ind)["tag"] = j["tag"]

    def solve_types(self):

        que = query("lht_ceiling_type_marks(where: {floor: {_eq: $floor}})".replace("$floor",self.floor), {"x", "y", "tag"})
        self.solve_kd()
        res = que(full_root=True)["data"]["lht_ceiling_type_marks"]
        list(map(self.tag_match, res))
        self.aaa()
        self.commit()
        return 200


async def update_types_async(data):
    try:
        for i, d in enumerate(data):
            TypeTag(**d)
        print(200)
    except Exception as err:
        print(err, file=sys.stderr)


def update_types(data):
    for i, d in enumerate(data):
        TypeTag(**d)
    return 200


class RunVars(str, Enum):
    run_async: str = "async"
    run_in_thread: str = "thread"
    run_simple: str = "simple"


def update_types_from_file(path: str, running_type: RunVars, debug=True):
    with open(path, "r") as f:
        data = decode_sv(json.load(f))
    match running_type:
        case RunVars.run_simple:
            update_types(data)
        case RunVars.run_async:
            asyncio.run(update_types_async(data), debug=debug)
        case RunVars.run_in_thread:

            _th = threading.Thread(target=update_types, args=(data,))
            _th.name = "Types-Updater-Thread"
            _th.start()
