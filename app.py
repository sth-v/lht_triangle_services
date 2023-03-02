import io
import json
import sys
import time
from abc import ABCMeta, abstractmethod
from enum import Enum

import numpy as np
import requests
import shapely
from jinja2.nativetypes import NativeEnvironment
from scipy.spatial import KDTree
from models.utils import decode_sv
import threading
import asyncio

from scipy.spatial import distance
import more_itertools
from mmcore.collections.multi_description import ElementSequence
from cxmdata import CxmData
from mmcore.baseitems import Matchable, DataDescriptor
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

from mmcore.geom.base import Polygon

from models.geom import PointRhp
from mmcore.services.redis import connect

redis_conn = connect.bootstrap_cloud()
PRIMARY = "runitime:lht:ceiling:"
TOLERANCE = 100

from models.graphql import GQlEntity, InsertTypeMarkOne, UpdateTypeMarksByPKPartialTemp, TypeMarksReg, \
    TypeMarksByPkPartial, UpdatePanelTriangleByPKPartialTemp, PanelTriangleByPkPartial
from models.connections import *

# import Rhino.Geometry as rgg
logger_buf = io.StringIO()

# triangles = CxmData(redis_conn.get(PRIMARY + "L2W:triangles")).decompress()

# mask = CxmData(redis_conn.get(PRIMARY + "L2W:mask")).decompress()
# pnl_ns = CxmData(redis_conn.get(PRIMARY + "L2W:pnl_namespace")).decompress()
# tps = [[10, 20], [11, 21], [12, 22], [10, 20]]

TOLERANCE = 100

from mmcore.services.redis import connect

connect.bootstrap_cloud()


# cutter = CutPanels(mask)


class GraphQlDescriptor(DataDescriptor, metaclass=ABCMeta):
    """
    Override mutation and query methods to use it
    """
    __descriptor_registry_name__ = "graphql_attribute"

    @abstractmethod
    def mutation(self):
        pass

    @abstractmethod
    def query(self):
        pass

    def __set_name__(self, owner, name):
        self.name = name
        if not hasattr(owner, f"__{self.__descriptor_registry_name__}_dict__"):
            setattr(owner, f"__{self.__descriptor_registry_name__}_dict__", {})
        owner.__graphql_attribute_dict__[name] = self

    def __get__(self, inst, own):
        r = self.query()(x=inst.x, y=inst.y)
        # print("r:  ", r)
        return r[0][self.name]

    def __set__(self, inst, v):
        self.mutation()(**{
            'x': inst.x,
            'y': inst.y,
            self.name: v
        }
                        )


import yaml


class GraphQlAbstractProtocol:
    __slots__ = "_typename", "name", "default", "path", "key"
    path: str
    key: str
    name: str
    default: str

    def __init_subclass__(cls, key="qql_type_notation.yml", path="gqlproto-pyclass.yml", **kwargs):
        cls.key = key
        cls.path = path
        super().__init_subclass__()

    def __init__(self, typename):
        super().__init__()

        self._typename = typename
        spc = self.spec()
        self.default = spc[typename][self.key]["default"]
        self.name = spc[typename][self.key]["name"]

    def spec(self) -> dict:
        return yaml.unsafe_load(self.path)["pyclass"]["spec"]

    def from_spec(self) -> str:
        return f"${self._typename}: {self.name} = {self.default}"

    def __repr__(self):
        return "GraphQl Protocol: " + self.from_spec()

    def __str__(self):
        return self.from_spec()


class GraphQlProtocol(GraphQlAbstractProtocol, key="qql_type_notation.yml", path="gqlproto-pyclass.yml"):

    def from_spec(self) -> str:
        return f"${self._typename}: {self.name} = {self.default}"


class GraphQlInlineSpec(GraphQlAbstractProtocol, key="qql_type_notation.yml", path="gqlproto-pyclass.yml"):

    def from_spec(self) -> str:
        return f"{self._typename}: ${self.name}"


class GQlProperty:
    def mutation(self, func):


        self._mutation=func
        return self
    def query(self, func):

        self._query = func

        # self.__dict__[self.name+"_query"]=wrapper

        return self

    gql_type: str

    def __init__(self, client, **kwargs):
        # self.mutate_j2env ,self.query_j2env=NativeEnvironment(), NativeEnvironment()
        super().__init__()

        self.mutate_j2env, self.query_j2env = NativeEnvironment(), NativeEnvironment()

        self.client = client
        self.kwargs = kwargs

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        self.fields = func.__code__.co_varnames
        return self

    def request(self, body, variables):
        if variables is None:
            variables={}
        request = requests.post(self.client.url,

                                headers=self.client.headers,
                                json={
                                    "query": body,
                                    "variables": variables

                                }
                                )
        # print(self.body, vars, request.json())

        ...

    def __get__(self, instance, own):


        return list(self.request(self._query(instance ),self.kwargs))[-1]

    def __set__(self, instance, variables):
        self._mutation(instance)(variables)


    def setup_vars(self, instance):
        self.kws = {}
        self.kws |= self.kwargs
        for k in self.kwargs.keys():
            if hasattr(instance, k):

                self.kws[k] = getattr(instance, k)
            else:
                continue
        self.kws["pkey"] = instance.pk
        return self.kws

    def generate_vars(self, variables):
        return {"inline_vars": f'({",".join(GraphQlProtocol(pkk).from_spec() for pkk in variables.keys())})',
                "outline_vars": f'{" ,".join(GraphQlInlineSpec(pkk).from_spec() for pkk in variables.keys())}'}


from mmcore.gql import client as gql_client

qclcl = gql_client.GQLClient(url=gql_client.GQL_PLATFORM_URL, headers={
        "content-type": "application/json",
        "user-agent": "JS GraphQL",
        "X-Hasura-Role": "admin",
        "X-Hasura-Admin-Secret": "mysecretkey"
    })
class A(Matchable):
    """
    lht_ceiling + _ + panels + _by_pk|... + ( $pk : String = "") + {

    {{ root }}_{{ table }}{{ postfix }}{{ inline_vars }} {
        recievs
    }
    """
    __match_args__ = "x", "y", "floor"

    @GQlProperty(client=qclcl)
    def uuid(self, uuid): return uuid

    @uuid.query
    def uuid(self, body="""
    query {
        lht_ceiling_type_marks_by_pk(x: $x, y: $y) {
            uuid
         }
        }"""):
        return body.replace("$x", str(self.x)).replace("$y", str(self.y))

    @GQlProperty(client=qclcl)
    def centroid(self, centroid):
        return PointRhp(centroid['x'], centroid['y'], centroid['z'])

    @centroid.query
    def centroid(self, body="""
            query {
                lht_ceiling_type_marks(where: {_and: {uuid: {_eq: $uuid}}}) {
                    modify
                    tag
                    x
                    y
                    z
                    floor
                    uuid
                      }
                    }
                    """):
        return body.replace("$uuid", self.uuid)


class TypeTagDesc(GraphQlDescriptor):

    def mutation(self):
        return GQlEntity(UpdateTypeMarksByPKPartialTemp(self.name))

    def query(self):
        return TypeMarksByPkPartial(self.name)


class PanelTriangleDesc(GraphQlDescriptor):

    def mutation(self): return GQlEntity(UpdatePanelTriangleByPKPartialTemp(self.name))

    def query(self): return PanelTriangleByPkPartial(self.name)


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


from mmcore.gql.client import query, mutate, GqlString


class Floor(str, Enum):
    L2W = "L2W"
    L2 = "L2"
    L1 = "L1"
    B1 = "B1"


class PanelTriangle(Matchable):
    __match_args__ =  "x", "y", "floor"

    floor: Floor = Floor.L2W
    fields=  (
            "mark",
            "centroid",
            "tag",
            "subtype",
            "floor",
            "uuid",
            "updated_at", "points")
    initial_fields = fields
    subtype = PanelTriangleDesc()
    extra = PanelTriangleDesc()
    dtype= PanelTriangleDesc()
    tag = PanelTriangleDesc()
    uuid = PanelTriangleDesc()
    centroid = PanelTriangleDesc()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = round(v / TOLERANCE, 0) * TOLERANCE

    @classmethod
    def get_sequence(cls) -> ElementSequence:
        return ElementSequence(
            list(PanelTriangle(dct["x"], dct["y"], dct["floor"], no_post=True) for dct in more_itertools.flatten(TypeMarksReg())))

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = round(v / TOLERANCE, 0) * TOLERANCE

    def distance(self, other):
        return distance.euclidean([self.x, self.y], [other.x, other.y])


class CompareSequences:
    def __init__(self):
        super().__init__()

    def __call__(self, a: ElementSequence, b: ElementSequence, **kwargs):
        ...


class Triangles(ElementSequence):
    floor: Floor
    query: query
    mutation: mutate
    _x = _y = 0

    def __init_subclass__(cls, floor: Floor = None, fields=(
            "mark",
            "centroid",
            "tag",
            "subtype",
            "floor",
            "uuid",
            "updated_at", "points"), **kwargs):
        cls.floor = floor
        cls.initial_fields = fields
        cls.query = query("lht_ceiling_panels(where: {floor: {_eq: \"$floor\"}})".replace("$floor", floor), set(fields))
        cls.mutation = mutate(GqlString("lht_ceiling", "panels"), fields={
            "mark",
            "tag",
            "floor",
            "mask",
            "uuid",
            "updated_at",
            "points",
            "x",
            "y"}, variables=dict(objects=[]))

        super().__init_subclass__(**kwargs)

    def __init__(self, seq):
        super().__init__(seq=seq)
        self.aaa()

    @classmethod
    def from_redis(cls):
        trianglesv2 = json.loads(redis_conn.get(PRIMARY + f"{cls.floor}:pnl_json").decode())
        sl = cls(trianglesv2)

        return sl

    def commit(self):
        return self.mutation(variables={"objects": self._seq})

    def fetch(self):
        self._seq = self.query(full_root=True)["data"]["lht_ceiling_panels"]
        self.aaa()

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
        return cls(cls.query(full_root=True)["data"]["lht_ceiling_panels"])

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

        que = query("lht_ceiling_type_marks(where: {floor: {_eq: $floor}})".replace("$floor", self.floor),
                    {"x", "y", "tag"})
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


class TrianglesL2w(Triangles, floor=Floor.L2W):
    ...


class TrianglesL2(Triangles, floor=Floor.L2):
    ...


class TrianglesB1(Triangles, floor=Floor.B1):
    ...


class TrianglesL1(Triangles, floor=Floor.L1):
    ...


def get_mask_from_redis(floor="L2W"):
    return CxmData(redis_conn.get(PRIMARY + floor + ":mask")).decompress()


class SimpleTriangle(Polygon):
    def to_shapely(self):
        return shapely.Polygon(self.points)

    def isintersect(self, other):
        return shapely.contains_properly(other)


crv = shapely.LineString()
