import json
import os
import sys
import threading
from typing import NewType, Any

# import Rhino.Display
# Rhino.DocObjects.ObjectAttributes.PlotColorSource.

import pandas as pd
import rpyc
import uvicorn
import dotenv
from mmcore.gql.client import GQLFileBasedQuery
que = GQLFileBasedQuery("data/query.graphql")

dotenv.load_dotenv()
GDRIVE = "/Users/andrewastakhov/Library/CloudStorage/GoogleDrive-aa@contextmachine.ru/Shared drives/CONTEXTMACHINE/PROJECTS/2206_Lahta_Triangles/Lahta_Trngls_FILES_WORK/MM/Actual_data"
import cxmdata

import rhino3dm
from collections import Counter
from mmcore.collections.multi_description import ES, ElementSequence
from mmcore.geom.materials import ColorRGB, ColorRGBA

from mmcore.services.service import RpycService
import sys
from fastapi import FastAPI
from mmcore.gql.client import GQLFileBasedQuery, uuid

from fastapi import FastAPI, Path, APIRouter
from fastapi import Query as FQuery

import strawberry

from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter


update_contour = GQLFileBasedQuery("data/update-contour.graphql")


floor_mapping = {
    "L2W": ("L2", "L2W"),
    "L2": ("L2", None),
    "L1": ("L1", None),
    "B1": ("B1", None)
}


class FloorWriter:
    _colors = []
    redis_primkey = "lht:triangle-ceiling:contours:"

    def __init__(self, floor, fname=None, init_from="gql"):
        super().__init__()
        self.floor = floor
        self.fname = floor if fname is None else fname
        self.filename = f"{self.fname}_base_contour.json"
        self.model = rhino3dm.File3dm()
        match init_from:
            case "gql":
                self.from_gql()
            case "gdrive":
                self.from_gdrive()
            case "redis":
                self.from_redis()
        self.gen_contours_ec()
        self.redis_key = self.redis_primkey + self.floor + ":" + self.fname

    @property
    def path(self):
        return self.floor + "/" + self.filename+".3dm"

    @property
    def fullpath(self):
        return f"{GDRIVE}/{self.path}"

    def from_gql(self):
        self._json = list(que(full_root=False).values())[0]
        return self._json

    def from_redis(self):
        #self._json = json.loads(rconn.get(self.redis_key).decode())
        #return self._json
        ...

    def to_redis(self):
        #rconn.set(self.redis_key, json.dumps(self._json).encode())
        ...

    def json(self):

        return self._json

    def from_gdrive(self):
        with open(self.fullpath) as f:
            import json
            self._json = json.load(f)
            return self._json

    def csv(self):

        return pd.DataFrame(self.json())

    def __ror__(self, other):
        jsn = self.json()
        jsn |= other.json()
        return jsn

    @property
    def contours(self):

        return traverse_cxm_data_json(self.json())

    def gen_contours_ec(self):

        self._cont_ec = ElementSequence(self.contours)

    def cont_ec(self):
        return self._cont_ec

    @property
    def details(self):
        return Counter(self._cont_ec["detail"])

    @property
    def colors(self):

        import numpy as np

        ncl = len(self.details) - len(self._colors)

        if ncl > 0:
            clrs = np.random.random((ncl, 3))
            for i in range(ncl):
                self._colors.append(ColorRGB(*clrs[i, :]))
        return self._colors

    def append_geometry(self, item):

        cold = dict(zip(self.details.keys(), self.colors))
        geo = item["curve"]
        attrs = rhino3dm.ObjectAttributes()
        attrs.SetUserString("detail", item["detail"])
        attrs.SetUserString('floor', item["floor"])
        attrs.ObjectColor = cold[item["detail"]] + (255,)
        attrs.PlotColor = cold[item["detail"]] + (255,)
        attrs.PlotColorSource = rhino3dm._rhino3dm.ObjectPlotColorSource.PlotColorFromObject
        attrs.ColorSource = rhino3dm._rhino3dm.ObjectColorSource.ColorFromObject

        self.model.Objects.Add(geo, attrs)

    def write_model(self):

        for item in self.contours:
            self.append_geometry(item)

        return self.model

    def dump_model(self):
        if self.model.Write(self.path, 7):
            return self.path
        else:
            return False

    @property
    def byte_array_model(self):
        return self.model.Encode()

    @property
    def archive_dict(self):

        return [rhino3dm.ArchivableDictionary.Encode(contour) for contour in self.contours]

    def generate_dict(self):
        cold = dict(zip(self.details.keys(), self.colors))
        for i, contour in enumerate(self.json()):
            contour["color"] = cold[contour["detail"]].to_dict()
            yield contour

    def ToJSON(self):
        return json.dumps(list(self.generate_dict()))


def traverse_cxm_data_json(dat):
    if isinstance(dat, dict):
        dct = {}
        for k, v in dat.items():
            if k == "cxmdata":
                dct = cxmdata.CxmData(v).decompress()
                break
            else:
                dct[k] = traverse_cxm_data_json(v)
        return dct
    elif isinstance(dat, (list, tuple)):
        return [traverse_cxm_data_json(i) for i in dat]
    else:
        return dat


class FloorContoursManager:
    def __init__(self, *args):
        super().__init__()
        self.floors = {}
        for floor, part in args:
            self.floors[(floor, part)] = FloorWriter(floor, part)

    def __getitem__(self, item):
        return self.floors[item]



contours = FloorContoursManager(*(("B1", None), ("L1", None), ("L2", None), ("L2", "L2W")))
