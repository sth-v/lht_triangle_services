
import threading

import uvicorn
import dotenv

from gql import graphql_app

dotenv.load_dotenv()
GDRIVE = "/Users/andrewastakhov/Library/CloudStorage/GoogleDrive-aa@contextmachine.ru/Shared drives/CONTEXTMACHINE/PROJECTS/2206_Lahta_Triangles/Lahta_Trngls_FILES_WORK/MM/Actual_data"

from mmcore.collections.multi_description import ES, ElementSequence


from mmcore.services.service import RpycService
import sys

from fastapi import Query as FQuery

from mm import contours

from fastapi import FastAPI



class MyService(RpycService,
                configs='http://storage.yandexcloud.net/lahta.contextmachine.online/svc/ceiling-contours.yaml'):
    ...


class ht:

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        v = instance._thread
        print(f"get thread {v}")
        return v

    def __set__(self, instance, v):
        instance._thread = v
        print(f"set thread {v}")

    def __delete__(self, instance):
        instance._thread.join(600)
        del instance._thread
        print("deleting thread")


class Handle:
    thread = ht()
    service = MyService

    def on_startup(self):
        thread = threading.Thread(target=lambda: sys.exit(self.service.run()))
        thread.start()

        self.thread = thread

    def on_shutdown(self):
        self.thread.join(600)
        del self.thread

    def __init__(self, app):
        super().__init__()
        self.app = app

    def __call__(self, *args, **kwargs):
        return self.app(*args, on_startup=[self.on_startup], on_shutdown=[self.on_shutdown], **kwargs)

@Handle
class CxmFastAPI(FastAPI):
    ...
contours_app = CxmFastAPI()

contours_app.include_router(graphql_app, prefix="/graphql")


@contours_app.get("/")
def home():
    return {"msg": "lht/triangle-ceiling/contours"}


from enum import Enum


class FloorAttribute(str, Enum):
    curve = "json"
    color = "file 3dm"
    offset = "archive dict"


class FloorResponseType(str, Enum):
    JSON = "json"
    FILE3DM = "file 3dm"
    ARCHIVEDICT = "archive dict"


response_dict = {
    FloorResponseType.JSON: lambda x: x.ToJSON().encode(),
    FloorResponseType.FILE3DM: lambda x: x.byte_array_model.encode(),
    FloorResponseType.ARCHIVEDICT: lambda x: x.archive_dict.encode()

}

from fastapi.responses import FileResponse


@contours_app.get("/{floor}/")
def get_floor(floor: str, part: str = FQuery(default="null")):
    if part == "null":
        part = None

    with open("myfile", "wb") as f:
        f.write(response_dict[FloorResponseType.JSON](contours[(floor, part)]))

    return FileResponse("myfile")


@contours_app.get("/{floor}/rhino")
def get_rhino(floor: str, part: str = FQuery(default="null"),
              ):
    if part == "null":
        part = None

    contours[(floor, part)].write_model()
    path = contours[(floor, part)].dump_model()



    return FileResponse(path, content_disposition_type="form-data", filename=floor,
                        headers={"Content-Type": "multipart/form-data"})


@contours_app.get("/{floor}/attribute/{attr}")
def get_floor_attribute(floor: str, attr: str, part: None | str = FQuery(default=None)):
    es = ElementSequence(list(contours[(floor, part)].generate_dict()))
    return {"msg": es[attr]}


@contours_app.get("/{floor}/names")
def get_floor_attribute_names(floor: str, part: None | str = FQuery(default=None)):
    es = ES(list(contours[(floor, part)].generate_dict()))

    return {"msg": list(es.keys())}


@contours_app.get("/{floor}/names")
def get_floor_attribute_names(floor: str, part: None | str = FQuery(default=None)):
    es = ES(list(contours[(floor, part)].generate_dict()))

    return {"msg": list(es.keys())}


@contours_app.post("/{floor}/attributes/{attr}/{item}")
def set_floor_attribute_names(floor: str, attr: str, part: None | str = FQuery(default=None), item: int | None = None,
                              data: dict = {}):
    if item is None:
        contours[(floor, part)]._json |= data
    else:
        contours[(floor, part)]._json[item] |= data

    return contours[(floor, part)]._json


if __name__ == "__main__":
    uvicorn.run("__main__:contours_app", host="0.0.0.0", port=8734, reload=True)
