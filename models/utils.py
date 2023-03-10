from typing import Iterator, Any

from cxmdata import CxmData
from mmcore.collections.multi_description import ElementSequence
from mmcore.services.redis.connect import bootstrap_stack
redis_conn = bootstrap_stack()

def paginate(data_):
    data = zip(*data_.values())
    while True:
        try:
            yield dict(zip(data_.keys(), next(data)))
        except StopIteration as err:
            break


def decode_sv(data):
    floor = list(data.keys())[0].upper()
    marks = []
    for i in list(data.values())[0]:
        (x, y, z), (typ_name,) = i
        marks.append(dict(x=x, y=y, z=z, tag=typ_name, floor=floor))
    return marks


class PaginateGhNamespace(Iterator):
    conn: Any

    def __init__(self, data):
        self.headers = data.keys()
        self.data = zip(*data.values())

    def _next_page(self):
        return dict(zip(self.headers, next(self.data)))

    @classmethod
    def from_redis(cls, key):
        return cls(data=CxmData(cls.conn.get(key)).decompress())

    def __iter__(self):
        return self

    def __next__(self):
        return self._next_page()


class RedisCxmDataBridge:
    iterator = PaginateGhNamespace
    name = None
    conn = None

    def __init__(self, key, primary, conn, **kwargs):
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
        return CxmData(self.conn.get(key)).decompress()

    def get(self):
        return self.iterator(self.get_data())

    def __get__(self, instance, owner):
        return self.get()


from collections import Counter


def ordered_compare(a, b):
    dct = dict(eq=[], not_eq=[])
    for k, (i, j) in enumerate(zip(a, b)):
        if i == j:
            rpr = lambda xxx: f"{k} : {i} == {j}"
            dct["eq"].append({"index": k, "a": i, "b": j, "representation": rpr, "eq": True})
        else:
            rpr = lambda xxx: f"{k} : {i} -> {j}"
            dct["not_eq"].append({"index": k, "a": i, "b": j, "representation": rpr, "eq": False})
    return dct


def compare_multi_keys(a, b, keys=()):
    dt = {}
    for k in keys:
        dt[k] = get_ordered_diff(a[k], b[k])
    return dt


class Compare(Counter):
    def __init__(self, tag, subtype):
        super().__init__()




def get_ordered_diff(a, b):
    res = ordered_compare(a, b)
    el = ElementSequence(res["not_eq"])
    d = Counter(zip(el["a"], el["b"]))
    return d


def abs_compare(a, b):
    dct = dict(eq=[], not_eq=[])
    for k, (i, j) in enumerate(zip(a, b)):
        if i == j:
            dct["eq"].append({"index": k, "a": i, "b": j, "eq": True})
        else:
            dct["not_eq"].append({"index": k, "a": i, "b": j, "eq": False})
    return dct
