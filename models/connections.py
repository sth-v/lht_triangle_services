import redis_om, redis

from mmcore.services.rhpyc import get_connection
import os

__all__ = [
    "rhpyc_conn",

    "rh",
    "rg",
    "rs",
    "rss"
]


rhpyc_conn = get_connection()
try:
    rh = rhpyc_conn.root.getmodule("Rhino")
    rg = rhpyc_conn.root.getmodule("Rhino.Geometry")
    rs = rhpyc_conn.root.getmodule("rhinoscript")
    rss = rhpyc_conn.root.getmodule("rhinoscriptsyntax")
except:
    import rhino3dm

    rhpyc_conn.root.execute("import clr;import sys;sys.path.extend(['C:/Program Files/Rhino 7/System'])")
    rhpyc_conn.root.execute("clr.AddReference('RhinoCommon')")
    rhpyc_conn.root.execute("import Rhino")
    rh = rhpyc_conn.root.getmodule("Rhino")
    rg = rhpyc_conn.root.getmodule("Rhino.Geometry")

    rs,rss= None, None
