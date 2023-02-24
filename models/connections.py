import redis_om, redis

from mmcore.services.rhpyc import get_connection
import os

__all__ = [
    "rhpyc_conn",
    "redis_conn",
    "rh",
    "rg",
    "rs",
    "rss"
]

if os.getenv("APPENV") == "development":
    os.environ["RHINO_RPYC_HOST"] = "localhost"
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_HOST = os.getenv("REDIS_HOST") if os.getenv("REDIS_HOST") is not None else "localhost"
    REDIS_PORT = os.getenv("REDIS_PORT") if os.getenv("REDIS_HOST") is not None else 6379
    redis_conn = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD
    )
    #REDIS_STACK_URL = os.getenv("REDIS_STACK_URL")
    #redis_conn = redis_om.get_redis_connection(url=REDIS_STACK_URL)
else:
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_HOST = os.getenv("REDIS_HOST") if os.getenv("REDIS_HOST") is not None else "localhost"
    REDIS_PORT = os.getenv("REDIS_PORT") if os.getenv("REDIS_HOST") is not None else 6379
    redis_conn = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD
    )

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
