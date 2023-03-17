cd /build
docker build --no-cache -f Dockerfile -t sthv/cxm-rhino3dm
docker run sthv/cxm-rhino3dm