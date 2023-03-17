
docker run --rm -d -v vol:/vol -v build:/build --name some-docker --privileged docker:dind-rootless

docker docker cp build some-docker:/build
docker exec -it some-docker docker-entrypoint.sh export DOCKER_BUILDKIT=0
docker exec -it some-docker docker-entrypoint.sh sh /build/buildme.sh
docker exec -it some-docker docker-entrypoint.sh docker run sthv/cxm-rhino3dm
docker info --format '{{ json .SecurityOptions }}'
