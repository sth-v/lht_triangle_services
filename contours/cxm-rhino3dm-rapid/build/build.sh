apt-get update
apt-get install -y cmake g++ gdb cmake ninja-build rsync zip git make python3
ARG repo=https://github.com/mcneel/rhino3dm.git
cd src
# syntax=docker/dockerfile:1
FROM alpine
WORKDIR /src
RUN --mount=target=. \
  make REVISION=$(git clone --recursive $repo) build


