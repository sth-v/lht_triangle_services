# syntax=docker/dockerfile:1

FROM emscripten/emsdk AS builder
RUN apt-get update && apt-get install -y ninja-build git && apt-get update
RUN python3 -m pip install --upgrade pip wheel setuptools cmake

FROM builder as build
WORKDIR /rhino3dm
RUN git clone --recursive https://github.com/mcneel/rhino3dm.git
RUN python3 setup.py bdist_wheel

FROM sthv/mmcore AS base
WORKDIR /tmp
COPY --from=build / .



