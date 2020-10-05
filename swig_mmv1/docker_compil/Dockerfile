FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --fix-missing --install-recommends \
x11proto-core-dev make cmake libx11-dev imagemagick gcc g++ \
exiv2 libimage-exiftool-perl libgeo-proj4-perl \
doxygen swig python3-dev graphviz cmake-curses-gui pkg-config libxext-dev

ADD compile_apipy.sh /compile_apipy.sh
