Docker builder for MicMac
=========================

Build MicMac form your own MicMac folder with an old Ubuntu distribution.

To get and update MicMac, use git on your machine.

The docker is to be used interactivelly to lanch compilation.

Build docker image
------------------

    docker build -t mm3ddocker .


Run image (with MicMac sources in ~/micmac)
-------------------------------------------

    docker build -t mm3ddocker .
    docker run -ti --rm -v ~/micmac:/mm mm3ddocker bash

Compilation
-----------

In this docker :

    cd /mm/build
    rm -Rf *
    cd build
    cmake \
  	  -DWITH_QT5=1 \
  	  -DWITH_APIPYTHON=1
    NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
    make -j$NBRP
    make apipy

Output will be in ~/micmac/swig_mmv1/build/lib.linux-x86_64-3.5/
