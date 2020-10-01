Docker builder for MicMac
=========================

Build MicMac form your own MicMac folder with an old Ubuntu distribution.

To get and update MicMac, use git on your machine.

The docker is to be used interactivelly to lanch compilation.

Build docker image
------------------

    docker build -t mm3ddocker1604 .


Run image (with MicMac sources in ~/micmac)
-------------------------------------------

    docker run -ti --rm -v ~/micmac:/mm mm3ddocker1604 bash

Compilation
-----------

In this docker :
    
    /compile_apipy.sh

Output will be in ~/micmac/swig_mmv1/build/lib.linux-x86_64-3.5/


Remove docker image
-------------------
Remove compilation files from the docker before removing image!

    docker rmi mm3ddocker1604

