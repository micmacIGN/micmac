Dependencies

libboost-all-dev
libeigen3-dev
#  libceres-dev => no longer needed 4 now


To compile :

Compile MicMac V1, then in MMVII/bin directory:

    make

or 

    make -f Mk-MMVII.makefile

On first compilation, SymDer code has to be generated:

    make
    ./MMVII  GenCodeSymDer
    make

To generate html doc
--------------------
In MMVII directory:

    doxygen Doxyfile 


To generate pdf doc
-------------------

In MMVII/Doc directory:

    make


