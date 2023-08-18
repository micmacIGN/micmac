Dependencies

libboost-all-dev
libeigen3-dev
#  libceres-dev => no longer needed 4 now


To compile :

Compile MicMac V1, then in MMVII/bin directory:

    make

On first compilation or SymDer-related runtime error, synbolic derivatives code has to be generated:

    make
    ./MMVII  GenCodeSymDer
    make

In case of SynDer-related compilation error, clear all generated code before compilation:

    make distclean
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


