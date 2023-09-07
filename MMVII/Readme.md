Dependencies
------------
  - Required: boost, eigen3, cmake.
  - Optional: OpenMP, ccache

  - Ubuntu 20.04:
    - `sudo apt install libboost-all-dev libeigen3-dev ccache cmake`
  If using CLang version XX and want OpenMP:
    - `sudo apt install libomp-XX-dev`


Compilation (short):
--------------------
    Compile MicMac V1
    cd MMVII/build
    cmake ..
    make or make VERBOSE=1

 On first compilation, generated code must be created:

    make
    ../bin/MMVII GenCodeSymDer
    make

 Tests :

    ../bin/MMVII Bench 1


 Clean :

    make clean     : delete build products
    make distclean : delete build products and generated codes
    rm -fr MMVII/build/*` : reinitialize the build configuration

 In case of SymDer-related compilation error, clear all generated code before compilation:

   make distclean
   make
   ../bin/MMVII  GenCodeSymDer
   make

Compilation (detail):
--------------------
 - You can use `cmake -G Ninja ..` to use Ninja build system instead of the native one. (Ninja must be installed ...)
 - Use `cmake --build . -j 8` or `cmake --build . -j 8 -v` instead of make (works with all build systems)
 - Use `cmake --build . --target clean` or `cmake --build . --target cleanall`
 - Use `ccmake ..` or `cmake-gui ..` to change config option:
   - CMAKE_BUILD_TYPE:
       . Debug : -g
       . RelWithDebInfo : -O3 -g  (default)
       . Release : -O3 -DNDEBUG
   - EIGEN3_INCLUDE_PATH:
       . Contains path to eigen3 source files. Auto filled if cmake found eigen3
       . You can set it to your own eigen3 directory if desired
       . You HAVE to set it if cmake didn't find it
       . You can clear it to force cmake to search for eigen3
    - CMAKE_CXX_COMPILER (advanced mode : 't'):
       . Allow to set compiler version and type (g++, clang)


Legacy :
--------
    Compile MicMac V1, then in MMVII/bin directory:
      make

   On first compilation or SymDer-related runtime error, synbolic derivatives code has to be generated:
 
    make
    ./MMVII  GenCodeSymDer
    make
 
   In case of SymDer-related compilation error, clear all generated code before compilation:
 
   make distclean
   make
   /MMVII  GenCodeSymDer
   make

  if Eigen is not installed in /usr/include/eigen3 (apt install libeigen3-dev does this correctly), use:
  `make EIGEN_DIR=/my_eigen_dir` in the above commands.


To generate html doc
--------------------
In MMVII directory:

    doxygen Doxyfile 


To generate pdf doc
-------------------

In MMVII/Doc directory:

    make

