Dependencies
------------
  - Required: cmake.
  - Optional: OpenMP, ccache

  - Ubuntu 20.04:
    - `sudo apt install ccache cmake`
    - If using CLang version XX and want OpenMP: `sudo apt install libomp-XX-dev`


Compilation (short), replace N with the number of processor threads:
--------------------
    Compile MicMac V1
    cd MMVII
    (mkdir -p build)
    cd build
    cmake ..
    make -j N (or make -j N VERBOSE=1 to see compile command line)

 On first compilation, generated code must be created:

    make -j N
    ../bin/MMVII GenCodeSymDer
    make -j N

 Tests :

    ../bin/MMVII Bench 1


 Clean :

    make clean     : delete build products
    make distclean : delete build products and generated codes
    rm -fr MMVII/build/* : reinitialize the build configuration

 In case of SymDer-related compilation error, clear all generated code before compilation:

    make distclean
    make
    ../bin/MMVII  GenCodeSymDer
    make

Compilation (detail):
--------------------
 - You can use `cmake -G Ninja ..` to use Ninja build system instead of the native one. (`sudo apt install ninja-build`)
 - Use `cmake --build . -j 8` or `cmake --build . -j 8 -v` instead of make (works with all build systems)
 - Use `cmake --build . --target clean` or `cmake --build . --target cleanall`
 - Use `ccmake ..` or `cmake-gui ..` to change config option:
   - CMAKE_BUILD_TYPE:
       . Debug : -g
       . RelWithDebInfo : -O3 -g  (default)
       . Release : -O3 -DNDEBUG
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


To generate html doc
--------------------
In MMVII directory:

    doxygen Doxyfile 


To generate pdf doc
-------------------

Require latex: `sudo apt install texalive`

In MMVII/Doc directory:

    make


Bash completion (beta)
----------------------

It is possible to have MMVII command completion for Linux bash

- Requires: bash-completion python3

   (Already installed by default on Ubuntu, just in case:  sudo apt install bash-completion python3)

- Configuration:
  - MMVII must be compiled
  - MMVII executable must be in your $PATH
  - Let your ${HOME}/.bashrc source the completion script:

   `[ -f ${HOME}/@MICMAC_SOURCE_DIR@/micmac/MMVII/bash-completion/mmvii-completion ] && . ${HOME}/@MICMAC_SOURCE_DIR@/micmac/MMVII/bash-completion/mmvii-completion ]`
  - Completion will be active in terminals opened after this modification.


vCommand (beta)
---------------
There is a GUI tool that can help for writing MMVII command : vMMVII

It will be automatically compiled with MMVII if development package Qt5 (or Qt6) is installed (Ubuntu 22.04: `sudo apt install qtbase5-dev`)

Usage: just type "vMMVII" in your working directory.

- Sorry, no documentation yet
- This tool is beta: some MMVII parameters may be misinterpreted or not have the good File Dialog helper.
