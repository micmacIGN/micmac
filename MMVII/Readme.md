Dependencies
------------
  - Required: cmake.
  - Optional: OpenMP, ccache

  - Ubuntu 20.04:
    - `sudo apt install ccache cmake`
    - If using CLang version XX and want OpenMP: `sudo apt install libomp-XX-dev`


Compilation (short) :
--------------------

 Compile MicMac V1, then (replace N with the number of processor threads) :

    cd MMVII
    mkdir -p build
    cd build
    cmake ..
    make -j N full (or make -j N full VERBOSE=1 to see compile command line)

  Following compilations could be run just with:

    make -j N


 Tests :

    ../bin/MMVII Bench 1


 Compilation targets :

    all|(none) : build MMVII
    full       : re-generate files for symbolic calculus and build MMVII
    rebuild    : distclean + full
    clean      : delete build products
    distclean  : delete build products and generated files for symbolic calculus

    rm -fr MMVII/build/* : reinitialize the build configuration. cmake needs to be rerun after that

 In case of SymDer-related compilation error, re-generate files for symbolic calculus :

    make full

Compilation (detail):
--------------------
 - You can use `cmake -G Ninja ..` to use Ninja build system instead of the native one. (`sudo apt install ninja-build`)
 - Use `cmake --build . -j 8 [--target TARGET]` or `cmake --build . -j 8 -v [--target TARGET]` instead of make (works with all build systems)
 - Use `cmake --build . --target clean` or `cmake --build . --target distclean`
 - Use `ccmake ..` or `cmake-gui ..` to change config option:
   - CMAKE_BUILD_TYPE:
     - Debug : -g
     - RelWithDebInfo : -O3 -g  (default)
     - Release : -O3 -DNDEBUG
   - CMAKE_CXX_COMPILER (advanced mode : 't'):
     - Allow to set compiler version and type (g++, clang)
   - vMMVII_BUILD (see bellow):
     - ON : enable compilation of vMMVII. Needs Qt >= 5.12.8
     - OFF: disable compilation of vMMVII
     - You may need to add 'CMAKE_PREFIX_PATH=<Qt dir>' on Windows (ex: c:\Qt\6.6.0\msvc2019_64)

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
  - MMVII must be (fully) compiled
  - MMVII executable must be in your $PATH
  - Add to your ${HOME}/.bashrc the following line:

   `[ -f ${HOME}/@MICMAC_SOURCE_DIR@/micmac/MMVII/bash-completion/mmvii-completion ] && . ${HOME}/@MICMAC_SOURCE_DIR@/micmac/MMVII/bash-completion/mmvii-completion ]`
  - Completion will be active in terminals opened after this modification.


vCommand (beta)
---------------
There is a GUI tool that can help for writing MMVII command : vMMVII

It will be automatically compiled with MMVII if development package Qt5 (or Qt6) is installed (Ubuntu 22.04: `sudo apt install qtbase5-dev`)

Windows: You may have to add Qt installation path when running cmake configuration :

`cmake .. -D CMAKE_PREFIX_PATH=C:\Qt\6.6.0\msvc2019_64`

Usage: just type "vMMVII" in your working directory.

- Sorry, no documentation yet

- This tool is beta: some MMVII parameters may be misinterpreted or not have the good File Dialog helper.
