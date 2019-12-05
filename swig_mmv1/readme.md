pymm3d: MicMac Python API
=========================

Introduction
------------

This is an API to a small part of MicMac for Python 3.

Download
--------

  * [2019/12/04 binary version for Python 3.6 with Ubuntu 18.04](../../bin/swig_mmv1-20191204.tar.bz2)


Usage
-----

copy _mm3d.so and mm3d.py in your working directory, then with python3:

```python
    import mm3d
    c=mm3d.CamOrientFromFile("Ori-FishEyeBasic/Orientation-Calibration_geo_14_001_01_015000.thm.dng_G.tif.xml")
    p=mm3d.Pt2dr(1000,1000)
    prof=1
    print(c.ImEtProf2Terrain(c.NormM2C(p),prof))
```

Documentation
-------------
See [main file doc](api__mm3d_8h.html)

Compilation
-----------
Only tested on Linux.

Dependances:
 - SWIG 3.0
 - Python 3.6 headers
 - for documentation: doxygen

Compile micmac (to have lib/libelise.a), then
cd swig_mmv1
make -f Makefile_swig_linux

The files to distribute are: swig_mmv1/_mm3d.so and swig_mmv1/mm3d.py

cd doc
doxygen Doxyfile

The documentation is in swig_mmv1/doc/html/index.html

TODO
----
 * for now the selected classes are copied in api_mm3d.h
 * make a script to automatically extract classes definitions from mm3d sources
 * create a header file for each subject

