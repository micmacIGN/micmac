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
See [files doc](files.html)

Compilation
-----------
Only tested on Linux.

Dependances:
 - SWIG 3.0
 - Python 3.6 headers
 - for documentation: doxygen, GraphViz

Compile micmac (to have lib/libelise.a), then
cd swig_mmv1
make -f Makefile_swig_linux all

The files to distribute are: swig_mmv1/_mm3d.so and swig_mmv1/mm3d.py

The documentation is in swig_mmv1/doc/html/index.html

TODO
----
 * use distutils?
 * fix mm modifications (#ifdef FORSWIG) 
 * for now the selected classes are copied in api_mm3d.h
 * make a script to automatically extract classes definitions from mm3d sources
 * create a header file for each subject
 * do not quit python on fatal errors?
 * see https://pybind11.readthedocs.io/en/stable/intro.html

 * createIdealCamXML: from quaternion, and add check points

Crash on import
---------------
 Python may crash if some references are not definied in _mm3d.so.
 This may happen when importing a class for which some methods are defined but not implemented nor used in MicMac.
 Swig will use them and _mm3d.so linking will not test it.
 
 Use
```
    make -f Makefile_swig_linux check
```
 To link a dummy executable to check if all methods are implemented in MicMac.

