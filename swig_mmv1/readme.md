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
 - for documentation: doxygen, GraphViz

Compile micmac (to have lib/libelise.a), then
cd swig_mmv1
make -f Makefile_swig_linux all

The files to distribute are: swig_mmv1/_mm3d.so and swig_mmv1/mm3d.py

The documentation is in swig_mmv1/doc/html/index.html

TODO
----
 * use distutils?
 * fix mm modifications (constructors) 
 * for now the selected classes are copied in api_mm3d.h
 * make a script to automatically extract classes definitions from mm3d sources
 * create a header file for each subject
 * do not quit python on fatal errors?
 * see https://pybind11.readthedocs.io/en/stable/intro.html

 * createIdealCamXML: from quaternion, and add check points

Crash on import
---------------
 in mm3d_wrap.cxx, the line
 result = ((ElPackHomologue const *)arg1)->BoxP1();
 makes python crash on import.
 it can be avoided without Box2dr BoxP1() const; in api_mm3d.h.
 Box2dr default constructor seems ok, and tPairPt  PMed() const; also crashes.
 It may be a problem with SWIG_ConvertPtr(obj0, &argp1,SWIGTYPE_p_ElPackHomologue, 0 |  0 );
