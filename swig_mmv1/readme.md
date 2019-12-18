pymm3d: MicMac Python API
=========================

Introduction
------------

This is an API to a small part of MicMac for Python 3.

Download
--------

  * [2019/12/17 binary version for Python 3.6 with Ubuntu 18.04](../../bin/swig_mmv1_20191217.tar.bz2)


Usage
-----

Copy _mm3d.so and mm3d.py in your working directory, then with python3:

```python
    import mm3d
    
    #read orientation xml
    try:
		c=mm3d.CamOrientFromFile("Ori-FishEyeBasic/Orientation-Calibration_geo_14_001_01_015000.thm.dng_G.tif.xml")
		p=mm3d.Pt2dr(1000,1000)
		prof=1
		print(c.ImEtProf2Terrain(c.NormM2C(p),prof))
	except RuntimeError as e:
		print(e)

    #get set of files from regex
    li = mm3d.getFileSet(".",".*.py")
    
    #read homol pack
	pack = mm3d.ElPackHomologue.FromFile("Zhenjue/Homol/PastisDSC_3115.JPG/DSC_3116.JPG.dat")
	print(pack.size())
	list_homol=pack.getList()
	for h in list_homol[0:10]:
	   print(h.P1(),h.P2())

	#create homol pack
	aPackOut=mm3d.ElPackHomologue()
	aCple=mm3d.ElCplePtsHomologues(mm3d.Pt2dr(10,10),mm3d.Pt2dr(20,20));
	aPackOut.Cple_Add(aCple);
	aPackOut.StdPutInFile("homol.dat");

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
 * make a script to automatically extract classes definitions from mm3d sources, add @file
 * create a header file for each subject
 * do not quit python on fatal errors? (add exceptions in micmac?)
 * see https://pybind11.readthedocs.io/en/stable/intro.html
 * include libs in _mm3d.so : ImportError: /lib/x86_64-linux-gnu/libm.so.6: versionGLIBC_2.27' not found

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

