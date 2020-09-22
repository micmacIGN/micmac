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

More usage examples can be found in swig_mmv1/examples_py

Documentation
-------------
See [files doc](files.html)

Compilation
-----------
Only tested on Linux.

Dependencies:
 - SWIG
 - Python 3 headers
 - for documentation: doxygen, GraphViz

On debian:
    apt install swig python3-dev doxygen graphviz


To compile, select "WITH_APIPYTHON" in cmake interface, then:
    make apipy

(if elise is to be updated, you have to run "make elise")

The files to distribute are: swig_mmv1/_mm3d.so and swig_mmv1/mm3d.py

To create documentation:

    cd swig_mmv1
    make -f Makefile_swig_linux doc

The documentation is in swig_mmv1/doc/html/index.html

Developement
------------

To add MM classes to python, add then to mm3d.i file (both #include and %include).
If you want to be able to use python lists for objects of these classes, use %template.

To check if everything is correct:

    make -f Makefile_swig_linux clean && make -f Makefile_swig_linux check

This way you can see every undefined references that you have to fix (by adding other files or hiding it with #ifndef SWIG).


TODO
----
 * confirm that ElExit shadowing is ok
 * add private/files.h
 * fix warnings (do not look at functions eToString etc?)
 * add it to cmake to support compilation with Qt
 * mm3d_wrap.cxx is > 8Mo, hide more classes?
 * how to automatically add every used implementation of templates (like MakeFileXML)?
 * try to use %naturalvar
 * make MM matrix to/from np.array conversion?
 * use distutils?
 * fix mm modifications (#if[n]def FORSWIG)
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


Changes in MicMac to check:
---------------------------
Parts removed are in #ifndef SWIG for now.
 * remove TheFileMMDIR (not used ?)
 * remove BanniereGlobale (not used ?)
 * xml2cpp enum definition synthax
 * eToString, ToXMLTree, BinaryDumpInFile overloads are shadowed, but new functions with _classname are created for python users. Why is there no problem with BinaryUnDumpFromFile and Mangling ??
