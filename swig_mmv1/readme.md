pymm3d: MicMacv2 Python API
===========================

Introduction
------------

This is an API to a small part of MicMac for Python 3.

Usage
-----

```python
    import mm3d
    
```

More usage examples can be found in swig_mmv1/examples_py/.


Documentation
-------------
See mm3d.html


Examples
--------

[Example](examples_py/ex1.py)


Compilation
-----------
Only tested on Linux.

Dependencies:
 - SWIG
 - pkg-config
 - Python 3 headers
 - PIP for Python 3
 - Wheel for Python 3
 - libclang for Python 3
 - pdoc3

On debian:
    apt install swig python3-dev pkg-config python3-pip
    pip3 install libclang wheel

To compile, select "WITH_APIPYTHON" in cmake interface, then:
    make apipy


Distribution
------------
Distribute file dist/mm3d-*.whl.
User can install it with
    pip3 install mm3d-*.whl

Update pip if needed:
    python3 -m pip install --upgrade pip

Developement
------------

To add MM3D classes to python, add then to gen_fix_classes.py file (in "all_headers" list).
To be able to use templates classes, use %template in mm3d.i
If you want to be able to use python lists for objects of these classes, use %template.

PIP package tutorial: https://packaging.python.org/tutorials/packaging-projects/
https://realpython.com/python-wheels/

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
 * fix mm modifications (#if[n]def FORSWIG)
 * for now the selected classes are copied in api_mm3d.h
 * make a script to automatically extract classes definitions from mm3d sources, add @file
 * create a header file for each subject
 * do not quit python on fatal errors? (add exceptions in micmac?)
 * see https://pybind11.readthedocs.io/en/stable/intro.html
 * createIdealCamXML: from quaternion, and add check points


Changes in MicMac to check:
---------------------------
 * Default constructors and destructors (with "for swig" comment)
 * xml2cpp enum definition synthax
 * eToString, ToXMLTree, BinaryDumpInFile overloads are shadowed, but new functions with _classname are created for python users. Why is there no problem with BinaryUnDumpFromFile and Mangling ??
