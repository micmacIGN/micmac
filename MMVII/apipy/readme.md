pymmv2: MicMacv2 Python API
===========================

Introduction
------------

This is an API to a small part of MicMacv2 for Python 3.

Usage
-----

```python
    import mmv2
    
```

More usage examples can be found in apipy/examples_py


Documentation
-------------
See doc/


Compilation
-----------
Only tested on Linux.

Dependencies:
 - SWIG
 - pkg-config
 - Python 3 headers
 - PIP for Python 3
 - Wheel for Python3
 - for documentation: doxygen, GraphViz

Compile MMv1 and MMv2.

On debian:
    apt install swig python3-dev pkg-config python3-pip python3-numpy python3-wheel
    apt install doxygen graphviz

Compilation:
    make -f Makefile_apipy_linux

If MMv1 was compiled with Qt:
    make -f Makefile_apipy_linux USEQT=ON


Distribution
------------
Distribute file dist/mmv2-*.whl.
User can install it with
    pip3 install mmv2-*.whl

Update pip if needed:
    python3 -m pip install --upgrade pip

Developement
------------

To add MMv2 classes to python, add then to mmv2.i file (in "classes to export" part).
Tu be able to use templates classes, use %template.
If you want to be able to use python lists for objects of these classes, use %template.

PIP package tutorial: https://packaging.python.org/tutorials/packaging-projects/
https://realpython.com/python-wheels/

TODO
----
 - test shared pointers behavior in python between cIm2D and cDataIm2D
 - test memory AllocExecuteDestruct, check Restoration
 - matrices with numpy (cRotation3d)
 - re-create benchs in py

