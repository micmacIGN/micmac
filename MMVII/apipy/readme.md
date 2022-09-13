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
See mmv2.html


Examples
--------

[Example](ex/ex1.py)


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
 - for documentation: doxygen and GraphViz, or pdoc3

On debian:
    apt install swig python3-dev pkg-config python3-pip
    pip3 install libclang wheel

First, compile MMv1 and MMv2.

Apipy compilation:
    make -f Makefile_apipy_linux

Distribution
------------
Distribute file dist/mmv2-*.whl.
User can install it with
    pip3 install mmv2-*.whl

Update pip if needed:
    python3 -m pip install --upgrade pip

Developement
------------

To add MMv2 classes to python, add then to gen_fix_classes.py file (in "all_headers" list).
To be able to use templates classes, use %template in mmv2.i
If you want to be able to use python lists for objects of these classes, use %template.

PIP package tutorial: https://packaging.python.org/tutorials/packaging-projects/
https://realpython.com/python-wheels/

TODO
----
 - test shared pointers behavior in python between cIm2D and cDataIm2D
 - test memory AllocExecuteDestruct, check Restoration
 - matrices with numpy (cRotation3d)
 - re-create benchs in py

