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
 - for documentation: doxygen, GraphViz

On debian:
    apt install swig python3-dev doxygen graphviz pkg-config

Compilation:
    make -f Makefile_apipy_linux

The module is automatically copied in ~/.local/lib/python3.*/site-packages/ and the xml files in ~/.local/mmv2/.

Developement
------------

To add MMv2 classes to python, add then to mmv2.i file (in "classes to export" part).
Tu be able to use templates classes, use %template.
If you want to be able to use python lists for objects of these classes, use %template.

PIP package tutorial: https://packaging.python.org/tutorials/packaging-projects/


TODO
----
 - test shared pointers behavior in python between cIm2D and cDataIm2D
 - test memory AllocExecuteDestruct, check Restoration
 - matrices with numpy (cRotation3d)
 - re-create benchs in py

