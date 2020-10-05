#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension

import sys

libs=[]
if "USEQT=ON" in sys.argv:
  print("With USEQT")
  libs=['GL', 'GLU', 'Qt5Core', 'Qt5Gui', 'Qt5Widgets', 'Qt5Concurrent', 'Qt5OpenGL', 'Qt5Xml']
else:
  print("Without USEQT")

#clear arguments for setup()
for arg in sys.argv[:]:
    if arg.startswith("USEQT"):
        sys.argv.remove(arg)

mm3d_module = Extension('_mm3d',
           define_macros = [('FORSWIG','')],
           sources=['mm3d_wrap.cxx', 'api/api_mm3d.cpp'],
           libraries = ['X11', 'Xext', 'm', 'dl', 'pthread']+libs,
           library_dirs = [],
           include_dirs = ['/usr/local/include', '.', '../include/'],
           language="c++",
           extra_objects=["../lib/libelise.a", "../lib/libANN.a"],
           extra_compile_args=['-std=c++11']
       )

setup (name = 'mm3d',
       version = '0.1',
       author      = "IGN",
       description = """MicMac Python API""",
       ext_modules = [mm3d_module],
       py_modules = ["mm3d"],
       )

#https://docs.python.org/3/extending/building.html
#swig -python -py3 -DFORSWIG -c++ -I. -I../include/ mm3d.i
#python3 setup.py clean
#rm *.so
#python3 setup.py build_ext --inplace
#sudo python3 setup.py install
