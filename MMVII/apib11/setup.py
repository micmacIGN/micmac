import sys
import glob
import os

# Available at setup time due to pyproject.toml
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

from pybind11.setup_helpers import ParallelCompile
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

__version__ = "0.0.2"

module_name = 'MMVII'

all_cpp_api_files =  glob.glob('./*.cpp')

cxxflags = ['-fopenmp', '-std=c++17', '-Wall', '-Werror', '-O3', '-fPIC']


ext_modules = [
    Pybind11Extension("_MMVII",
        all_cpp_api_files,
        libraries = ['X11', 'Xext', 'm', 'dl', 'pthread', 'stdc++fs', 'gomp'],
        library_dirs = [],
        include_dirs = ['/usr/local/include', '.', '..', '../include/', '../ExternalInclude/'],
        language = 'c++',
        extra_objects = ['../bin/libP2007.a', '../../lib/libelise.a', '../../lib/libANN.a'],
        extra_compile_args = cxxflags,
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

xml_micmac_files = glob.glob('../../include/XML_MicMac/*.xml')
xml_gen_files = glob.glob('../../include/XML_GEN/*.xml')

setup(
    name=module_name,
    version=__version__,
    author="IGN",
    #author_email="",
    url="https://github.com/micmacIGN",
    description="MicMac v2 Python API",
    long_description="",
    ext_modules=ext_modules,
    py_modules=['MMVII'],
    data_files = [
                  (module_name+"/MMVII/bin", ['../bin/MMVII']),
                  (module_name+"/include/XML_MicMac", xml_micmac_files),
                  (module_name+"/include/XML_GEN", xml_gen_files)
                 ],
    platforms  = ['x86_64'],
    install_requires=['numpy'],
    zip_safe=False,
    python_requires=">=3.7",
)
