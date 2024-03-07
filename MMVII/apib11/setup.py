import sys
import glob
import os

# Available at setup time due to pyproject.toml
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.dist import Distribution

from pybind11.setup_helpers import ParallelCompile
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

__version__ = "0.0.2"

module_name = 'MMVII'




all_cpp_api_files =  os.getenv('SRCS').split()
cxxflags = os.getenv('CXXFLAGS').split()
xml_micmac_files = os.getenv('XML_MICMAC_FILES').split()
xml_mmvii_localparamters = os.getenv('XML_MMVII_LOCALPARAMETERS').split()
xml_mmvii_localparamters_default = os.getenv('XML_MMVII_LOCALPARAMETERS_DEFAULT').split()
xml_gen_files = os.getenv('XML_GEN_FILES').split()
extra_objects = os.getenv('EXTRA_OBJECTS').split()
mmv2_bin = os.getenv('MMVII_BIN').split()


ext_modules = [
    Pybind11Extension("_MMVII",
        all_cpp_api_files,
        libraries = ['X11', 'Xext', 'm', 'dl', 'pthread', 'stdc++fs', 'gomp'],
        library_dirs = [],
        include_dirs = ['/usr/local/include', '.', '..', '../include/', '../ExternalInclude/eigen-3.4.0'],
        language = 'c++',
        extra_objects = extra_objects,
        extra_compile_args = cxxflags,
        define_macros = [('VERSION_INFO', __version__)],
        ),
]


def wheel_name(**kwargs):
    # create a fake distribution from arguments
    dist = Distribution(attrs=kwargs)
    # finalize bdist_wheel command
    bdist_wheel_cmd = dist.get_command_obj('bdist_wheel')
    bdist_wheel_cmd.ensure_finalized()
    # assemble wheel file name
    distname = bdist_wheel_cmd.wheel_dist_name
    tag = '-'.join(bdist_wheel_cmd.get_tag())
    return f'{distname}-{tag}.whl'


setup_kwargs = dict(
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
                  (module_name+"/MMVII/bin", mmv2_bin),
                  (module_name+"/include/XML_MicMac", xml_micmac_files),
                  (module_name+"/MMVII/MMVII-LocalParameters", xml_mmvii_localparamters),
                  (module_name+"/MMVII/MMVII-LocalParameters/Default", xml_mmvii_localparamters_default),
                  (module_name+"/include/XML_GEN", xml_gen_files)
                 ],
    platforms  = ['x86_64'],
    install_requires=[],
    zip_safe=False,
    python_requires=">=3.7",
)

if len(sys.argv) == 2 and sys.argv[1] == 'module_name':
    print ('dist/' + wheel_name(**setup_kwargs))
    sys.exit(0)
setup(**setup_kwargs)
