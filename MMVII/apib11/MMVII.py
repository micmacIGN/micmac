import os
import sys
from pathlib import Path

mmv2_src_relative_path = '../'
mmv2_install_relative_path = '../../../MMVII/MMVII'

mm_data_path = None

# test if run from MMVII sources
test_path = os.path.normpath(os.path.dirname(__file__) + '/' + mmv2_src_relative_path)
if os.path.isfile(test_path + '/bin/MMVII'):
    print("Using local MMVII")
    mm_data_path = os.path.normpath(os.path.dirname(__file__) + '/' + mmv2_src_relative_path)
    #search for .so file
    so_path = None
    for path in Path('build').rglob('_MMVII.*.so'):
        if so_path is None:
            so_path = path.parent
        else :
            print('Warning: multiple _MMVII.*.so files found in build directory')
    if so_path is not None:
        print("Using _MMVII.*.so file from directory ",so_path)
        sys.path = [ str(so_path) ] + sys.path
else:
    # test if installed
    for p in sys.path:
        test_path = os.path.normpath(os.path.dirname(__file__) + '/' + mmv2_install_relative_path)
        if os.path.isfile(test_path + '/bin/MMVII'):
            mm_data_path = os.path.normpath(os.path.dirname(__file__) + '/' + mmv2_install_relative_path)
            break

if mm_data_path is not None:
    print('MMVII path:', mm_data_path)
    from _MMVII import *
    themodule = MM_Module(mm_data_path)
else:
    print('Error: impossible to find MMVII binary. Error at installation?')

