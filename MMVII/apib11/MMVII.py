import os
mm_install_path = os.path.normpath(os.path.dirname(__file__) + '/../../../MMVII/MMVII')

if os.path.isdir(mm_install_path):
    print('MMVII path:', mm_install_path)
    from _MMVII import *
    themodule = MM_Module(mm_install_path)
else:
    print('Error: please do not run from this directory')

