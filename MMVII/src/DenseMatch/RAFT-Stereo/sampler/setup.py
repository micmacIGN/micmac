from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_flags = {}
extra_link_arguments = {}

TORCHINCLUDE="-I/home/MAChebbi/opt/micmac_SimLearning/MMVII/libtorch/include -I/home/MAChebbi/opt/micmac_SimLearning/MMVII/libtorch/include/torch/csrc/api/include"
TORCHLIB_NVCC="-L/home/MAChebbi/opt/micmac_SimLearning/MMVII/libtorch/lib -Xlinker -rpath=/home/MAChebbi/opt/micmac_SimLearning/MMVII/libtorch/lib"

gencodes = ['-arch=sm_50',
            '-gencode', 'arch=compute_50,code=sm_50',
            '-gencode', 'arch=compute_52,code=sm_52',
            '-gencode', 'arch=compute_60,code=sm_60',
            '-gencode', 'arch=compute_61,code=sm_61',
            '-gencode', 'arch=compute_70,code=sm_70',
            '-gencode', 'arch=compute_75,code=sm_75',
            '-gencode', 'arch=compute_75,code=compute_75',]

#extra_compile_flags['nvcc'] = gencodes
#extra_compile_flags['nvcc'].append(TORCHINCLUDE)
#extra_compile_flags['nvcc'].append(TORCHLIB_NVCC)

#print( extra_compile_flags)

setup(
    name='corr_sampler',
    ext_modules=[
        CUDAExtension('corr_sampler', [
            'sampler.cpp', 'sampler_kernel.cu',
        ],
      extra_compile_args=extra_compile_flags)
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


