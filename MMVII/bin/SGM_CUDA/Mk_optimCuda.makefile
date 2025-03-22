TORCHLIBPath=$(HOME)/opt/micmac/MMVII/libtorch
TORCHINCLUDE=-I$(TORCHLIBPath)/include -I$(TORCHLIBPath)/include/torch/csrc/api/include
TORCHLIB_NVCC=-L$(TORCHLIBPath)/lib -Xlinker -rpath=$(TORCHLIBPath)/lib

all:cudInfer.o cudInfer.a 

cudInfer.o: cudInfer.cu
	nvcc -arch sm_75 -O3 -DNDEBUG --compiler-options '-fPIC' -o cudInfer.o -c cudInfer.cu $(TORCHINCLUDE) $(TORCHLIB_NVCC)

cudInfer.a:cudInfer.o
	ar rcs libcudInfer.a cudInfer.o
clean:
	rm -f cudInfer.o libcudInfer.a
