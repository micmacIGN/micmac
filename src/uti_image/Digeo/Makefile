CXX = clang++
CXX_FLAGS = -Wall -D__DEBUG=1 -D__UNIX=1 -O3

all: test_ConvolutionKernel1D test_GaussianConvolutionKernel1D test_Convolution

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

TypeTraits.h: base_types.h
ConvolutionKernel1D.h: TypeTraits.h
ConvolutionKernel1D.o: ConvolutionKernel1D.h debug.h
GaussianConvolutionKernel1D.h: ConvolutionKernel1D.h
GaussianConvolutionKernel1D.o: GaussianConvolutionKernel1D.h debug.h
cConvolSpec.h: ConvolutionKernel1D.h
cConvolSpec.o: cConvolSpec.h GenConvolSpec.u_int1.h GenConvolSpec.u_int2.h GenConvolSpec.real4.h debug.h
test_GaussianConvolutionKernel1D.o: GaussianConvolutionKernel1D.h ConvolutionKernel1D.h TypeTraits.h base_types.h
test_Convolution.o: cConvolSpec.h DiamondSquare.h GaussianConvolutionKernel1D.h
Times.o: Times.h Times.inline.h

test_ConvolutionKernel1D: ConvolutionKernel1D.o test_ConvolutionKernel1D.o ConvolutionKernel1D.h
	$(CXX) $(CXX_FLAGS) ConvolutionKernel1D.cpp test_ConvolutionKernel1D.cpp -o test_ConvolutionKernel1D

test_GaussianConvolutionKernel1D: GaussianConvolutionKernel1D.o ConvolutionKernel1D.o cConvolSpec.o test_GaussianConvolutionKernel1D.o
	$(CXX) $(CXX_FLAGS) GaussianConvolutionKernel1D.o ConvolutionKernel1D.o test_GaussianConvolutionKernel1D.o cConvolSpec.o -D__DEBUG=1 -Wall -o test_GaussianConvolutionKernel1D

test_Convolution: cConvolSpec.h DiamondSquare.h GaussianConvolutionKernel1D.h test_Convolution.o ConvolutionKernel1D.o GaussianConvolutionKernel1D.o cConvolSpec.o Times.o
	$(CXX) $(CXX_FLAGS) test_Convolution.o ConvolutionKernel1D.o GaussianConvolutionKernel1D.o cConvolSpec.o Times.o -D__DEBUG=1 -Wall -o test_Convolution

clean:
	rm -f *.o
	rm -f test_ConvolutionKernel1D
	rm -f test_GaussianConvolutionKernel1D
	rm -f test_Convolution

clean_convolutions:
	echo "" > GenConvolSpec.u_int1.h
	echo "" > GenConvolSpec.u_int2.h
	echo "" > GenConvolSpec.real4.h
