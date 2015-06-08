CXX ?= clang++
CXX_FLAGS += -Wall -O3 -DNO_ELISE
LINK_FLAGS += -Wall
DIGEO_DIR=../../../src/uti_image/Digeo
CPP_FILES = test_ConvolutionKernel1D.cpp test_GaussianConvolutionKernel1D.cpp test_Convolution.cpp\
 $(DIGEO_DIR)/ConvolutionKernel1D.cpp $(DIGEO_DIR)/GaussianConvolutionKernel1D.cpp $(DIGEO_DIR)/cConvolSpec.cpp $(DIGEO_DIR)/Times.cpp 
D_FILES = $(CPP_FILES:.cpp=.d)
O_FILES = $(CPP_FILES:.cpp=.o)
ALL_CXX_FLAGS = $(CXX_FLAGS) -I$(DIGEO_DIR)
EXECUTABLES = test_ConvolutionKernel1D test_GaussianConvolutionKernel1D test_Convolution

all: $(EXECUTABLES)

debug: CXX_FLAGS += -DDEBUG
debug: all

-include $(D_FILES)

depends: $(D_FILES)

%.d: %.cpp
	@set -e; rm -f $@;\
	$(CXX) $(ALL_CXX_FLAGS) -MM $< > $@.tmp;\
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@;\
	rm $@.tmp

test_ConvolutionKernel1D: test_ConvolutionKernel1D.o ConvolutionKernel1D.o
	$(CXX) $(LINK_FLAGS) test_ConvolutionKernel1D.o ConvolutionKernel1D.o -o test_ConvolutionKernel1D

test_GaussianConvolutionKernel1D: test_GaussianConvolutionKernel1D.o GaussianConvolutionKernel1D.o ConvolutionKernel1D.o
	$(CXX) $(LINK_FLAGS) test_GaussianConvolutionKernel1D.o GaussianConvolutionKernel1D.o ConvolutionKernel1D.o -o test_GaussianConvolutionKernel1D

test_Convolution: test_Convolution.o cConvolSpec.o GaussianConvolutionKernel1D.o ConvolutionKernel1D.o  Times.o
	$(CXX) $(LINK_FLAGS) test_Convolution.o cConvolSpec.o GaussianConvolutionKernel1D.o ConvolutionKernel1D.o  Times.o -o test_Convolution

ConvolutionKernel1D.o: $(DIGEO_DIR)/ConvolutionKernel1D.cpp
	$(CXX) -c $(ALL_CXX_FLAGS) $(DIGEO_DIR)/ConvolutionKernel1D.cpp -o ConvolutionKernel1D.o

GaussianConvolutionKernel1D.o: $(DIGEO_DIR)/GaussianConvolutionKernel1D.cpp
	$(CXX) -c $(ALL_CXX_FLAGS) $(DIGEO_DIR)/GaussianConvolutionKernel1D.cpp -o GaussianConvolutionKernel1D.o

cConvolSpec.o: $(DIGEO_DIR)/cConvolSpec.cpp
	$(CXX) -c $(ALL_CXX_FLAGS) $(DIGEO_DIR)/cConvolSpec.cpp -o cConvolSpec.o

Times.o: $(DIGEO_DIR)/Times.cpp
	$(CXX) -c $(ALL_CXX_FLAGS) $(DIGEO_DIR)/Times.cpp -o Times.o

%.o: %.cpp
	$(CXX) -c $(ALL_CXX_FLAGS) $< -o $@

clean_depends:
	rm -f *.d
	rm -f *.tmp

clean: clean_depends
	rm -f $(EXECUTABLES)
	rm -f *.o
	rm -f *.txt
	rm -f *.jpg
	rm -f *.exe

