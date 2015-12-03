CXX ?= clang++
CXX_FLAGS += -std=c++11 -Wall -O3 -DNO_ELISE
ELISE_ROOT_DIR=../../..
ELISE_LIBRARY_DIR=$(ELISE_ROOT_DIR)/lib
ELISE_INCLUDE_DIR=$(ELISE_ROOT_DIR)/include
DIGEO_DIR=$(ELISE_ROOT_DIR)/src/uti_image/Digeo
CPP_FILES = rotate.cpp\
 test_ConvolutionKernel1D.cpp test_GaussianConvolutionKernel1D.cpp test_Convolution.cpp\
 $(DIGEO_DIR)/ConvolutionKernel1D.cpp $(DIGEO_DIR)/GaussianConvolutionKernel1D.cpp $(DIGEO_DIR)/cConvolSpec.cpp $(DIGEO_DIR)/Times.cpp 
D_FILES = $(CPP_FILES:.cpp=.d)
O_FILES = $(CPP_FILES:.cpp=.o)
ALL_CXX_FLAGS = $(CXX_FLAGS) -I$(DIGEO_DIR) -I$(ELISE_INCLUDE_DIR)
EXECUTABLES = test_ConvolutionKernel1D test_GaussianConvolutionKernel1D test_Convolution rotate dumb_matcher
LIB_X11 = -lX11
LINK_FLAGS = -Wall -L$(ELISE_LIBRARY_DIR) -lelise $(LIB_X11)

all: $(EXECUTABLES)

debug: DEBUG_FLAGS += -D__DEBUG
debug: all

-include $(D_FILES)

depends: $(D_FILES)

%.d: %.cpp
	@set -e; rm -f $@;\
	$(CXX) $(ALL_CXX_FLAGS) -MM $< > $@.tmp;\
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@;\
	rm $@.tmp

rotate.o: rotate.cpp
	$(CXX) -c -std=c++11 $(DEBUG_FLAGS) -Wall -O3 -I$(DIGEO_DIR) -I$(ELISE_INCLUDE_DIR) rotate.cpp -o rotate.o

rotate: rotate.o $(ELISE_LIBRARY_DIR)/libelise.a
	$(CXX) rotate.o $(LINK_FLAGS) -o rotate

dumb_matcher: dumb_matcher.o $(ELISE_LIBRARY_DIR)/libelise.a
	$(CXX) dumb_matcher.o $(LINK_FLAGS) -o dumb_matcher

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
	$(CXX) -c $(DEBUG_FLAGS) $(ALL_CXX_FLAGS) $< -o $@

clean_depends:
	rm -f $(D_FILES)
	rm -f *.tmp

clean: clean_depends
	rm -f $(EXECUTABLES)
	rm -f *.o
	rm -f *.txt
	rm -f *.jpg
	rm -f *.exe
	rm -fr output
