all: test_multichannel test_convolution

CULTURE3D_DIR=../../../
INCLUDE_DIR=$(CULTURE3D_DIR)include
LIBRARY_DIR=$(CULTURE3D_DIR)lib
LIBRARIES=-lelise -lX11
ELISE_LIB=$(LIBRARY_DIR)/libelise.a

test_multichannel: $(ELISE_LIB) multichannel.cpp
	$(CXX) multichannel.cpp -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) $(LIBRARIES) -o test_multichannel

test_convolution: $(ELISE_LIB) convolution.cpp
	$(CXX) convolution.cpp -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) $(LIBRARIES) -o test_convolution
