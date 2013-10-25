CXX = g++
INC_DIR = -I../../include
LIB_DIR = -L../../lib
LIBS = -lelise -lX11
CXXFLAGS = -O -Werror -Wall

OBJ = b_0_35.o\
      b_0_38.o\
#      b_0_56.cpp

.cpp.o:
	$(CXX) -c $(CCFLAG) $(INC_DIR) $< -o $@

all: b_0_all.cpp $(OBJ)
	g++ b_0_all.cpp $(OBJ) -o bench $(CXXFLAGS) $(INC_DIR) $(LIB_DIR) $(LIBS)
