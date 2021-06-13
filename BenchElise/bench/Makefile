CXX = g++
INC_DIR = -I../../include
LIB_DIR = -L../../lib
LIBS = -lelise -lX11
CXXFLAGS = -O -Werror -Wall

OBJ = b_0_35.o\
      b_0_38.o\
      b_0_56.o\
      b_0_0.o\
      b_0_1.o\
      b_0_2.o\
      b_0_3.o\
      b_0_4.o\
      b_0_5.o\
      b_0_6.o\
      b_0_7.o\
      b_0_8.o\
      b_0_9.o\
      b_0_10.o\
      b_0_11.o\
      b_0_12.o\
      b_0_13.o\
      b_0_14.o\
      b_0_15.o\
      b_0_16.o\
      b_0_17.o\
      b_0_18.o\
      b_0_18_1.o\
      b_0_19.o\
      b_0_19_1.o\
      b_0_19_2.o\
      b_0_20.o\
      b_0_21.o\
      b_0_22.o\
      b_0_23.o\
      b_0_24.o\
      b_0_25.o\
      b_0_26.o\
      b_0_26_1.o\
      b_0_27.o\
      b_0_28.o\
      b_0_29.o\
      b_0_30.o\
      b_0_31.o\
      b_0_32.o\
      b_0_33.o\
      b_0_34.o\
      b_0_34_1.o\
      b_0_36.o\
      b_0_36_1.o\
      b_0_37.o\
      b_0_39.o\
      b_0_40.o\
      b_0_41.o\
      b_0_43.o\
      b_0_45.o\
      b_0_44.o\
      b_0_42.o\
      b_0_46_1.o\
      b_0_46_5.o\
      b_0_46.o\
      b_0_47.o\
      b_0_48.o\
      b_0_49.o\
      b_0_50.o\
      b_0_46_8.o\
      b_0_48_3.o\
      b_0_48_4.o\
      b_0_51.o\
      b_0_52.o\
      b_0_53.o\
      b_0_54.o\
      b_0_46_4.o\
      b_0_55.o\
      b_0_54_1.o

.cpp.o:
	$(CXX) -c $(CCFLAG) $(INC_DIR) $< -o $@

bench: b_0_all.cpp $(OBJ)
	g++ b_0_all.cpp $(OBJ) -o bench $(CXXFLAGS) $(INC_DIR) $(LIB_DIR) $(LIBS)

all: bench

