MMV1_DIR=../..

MMV1_LIBDIR=${MMV1_DIR}/lib
MMV2_DIR=${MMV1_DIR}/MMVII
MMV2_BINDIR=${MMV2_DIR}/bin
MMV2_INCDIR=${MMV2_DIR}/include
export MODULE_NAME = MMVII

DOC_INCLUDES=../include/MMVII_PCSens.h ../include/MMVII_memory.h ../include/MMVII_AimeTieP.h ../include/MMVII_Geom3D.h ../include/MMVII_Mappings.h ../include/MMVII_MeasuresIm.h


# =========== Includes & Libraries
MMV1_INCLUDES=-I ${MMV1_DIR}/include -I${MMV1_DIR}
MMV2_INCLUDES=-I ${MMV2_DIR} -I ${MMV2_INCDIR} -I /usr/include/eigen3
PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes) 

MMV2_LIBNAME=P2007
MMV2_LIB=${MMV2_BINDIR}/lib${MMV2_LIBNAME}.a
MMV1_LIBS=${MMV1_LIBDIR}/libelise.a ${MMV1_LIBDIR}/libANN.a
## MacOS : may be -lstdc++fs should be replaced by -lc++experimental
EXT_LIBS=-lpthread -lX11 -lstdc++fs


# =========== Compiler & Flags
CXX=g++
LD=g++
export CXXFLAGS=-fopenmp -std=c++17 -Wall  -Werror -O3 -fPIC 
ALL_INC=${MMV2_INCLUDES} ${MMV1_INCLUDES}  ${PYBIND_INCLUDES}

PYTHON = python3
PIP = pip3

# ===========  Use ccache if found
CCACHE:=${shell which ccache}
ifneq (,${CCACHE})
  override CXX:=${CCACHE} ${CXX}
endif

export SRCS:=${wildcard *.cpp}

# ============ Targets
export XML_MICMAC_FILES=$(wildcard ../../include/XML_MicMac/*.xml)
export XML_GEN_FILES=$(wildcard ../../include/XML_GEN/*.xml)
export EXTRA_OBJECTS=../bin/libP2007.a ../../lib/libelise.a ../../lib/libANN.a
export MMVII_BIN=../bin/MMVII

MODULE=$(shell export CC="$$CXX" SRCS="$$SRCS"  CXXFLAGS="$$CXXFLAGS" XML_MICMAC_FILES="$$XML_MICMAC_FILES" XML_GEN_FILES="$$XML_GEN_FILES" EXTRA_OBJECTS="$$EXTRA_OBJECTS" MMVII_BIN="$$MMVII_BIN" ; $(PYTHON) setup.py module_name)

.phony: all clean clean_python install uninstall

all: $(MODULE)

clean: uninstall clean_python
	rm -f docstrings.h
	rm -fr xml

clean_python:
	rm -Rf build/ dist/ $(MODULE_NAME).egg-info/ __pycache__/
	$(PYTHON) setup.py clean

docstrings.h: $(DOC_INCLUDES) Makefile
	rm -f $@
	rm -fr xml
	(cat Doxyfile ; echo "INPUT=$(DOC_INCLUDES)") | doxygen -
	python3 makedoc.py xml/class*xml xml/namespaceMMVII.xml > $@ || rm -f $@
	rm -fr xml

$(MODULE): export CC=${CXX} 

$(MODULE): docstrings.h ${SRCS} $(MMVII_BIN) $(EXTRA_OBJECTS) $(XML_MICMAC_FILES) $(XML_GEN_FILES)
	$(MAKE) clean_python
	$(PYTHON) setup.py build
	$(PYTHON) setup.py bdist_wheel

install: $(MODULE)
	$(PIP) install --force-reinstall --user dist/*.whl

uninstall:
	$(PIP) uninstall -y $(MODULE_NAME)

