MMV1_DIR=../..

MMV1_LIBDIR=${MMV1_DIR}/lib
MMV2_DIR=${MMV1_DIR}/MMVII
MMV2_BINDIR=${MMV2_DIR}/bin
MMV2_INCDIR=${MMV2_DIR}/include
MODULE_NAME = MMVII

DOC_INCLUDES=../include/MMVII_PCSens.h ../include/MMVII_memory.h ../include/MMVII_AimeTieP.h ../include/MMVII_Geom3D.h ../include/MMVII_Mappings.h


# =========== Includes & Libraries
MMV1_INCLUDES=-I ${MMV1_DIR}/include -I${MMV1_DIR}
MMV2_INCLUDES=-I ${MMV2_DIR} -I ${MMV2_INCDIR} -I ${MMV2_DIR}/ExternalInclude
PYBIND_INCLUDES=$(shell python3 -m pybind11 --includes) 

MMV2_LIBNAME=P2007
MMV2_LIB=${MMV2_BINDIR}/lib${MMV2_LIBNAME}.a
MMV1_LIBS=${MMV1_LIBDIR}/libelise.a ${MMV1_LIBDIR}/libANN.a
## MacOS : may be -lstdc++fs should be replaced by -lc++experimental
EXT_LIBS=-lpthread -lX11 -lstdc++fs


# =========== Compiler & Flags
CXX=g++
LD=g++
CXXFLAGS=-fopenmp -std=c++17 -Wall  -Werror -O3 -fPIC ${MMV2_INCLUDES} ${MMV1_INCLUDES}  ${PYBIND_INCLUDES}

PYTHON = python3
PIP = pip3

# ===========  Use ccache if found
CCACHE:=${shell which ccache}
ifneq (,${CCACHE})
  override CXX:=${CCACHE} ${CXX}
endif

SRCS:=${wildcard *.cpp}

# ============ Targets
.phony: all module clean install uninstall
all: module

clean: uninstall
	rm -f docstrings.h
	rm -Rf build/ dist/ $(MODULE_NAME).egg-info/ __pycache__/
	$(PYTHON) setup.py clean

docstrings.h: ${DOC_INCLUDES} Makefile
	pybind11-mkdoc -o $@ ${DOC_INCLUDES} ${CXXFLAGS} >/dev/null

module: docstrings.h ${SRCS}
	CC="${CXX}" $(PYTHON) setup.py build
	CC="${CXX}" $(PYTHON) setup.py bdist_wheel

install: module
	$(PIP) install --user dist/*.whl

uninstall:
	$(PIP) uninstall -y $(MODULE_NAME)

