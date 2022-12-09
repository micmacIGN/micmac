MMV1_DIR=../..

MMV1_LIBDIR=${MMV1_DIR}/lib
MMV2_DIR=${MMV1_DIR}/MMVII
MMV2_BINDIR=${MMV2_DIR}/bin
MMV2_INCDIR=${MMV2_DIR}/include
MODULE=MMVII$(shell python3-config --extension-suffix)



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
CXXFLAGS=-fopenmp -std=c++17 -Wall  -Werror -O3 -fPIC ${MMV2_INCLUDES} ${MMV1_INCLUDES}  ${PYBIND_INCLUDES} '-DPYBIND_MMVII_DIR="${MMV2_DIR}"'
LDFLAGS=-fopenmp -shared


# ===========  Use ccache if found
CCACHE:=${shell which ccache}
ifneq (,${CCACHE})
  override CXX:=${CCACHE} ${CXX}
endif


# ============ Targets
.phony: all clean distclean docstr
all: docstr ${MODULE}

clean:
	rm -f ${OBJS} ${DEPS} ${MMV2_EXE} ${MMV2_LIB}

distclean: clean
	-rm -f ${MMV2_GENDIR}/*

docstr:
	pybind11-mkdoc -o docstrings.h ../include/MMVII_PCSens.h ${CXXFLAGS}

# =========== Build Objs

SRCS:=${wildcard *.cpp}
OBJS:=${SRCS:.cpp=.o}


%.o: %.cpp Makefile
	${CXX} -c  ${CXXFLAGS} $(python3 -m pybind11 --includes) $< -o $@


# ========== Build Module
${MODULE}: ${OBJS} ${MMV2_LIB} ${MMV1_LIBS}
	${CXX} ${LDFLAGS} $^ ${EXT_LIBS}  -o ${MODULE}


# =========== Auto dependancies
DEPS=${join ${addsuffix ., ${dir ${OBJS}}}, ${notdir ${OBJS:.o=.d}}}

.%.d: %.cpp
	@set -e; rm -f $@; \
	${CXX} -MM -MP -MT '${patsubst %.cpp,%.o,$<} $@' -MF $@ ${CXXFLAGS} $<   2>/dev/null

ifneq (${MAKECMDGOALS},clean)
ifneq (${MAKECMDGOALS},distclean)
-include ${DEPS}
endif
endif
