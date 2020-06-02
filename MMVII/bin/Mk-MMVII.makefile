#
#== Directories
#
MMDir=../../
MMV2Dir=${MMDir}MMVII/
MMV2DirSrc=${MMV2Dir}src/
MMV2DirBin=${MMV2Dir}bin/
MMV2Objects=${MMV2Dir}object/
MMV2DirIncl=${MMV2Dir}include/
MMV2ElisePath=${MMDir}/lib/libelise.a
MMV2Exe=MMVII

MMSymbDerDir=${MMV2DirIncl}/SymbDer
MMSymbDerHeader=$(wildcard ${MMSymbDerDir}/*.h)

all : ${MMV2DirBin}${MMV2Exe}

#
#
#       ===== INSTALLATION ========================================
#       => for setting MMVII directory
#
MMV2ResultInstal=${MMV2DirSrc}ResultInstall/ResultInstall.cpp
MMV2SrcInstal=${MMV2DirSrc}BinaireInstall/InstalMM.cpp
MMV2BinInstal=../Instal2007
#
#=========== Sous directory des sources
#
MMV2DirTLE=${MMV2DirSrc}TestLibsExtern/
SrcTLE=$(wildcard ${MMV2DirTLE}*.cpp)
ObjTLE=$(SrcTLE:.cpp=.o) 
#
MMV2DirT4MkF=${MMV2DirSrc}Test4Mkfile/
Src4Mkf= $(wildcard ${MMV2DirT4MkF}*.cpp)
ObjMkf= $(Src4Mkf:.cpp=.o) 
#
MMV2DirBench=${MMV2DirSrc}Bench/
SrcBench= $(wildcard ${MMV2DirBench}*.cpp)
ObjBench= $(SrcBench:.cpp=.o) 
#
#
MMV2DirAppli=${MMV2DirSrc}Appli/
SrcAppli= $(wildcard ${MMV2DirAppli}*.cpp)
ObjAppli= $(SrcAppli:.cpp=.o) 
#
#
MMV2DirUtils=${MMV2DirSrc}Utils/
SrcUtils= $(wildcard ${MMV2DirUtils}*.cpp)
ObjUtils= $(SrcUtils:.cpp=.o) 
#
#
MMV2DirUtiMaths=${MMV2DirSrc}UtiMaths/
SrcUtiMaths= $(wildcard ${MMV2DirUtiMaths}*.cpp)
ObjUtiMaths= $(SrcUtiMaths:.cpp=.o) 
#
#
MMV2DirSerial=${MMV2DirSrc}Serial/
SrcSerial= $(wildcard ${MMV2DirSerial}*.cpp)
ObjSerial= $(SrcSerial:.cpp=.o) 
#
#
MMV2DirMMV1=${MMV2DirSrc}MMV1/
SrcMMV1=$(wildcard ${MMV2DirMMV1}*.cpp)
ObjMMV1=$(SrcMMV1:.cpp=.o) 
#
#
MMV2DirCmdSpec=${MMV2DirSrc}CmdSpec/
SrcCmdSpec=$(wildcard ${MMV2DirCmdSpec}*.cpp)
ObjCmdSpec=$(SrcCmdSpec:.cpp=.o) 
#
#
MMV2DirImagesBase=${MMV2DirSrc}ImagesBase/
SrcImagesBase=$(wildcard ${MMV2DirImagesBase}*.cpp)
ObjImagesBase=$(SrcImagesBase:.cpp=.o) 
#
#
MMV2DirImagesFiltrLinear=${MMV2DirSrc}ImagesFiltrLinear/
SrcImagesFiltrLinear=$(wildcard ${MMV2DirImagesFiltrLinear}*.cpp)
ObjImagesFiltrLinear=$(SrcImagesFiltrLinear:.cpp=.o) 
#
#
MMV2DirImagesInfoExtract=${MMV2DirSrc}ImagesInfoExtract/
SrcImagesInfoExtract=$(wildcard ${MMV2DirImagesInfoExtract}*.cpp)
ObjImagesInfoExtract=$(SrcImagesInfoExtract:.cpp=.o) 
#
#
MMV2DirMatrix=${MMV2DirSrc}Matrix/
SrcMatrix=$(wildcard ${MMV2DirMatrix}*.cpp)
ObjMatrix=$(SrcMatrix:.cpp=.o) 
#
#
MMV2DirDIB=${MMV2DirSrc}DescIndexBinaire/
SrcDIB=$(wildcard ${MMV2DirDIB}*.cpp)
ObjDIB=$(SrcDIB:.cpp=.o)
#
#
MMV2DirPerso=${MMV2DirSrc}Perso/
SrcPerso=$(wildcard ${MMV2DirPerso}*.cpp)
ObjPerso=$(SrcPerso:.cpp=.o)
#
#
MMV2DirCalcDescriptPCar=${MMV2DirSrc}CalcDescriptPCar/
SrcCalcDescriptPCar=$(wildcard ${MMV2DirCalcDescriptPCar}*.cpp)
ObjCalcDescriptPCar=$(SrcCalcDescriptPCar:.cpp=.o)
#
#
MMV2DirMatchTieP=${MMV2DirSrc}MatchTieP/
SrcMatchTieP=$(wildcard ${MMV2DirMatchTieP}*.cpp)
ObjMatchTieP=$(SrcMatchTieP:.cpp=.o)
#
#
MMV2DirGraphs=${MMV2DirSrc}Graphs/
SrcGraphs=$(wildcard ${MMV2DirGraphs}*.cpp)
ObjGraphs=$(SrcGraphs:.cpp=.o) 
#
#    => Le Main
MAIN=${MMV2DirSrc}main.cpp
#============ Calcul des objets
#
OBJ= ${ObjMatchTieP} ${ObjCalcDescriptPCar} ${ObjImagesBase}  ${ObjMMV1}  ${ObjUtiMaths} ${ObjImagesInfoExtract} ${ObjImagesFiltrLinear} ${ObjCmdSpec} ${ObjBench} ${ObjMatrix} ${ObjAppli} ${ObjDIB}   ${ObjTLE} ${ObjMkf} ${ObjUtils} ${ObjSerial}  ${ObjPerso}  ${ObjGraphs}
#
#=========  Header ========
#
#
HEADER=$(wildcard ${MMV2DirIncl}*.h)
#
# ========== Code Generator
#
MMV2_CGEN_DIR=${MMV2DirSrc}TestCodeGen
MMV2_CGEN_SRC=${MMV2_CGEN_DIR}/CodeGenerator.cpp
MMV2_CODEGEN=${MMV2DirBin}CodeGenerator

${MMV2_CODEGEN}: ${MMV2_CGEN_SRC} ${MMV2_CGEN_DIR}/Formula_*.h $(filter-out ${MMV2DirIncl}CodeGen_%.h,${HEADER})
	${CXX}  $< ${CFlags}  -o $@

${MMV2DirIncl}/CodeGen_IncludeAll.h: ${MMV2_CODEGEN}
	@echo "* Generating Formulas code"
	( cd ${MMV2DirIncl} && ${MMV2_CODEGEN} )

HEADER:=$(filter-out ${MMV2DirIncl}CodeGen%.h,${HEADER})

${MMV2_CGEN_DIR}/%.o :  ${MMV2_CGEN_DIR}/%.cpp ${HEADER} ${MMSymbDerHeader}
	${CXX} -c  $< ${CFlags} -o $@

MMV2_TCGEN_SRCS=$(filter-out ${MMV2_CGEN_SRC},$(wildcard ${MMV2_CGEN_DIR}/*.cpp))
MMV2_TCGEN_OBJS=${MMV2_TCGEN_SRCS:.cpp=.o}
OBJ:=${OBJ} ${MMV2_TCGEN_OBJS}

${MMV2_TCGEN_OBJS}: ${HEADER} ${MMV2_CGEN_DIR}/TestCodeGenTpl.h ${MMV2DirIncl}/CodeGen_IncludeAll.h

${OBJ}: | ${MMV2_CODEGEN}


#
#  Binaries
#== CFLAGS etc...
#
CXX=g++
CFlags= "-fopenmp" "-std=c++14" "-Wall"  "-Werror" "-O4" "-march=native" -I${MMV2Dir} -I${MMDir}/include/ -I${MMDir}
BOOST_LIBS= -lboost_system -lboost_serialization -lboost_regex -lboost_filesystem
QTAnnLibs= -lXext /usr/lib/x86_64-linux-gnu/libQt5Core.so /usr/lib/x86_64-linux-gnu/libQt5Gui.so /usr/lib/x86_64-linux-gnu/libQt5Xml.so /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so -lGLU -lGL  -ldl -lpthread /usr/lib/x86_64-linux-gnu/libQt5Xml.so /usr/lib/x86_64-linux-gnu/libQt5Concurrent.so /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so /usr/lib/x86_64-linux-gnu/libQt5Widgets.so /usr/lib/x86_64-linux-gnu/libQt5Gui.so /usr/lib/x86_64-linux-gnu/libQt5Core.so ../../lib/libANN.a
LibsFlags= ${MMV2ElisePath} -lX11  ${BOOST_LIBS}  ${QTAnnLibs}
#
${MMV2DirBin}${MMV2Exe} :  ${OBJ} ${MAIN} ${MMV2ResultInstal} ${MMV2ElisePath}
	${CXX}  ${MAIN} ${CFlags}  ${OBJ}  ${LibsFlags}  -o ${MMV2DirBin}${MMV2Exe} 
	ar rvs P2007.a    ${OBJ}  
#
# ==========    INSTALLATION =================
#
${MMV2ResultInstal} : ${MMV2SrcInstal}
	${CXX} ${MMV2SrcInstal} -o ${MMV2BinInstal} ${BOOST_LIBS}
	mkdir -p `dirname ${MMV2ResultInstal}`
	${MMV2BinInstal}
	rm ${MMV2BinInstal}
#
# ================ Objects ==================
#
${MMV2DirMatchTieP}%.o :  ${MMV2DirMatchTieP}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirCalcDescriptPCar}%.o :  ${MMV2DirCalcDescriptPCar}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirPerso}%.o :  ${MMV2DirPerso}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirMMV1}%.o :  ${MMV2DirMMV1}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirCmdSpec}%.o :  ${MMV2DirCmdSpec}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirTLE}%.o :  ${MMV2DirTLE}%.cpp   ${HEADER} ${MMSymbDerHeader}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirT4MkF}%.o :  ${MMV2DirT4MkF}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirBench}%.o :  ${MMV2DirBench}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirAppli}%.o :  ${MMV2DirAppli}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirUtils}%.o :  ${MMV2DirUtils}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirUtiMaths}%.o :  ${MMV2DirUtiMaths}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirSerial}%.o :  ${MMV2DirSerial}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirImagesBase}%.o :  ${MMV2DirImagesBase}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirImagesFiltrLinear}%.o :  ${MMV2DirImagesFiltrLinear}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirImagesInfoExtract}%.o :  ${MMV2DirImagesInfoExtract}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirMatrix}%.o :  ${MMV2DirMatrix}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirDIB}%.o :  ${MMV2DirDIB}%.cpp   ${HEADER} ${MMV2DirDIB}*.h
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirGraphs}%.o :  ${MMV2DirGraphs}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
#
#       ===== TEST ========================================
#
Show:
	echo ${SrcCalcDescriptPCar}
	echo DU=${MMV2DirCalcDescriptPCar}
	echo ObjCalcDescriptPCar : ${ObjCalcDescriptPCar}
	echo SrcCalcDescriptPCar: ${SrcCalcDescriptPCar}
	echo MMV2DirCalcDescriptPCar: ${MMV2DirCalcDescriptPCar}
clean :
	rm -f ${OBJ} ${MMV2_CODEGEN} ${MMV2DirIncl}CodeGen_*.h
#
#
