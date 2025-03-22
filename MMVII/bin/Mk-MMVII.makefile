#
#== Directories
#
MMDir=../../
MMV2Dir=${MMDir}MMVII/
MMV2DirSrc=${MMV2Dir}src/
MMV2DirBin=${MMV2Dir}bin/
MMV2Objects=${MMV2Dir}object/
MMV2DirIncl=${MMV2Dir}include
MMV2ElisePath=${MMDir}/lib/libelise.a
MMV2_EXTINCDIR=${MMV2Dir}ExternalInclude
# =========== Includes & Libraries
EIGEN_DIR=${MMV2_EXTINCDIR}/eigen-3.4.0
# LIBRARY TORCH LIBS AND INLCUDES 
#==========================================================#
TORCHLIBPath=$(HOME)/Documents/Evaluation_MCC-CNN_CODE/Code_Inference_Marc_2/libtorch
TORCHLIB_NVCC=-L$(TORCHLIBPath)/lib -Xlinker -rpath=$(TORCHLIBPath)/lib
TORCHLIB_CPP=-L$(TORCHLIBPath)/lib -Wl,-rpath=$(TORCHLIBPath)/lib
TORCHINCLUDE=-I$(TORCHLIBPath)/include -I$(TORCHLIBPath)/include/torch/csrc/api/include
#QTINCLUDE=-I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtWidgets -I/usr/include/x86_64-linux-gnu/qt5/QtGui -I/usr/include/x86_64-linux-gnu/qt5/QtCore -I/usr/include/x86_64-linux-gnu/qt5/QtConcurrent -I/usr/include/x86_64-linux-gnu/qt5/QtXml -I/usr/include/x86_64-linux-gnu/qt5/QtOpenGL
TORCHLIBS=-Wl,--no-as-needed -ltorch -ltorch_cuda_cpp -ltorch_cuda_cu -ltorch_cuda -lc10d_cuda_test -ltorch_cpu -lc10 -lc10_cuda $(TORCHLIBPath)/lib/libnvrtc-builtins-4730a239.so.11.3 -lnvToolsExt -lcudart
#===========================================================#
# libs to exclude: -lcaffe2_nvrtc
MMV2Exe=MMVII
MMV2_INSTALL_PATH=${abspath ${MMV2DirBin}}/

MMSymbDerHeader=$(wildcard ${MMV2DirIncl}/SymbDer/*.h)
MMKaptureHeader=$(wildcard ${MMV2DirSrc}/kapture/*.h)

all : ${MMV2DirBin}${MMV2Exe}

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
MMV2DirMappings=${MMV2DirSrc}Mappings/
SrcMappings=$(wildcard ${MMV2DirMappings}*.cpp)
ObjMappings=$(SrcMappings:.cpp=.o) 
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
#
MMV2DirDenseMatch=${MMV2DirSrc}DenseMatch/
SrcDenseMatch=$(wildcard ${MMV2DirDenseMatch}*.cpp)
ObjDenseMatch=$(SrcDenseMatch:.cpp=.o) 
#
#
MMV2DirCodedTarget=${MMV2DirSrc}CodedTarget/
SrcCodedTarget=$(wildcard ${MMV2DirCodedTarget}*.cpp)
ObjCodedTarget=$(SrcCodedTarget:.cpp=.o) 
#
#
MMV2DirLearnMatch=${MMV2DirSrc}LearningMatching/
SrcLearnMatch=$(wildcard ${MMV2DirLearnMatch}*.cpp)
ObjLearnMatch=$(SrcLearnMatch:.cpp=.o) 
#
#
MMV2DirBenchSNL=${MMV2DirSrc}TutoBenchTrianguRSNL/
SrcBenchSNL=$(wildcard ${MMV2DirBenchSNL}*.cpp)
ObjBenchSNL=$(SrcBenchSNL:.cpp=.o) 
#
MMV2DirBenchSEN=${MMV2DirSrc}Sensors/
SrcBenchSEN=$(wildcard ${MMV2DirBenchSEN}*.cpp)
ObjBenchSEN=$(SrcBenchSEN:.cpp=.o) 
#
#
MMV2DirSymbDerGen=${MMV2DirSrc}SymbDerGen/
SrcSymbDerGen=$(wildcard ${MMV2DirSymbDerGen}*.cpp)
ObjSymbDerGen=$(SrcSymbDerGen:.cpp=.o)

#
#
MMV2DirBundleAdjustment=${MMV2DirSrc}BundleAdjustment/
SrcBA=$(wildcard ${MMV2DirBundleAdjustment}*.cpp)
ObjBA=$(SrcBA:.cpp=.o)
#
#
MMV2DirCnvrtFormal=${MMV2DirSrc}ConvertFormat/
SrcCnvrtFormat=$(wildcard ${MMV2DirCnvrtFormal}*.cpp)
ObjCnvrtFormat=$(SrcCnvrtFormat:.cpp=.o)
#
#

MMV2DirExoBloc=${MMV2DirSrc}ExoBloc/
SrcExoBloc=$(wildcard ${MMV2DirExoBloc}*.cpp)
ObjExoBloc=$(SrcExoBloc:.cpp=.o)
#
#
MMV2DirInstrumental=${MMV2DirSrc}Instrumental/
SrcInstrumental=$(wildcard ${MMV2DirInstrumental}*.cpp)
ObjInstrumental=$(SrcInstrumental:.cpp=.o)
#
#

SRC_REGGEN=${MMV2DirGeneratedCodes}cName2CalcRegisterAll.cpp
MMV2DirGeneratedCodes=${MMV2DirSrc}GeneratedCodes/
SrcGeneratedCodes:=$(wildcard ${MMV2DirGeneratedCodes}*.cpp)
## Force ${REGEN} to be built, but assure only once in SrcGeneratedCodes :
SrcGeneratedCodes:=${filter-out ${SRC_REGGEN},${SrcGeneratedCodes}} ${SRC_REGGEN}
ObjGeneratedCodes=$(SrcGeneratedCodes:.cpp=.o)

${SRC_REGGEN}:
	mkdir -p ${MMV2DirGeneratedCodes}
	cp  ${MMV2DirIncl}/SymbDer/cName2CalcRegisterAll.cpp.tmpl $@
#
#
MMV2DirGeoms=${MMV2DirSrc}Geoms/
SrcGeoms=$(wildcard ${MMV2DirGeoms}*.cpp)
ObjGeoms=$(SrcGeoms:.cpp=.o) 
#
#
MMV2DirGeom2D=${MMV2DirSrc}Geom2D/
SrcGeom2D=$(wildcard ${MMV2DirGeom2D}*.cpp)
ObjGeom2D=$(SrcGeom2D:.cpp=.o) 
#
#
MMV2DirGeom3D=${MMV2DirSrc}Geom3D/
SrcGeom3D=$(wildcard ${MMV2DirGeom3D}*.cpp)
ObjGeom3D=$(SrcGeom3D:.cpp=.o) 
#
#
MMV2DirMesh=${MMV2DirSrc}Mesh/
SrcMesh=$(wildcard ${MMV2DirMesh}*.cpp)
ObjMesh=$(SrcMesh:.cpp=.o) 
#
#
MMV2DirMeshDispl=${MMV2DirSrc}MeshDisplacement/
SrcMeshDispl=$(wildcard ${MMV2DirMeshDispl}*.cpp)
ObjMeshDispl=$(SrcMeshDispl:.cpp=.o) 
#
#
MMV2DirOrientReport=${MMV2DirSrc}OrientReport/
SrcOrientReport=$(wildcard ${MMV2DirOrientReport}*.cpp)
ObjOrientReport=$(SrcOrientReport:.cpp=.o) 
#
#
MMV2DirPoseEstim=${MMV2DirSrc}PoseEstim/
SrcPoseEstim=$(wildcard ${MMV2DirPoseEstim}*.cpp)
ObjPoseEstim=$(SrcPoseEstim:.cpp=.o) 
#
#
MMV2DirRadiom=${MMV2DirSrc}Radiom/
SrcRadiom=$(wildcard ${MMV2DirRadiom}*.cpp)
ObjRadiom=$(SrcRadiom:.cpp=.o) 
#
#
MMV2DirTieP=${MMV2DirSrc}TieP/
SrcTieP=$(wildcard ${MMV2DirTieP}*.cpp)
ObjTieP=$(SrcTieP:.cpp=.o) 
#
#
MMV2DirTopo=${MMV2DirSrc}Topo/
SrcTopo=$(wildcard ${MMV2DirTopo}*.cpp)
ObjTopo=$(SrcTopo:.cpp=.o) 
#
MMV2DirSysCo=${MMV2DirSrc}SysCo/
SrcSysCo=$(wildcard ${MMV2DirSysCo}*.cpp)
ObjSysCo=$(SrcSysCo:.cpp=.o)
#
MMV2DirExports=${MMV2DirSrc}Exports/
SrcExports=$(wildcard ${MMV2DirExports}*.cpp)
ObjExports=$(SrcExports:.cpp=.o)
#
MMV2DirMorphoMat =${MMV2DirSrc}MorphoMat/
SrcMorphoMat=$(wildcard ${MMV2DirMorphoMat}*.cpp)
ObjMorphoMat=$(SrcMorphoMat:.cpp=.o)
#
MMV2DirKapture=${MMV2DirSrc}kapture/
#SrcKapture=$(fiter-out ${MMV2DirKapture}kpt_test.cpp $(wildcard ${MMV2DirKapture}*.cpp))
SrcKapture=$(wildcard ${MMV2DirKapture}*.cpp)
ObjKapture=$(SrcKapture:.cpp=.o)
#
#    => Le Main
MAIN=${MMV2DirSrc}main.cpp
#============ Calcul des objets
#
OBJ= ${ObjMatchTieP} ${ObjCalcDescriptPCar} ${ObjImagesBase}  ${ObjMMV1}  ${ObjUtiMaths} ${ObjImagesInfoExtract} ${ObjImagesFiltrLinear} ${ObjCmdSpec} ${ObjBench} ${ObjMatrix} ${ObjAppli} ${ObjDIB}   ${ObjTLE} ${ObjMkf} ${ObjUtils} ${ObjSerial}  ${ObjPerso}  ${ObjGraphs} ${ObjDenseMatch} ${ObjSymbDerGen} ${ObjGeneratedCodes} ${ObjGeoms} ${ObjGeom2D} ${ObjGeom3D}  ${ObjMappings} ${ObjLearnMatch} ${ObjKapture} ${ObjSysCo} ${ObjExports} ${ObjMorphoMat} ${ObjCodedTarget} ${ObjBenchSNL} ${ObjBenchSEN} ${ObjBA} ${ObjCnvrtFormat} ${ObjExoBloc} ${ObjInstrumental} ${ObjMesh} ${ObjMeshDispl} ${ObjOrientReport} ${ObjPoseEstim} ${ObjRadiom} ${ObjTieP} ${ObjTopo} ./SGM_CUDA/cudInfer.a
#
#=========  Header ========
#
#
HEADER=$(wildcard ${MMV2DirIncl}*.h)

#
#  Binaries
#== CFLAGS etc...
#
CXX=g++
CFlags= "-fopenmp" "-std=c++17" "-Wall"  "-Werror" "-O4" "-fPIC" -I${MMV2Dir} -I ${MMV2Dir}/ExternalInclude -I ${MMV2DirIncl} -I ${EIGEN_DIR} -I ${MMDir}/include -I${MMDir} ${TORCHINCLUDE} -D'MMVII_INSTALL_PATH="${MMV2_INSTALL_PATH}"'
#CFlags= "-fopenmp" "-std=c++17" "-Wall"  "-Werror" "-O4" "-march=native" "-fPIC" -I${MMV2Dir} -I${MMV2Dir}/ExternalInclude -I${MMDir}/include/ -I${MMDir} -D'MMVII_INSTALL_PATH="${MMV2_INSTALL_PATH}"'

BOOST_LIBS=
QTAnnLibs= -lXext /usr/lib/x86_64-linux-gnu/libQt5Core.so /usr/lib/x86_64-linux-gnu/libQt5Gui.so /usr/lib/x86_64-linux-gnu/libQt5Xml.so /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so -lGLU -lGL  -ldl -lpthread /usr/lib/x86_64-linux-gnu/libQt5Xml.so /usr/lib/x86_64-linux-gnu/libQt5Concurrent.so /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so /usr/lib/x86_64-linux-gnu/libQt5Widgets.so /usr/lib/x86_64-linux-gnu/libQt5Gui.so /usr/lib/x86_64-linux-gnu/libQt5Core.so ../../lib/libANN.a
## MacOS : may be -lstdc++fs should be replaced by -lc++experimental
LibsFlags= ${MMV2ElisePath} -lX11  ${BOOST_LIBS}  ${QTAnnLibs} ${TORCHLIBS}  -lstdc++fs -lproj
#
${MMV2DirBin}${MMV2Exe} :  ${OBJ} ${MAIN} ${MMV2ElisePath}
	${CXX}  ${MAIN} ${CFlags}  ${OBJ}  ${TORCHLIB_CPP} ${TORCHLIB_NVCC} ${LibsFlags}  -o ${MMV2DirBin}${MMV2Exe} 
	rm -f libP2007.a
	ar rvs libP2007.a  ${OBJ}
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
${MMV2DirMappings}%.o :  ${MMV2DirMappings}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirDIB}%.o :  ${MMV2DirDIB}%.cpp   ${HEADER} ${MMV2DirDIB}*.h
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirGraphs}%.o :  ${MMV2DirGraphs}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirDenseMatch}%.o :  ${MMV2DirDenseMatch}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirLearnMatch}%.o :  ${MMV2DirLearnMatch}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirBenchSNL}%.o :  ${MMV2DirBenchSNL}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirBenchSEN}%.o :  ${MMV2DirBenchSEN}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirSymbDerGen}%.o :  ${MMV2DirSymbDerGen}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirGeneratedCodes}%.o :  ${MMV2DirGeneratedCodes}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirGeoms}%.o :  ${MMV2DirGeoms}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirGeom2D}%.o :  ${MMV2DirGeom2D}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirGeom3D}%.o :  ${MMV2DirGeom3D}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirKapture}%.o :  ${MMV2DirKapture}%.cpp   ${MMKaptureHeader}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirSysCo}%.o :  ${MMV2DirSysCo}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -I /usr/include -o $@
${MMV2DirExports}%.o :  ${MMV2DirExports}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirMorphoMat}%.o :  ${MMV2DirMorphoMat}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirCodedTarget}%.o :  ${MMV2DirCodedTarget}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@

${MMV2DirBundleAdjustment}%.o :  ${MMV2DirBundleAdjustment}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirCnvrtFormal}%.o :  ${MMV2DirCnvrtFormal}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirExoBloc}%.o :  ${MMV2DirExoBloc}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirInstrumental}%.o :  ${MMV2DirInstrumental}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@

${MMV2DirMesh}%.o :  ${MMV2DirMesh}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirMeshDispl}%.o :  ${MMV2DirMeshDispl}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirOrientReport}%.o :  ${MMV2DirOrientReport}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirPoseEstim}%.o :  ${MMV2DirPoseEstim}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirRadiom}%.o :  ${MMV2DirRadiom}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirTieP}%.o :  ${MMV2DirTieP}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirTopo}%.o :  ${MMV2DirTopo}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
#       ===== TEST ========================================
#
Show:
	echo ${SrcCalcDescriptPCar}
	echo DU=${MMV2DirCalcDescriptPCar}
	echo ObjCalcDescriptPCar : ${ObjCalcDescriptPCar}
	echo SrcCalcDescriptPCar: ${SrcCalcDescriptPCar}
	echo MMV2DirCalcDescriptPCar: ${MMV2DirCalcDescriptPCar}

clean :
	rm -f ${OBJ}

distclean: clean
	-rm -f ${MMV2DirGeneratedCodes}*
#
#
