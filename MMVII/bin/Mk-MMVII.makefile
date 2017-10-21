#
#== Directories
#
MMDir=/home/ubuntu/Desktop/MMM/micmac/
MMV2Dir=${MMDir}MMVII/
MMV2DirSrc=${MMV2Dir}src/
MMV2DirBin=${MMV2Dir}bin/
MMV2Objects=${MMV2Dir}object/
MMV2DirIncl=${MMV2Dir}include/
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
#    => Le Main
#
MAIN=${MMV2DirSrc}main.cpp
#============ Calcul des objets
#
OBJ= ${ObjTLE} ${ObjMkf} ${ObjBench} ${ObjAppli} ${ObjUtils}
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
CFlags="-std=c++11" -I${MMV2Dir}
LibsFlags= ${MMDir}/lib/libelise.a -lX11 
MMV2Exe=MMVII
#
${MMV2DirBin}${MMV2Exe} :  ${OBJ} ${MAIN}
	${CXX}  ${MAIN} ${CFlags}  ${OBJ}  ${LibsFlags}  -o ${MMV2DirBin}${MMV2Exe}
#
#
# Objects
#
#
${MMV2DirTLE}%.o :  ${MMV2DirTLE}%.cpp   ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirT4MkF}%.o :  ${MMV2DirT4MkF}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirBench}%.o :  ${MMV2DirBench}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirAppli}%.o :  ${MMV2DirAppli}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
${MMV2DirUtils}%.o :  ${MMV2DirUtils}%.cpp ${HEADER}
	${CXX} -c  $< ${CFlags} -o $@
#
#       ===== TEST ========================================
#
all : ${MMV2DirBin}${MMV2Exe} ${OBJ}
	${CXX}  ${MAIN} ${CFlags}  ${OBJ}  ${LibsFlags}  -o ${MMV2DirBin}${MMV2Exe}
Show:
	echo ${SrcUtils}
	echo DU=${MMV2DirUtils}
#
