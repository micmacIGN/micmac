# compile micmac
rm -rf build
mkdir build
cd build

#if [ "$1" = "serveur" ]
#then
#	SERVER_PARAMS="-DKAKADU_NO_FIND=1 -DKAKADU_DIR=/usr/kakadu-6.0 -DWITH_HEADER_PRECOMP=0"
#fi

#cmake ../ $SERVER_PARAMS
cmake .. -DWITH_KAKADU=1 -DKAKADU_DIR=$PWD/../../kakadu

OS=$(uname -s)

REV_NUMBER=$(hg log -r tip --template "{rev}")
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
make install -j$NBRP


# créer un répertoire pour l'archive
BIN_DIR=micmac
rm -fr $BIN_DIR
mkdir $BIN_DIR

# copie les XML nécessaires à l'execution
cp -r ../bin $BIN_DIR
cp -r ../data $BIN_DIR
mkdir $BIN_DIR/include
cp -r ../include/XML_GEN $BIN_DIR/include
cp -r ../include/XML_MicMac $BIN_DIR/include
cp ../README $BIN_DIR
cp ../LISEZMOI $BIN_DIR

# copie les outils tiers
if [ $OS = "Linux" ]
then
	mkdir $BIN_DIR/binaire-aux
	cp ../binaire-aux/siftpp_tgi.LINUX $BIN_DIR/binaire-aux
	cp ../binaire-aux/ann_mec_filtre.LINUX $BIN_DIR/binaire-aux
else
	mkdir $BIN_DIR/binaire-aux
	cp ../binaire-aux/ann_samplekey200filtre.OSX $BIN_DIR/binaire-aux
	cp ../binaire-aux/siftpp_tgi.OSX $BIN_DIR/binaire-aux
fi

# créer l'archive
ARCH=${OS}_$(uname -p)
tar -czf bin_${ARCH}_rev${REV_NUMBER}.tar.gz $BIN_DIR
