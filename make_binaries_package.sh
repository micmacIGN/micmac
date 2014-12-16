# compile micmac
rm -rf build
mkdir build
cd build

if [ "$1" == "serveur" ]
then
	SERVER_PARAMS="-DWITH_KAKADU=1 -DKAKADU_DIR=$PWD/../../kakadu -DWITH_HEADER_PRECOMP=0 -DNO_X11=ON"
fi

cmake .. $SERVER_PARAMS

OS=$(uname -s)

REV_NUMBER=$(hg log -r tip --template "{rev}")
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
echo "number of cores : $NBRP"
make install -j$NBRP

# créer un répertoire pour l'archive
BIN_DIR=micmac
rm -fr $BIN_DIR
mkdir $BIN_DIR

if [ "$1" != "serveur" ]
then
	# la création d'archive sur le serveur est gérée par maven

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
fi
