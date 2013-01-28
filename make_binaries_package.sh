# compile micmac
rm -rf build
mkdir build
cd build
cmake ../
REV_NUMBER=$(hg log -r tip --template "{rev}")
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
make -j$NBRP
make install

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
cp README.fr $BIN_DIR
cp README.en $BIN_DIR

# copie les outils tiers
OS=$(uname -s)
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
