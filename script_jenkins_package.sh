# compile micmac
rm -rf build
mkdir build
cd build
cmake ../
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
make -j$NBRP
make install

# créer un répertoire pour l'archive
BIN_DIR=MicMac
rm -fr $BIN_DIR
mkdir $BIN_DIR

# copie les XML nécessaires à l'execution
cp -r ../bin $BIN_DIR
mkdir $BIN_DIR/include
cp -r ../include/XML_GEN $BIN_DIR/include
cp -r ../include/XML_MicMac $BIN_DIR/include

# copie les outils tiers
mkdir $BIN_DIR/binaire-aux
cp ../binaire-aux/siftpp_tgi.LINUX $BIN_DIR/binaire-aux
cp ../binaire-aux/ann_mec_filtre.LINUX $BIN_DIR/binaire-aux

# créer l'archive
ARCH=`uname`_`uname -p`
tar -czf bin_$ARCH.tar.gz $BIN_DIR
