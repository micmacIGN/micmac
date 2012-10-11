rm -rf bin
mkdir build
cd build
cmake ..
make clean
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
echo "Nbre de coeurs : " $NBRP
make install -j$NBRP
cd ..
chmod +x ./binaire-aux/siftpp_tgi.LINUX
chmod +x ./binaire-aux/ann_mec_filtre.LINUX

