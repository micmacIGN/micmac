rm -rf build
mkdir build
cd build
cmake ..
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
make -j$NBRP
make install
cd ..
chmod +x ./binaire-aux/siftpp_tgi.LINUX
chmod +x ./binaire-aux/ann_mec_filtre.LINUX

