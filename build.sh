rm -rf bin
mkdir build
cd build
cmake ..
make install
cd ..
chmod +x ./binaire-aux/siftpp_tgi.LINUX
chmod +x ./binaire-aux/ann_mec_filtre.LINUX
