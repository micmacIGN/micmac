rm -rf build
mkdir build
cd build
cmake ..
nb_of_cpu=$(sysctl -n hw.ncpu)
make -j$nb_of_cpu
make install
cd ..

