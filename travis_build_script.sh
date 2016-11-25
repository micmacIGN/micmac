# travis building script (MacOS and Linux)
mkdir build
cd build
cmake .. -DWITH_QT5=On
make 
make install
cd ..
pwd
ls
