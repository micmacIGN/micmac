rm -rf bin
mkdir build
cd build
cmake ..
make install
cd ..
mkdir data
cd data
svn co http://www.micmac.ign.fr/svn/micmac_data/trunk/ExempleDoc/Boudha/
cd ..
chmod +x ./binaire-aux/siftpp_tgi.LINUX
chmod +x ./binaire-aux/ann_mec_filtre.LINUX
