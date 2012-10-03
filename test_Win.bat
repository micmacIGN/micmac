mkdir data
cd data
svn co http://www.micmac.ign.fr/svn/micmac_data/trunk/ExempleDoc/Boudha/
cd ../bin
Tapioca MulScale ../data/Boudha/IMG_[0-9]{4}.tif 300 -1 ExpTxt=1
Apero  ../data/Boudha/Apero-5.xml
MICMAC  ../data/Boudha/Param-6-Ter.xml

