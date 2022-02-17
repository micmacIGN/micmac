nomArchive=micmac_$1
echo 'creation de l archive binaire : '$nomArchive
mkdir $nomArchive
cp -R ../bin $nomArchive
cp -R ../binaire-aux $nomArchive
cp -R ../data $nomArchive
mkdir $nomArchive/include
mkdir $nomArchive/data
cp -R ../include/XML_MicMac $nomArchive/include
cp -R ../include/XML_GEN $nomArchive/include
cp -R ../include/qt $nomArchive/include
cp -R ../data $nomArchive/data
tar -czf $nomArchive.tgz $nomArchive
