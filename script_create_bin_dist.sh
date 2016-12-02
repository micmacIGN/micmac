nomArchive=micmac_$1
echo 'creation de l archive binaire : '$nomArchive
mkdir $nomArchive
cp -R ../bin $nomArchive
cp -R ../binaire-aux $nomArchive
mkdir $nomArchive/include
cp -R ../include/XML_MicMac $nomArchive/include
cp -R ../include/XML_GEN $nomArchive/include
tar -czf $nomArchive.tgz $nomArchive
