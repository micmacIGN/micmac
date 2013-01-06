MICMAC_REPOSITORY=`basename $PWD`
REV_NUMBER=`hg log -r tip --template "{rev}"`
TEMP_DIRECTORY=micmac_sources_package
ARCHIVE_NAME=micmac_source_rev$REV_NUMBER
if [ -f $ARCHIVE_NAME.tar.gz ]
then
	echo l\'archive $ARCHIVE_NAME.tar.gz existe déjà
	exit -1
fi
cd ../
if [ -d $TEMP_DIRECTORY ]
then
	echo le repertoire $TEMP_DIRECTORY existe déjà
	exit -1
fi
cp -r $MICMAC_REPOSITORY $TEMP_DIRECTORY
cd $TEMP_DIRECTORY
rm -fr ./.hg
rm -fr ./interface
rm -fr ./bin
rm -fr ./build*
rm -fr ./applis
rm -fr ./include/StdAfx.h.gch
rm -fr ./Documentation
rm -fr ./BenchElise
rm -fr ./binaire-aux
rm -fr ./data/Tabul/.svn
tar czf ../$MICMAC_REPOSITORY/$ARCHIVE_NAME.tar.gz ./
cd ..
rm -fr $TEMP_DIRECTORY
