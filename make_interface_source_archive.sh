#!/bin/sh

MICMAC_REPOSITORY=`basename $PWD`
REV_NUMBER=`hg log -r tip --template "{rev}"`
TEMP_DIRECTORY=micmac
ARCHIVE_NAME=micmac_interface_source_rev$REV_NUMBER
if [ -f $ARCHIVE_NAME.tar.gz ]
then
	echo l\'archive $ARCHIVE_NAME.tar.gz existe déjà
	exit 1
fi
if [ -d $TEMP_DIRECTORY ]
then
	echo le repertoire $TEMP_DIRECTORY existe déjà
	exit 1
fi
mkdir $TEMP_DIRECTORY
mkdir $TEMP_DIRECTORY/src
cp -R interface $TEMP_DIRECTORY
cp -R src/interface $TEMP_DIRECTORY/src

tar czf $ARCHIVE_NAME.tar.gz $TEMP_DIRECTORY
rm -fr $TEMP_DIRECTORY
