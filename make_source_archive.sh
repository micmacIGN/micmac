#!/bin/sh

MICMAC_REPOSITORY=`basename $PWD`
REV_NUMBER=`hg log -r tip --template "{rev}"`
TEMP_DIRECTORY=micmac
ARCHIVE_NAME=micmac_source_rev$REV_NUMBER
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
cp -R CodeGenere $TEMP_DIRECTORY
cp -R CodeExterne $TEMP_DIRECTORY
cp -R data $TEMP_DIRECTORY
cp -R include $TEMP_DIRECTORY
cp -R src $TEMP_DIRECTORY
cp CMakeLists.txt $TEMP_DIRECTORY
cp Makefile-XML2CPP $TEMP_DIRECTORY
cp precompiled_headers.cmake $TEMP_DIRECTORY

rm -fr $TEMP_DIRECTORY/include/StdAfx.h.gch
rm -fr $TEMP_DIRECTORY/data/Tabul/.svn
rm -fr $TEMP_DIRECTORY/src/interface

tar czf $ARCHIVE_NAME.tar.gz $TEMP_DIRECTORY
rm -fr $TEMP_DIRECTORY
