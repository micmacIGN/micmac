#!/bin/bash

MICMAC_REPOSITORY=`basename $PWD`
#REV_NUMBER=`hg log -r tip --template "{rev}"`
REV_NUMBER=`hg identify --num`
OUT_MICMAC_DIR=micmac
ARCHIVE_NAME=micmac_source_rev$REV_NUMBER
if [ -f $ARCHIVE_NAME.tar.gz ]
then
	echo l\'archive $ARCHIVE_NAME.tar.gz existe déjà
	exit 1
fi
if [ -d $OUT_MICMAC_DIR ]
then
	echo le repertoire $OUT_MICMAC_DIR existe déjà
	exit 1
fi
mkdir $OUT_MICMAC_DIR
cp -R CodeGenere $OUT_MICMAC_DIR
cp -R CodeExterne $OUT_MICMAC_DIR
cp -R data $OUT_MICMAC_DIR
cp -R include $OUT_MICMAC_DIR
cp -R src $OUT_MICMAC_DIR
cp CMakeLists.txt $OUT_MICMAC_DIR
cp Makefile-XML2CPP $OUT_MICMAC_DIR
cp precompiled_headers.cmake $OUT_MICMAC_DIR
cp README $OUT_MICMAC_DIR
cp LISEZMOI $OUT_MICMAC_DIR

rm -fr $OUT_MICMAC_DIR/include/StdAfx.h.gch
rm -fr $OUT_MICMAC_DIR/data/Tabul/.svn
rm -fr $OUT_MICMAC_DIR/src/interface

mkdir $OUT_MICMAC_DIR/binaire-aux
cp -r binaire-aux/BIN_AUX_WIN32 $OUT_MICMAC_DIR/binaire-aux
cp binaire-aux/ann_mec_filtre.LINUX $OUT_MICMAC_DIR/binaire-aux
cp binaire-aux/siftpp_tgi.LINUX $OUT_MICMAC_DIR/binaire-aux
cp binaire-aux/ann_samplekey200filtre.OSX $OUT_MICMAC_DIR/binaire-aux
cp binaire-aux/siftpp_tgi.OSX $OUT_MICMAC_DIR/binaire-aux
cp binaire-aux/siftpp_tgi.exe $OUT_MICMAC_DIR/binaire-aux
cp binaire-aux/ann_samplekeyfiltre.exe $OUT_MICMAC_DIR/binaire-aux

tar czf $ARCHIVE_NAME.tar.gz $OUT_MICMAC_DIR
rm -fr $OUT_MICMAC_DIR
