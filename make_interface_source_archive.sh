#!/bin/bash

MICMAC_REPOSITORY=`basename $PWD`
REV_NUMBER=`hg log -r tip --template "{rev}"`
OUT_MICMAC_DIR=micmac
ARCHIVE_NAME=micmac_interface_source_rev$REV_NUMBER
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
mkdir ${OUT_MICMAC_DIR}/src
cp -R interface $OUT_MICMAC_DIR
cp -R src/interface ${OUT_MICMAC_DIR}/src
rm -fR ${OUT_MICMAC_DIR}/interface/documentation

tar czf ${ARCHIVE_NAME}.tar.gz $OUT_MICMAC_DIR
rm -fr $OUT_MICMAC_DIR
