MICMAC_BASE_DIR=micmac-1.0

if [ ! -f bin/mm3d ]
then
	echo "no bin/mm3d found"
	exit 1
fi

#~ ARCH=$(bin/mm3d CheckDependencies version)
#~ REV=$(bin/mm3d CheckDependencies rev)
REV=$(hg identify --num)
if [[ "$REV" == *"+" ]]
then
	echo "repository has uncommitted changes: removing +"
	REV="${REV%?}"
fi
TARBALL=../micmac-1.0.$REV.tar.gz
echo "removing [$TARBALL]"
rm -f $TARBALL
echo "removing [$MICMAC_BASE_DIR]"
rm -fr $MICMAC_BASE_DIR

mkdir $MICMAC_BASE_DIR
mkdir $MICMAC_BASE_DIR/bin
cp bin/* $MICMAC_BASE_DIR/bin

mkdir $MICMAC_BASE_DIR/binaire-aux
cp -r binaire-aux/linux $MICMAC_BASE_DIR/binaire-aux/linux

mkdir $MICMAC_BASE_DIR/Documentation
cp Documentation/DocMicMac.pdf $MICMAC_BASE_DIR/Documentation

mkdir $MICMAC_BASE_DIR/InterfaceCEREMA
cp InterfaceCEREMA/AperoDeDenis.py $MICMAC_BASE_DIR/InterfaceCEREMA
cp InterfaceCEREMA/logoCerema.jpg $MICMAC_BASE_DIR/InterfaceCEREMA
cp InterfaceCEREMA/logoIGN.jpg $MICMAC_BASE_DIR/InterfaceCEREMA
cp InterfaceCEREMA/Notice\ Installation\ interface\ graphique\ MicMac.pdf $MICMAC_BASE_DIR/InterfaceCEREMA

mkdir $MICMAC_BASE_DIR/CodeExterne
cp -r CodeExterne/ANN $MICMAC_BASE_DIR/CodeExterne
cp -r CodeExterne/Poisson $MICMAC_BASE_DIR/CodeExterne
cp -r CodeExterne/rnx2rtkp $MICMAC_BASE_DIR/CodeExterne

cp -r include $MICMAC_BASE_DIR
rm -fr $MICMAC_BASE_DIR/include/StdAfx.h.gch
echo $REV >$MICMAC_BASE_DIR/include/rev

cp -r CodeGenere $MICMAC_BASE_DIR
cp -r data $MICMAC_BASE_DIR
cp -r ExtDataPrep $MICMAC_BASE_DIR
cp -r fdsc $MICMAC_BASE_DIR
cp -r src $MICMAC_BASE_DIR

cp CMakeLists.txt COPYING LISEZMOI precompiled_headers.cmake README $MICMAC_BASE_DIR/

tar czf $TARBALL $MICMAC_BASE_DIR
rm -fr $MICMAC_BASE_DIR
