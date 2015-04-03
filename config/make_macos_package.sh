QT_DIR=/Users/admin/dev/Qt5.4.1/5.4/clang_64
FRAMEWORKS_FROM_BIN_PATH=../Frameworks
MM_PATH=..
PACKAGE_PATH=$1
FRAMEWORKS_PATH=$PACKAGE_PATH/Frameworks

if [[ -z "$PACKAGE_PATH"  ]]; then
	echo necessite au moins un argument : le nom de dossier temporaire pour la creation de l\'archive
	exit 1
fi

rm -rf $PACKAGE_PATH
mkdir $PACKAGE_PATH

echo "copy Qt frameworks"
mkdir $FRAMEWORKS_PATH
cp -R $QT_DIR/lib/QtCore.framework $FRAMEWORKS_PATH/
cp -R $QT_DIR/lib/QtGui.framework $FRAMEWORKS_PATH/
cp -R $QT_DIR/lib/QtWidgets.framework $FRAMEWORKS_PATH/
cp -R $QT_DIR/lib/QtXml.framework $FRAMEWORKS_PATH/
cp -R $QT_DIR/lib/QtConcurrent.framework $FRAMEWORKS_PATH/
cp -R $QT_DIR/lib/QtOpenGL.framework $FRAMEWORKS_PATH/
cp -R $QT_DIR/lib/QtPrintSupport.framework $FRAMEWORKS_PATH/

echo "copy Qt plugins"
mkdir $FRAMEWORKS_PATH/platforms/
cp $QT_DIR/plugins/platforms/libqcocoa.dylib $FRAMEWORKS_PATH/platforms/
cp -R $QT_DIR/plugins/imageformats $FRAMEWORKS_PATH/

echo "remove Qt debug libraries"
rm $(find $FRAMEWORKS_PATH/ | grep debug)

echo "remove Qt headers"
rm -fr $(find  $FRAMEWORKS_PATH -maxdepth 2 | grep Headers)
rm -fr $(find $FRAMEWORKS_PATH -maxdepth 4 | grep Headers)

echo "copy binaries"
cp -R $MM_PATH/bin $PACKAGE_PATH
mkdir $PACKAGE_PATH/binaire-aux
cp $MM_PATH/binaire-aux/*.OSX $PACKAGE_PATH/binaire-aux/
cp $MM_PATH/binaire-aux/PoissonRecon $PACKAGE_PATH/binaire-aux/
cp $MM_PATH/binaire-aux/SurfaceTrimmer $PACKAGE_PATH/binaire-aux/

echo "copy XML parameters file"
mkdir $PACKAGE_PATH/include
cp -R $MM_PATH/include/XML_GEN $PACKAGE_PATH/include
rm $PACKAGE_PATH/include/XML_GEN/*.h
cp -R $MM_PATH/include/XML_MicMac $PACKAGE_PATH/include

echo "copy fdsc"
cp -R $MM_PATH/fdsc $PACKAGE_PATH/

echo "copy miscellaneous file"
mkdir -p $PACKAGE_PATH/data/Tabul
cp $MM_PATH/data/Tabul/erod_8 $PACKAGE_PATH/data/Tabul
mkdir -p $PACKAGE_PATH/include/qt/translations
cp $MM_PATH/include/qt/style.qss $PACKAGE_PATH/include/qt
cp $MM_PATH/include/qt/translations/*.qm $PACKAGE_PATH/include/qt/translations
cp $MM_PATH/README $PACKAGE_PATH/
cp $MM_PATH/LISEZMOI $PACKAGE_PATH/

echo "set mm3d and SaisieQT librairies to relative path"
./chg_lib_path.sh $PACKAGE_PATH

TARBALL_NAME=micmac_macosx_x86_64_rev$(hg identify --num -r .).tar.gz
echo "create tarball ["$TARBALL_NAME"]"
tar czf $TARBALL_NAME $PACKAGE_PATH
