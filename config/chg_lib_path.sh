QT_DIR=/Users/admin/dev/Qt5.4.1/5.4/clang_64
FRAMEWORKS_FROM_BIN_PATH=../Frameworks
FRAMEWORKS_PATH=Frameworks
STARTING_DIRECTORY=$PWD
cd $1
MM_PATH=$PWD
cd $STARTING_DIRECTORY

#| grep ^$'\t' | cut -d' ' -f1 | tr -d $'\t')

function change_lib_path
{
	SRC_FILE=$1
	SRC_PATH=$2
	DST_PATH=$3
	#echo replace $SRC_PATH by $DST_PATH in $SRC_FILE
	for lib in $(otool -L $SRC_FILE)
	do
		if [[ "$lib" =~ "$SRC_PATH" ]]
		then
			lib_without_path=${lib#${SRC_PATH}}
			newlib=$DST_PATH$lib_without_path
			#echo "$lib -> $newlib"
			install_name_tool -change $lib $newlib $SRC_FILE
		fi
	done
}

function process_framework
{
	FRAMEWORK_NAME=$1
	#install_name_tool -id $FRAMEWORKS_PATH/$FRAMEWORK_NAME.framework/Versions/5/$FRAMEWORK_NAME $FRAMEWORKS_PATH/$FRAMEWORK_NAME.framework/$FRAMEWORK_NAME
	cd $MM_PATH/$FRAMEWORKS_PATH/$FRAMEWORK_NAME.framework/Versions/5/
	install_name_tool -id $FRAMEWORK_NAME $FRAMEWORK_NAME
	cd $MM_PATH
	change_lib_path $FRAMEWORKS_PATH/$FRAMEWORK_NAME.framework/Versions/5/$FRAMEWORK_NAME $QT_DIR/lib/ @executable_path/$FRAMEWORKS_FROM_BIN_PATH/
}

process_framework QtCore
process_framework QtGui
process_framework QtOpenGL
process_framework QtConcurrent
process_framework QtPrintSupport
process_framework QtWidgets
process_framework QtXml

change_lib_path bin/mm3d $QT_DIR/lib/ @executable_path/$FRAMEWORKS_FROM_BIN_PATH/
change_lib_path bin/SaisieQT $QT_DIR/lib/ @executable_path/$FRAMEWORKS_FROM_BIN_PATH/
change_lib_path $FRAMEWORKS_PATH/platforms/libqcocoa.dylib $QT_DIR/lib/ @executable_path/$FRAMEWORKS_FROM_BIN_PATH/

IMAGE_FORMATS_PATH=$FRAMEWORKS_PATH/imageformats
for f in $(ls $IMAGE_FORMATS_PATH)
do
	change_lib_path $IMAGE_FORMATS_PATH/$f $QT_DIR/lib/ @executable_path/$FRAMEWORKS_FROM_BIN_PATH/
done
