#/bin/sh

MACOS_10_9_LIB=/opt/X11/lib/
MACOS_inf_10_9_LIB=/opt/local/lib/

version_string=$(sw_vers | grep ProductVersion | cut -f2)
version=( ${version_string//./ } )

#if [ ${version[1]} -ge 9 ]
#then
	SRC_LIB=$MACOS_inf_10_9_LIB
	DST_LIB=$MACOS_10_9_LIB
#else
#	SRC_LIB=$MACOS_10_9_LIB
#	DST_LIB=$MACOS_inf_10_9_LIB
#fi

#loop on file in bin
for f in bin/mm3d
do
	echo "->" $f
	#loop on need libraries
	for lib in $(otool -L $f | grep ^$'\t' | cut -d' ' -f1 | tr -d $'\t')
	do
		old_path=${lib%/*}
		echo $old_path
		if [ $old_path = $SRC_LIB ]
		then
			echo "old path found in [" $lib "]"
		fi
		
		echo $lib" -> "$DST_LIB${lib##*/}
		#sed 's|$SRC_LIB|$DST_LIB|g' $lib
		#if [ ! -f $lib ]
		#then
		#	if []
		#fi
	done
	#if [[ $i =~ 200[78] ]] ; then
	#echo "OK"
	#	else
	#echo "not OK"
	#fi
done
