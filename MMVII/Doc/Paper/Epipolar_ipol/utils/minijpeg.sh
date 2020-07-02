#! /bin/sh
#
# JPEG file optimizer
#
# Reduce the file of a JPEG file without modifying any pixel value by:
# - remove metadata (EXIF etc.) with jpegtran
# - optimize Huffman coding with jpegtran
#
# TODO: check for tools beforehand
#       usage help
#       less redundancy

set -e

do_remove_markers() {
  IN=$1
  TMP=$(tempfile)
  jpegtran -copy none $IN > $TMP
  touch -r $IN $TMP
  mv $TMP $IN
}

do_optimize_compression() {
  IN=$1
  TMP=$(tempfile)
  jpegtran -optimize $IN > $TMP
  touch -r $IN $TMP
  mv $TMP $IN
  jpegtran -optimize -progressive $IN > $TMP
  touch -r $IN $TMP
  if [ $(get_size $TMP) -lt $(get_size $IN) ]; then
      mv $TMP $IN
  fi;
}

get_size() {
  du -b $1 | cut -f1
}

# do stuff

for OLD in "$@"; do
    NEW=$(tempfile)
    cp $OLD $NEW
    do_remove_markers $NEW
    do_optimize_compression $NEW
    touch -r $OLD $NEW
    if [ $(get_size $NEW) -lt $(get_size $OLD) ]; then
	mv $NEW $OLD
    else
	rm $NEW
    fi;
done
