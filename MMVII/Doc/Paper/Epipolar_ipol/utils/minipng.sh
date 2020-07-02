#! /bin/sh
#
# PNG file optimizer
#
# Reduce the file of a PNG file without modifying any pixel value:
# - remove metadata (comment etc.) with pngcrush
# - optimize filters with optipng
# - optimize compression with advdef
#
# TODO: check for tools beforehand
#       usage help
#       less redundancy
#       use pngzop instead of advdef

set -e

do_remove_chunks() {
  IN=$1
  TMP=kk1.png
  pngcrush -q -f 0 -l 0 -m 1 \
      -rem iTXt -rem tEXt -rem zTXt \
      -rem bKGD -rem hIST -rem pHYs \
      -rem sPLT -rem sTER -rem tIME -rem tRNS \
      $IN $TMP
  touch -r $IN $TMP
  mv $TMP $IN
}

do_optimize_filters() {
  IN=$1
  TMP=kk2.png
  cp $IN $TMP
  optipng -quiet -o1 -f0-5 $TMP
  touch -r $IN $TMP
  mv $TMP $IN
}

do_optimize_deflate() {
  IN=$1
  TMP=kk3.png
  cp $IN $TMP
  advdef -z -3 $TMP
  touch -r $IN $TMP
  mv $TMP $IN
}

get_size() {
  du -b $1 | cut -f1
}

# do stuff

for OLD in "$@"; do
    NEW=kk.png
    cp $OLD $NEW
    do_remove_chunks $NEW
    do_optimize_filters $NEW
    do_optimize_deflate $NEW > /dev/null
    touch -r $OLD $NEW
    mv $NEW $OLD
done
