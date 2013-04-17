#!/bin/sh

BIN_DIR=$1
CHANT_DIR=$2

"$BIN_DIR/Tapioca" MulScale "$CHANT_DIR/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
"$BIN_DIR/Apero" "$CHANT_DIR/Apero-5.xml"
"$BIN_DIR/MICMAC" "$CHANT_DIR/Param-6-Ter.xml"
