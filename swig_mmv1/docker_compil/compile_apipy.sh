#! /bin/bash
cd /mm/
mkdir -p build_16.04/
cd build_16.04/
cmake .. -DWITH_APIPYTHON=1
NBRP=$(cat /proc/cpuinfo | grep processor | wc -l)
make -j$NBRP
make apipyclean
make apipy

