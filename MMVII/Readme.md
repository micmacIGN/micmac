Dependences :
-------------

libboost-all-dev
#  libceres-dev => no longer needed 4 now


To compile :
------------

Compile MicMac V1, then in MMVII directory :

cd bin/
make -f Mk-MMVII.makefile
or 
make ....

First time :
-----------
make
./MMVII  GenCodeSymDer
make

To generate html doc :
----------------------
doxygen Doxyfile 


To generate pdf doc :
---------------------

cd Doc/
pdflatex  Doc2007.tex
pdflatex  Doc2007.tex


