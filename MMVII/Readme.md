To install :
------------

Compile MicMac V1 without Qt support

cp micmac/lib/libelise.a micmac/lib/libelise_SsQt.a

cd bin/
make -f Mk-MMVII.makefile


To generate html doc :
----------------------
doxygen Doxyfile 


To generate pdf doc :
---------------------

cd Doc/
pdflatex  Doc2007.tex
pdflatex  Doc2007.tex


