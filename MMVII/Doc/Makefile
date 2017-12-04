#       DocUtil/DUGen.tex\
#
FTEX=   Doc2007.tex
DocMicMac.pdf : $(FTEX) 
	pdflatex Doc2007.tex
	makeindex Doc2007.idx
	pdflatex Doc2007.tex
clean :
	find . -iname "*.aux" -exec rm {} +
	rm -f DocMicMac.pdf DocMicMac.dvi
