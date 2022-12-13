FTEX =   Doc2007

.PHONY: clean a4 a5

a4: $(FTEX)_a4.pdf
a5: $(FTEX)_a5.pdf
all: a4 a5

SRCS:=$(FTEX).tex ${wildcard */*.tex}

$(FTEX)_a4.pdf : $(FTEX)_a4.tex ${SRCS}
	pdflatex $<
	makeindex $(<:.tex=.idx)
	pdflatex $<

$(FTEX)_a5.pdf: $(FTEX)_a5.tex ${SRCS}
	pdflatex $<
	makeindex $(<:.tex=.idx)
	pdflatex $<

clean :
	rm -f *.aux */*.aux *.toc *.dvi *.log *.out *.idx *.ind *.ilg $(FTEX)_a4.pdf $(FTEX)_a5.pdf
