#voir https://github.com/jeromyanglim/rmarkdown-rmeetup-2012/tree/master/talk

pdf:
	pandoc presentation.md  --slide-level 3 -t beamer -o presentation.tex
	pdflatex beamer.tex
	pdflatex beamer.tex

clean:
	rm -f *.aux *.log *.out *.toc *.snm *.nav
