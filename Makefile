#       DocUtil/DUGen.tex\
#
FTEX=   DocMicMac.tex\
	Generalites/Intro.tex\
	Generalites/SomReal.tex\
	Generalites/QuickStart.tex\
	Generalites/QuickStart_Apero.tex\
	Generalites/QuickStartOthers.tex\
	Generalites/QuickStartSimplified-Tools.tex\
	Generalites/QuickStartSimplified-Tools-2.tex\
	Generalites/ParamSpec.tex\
	Generalites/FullAutomaticTools.tex\
	Generalites/NameFileSpec.tex\
	Generalites/InteractiveTool.tex\
	Generalites/UseCase2D.tex\
	DocRef/ConvTool.tex\
	DocRef/GeoLocalisation.tex\
	DocRef/Advanced-TieP.tex\
	DocRef/Advanced-Apero1.tex\
	DocRef/Advanced-MicMac1.tex\
	DocRef/Advanced-MicMac2.tex\
	DocAlgo/DAMultiRes.tex\
	DocAlgo/DAGeneralites.tex\
	DocAlgo/DACorrel.tex\
	DocAlgo/DAEnergetik.tex\
	DocAlgo/DAOrient.tex\
        DocUtil/DUBins.tex\
        DocUtil/DUMec.tex\
        DocUtil/DUMecaGen.tex\
        DocUtil/DUAutreSec.tex\
        Annexes/Formats.tex
DocMicMac.pdf : $(FTEX) 
	pdflatex DocMicMac
	pdflatex DocMicMac
	pdflatex DocMicMac
DocMicMac.dvi : $(FTEX)
	latex DocMicMac
