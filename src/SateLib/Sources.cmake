set(SateLib_Src_Files
	${SATELIB_DIR}/CPP_RPC.cpp
	${SATELIB_DIR}/CPP_AsterDestrip.cpp
	${SATELIB_DIR}/Dimap2Grid.cpp
	${SATELIB_DIR}/RefineModel.cpp
	${SATELIB_DIR}/RefineASTER.cpp
	${SATELIB_DIR}/CPP_ApplyParralaxCor.cpp
	${SATELIB_DIR}/DigitalGlobe2Grid.cpp
	${SATELIB_DIR}/Aster2Grid.cpp
	${SATELIB_DIR}/CPP_SATtoBundle.cpp
	${SATELIB_DIR}/CPP_SATDef2D.cpp
	${SATELIB_DIR}/ASTERGT2MM.cpp
	${SATELIB_DIR}/ASTER_PostProc.cpp
)

# JE NE SAIS PAS SI CA SERT ???
source_group(SateLib FILES ${SateLib_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${SateLib_Src_Files}
)
