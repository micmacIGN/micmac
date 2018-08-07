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
	
	${SATELIB_DIR}/ALGLIB/alglibinternal.cpp
	${SATELIB_DIR}/ALGLIB/alglibinternal.h
	${SATELIB_DIR}/ALGLIB/alglibmisc.cpp
	${SATELIB_DIR}/ALGLIB/alglibmisc.h
	${SATELIB_DIR}/ALGLIB/ap.cpp
	${SATELIB_DIR}/ALGLIB/ap.h
	${SATELIB_DIR}/ALGLIB/dataanalysis.cpp
	${SATELIB_DIR}/ALGLIB/dataanalysis.h
	${SATELIB_DIR}/ALGLIB/diffequations.cpp
	${SATELIB_DIR}/ALGLIB/diffequations.h
	${SATELIB_DIR}/ALGLIB/fasttransforms.cpp
	${SATELIB_DIR}/ALGLIB/fasttransforms.h
	${SATELIB_DIR}/ALGLIB/integration.cpp
	${SATELIB_DIR}/ALGLIB/integration.h
	${SATELIB_DIR}/ALGLIB/interpolation.cpp
	${SATELIB_DIR}/ALGLIB/interpolation.h
	${SATELIB_DIR}/ALGLIB/linalg.cpp
	${SATELIB_DIR}/ALGLIB/linalg.h
	${SATELIB_DIR}/ALGLIB/optimization.cpp
	${SATELIB_DIR}/ALGLIB/optimization.h
	${SATELIB_DIR}/ALGLIB/solvers.cpp
	${SATELIB_DIR}/ALGLIB/solvers.h
	${SATELIB_DIR}/ALGLIB/specialfunctions.cpp
	${SATELIB_DIR}/ALGLIB/specialfunctions.h
	${SATELIB_DIR}/ALGLIB/statistics.cpp
	${SATELIB_DIR}/ALGLIB/statistics.h
	${SATELIB_DIR}/ALGLIB/stdafx.h
)

# JE NE SAIS PAS SI CA SERT ???
source_group(SateLib FILES ${SateLib_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${SateLib_Src_Files}
)
