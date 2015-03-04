set(SateLib_Src_Files
	${SATELIB_DIR}/CPP_RPC.cpp
	${SATELIB_DIR}/Dimap2Grid.cpp
	${SATELIB_DIR}/RefineModel.cpp
)

# JE NE SAIS PAS SI CA SERT ???
SOURCE_GROUP(SateLib FILES ${SateLib_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${SateLib_Src_Files}
)
