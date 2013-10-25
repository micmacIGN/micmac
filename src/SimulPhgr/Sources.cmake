set(SimulPhgr_Src_Files
	${SIMULPH_DIR}/cSimulCamera.cpp
)

SOURCE_GROUP(SimulPhgr FILES ${SimulPhgr_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${SimulPhgr_Src_Files}
)
