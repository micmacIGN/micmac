set(Src_TD_PPMD
	${TDPPMD_DIR}/cTD_Camera.cpp
	${TDPPMD_DIR}/cTD_SetAppuis.cpp
	${TDPPMD_DIR}/TD_Exemple.cpp
	${TDPPMD_DIR}/cTD_Im.cpp
)

#SOURCE_GROUP(Util FILES ${Util_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Src_TD_PPMD}
)
