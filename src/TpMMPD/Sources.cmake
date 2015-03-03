set(Src_TD_PPMD
	${TDPPMD_DIR}/cTD_Camera.cpp
	${TDPPMD_DIR}/cTD_SetAppuis.cpp
	${TDPPMD_DIR}/TD_Exemple.cpp
	${TDPPMD_DIR}/cTD_Im.cpp
	${TDPPMD_DIR}/ExoMM_CorrelMulImage.cpp
	${TDPPMD_DIR}/Match1.cpp
	${TDPPMD_DIR}/ExoMM_SimulTieP.cpp
	${TDPPMD_DIR}/ServiceGeoSud.cpp
	${TDPPMD_DIR}/Surf.cpp
	${TDPPMD_DIR}/cTD_Epip.cpp
	${TDPPMD_DIR}/ReprojImg.cpp
	${TDPPMD_DIR}/cPseudoIntersect.cpp
	${TDPPMD_DIR}/Export2Ply.cpp
)

#SOURCE_GROUP(Util FILES ${Util_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Src_TD_PPMD}
)
