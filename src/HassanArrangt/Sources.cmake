set(HassanArrangt_Src_Files
	${HASSA_DIR}/cElHJSol3D.cpp
	${HASSA_DIR}/cElHJaArrangt.cpp
	${HASSA_DIR}/cElHJaArrangt_Visu.cpp
	${HASSA_DIR}/cElHJaAttrArcPlani.cpp
	${HASSA_DIR}/cElHJaAttrSomPlani.cpp
	${HASSA_DIR}/cElHJaDroite.cpp
	${HASSA_DIR}/cElHJaFacette.cpp
	${HASSA_DIR}/cElHJaPlan3D.cpp
	${HASSA_DIR}/cElHJaPoint.cpp
	${HASSA_DIR}/cElHJaSomEmpr.cpp
	${HASSA_DIR}/cElHJa_InstanceGraphe.cpp
	${HASSA_DIR}/cFullSubGrWithP.cpp
)

source_group(HassanArrangt FILES ${HassanArrangt_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${HassanArrangt_Src_Files}
)
