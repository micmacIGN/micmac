set(Algo_speciaux_Src_Files
	${ALOG_SPE_DIR}/deriche.cpp
	${ALOG_SPE_DIR}/hongr.cpp
	${ALOG_SPE_DIR}/hongrois.cpp
	${ALOG_SPE_DIR}/opb_deb_phase.cpp
	${ALOG_SPE_DIR}/FiltreDepthMaps.cpp
)

source_group(Algo_speciaux FILES ${Algo_speciaux_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Algo_speciaux_Src_Files}
)
