set(OriPhoto_Src_Files
	${ORIPHO_DIR}/astro.cpp
	#${ORIPHO_DIR}/cOriMntCarto.cpp
	${ORIPHO_DIR}/elise_fonc.cpp
	${ORIPHO_DIR}/elise_interface.cpp
	${ORIPHO_DIR}/filmdist.cpp
	${ORIPHO_DIR}/lambgeo.cpp
	${ORIPHO_DIR}/matrices.cpp
	${ORIPHO_DIR}/orilib.cpp
	${ORIPHO_DIR}/orisol.cpp
)

source_group(OriPhoto FILES ${OriPhoto_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${OriPhoto_Src_Files}
)
