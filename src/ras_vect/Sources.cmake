set(Ras_vect_Files
	${RAS_DIR}/prolong_seg_dr.cpp
	${RAS_DIR}/seg_dr.cpp
)

source_group(Ras_vect FILES ${Ras_vect_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Ras_vect_Files}
)
