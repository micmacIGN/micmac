set(Neighboor_Src_Files
	${NEIGH_DIR}/b2d_spec_neigh.cpp
	${NEIGH_DIR}/dilat_conc.cpp
	${NEIGH_DIR}/neigh_filter.cpp
	${NEIGH_DIR}/neigh_general.cpp
	${NEIGH_DIR}/red_op_neigh.cpp
)

SOURCE_GROUP(Neighboor FILES ${Neighboor_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Neighboor_Src_Files}
)
