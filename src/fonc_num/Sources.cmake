set(Fonc_num_Src_Files
	${FONC_NUM_DIR}/compos.cpp
	${FONC_NUM_DIR}/coord.cpp
	${FONC_NUM_DIR}/fcteurs_uti.cpp
	${FONC_NUM_DIR}/fnum_compile.cpp
	${FONC_NUM_DIR}/fnum_compile2.cpp
	${FONC_NUM_DIR}/fnum_gen_tpl.cpp
	${FONC_NUM_DIR}/fnum_general.cpp
	${FONC_NUM_DIR}/linear_filter.cpp
	${FONC_NUM_DIR}/linear_filtre_mne.cpp
	${FONC_NUM_DIR}/linear_order.cpp
	${FONC_NUM_DIR}/linear_proj.cpp
	${FONC_NUM_DIR}/proj_brd.cpp
	${FONC_NUM_DIR}/symb_fnum.cpp
)

source_group(Fonc_num FILES ${Fonc_num_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Fonc_num_Src_Files}
)
