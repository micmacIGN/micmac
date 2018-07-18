set(Bitm_Src_Files
	${BITM_DIR}/bitm_op_ass.cpp
	${BITM_DIR}/cCpleImMasq.cpp
	${BITM_DIR}/comput.cpp
	${BITM_DIR}/comput_tpl.cpp
	${BITM_DIR}/flx_out_im2.cpp
	${BITM_DIR}/font_bitm.cpp
	${BITM_DIR}/im2d_bits.cpp
	${BITM_DIR}/im2d_reech_grid.cpp
	${BITM_DIR}/im2d_tpl.cpp
	${BITM_DIR}/im3d_tpl.cpp
	${BITM_DIR}/im_compr.cpp
	${BITM_DIR}/imxd.cpp
	${BITM_DIR}/lpts.cpp
	${BITM_DIR}/matrix.cpp
	${BITM_DIR}/pj_eq234.cpp
	${BITM_DIR}/polyn2DReel.cpp
	${BITM_DIR}/polynome.cpp
	${BITM_DIR}/real_poly.cpp
	${BITM_DIR}/scale_im_compr.cpp
)

source_group(Amd FILES ${Bitm_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Bitm_Src_Files}
)
