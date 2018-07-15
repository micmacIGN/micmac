set(Op_buf_Src_Files
	${OP_BUF_DIR}/opb_assoc.cpp
	${OP_BUF_DIR}/opb_cany.cpp
	${OP_BUF_DIR}/opb_chamfer.cpp
	${OP_BUF_DIR}/opb_chscale.cpp
	${OP_BUF_DIR}/opb_conc.cpp
	${OP_BUF_DIR}/opb_etiq.cpp
	${OP_BUF_DIR}/opb_flag.cpp
	${OP_BUF_DIR}/opb_fonc_a_trou.cpp
	${OP_BUF_DIR}/opb_general.cpp
	${OP_BUF_DIR}/opb_max_loc_dir.cpp
	${OP_BUF_DIR}/opb_ord.cpp
	${OP_BUF_DIR}/opb_pts_interets.cpp
	${OP_BUF_DIR}/opb_simple.cpp
	${OP_BUF_DIR}/opb_skel.cpp
	${OP_BUF_DIR}/opb_som_rvar.cpp
	${OP_BUF_DIR}/users_opb.cpp
)

source_group(Op_buf FILES ${Op_buf_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Op_buf_Src_Files}
)
