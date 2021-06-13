set(Xinterf_Src_Files
	${XINTER_DIR}/Ok_eLise.cpp
	${XINTER_DIR}/bi_scroller.cpp
	${XINTER_DIR}/bitm_win.cpp
	${XINTER_DIR}/cElImageFlipper.cpp
	${XINTER_DIR}/display.cpp
	${XINTER_DIR}/fen_graph_window.cpp
	${XINTER_DIR}/fen_windows.cpp
	${XINTER_DIR}/fen_x11.cpp
	${XINTER_DIR}/gen_win.cpp
	${XINTER_DIR}/graphics.cpp
	${XINTER_DIR}/image_interactor.cpp
	${XINTER_DIR}/imfile_scrol.cpp
	${XINTER_DIR}/incruster.cpp
	${XINTER_DIR}/integ_out.cpp
	${XINTER_DIR}/pop_up_menu_transp.cpp
	${XINTER_DIR}/rle_out.cpp
	${XINTER_DIR}/scroller.cpp
	${XINTER_DIR}/xcolour.cpp
)

source_group(Xinterf FILES ${Xinterf_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Xinterf_Src_Files}
)
