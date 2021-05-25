set(Api_Src_Files
	${API_DIR}/vecto_hough.cpp
	${API_DIR}/vecto_skel.cpp

)

source_group(Api FILES ${Api_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Api_Src_Files}
)
