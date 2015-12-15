set(PostScript_Src_Files
	${PSC_DIR}/disp.cpp
	${PSC_DIR}/dxf.cpp
	${PSC_DIR}/filters.cpp
	${PSC_DIR}/iocomp.cpp
	${PSC_DIR}/prim_graph.cpp
	${PSC_DIR}/win_bitm.cpp
	${PSC_DIR}/wind.cpp
)

SOURCE_GROUP(PostScript FILES ${PostScript_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${PostScript_Src_Files}
)
