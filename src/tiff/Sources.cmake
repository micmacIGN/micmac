set(Tiff_Src_Files
	${TIFF_DIR}/cLazyTiffFile.cpp
	${TIFF_DIR}/tiff_args_opt.cpp
	${TIFF_DIR}/tiff_compr_tiles.cpp
	${TIFF_DIR}/tiff_header.cpp
	${TIFF_DIR}/tiff_nc_tiles.cpp
	${TIFF_DIR}/tiff_pckbit_un_load.cpp
	${TIFF_DIR}/tiff_tags.cpp
	#${TIFF_DIR}/el_dcraw.c
)

source_group(Tiff FILES ${Tiff_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Tiff_Src_Files}
)
