set(File_image_Src_Files
	${FILE_IMAGE_DIR}/bmp.cpp
	${FILE_IMAGE_DIR}/elise_format.cpp
	${FILE_IMAGE_DIR}/fich_2d_gen.cpp
	${FILE_IMAGE_DIR}/gef.cpp
	${FILE_IMAGE_DIR}/gif.cpp
	${FILE_IMAGE_DIR}/pnm.cpp
	${FILE_IMAGE_DIR}/tga.cpp
)

source_group(File_image FILES ${File_image_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${File_image_Src_Files}
)
