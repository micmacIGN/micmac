set(Hough_Src_Files
	${HOUGH_DIR}/cPtOfCorrel.cpp
	${HOUGH_DIR}/hough_algo.cpp
	${HOUGH_DIR}/hough_basic.cpp
	${HOUGH_DIR}/hough_clip.cpp
	${HOUGH_DIR}/hough_extrac_basic.cpp
	${HOUGH_DIR}/hough_file.cpp
	${HOUGH_DIR}/hough_filtr_seg_basic.cpp
	${HOUGH_DIR}/hough_gen.cpp
	${HOUGH_DIR}/hough_inst_fseg.cpp
	${HOUGH_DIR}/hough_merger.cpp
	${HOUGH_DIR}/hough_sub_pixel.cpp
)

source_group(Hough FILES ${Hough_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Hough_Src_Files}
)
