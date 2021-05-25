set(Morpho_cabl_Src_Files
	${MORPHO_DIR}/chamfer_dist.cpp
	${MORPHO_DIR}/dequantif.cpp
	${MORPHO_DIR}/morph_rle.cpp
	${MORPHO_DIR}/skel_interface.cpp
	${MORPHO_DIR}/skel_vein.cpp
	${MORPHO_DIR}/graphe_region.cpp
	${MORPHO_DIR}/class_morpho.cpp
)

source_group(Morpho FILES ${Morpho_cabl_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Morpho_cabl_Src_Files}
)
