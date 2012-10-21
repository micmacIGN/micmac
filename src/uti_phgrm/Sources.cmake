set(uti_phgrm_Src_Files
	${UTI_PHGRM_DIR}/CPP_AperiCloud.cpp
	${UTI_PHGRM_DIR}/CPP_Malt.cpp
)

SOURCE_GROUP(uti_phgrm FILES ${uti_phgrm_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${uti_phgrm_Src_Files}
)
