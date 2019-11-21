set(PostProcessing_Src_Files
	${POST_PROCESSING_DIR}/CPP_Banana.cpp
)

# JE NE SAIS PAS SI CA SERT ???
source_group(PostProcessing FILES ${PostProcessing_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${PostProcessing_Src_Files}
)
