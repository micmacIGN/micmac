set(Top_level_Src_Files
	top_level/copy.cpp
)

SOURCE_GROUP(Top_level FILES ${Top_level_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Top_level_Src_Files}
)
