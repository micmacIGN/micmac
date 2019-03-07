set(Recipes_Src_Files
	${RECIPES_DIR}/refft.cpp
	${RECIPES_DIR}/regaussj.cpp
	${RECIPES_DIR}/rejacobi.cpp
	${RECIPES_DIR}/resparse.cpp
)

source_group(Recipes FILES ${Recipes_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Recipes_Src_Files}
)
