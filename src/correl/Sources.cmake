set(Correl_Src_Files
	${CORREL_DIR}/correl_2D.cpp
	${CORREL_DIR}/correl_init.cpp
	${CORREL_DIR}/correl_special.cpp
)

source_group(Correl FILES ${Correl_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Correl_Src_Files}
)
