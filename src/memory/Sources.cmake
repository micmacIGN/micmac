set(Memory_Src_Files
	${MEMORY_DIR}/cpt_ref.cpp
	${MEMORY_DIR}/liste.cpp
	${MEMORY_DIR}/new_cpt_ref.cpp
	${MEMORY_DIR}/smart_pointeur.cpp
	${MEMORY_DIR}/tab_prov.cpp
)

source_group(Memory FILES ${Memory_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Memory_Src_Files}
)
