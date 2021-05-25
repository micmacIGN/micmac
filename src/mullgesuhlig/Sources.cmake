set(Mullgesuhlig_Src_Files
	${MULLGES_DIR}/mubasic.cpp
	${MULLGES_DIR}/muflaguer.cpp
	${MULLGES_DIR}/mufmueller.cpp
	${MULLGES_DIR}/muvmblock.cpp
)

source_group(Mullgesuhlig FILES ${Mullgesuhlig_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Mullgesuhlig_Src_Files}
)
