set(Radiom_Src_Files
	${RADIOM_DIR}/egal_radiom.cpp
)

SOURCE_GROUP(Radiom FILES ${Radiom_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Radiom_Src_Files}
)
