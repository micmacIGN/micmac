set(Output_Src_Files
	${OUTPUT_DIR}/oper.cpp
	${OUTPUT_DIR}/out_general.cpp
	${OUTPUT_DIR}/reduction.cpp
	${OUTPUT_DIR}/transform_intern.cpp
)

source_group(Output FILES ${Output_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Output_Src_Files}
)
