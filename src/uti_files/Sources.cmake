set(uti_files_Src_Files
    ${UTI_FILES_DIR}/CPP_MapCmd.cpp
    ${UTI_FILES_DIR}/CPP_BatchFDC.cpp
    ${UTI_FILES_DIR}/CPP_MyRename.cpp
    ${UTI_FILES_DIR}/CPP_cod.cpp
)

SOURCE_GROUP(uti_files FILES ${uti_files_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${uti_files_Src_Files}
)
