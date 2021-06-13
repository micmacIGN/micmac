set(Amd_Src_Files
	${AMD_DIR}/amd_1.cpp
	${AMD_DIR}/amd_2.cpp
	${AMD_DIR}/amd_aat.cpp
	${AMD_DIR}/amd_control.cpp
	${AMD_DIR}/amd_defaults.cpp
	${AMD_DIR}/amd_demo_1.cpp
	${AMD_DIR}/amd_dump.cpp
	${AMD_DIR}/amd_global.cpp
	${AMD_DIR}/amd_info.cpp
	${AMD_DIR}/amd_order.cpp
	${AMD_DIR}/amd_post_tree.cpp
	${AMD_DIR}/amd_postorder.cpp
	${AMD_DIR}/amd_preprocess.cpp
	${AMD_DIR}/amd_valid.cpp
)

source_group(Amd FILES ${Amd_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${Amd_Src_Files}
)
