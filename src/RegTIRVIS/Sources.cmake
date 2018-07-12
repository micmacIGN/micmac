set(RegTIRVIS_Src_Files ${REGTIRVIS_DIR}/RegTIRVIS.cpp
                        ${REGTIRVIS_DIR}/Image.cpp
                        ${REGTIRVIS_DIR}/msd.cpp
                        ${REGTIRVIS_DIR}/Keypoint.cpp
                        ${REGTIRVIS_DIR}/msdImgPyramid.cpp
                        ${REGTIRVIS_DIR}/DescriptorExtractor.cpp
                        ${REGTIRVIS_DIR}/Arbre.cpp
)

source_group(REGTIRVIS FILES ${RegTIRVIS_Src_Files})

set(Elise_Src_Files
	${Elise_Src_Files}
	${RegTIRVIS_Src_Files}
)
