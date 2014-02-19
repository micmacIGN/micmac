set(uti_phgrm_SaisiePts_Src_Files
    ${UTI_PHGRM_SAISIEPTS_DIR}/cAppliInput.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cImage.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cPointeIm.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cWCreatePt.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cAppliSaisiePts.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cParamSaisiePts.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cPointGlob.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cWinIm.cpp
    ${UTI_PHGRM_SAISIEPTS_DIR}/cX11Interface.cpp
)

#SOURCE_GROUP(uti_phgrm.Apero FILES ${uti_phgrm_Apero_Src_Files})

list( APPEND uti_phgrm_Src_Files
	${uti_phgrm_SaisiePts_Src_Files}
)


