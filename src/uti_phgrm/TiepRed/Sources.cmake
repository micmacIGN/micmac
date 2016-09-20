set(uti_image_tiepred
    ${UTI_PHGRM_TieRed_DIR}/cAppliTiepRed.cpp
    ${UTI_PHGRM_TieRed_DIR}/cAppliTiepRed_Algo.cpp
    ${UTI_PHGRM_TieRed_DIR}/cLnk2ImTiepRed.cpp
    ${UTI_PHGRM_TieRed_DIR}/cPMulTiepRed.cpp
    ${UTI_PHGRM_TieRed_DIR}/cImageTiepRed.cpp
   ${UTI_PHGRM_TieRed_DIR}/cImageGrid.cpp
)


list( APPEND uti_phgrm_Src_Files
        ${uti_image_tiepred}
)


