set(uti_image_Oritiepred
    ${UTI_PHGRM_OriTieRed_DIR}/cOriAppliTieRed.cpp
    ${UTI_PHGRM_OriTieRed_DIR}/cOriAppliTieRed_Algo.cpp
    ${UTI_PHGRM_OriTieRed_DIR}/cOriCameraTiepRed.cpp
    ${UTI_PHGRM_OriTieRed_DIR}/cOriLnk2ImTiepRed.cpp
    ${UTI_PHGRM_OriTieRed_DIR}/cOriPMulTiepRed.cpp
    ${UTI_PHGRM_OriTieRed_DIR}/cOriVonGruber_TieRed.cpp
    ${UTI_PHGRM_OriTieRed_DIR}/cRedTieDepFromGraph.cpp
)


list( APPEND uti_phgrm_Src_Files
        ${uti_image_Oritiepred}
)


