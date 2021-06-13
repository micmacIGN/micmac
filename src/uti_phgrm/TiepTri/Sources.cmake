set(uti_image_TiePTri
    ${UTI_PHGRM_TiePTri_DIR}/cAppliTieTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/TiepTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/cImMasterTieTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/cImSecTieTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/cImTieTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/CorrelTiepTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/FineMultiCorrel.cpp
    ${UTI_PHGRM_TiePTri_DIR}/LsqCorrel.cpp
    ${UTI_PHGRM_TiePTri_DIR}/FiltrageSpatialTiePTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/cHomolPackTiepTri.cpp
    ${UTI_PHGRM_TiePTri_DIR}/MultTieP.cpp
    ${UTI_PHGRM_TiePTri_DIR}/cResulCorrelTieTri.cpp
)


list( APPEND uti_phgrm_Src_Files
        ${uti_image_TiePTri}
)


