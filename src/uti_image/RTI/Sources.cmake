set(uti_image_rti
    ${UTI_IMAGE_RTI_DIR}/cAppli_RTI.cpp
    ${UTI_IMAGE_RTI_DIR}/cOneImRTI.cpp
    ${UTI_IMAGE_RTI_DIR}/RTI_RecalGeom.cpp
    ${UTI_IMAGE_RTI_DIR}/RTI_RecalRadiom.cpp
    ${UTI_IMAGE_RTI_DIR}/RTI_PolynRecalRadiom.cpp
    ${UTI_IMAGE_RTI_DIR}/RTI.cpp
)


list(APPEND uti_phgrm_Src_Files
        ${uti_image_rti}
)


