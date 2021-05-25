set(uti_image_vino
    ${UTI_IMAGE_VINO_DIR}/Vino.cpp
    ${UTI_IMAGE_VINO_DIR}/cAppliVino.cpp
    ${UTI_IMAGE_VINO_DIR}/Vino_Geom.cpp
    ${UTI_IMAGE_VINO_DIR}/Extern_Vino.cpp
    ${UTI_IMAGE_VINO_DIR}/Extern_XmlX11.cpp
    ${UTI_IMAGE_VINO_DIR}/Vino_PopUp.cpp
    ${UTI_IMAGE_VINO_DIR}/Vino_Radiom.cpp
    ${UTI_IMAGE_VINO_DIR}/Vino_Messages.cpp
    ${UTI_IMAGE_VINO_DIR}/MMVII_Test.cpp
    ${UTI_IMAGE_VINO_DIR}/MMVII_Visu.cpp
)


list(APPEND uti_phgrm_Src_Files
        ${uti_image_vino}
)


