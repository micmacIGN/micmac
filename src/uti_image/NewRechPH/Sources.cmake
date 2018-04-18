set(uti_image_NewRechPH_Src_Files
    ${UTI_IMAGE_NewRechPH_DIR}/NewRechPH.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/cOneScaleImRechPH.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/ExternCalcPRemark.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/cParamNewRechPH.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/cOSIR_Corner.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/cOSIR_SIFT.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/cOSIR_Topo.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/cOSIR_Gaussian.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/StatPHom.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/NH_InvarRad.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/Match_Binaire_Compact.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/Match_Binaire.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/Match_Image.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/Match_InvRad.cpp
    ${UTI_IMAGE_NewRechPH_DIR}/Match_ImageSecPhase.cpp
)


list( APPEND uti_image_Src_Files ${uti_image_NewRechPH_Src_Files} )
