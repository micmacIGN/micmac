set(UTI_PHGRM_APERO_DIR ${UTI_PHGRM_DIR}/Apero)
set(UTI_PHGRM_MICMAC_DIR ${UTI_PHGRM_DIR}/MICMAC)
set(UTI_PHGRM_REDUCHOM_DIR ${UTI_PHGRM_DIR}/ReducHom)
set(UTI_PHGRM_PORTO_DIR ${UTI_PHGRM_DIR}/Porto)
set(UTI_PHGRM_SAISIEPTS_DIR ${UTI_PHGRM_DIR}/SaisiePts)

set(SrcGrp_Uti_PHGRM uti_phgrm)


INCLUDE (${UTI_PHGRM_APERO_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_MICMAC_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_REDUCHOM_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_PORTO_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_SAISIEPTS_DIR}/Sources.cmake)

set( Applis_phgrm_Src_Files
    ${UTI_PHGRM_DIR}/CPP_AperiCloud.cpp
    ${UTI_PHGRM_DIR}/CPP_Apero.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisiePts.cpp
    ${UTI_PHGRM_DIR}/CPP_Bascule.cpp
    ${UTI_PHGRM_DIR}/CPP_CmpCalib.cpp
    ${UTI_PHGRM_DIR}/CPP_Gri2Bin.cpp
    ${UTI_PHGRM_DIR}/CPP_GCPBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_MakeGrid.cpp
    ${UTI_PHGRM_DIR}/CPP_Malt.cpp
    ${UTI_PHGRM_DIR}/CPP_MICMAC.cpp
    ${UTI_PHGRM_DIR}/CPP_Nuage2Ply.cpp
    ${UTI_PHGRM_DIR}/CPP_Pasta.cpp
    ${UTI_PHGRM_DIR}/CPP_Pastis.cpp
    ${UTI_PHGRM_DIR}/CPP_Porto.cpp
    ${UTI_PHGRM_DIR}/CPP_ReducHom.cpp
    ${UTI_PHGRM_DIR}/CPP_RepLocBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_ScaleNuage.cpp
    ${UTI_PHGRM_DIR}/CPP_SBGlobBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_Tapas.cpp
    ${UTI_PHGRM_DIR}/CPP_Tapioca.cpp
    ${UTI_PHGRM_DIR}/CPP_Tarama.cpp
    ${UTI_PHGRM_DIR}/CPP_Tawny.cpp
    ${UTI_PHGRM_DIR}/CPP_TestCam.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieMasq.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieAppuisPredic.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieAppuisInit.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieBasc.cpp
)

SOURCE_GROUP(${SrcGrp_Uti_PHGRM} FILES ${uti_phgrm_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Applis FILES ${Applis_phgrm_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Apero FILES ${uti_phgrm_Apero_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\MicMac FILES ${uti_phgrm_MICMAC_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Porto FILES ${uti_phgrm_Porto_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\ReducHom FILES ${uti_phgrm_Porto_Src_Files})


list( APPEND uti_phgrm_Src_Files ${Applis_phgrm_Src_Files})
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_Apero_Src_Files})
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_MICMAC_Src_Files} )
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_Porto_Src_Files})
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_ReducHom_Src_Files})

list( APPEND Elise_Src_Files ${uti_phgrm_Src_Files})
