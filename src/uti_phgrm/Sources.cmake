set(UTI_PHGRM_APERO_DIR ${UTI_PHGRM_DIR}/Apero)
set(UTI_PHGRM_MICMAC_DIR ${UTI_PHGRM_DIR}/MICMAC)
set(UTI_PHGRM_REDUCHOM_DIR ${UTI_PHGRM_DIR}/ReducHom)
set(UTI_PHGRM_PORTO_DIR ${UTI_PHGRM_DIR}/Porto)

INCLUDE (${UTI_PHGRM_APERO_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_MICMAC_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_REDUCHOM_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_PORTO_DIR}/Sources.cmake)

list( APPEND uti_phgrm_Src_Files
    ${UTI_PHGRM_DIR}/CPP_AperiCloud.cpp
    ${UTI_PHGRM_DIR}/CPP_Apero.cpp
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
)

SOURCE_GROUP(uti_phgrm FILES ${uti_phgrm_Src_Files})

list( APPEND Elise_Src_Files
	${uti_phgrm_Src_Files}
)
