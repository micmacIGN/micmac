set(UTI_PHGRM_APERO_DIR ${UTI_PHGRM_DIR}/Apero)
set(UTI_PHGRM_MICMAC_DIR ${UTI_PHGRM_DIR}/MICMAC)
set(UTI_PHGRM_MAXFLOW_DIR ${UTI_PHGRM_DIR}/MaxFlow)
set(UTI_PHGRM_REDUCHOM_DIR ${UTI_PHGRM_DIR}/ReducHom)
set(UTI_PHGRM_PORTO_DIR ${UTI_PHGRM_DIR}/Porto)
set(UTI_PHGRM_SAISIEPTS_DIR ${UTI_PHGRM_DIR}/SaisiePts)
set(UTI_PHGRM_GPGPU_DIR ${UTI_PHGRM_DIR}/GpGpu)

set(UTI_PHGRM_FUSION_NUAGES ${UTI_PHGRM_DIR}/FusionNuage)

set(SrcGrp_Uti_PHGRM uti_phgrm)

INCLUDE (${UTI_PHGRM_APERO_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_MICMAC_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_MAXFLOW_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_REDUCHOM_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_PORTO_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_SAISIEPTS_DIR}/Sources.cmake)
INCLUDE (${UTI_PHGRM_FUSION_NUAGES}/Sources.cmake)

if(${CUDA_ENABLED})
    INCLUDE (${UTI_PHGRM_GPGPU_DIR}/Sources.cmake)
    configure_file(
        ${UTI_PHGRM_GPGPU_DIR}/GpGpuDefines.h.in
        ${PROJECT_SOURCE_DIR}/include/GpGpu/GpGpuDefines.h
    )

else()
    configure_file(
        ${UTI_PHGRM_GPGPU_DIR}/GpGpuNoDefines.h.in
        ${PROJECT_SOURCE_DIR}/include/GpGpu/GpGpuDefines.h
    )
endif()

set( Applis_phgrm_Src_Files
    ${UTI_PHGRM_DIR}/CPP_NuageBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_CreateEpip.cpp
    ${UTI_PHGRM_DIR}/CPP_VideoVisage.cpp
    ${UTI_PHGRM_DIR}/CPP_SEL.cpp
    ${UTI_PHGRM_DIR}/CPP_AperiCloud.cpp
    ${UTI_PHGRM_DIR}/CPP_AperoChImMM.cpp
    ${UTI_PHGRM_DIR}/CPP_MMPyram.cpp
    ${UTI_PHGRM_DIR}/CPP_MMModelInitial.cpp
    ${UTI_PHGRM_DIR}/CPP_MMAllAuto.cpp
    ${UTI_PHGRM_DIR}/CPP_MM2DPostSism.cpp
    ${UTI_PHGRM_DIR}/CPP_Apero.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisiePts.cpp
    ${UTI_PHGRM_DIR}/CPP_Bascule.cpp
    ${UTI_PHGRM_DIR}/CPP_Campari.cpp
    ${UTI_PHGRM_DIR}/CPP_CmpCalib.cpp
    ${UTI_PHGRM_DIR}/CPP_Gri2Bin.cpp
    ${UTI_PHGRM_DIR}/CPP_GCPBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_CenterBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_MakeGrid.cpp
    ${UTI_PHGRM_DIR}/CPP_Malt.cpp
    ${UTI_PHGRM_DIR}/CPP_Mascarpone.cpp
    ${UTI_PHGRM_DIR}/CPP_MergePly.cpp
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
    ${UTI_PHGRM_DIR}/CPP_ChgSysCo.cpp
    ${UTI_PHGRM_DIR}/CPP_XYZ2Im.cpp
)

SOURCE_GROUP(${SrcGrp_Uti_PHGRM} FILES ${uti_phgrm_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Applis FILES ${Applis_phgrm_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Apero FILES ${uti_phgrm_Apero_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\MicMac FILES ${uti_phgrm_MICMAC_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\MaxFlow FILES ${uti_phgrm_MaxFlow_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Porto FILES ${uti_phgrm_Porto_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\ReducHom FILES ${uti_phgrm_Porto_Src_Files})

if(${CUDA_ENABLED})
        SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\GpGpu FILES ${uti_phgrm_GpGpu_Src_Files})
        SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\GpGpu FILES ${GpGpuTools_Src_Files})
endif()

list( APPEND uti_phgrm_Src_Files ${Applis_phgrm_Src_Files})
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_Apero_Src_Files})
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_MICMAC_Src_Files} )
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_MaxFlow_Src_Files} )
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_Porto_Src_Files})
list( APPEND uti_phgrm_Src_Files ${uti_phgrm_ReducHom_Src_Files})

list( APPEND Elise_Src_Files ${uti_phgrm_Src_Files})
