set(UTI_PHGRM_APERO_DIR ${UTI_PHGRM_DIR}/Apero)
set(UTI_PHGRM_MICMAC_DIR ${UTI_PHGRM_DIR}/MICMAC)
set(UTI_PHGRM_GRAPHCUT_DIR ${UTI_PHGRM_DIR}/GraphCut)
set(UTI_PHGRM_REDUCHOM_DIR ${UTI_PHGRM_DIR}/ReducHom)
set(UTI_PHGRM_RHH_DIR ${UTI_PHGRM_DIR}/RHH)
set(UTI_PHGRM_CASA_DIR ${UTI_PHGRM_DIR}/CASA)
set(UTI_PHGRM_TieRed_DIR ${UTI_PHGRM_DIR}/TiepRed)
set(UTI_PHGRM_OriTieRed_DIR ${UTI_PHGRM_DIR}/OriTiepRed)
set(UTI_PHGRM_TieGeo_DIR ${UTI_PHGRM_DIR}/TiepGeoref)

set(UTI_PHGRM_TiePTri_DIR ${UTI_PHGRM_DIR}/TiepTri)
set(UTI_PHGRM_TiePHisto_DIR ${UTI_PHGRM_DIR}/TiePHistorical)

set(UTI_PHGRM_PORTO_DIR ${UTI_PHGRM_DIR}/Porto)
set(UTI_PHGRM_SAISIEPTS_DIR ${UTI_PHGRM_DIR}/SaisiePts)
set(UTI_PHGRM_GPGPU_DIR ${UTI_PHGRM_DIR}/GpGpu)

set(UTI_PHGRM_FUSION_NUAGES ${UTI_PHGRM_DIR}/FusionNuage)
set(UTI_PHGRM_MERGE_CLOUD ${UTI_PHGRM_DIR}/MergeCloud)
set(UTI_PHGRM_NEW_ORI ${UTI_PHGRM_DIR}/NewOri)
set(UTI_PHGRM_SAT_PHYS_MOD ${UTI_PHGRM_DIR}/SatPhysMod)
set(UTI_PHGRM_TEXT_DIR ${UTI_PHGRM_DIR}/TexturePacker)

set(UTI_PHGRM_MAXFLOW_DIR ${UTI_PHGRM_GRAPHCUT_DIR}/MaxFlow)
set(UTI_PHGRM_QPBO_DIR ${UTI_PHGRM_GRAPHCUT_DIR}/QPBO-v1.4)
set(UTI_PHGRM_SAT4GEO_DIR ${UTI_PHGRM_DIR}/SAT4GEO)

set(SrcGrp_Uti_PHGRM uti_phgrm)
set(SrcGrp_Graph_Cut uti_phgrm/GraphCut)

include(${UTI_PHGRM_APERO_DIR}/Sources.cmake)
include(${UTI_PHGRM_MICMAC_DIR}/Sources.cmake)
include(${UTI_PHGRM_MAXFLOW_DIR}/Sources.cmake)
include(${UTI_PHGRM_QPBO_DIR}/Sources.cmake)
include(${UTI_PHGRM_REDUCHOM_DIR}/Sources.cmake)
include(${UTI_PHGRM_RHH_DIR}/Sources.cmake)
include(${UTI_PHGRM_PORTO_DIR}/Sources.cmake)
include(${UTI_PHGRM_SAISIEPTS_DIR}/Sources.cmake)
include(${UTI_PHGRM_FUSION_NUAGES}/Sources.cmake)
include(${UTI_PHGRM_MERGE_CLOUD}/Sources.cmake)
include(${UTI_PHGRM_CASA_DIR}/Sources.cmake)
include(${UTI_PHGRM_TieRed_DIR}/Sources.cmake)
include(${UTI_PHGRM_OriTieRed_DIR}/Sources.cmake)
include(${UTI_PHGRM_TieGeo_DIR}/Sources.cmake)
include(${UTI_PHGRM_NEW_ORI}/Sources.cmake)
include(${UTI_PHGRM_SAT_PHYS_MOD}/Sources.cmake)
include(${UTI_PHGRM_TEXT_DIR}/Sources.cmake)
include(${UTI_PHGRM_TiePTri_DIR}/Sources.cmake)
include(${UTI_PHGRM_SAT4GEO_DIR}/Sources.cmake)
include(${UTI_PHGRM_TiePHisto_DIR}/Sources.cmake)

#define __CUDA_API_VERSION 0x5050

if(${CUDA_ENABLED})
	set(OptionCuda 1)

#        if("${CUDA_VERSION}" MATCHES "6.0")
#            set(__CUDA_API_VERSION 0x6000)
#        elseif("${CUDA_VERSION}" MATCHES "5.5")
#            set(__CUDA_API_VERSION 0x5050)
#        elseif("${CUDA_VERSION}" MATCHES "5.0")
#            set(__CUDA_API_VERSION 0x5000)
#        elseif("${CUDA_VERSION}" MATCHES "4.0")
#            set(__CUDA_API_VERSION 0x4000)
#            elseif("${CUDA_VERSION}" MATCHES "3.2")
#            set(__CUDA_API_VERSION 0x3020)
#        elseif("${CUDA_VERSION}" MATCHES "3.0")
#            set(__CUDA_API_VERSION 0x3000)
#        endif()

    INCLUDE (${UTI_PHGRM_GPGPU_DIR}/Sources.cmake)
else()
    set(OptionCuda 0)

#    set(__CUDA_API_VERSION 0x0000)

endif()

if(${WITH_OPENCL})
    set(OPENCL_ENABLED 1)
else()
    set(OPENCL_ENABLED  0)
endif()

if(${CUDA_CPP11THREAD_NOBOOSTTHREAD})
    set(CPP11THREAD_NOBOOSTTHREAD 1)
else()
    set(CPP11THREAD_NOBOOSTTHREAD  0)
endif()

if(${CUDA_NVTOOLS})
    set(NVTOOLS 1)
else()
    set(NVTOOLS  0)
endif()

if(${WITH_OPEN_MP})
    set(USE_OPEN_MP 1)
else()
    set(USE_OPEN_MP 0)
endif()

configure_file(
    ${UTI_PHGRM_GPGPU_DIR}/GpGpu_BuildOptions.h.in
    ${PROJECT_SOURCE_DIR}/include/GpGpu/GpGpu_BuildOptions.h
)


set( Applis_phgrm_Src_Files
    ${UTI_PHGRM_DIR}/CPP_ChamVec3D.cpp
    ${UTI_PHGRM_DIR}/CPP_HomolFromProfEtPx.cpp
    ${UTI_PHGRM_DIR}/CPP_NuageBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_ReechInvEpip.cpp
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
    ${UTI_PHGRM_DIR}/CPP_MMTestOrient.cpp
    ${UTI_PHGRM_DIR}/CPP_MMHomCorOri.cpp
    ${UTI_PHGRM_DIR}/CPP_CmpCalib.cpp
    ${UTI_PHGRM_DIR}/CPP_Gri2Bin.cpp
    ${UTI_PHGRM_DIR}/CPP_GCPBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_Block.cpp
    ${UTI_PHGRM_DIR}/CPP_Stereopolis.cpp
    ${UTI_PHGRM_DIR}/CPP_CenterBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_MakeGrid.cpp
    ${UTI_PHGRM_DIR}/CPP_Malt.cpp
    ${UTI_PHGRM_DIR}/CPP_MMByPair.cpp
    ${UTI_PHGRM_DIR}/CPP_MergePly.cpp
    ${UTI_PHGRM_DIR}/CPP_MICMAC.cpp
    ${UTI_PHGRM_DIR}/CPP_Nuage2Ply.cpp
    ${UTI_PHGRM_DIR}/CPP_Nuage2Homol.cpp
    ${UTI_PHGRM_DIR}/CPP_Pasta.cpp
    ${UTI_PHGRM_DIR}/CPP_Pastis.cpp
    ${UTI_PHGRM_DIR}/CPP_Porto.cpp
    ${UTI_PHGRM_DIR}/CPP_ReducHom.cpp
    ${UTI_PHGRM_DIR}/CPP_RepLocBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_ScaleNuage.cpp
    ${UTI_PHGRM_DIR}/CPP_SBGlobBascule.cpp
    ${UTI_PHGRM_DIR}/CPP_Tapas.cpp
    ${UTI_PHGRM_DIR}/CPP_NewTapas.cpp
    ${UTI_PHGRM_DIR}/CPP_Tapioca.cpp
    ${UTI_PHGRM_DIR}/CPP_Tarama.cpp
    ${UTI_PHGRM_DIR}/CPP_Tawny.cpp
    ${UTI_PHGRM_DIR}/CPP_Tequila.cpp
    ${UTI_PHGRM_DIR}/CPP_TestCam.cpp
    ${UTI_PHGRM_DIR}/CPP_TestChantier.cpp
    ${UTI_PHGRM_DIR}/CPP_TiPunch.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieMasq.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieQT.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieAppuisPredic.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieAppuisInit.cpp
    ${UTI_PHGRM_DIR}/CPP_SaisieBasc.cpp
    ${UTI_PHGRM_DIR}/CPP_SysCoordPolyn.cpp
    ${UTI_PHGRM_DIR}/CPP_ChgSysCo.cpp
    ${UTI_PHGRM_DIR}/CPP_XYZ2Im.cpp
    ${UTI_PHGRM_DIR}/CPP_GrapheHom.cpp
    ${UTI_PHGRM_DIR}/CPP_MMOnePair.cpp
    ${UTI_PHGRM_DIR}/CPP_BasicEpip.cpp
    ${UTI_PHGRM_DIR}/CPP_VisuCoupeEpip.cpp
    ${UTI_PHGRM_DIR}/CPP_HomFilterMasq.cpp
    ${UTI_PHGRM_DIR}/CPP_InitCamFromAppuis.cpp
    ${UTI_PHGRM_DIR}/CPP_Sake.cpp
    ${UTI_PHGRM_DIR}/CPP_Liquor.cpp
    ${UTI_PHGRM_DIR}/CPP_Luxor.cpp 
    ${UTI_PHGRM_DIR}/CPP_Morito.cpp
    ${UTI_PHGRM_DIR}/CPP_C3DC.cpp
    ${UTI_PHGRM_DIR}/CPP_GIMMI.cpp
    ${UTI_PHGRM_DIR}/CPP_Bundler2MM.cpp
    ${UTI_PHGRM_DIR}/CPP_MM2OpenMVG.cpp
    ${UTI_PHGRM_DIR}/CPP_MMToAerial.cpp
    ${UTI_PHGRM_DIR}/CPP_Sat3DP.cpp
    ${UTI_PHGRM_DIR}/CPP_Line3D.cpp
    ${UTI_PHGRM_DIR}/CPP_TiePHistoP.cpp
)



  
  


SOURCE_GROUP(${SrcGrp_Uti_PHGRM} FILES ${uti_phgrm_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Applis FILES ${Applis_phgrm_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Apero FILES ${uti_phgrm_Apero_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\MicMac FILES ${uti_phgrm_MICMAC_Src_Files})
SOURCE_GROUP(${SrcGrp_Graph_Cut}\\MaxFlow FILES ${uti_phgrm_MaxFlow_Src_Files})
SOURCE_GROUP(${SrcGrp_Graph_Cut}\\QPBO FILES ${uti_phgrm_qpbo_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\Porto FILES ${uti_phgrm_Porto_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\ReducHom FILES ${uti_phgrm_Porto_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_PHGRM}\\SAT4GEO FILES ${uti_phgrm_Sat4Geo_Src_Files})


if(${CUDA_ENABLED})
	source_group(${SrcGrp_Uti_PHGRM}\\GpGpu FILES ${uti_phgrm_GpGpu_Src_Files})
	source_group(${SrcGrp_Uti_PHGRM}\\GpGpu FILES ${GpGpuTools_Src_Files})
endif()

list(APPEND uti_phgrm_Src_Files ${Applis_phgrm_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_Apero_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_MICMAC_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_MaxFlow_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_qpbo_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_Porto_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_ReducHom_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_RHH_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_Casa_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_Text_Src_Files})
list(APPEND uti_phgrm_Src_Files ${uti_phgrm_Sat4Geo_Src_Files})
list(APPEND Elise_Src_Files ${uti_phgrm_Src_Files})
