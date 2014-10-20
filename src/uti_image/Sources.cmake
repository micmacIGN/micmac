set(UTI_IMAGE_MPDCRAW_DIR ${UTI_IMAGE_DIR}/MpDcraw)
set(UTI_IMAGE_DIGEO_DIR ${UTI_IMAGE_DIR}/Digeo)
set(UTI_IMAGE_SIFT_DIR ${UTI_IMAGE_DIR}/Sift)
set(UTI_IMAGE_ANN_DIR ${UTI_IMAGE_DIR}/Ann)
set(SrcGrp_Uti_Image uti_image)

INCLUDE (${UTI_IMAGE_MPDCRAW_DIR}/Sources.cmake)
INCLUDE (${UTI_IMAGE_DIGEO_DIR}/Sources.cmake)
INCLUDE (${UTI_IMAGE_SIFT_DIR}/Sources.cmake)
INCLUDE (${UTI_IMAGE_ANN_DIR}/Sources.cmake)

list( APPEND uti_image_Src_Files ${uti_image_Sift_Src_Files})
list( APPEND uti_image_Src_Files ${uti_image_Ann_Src_Files})

list( APPEND uti_image_Src_Files
        ${UTI_IMAGE_DIR}/CPP_Vignette.cpp
        ${UTI_IMAGE_DIR}/CPP_Arsenic.cpp
        ${UTI_IMAGE_DIR}/CPP_LumRas.cpp
    ${UTI_IMAGE_DIR}/CPP_Undist.cpp
    ${UTI_IMAGE_DIR}/CPP_CoherEpi.cpp
    ${UTI_IMAGE_DIR}/QualDepthMap.cpp
    ${UTI_IMAGE_DIR}/CPP_CmpIm.cpp
    ${UTI_IMAGE_DIR}/CPP_EstimFlatField.cpp
    ${UTI_IMAGE_DIR}/CPP_Dequant.cpp
    ${UTI_IMAGE_DIR}/CPP_Devlop.cpp
    ${UTI_IMAGE_DIR}/CPP_ElDcraw.cpp
    ${UTI_IMAGE_DIR}/CPP_GenXML2Cpp.cpp
    ${UTI_IMAGE_DIR}/CPP_GenMire.cpp
    ${UTI_IMAGE_DIR}/CPP_GrShade.cpp
    ${UTI_IMAGE_DIR}/CPP_MpDcraw.cpp
    ${UTI_IMAGE_DIR}/CPP_PastDevlop.cpp
    ${UTI_IMAGE_DIR}/CPP_Reduc2MM.cpp
    ${UTI_IMAGE_DIR}/CPP_ScaleIm.cpp
    ${UTI_IMAGE_DIR}/CPP_ConvertIm.cpp
    ${UTI_IMAGE_DIR}/CPP_MakePlancheImage.cpp
    ${UTI_IMAGE_DIR}/CPP_tiff_info.cpp
    ${UTI_IMAGE_DIR}/CPP_to8Bits.cpp
    ${UTI_IMAGE_DIR}/CPP_mmxv.cpp
    ${UTI_IMAGE_DIR}/CPP_Drunk.cpp
    ${UTI_IMAGE_DIR}/CPP_Impainting.cpp
    ${UTI_IMAGE_DIR}/CPP_CalSzWCor.cpp
    ${UTI_IMAGE_DIR}/CPP_MPDtest.cpp
    ${UTI_IMAGE_DIR}/CPP_Sift.cpp
    ${UTI_IMAGE_DIR}/CPP_Ann.cpp
    ${UTI_IMAGE_DIR}/CPP_StatImage.cpp
    ${UTI_IMAGE_DIR}/CPP_SplitMPOFormat.cpp
    ${UTI_IMAGE_DIR}/CPP_SupMntIm.cpp
    ${UTI_IMAGE_DIR}/CPP_Digeo.cpp
    ${UTI_IMAGE_DIR}/CPP_DevVideo.cpp
    ${UTI_IMAGE_DIR}/CPP_SupMntIm.cpp )

SOURCE_GROUP(${SrcGrp_Uti_Image}\\outils FILES ${uti_image_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_Image}\\outils\\SIFT FILES ${uti_image_Sift_Src_Files})
SOURCE_GROUP(${SrcGrp_Uti_Image}\\outils\\ANN FILES ${uti_image_Ann_Src_Files})

list( APPEND Elise_Src_Files ${uti_image_Src_Files} )
