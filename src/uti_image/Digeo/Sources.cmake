set(uti_image_Digeo_Src_Files
    ${UTI_IMAGE_DIGEO_DIR}/cParamDigeo.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cAppliDigeo.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cConvolSpec.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cDigeo_Topo.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cImDigeo.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cImInMem.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cOctaveDigeo.cpp
    ${UTI_IMAGE_DIGEO_DIR}/Digeo.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/cVisuCarac.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/Digeo_Detecteurs.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/Digeo_GaussFilter.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/Digeo_Pyram.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/GenConvolSpec.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/TestDigeoExt.cpp
    ${UTI_IMAGE_DIGEO_DIR}/DigeoPoint.cpp
    ${UTI_IMAGE_DIGEO_DIR}/Expression.cpp
    ${UTI_IMAGE_DIGEO_DIR}/Times.cpp
    ${UTI_IMAGE_DIGEO_DIR}/MultiChannel.cpp
    ${UTI_IMAGE_DIGEO_DIR}/GaussianConvolutionKernel1D.cpp
#    ${UTI_IMAGE_DIGEO_DIR}/Convolution.cpp
    ${UTI_IMAGE_DIGEO_DIR}/cConvolSpec.cpp
    ${UTI_IMAGE_DIGEO_DIR}/ConvolutionKernel1D.cpp
)


list( APPEND uti_image_Src_Files ${uti_image_Digeo_Src_Files} )
