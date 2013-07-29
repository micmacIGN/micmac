#include "GpGpu/GpGpu_Data.h"

bool  AImageCuda::bindTexture( textureReference& texRef )
{
    cudaChannelFormatDesc desc;
    bool bCha	= CData::ErrorOutput(cudaGetChannelDesc(&desc, GetCudaArray()),"Bind Texture / cudaGetChannelDesc");
    bool bBind	= CData::ErrorOutput(cudaBindTextureToArray(&texRef,GetCudaArray(),&desc),"Bind Texture / Bind");
    return bCha && bBind;
}

cudaArray* AImageCuda::GetCudaArray()
{
    return CData<cudaArray>::pData();
}

bool AImageCuda::Memset( int val )
{
    std::cout << "PAS DE MEMSET POUR CUDA ARRAY" << "\n";
    return true;
}

bool AImageCuda::abDealloc()
{
    return (cudaFreeArray( CData<cudaArray>::pData()) == cudaSuccess) ? true : false;
}
